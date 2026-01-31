import os
import time
import torch
import psutil
import numpy as np
from datasets import load_dataset
from evaluate import load
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Imports for Quantization
import onnx
import torch.ao.quantization
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.onnx import export
import onnxruntime as ort
from transformers.utils.fx import symbolic_trace
from onnxruntime.quantization import quantize_dynamic, QuantType
# from onnxruntime.quantization.preprocess import quant_pre_process # Failed

# Set parallelism for ONNX Runtime to avoid contention
os.environ["OMP_NUM_THREADS"] = "1"

# Set PyTorch Quantization Engine for Apple Silicon
if os.uname().sysname == "Darwin" and os.uname().machine == "arm64":
    torch.backends.quantized.engine = "qnnpack"


def get_memory_usage():
    """获取当前进程的内存占用（MB）"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def measure_latency(model_fn, inputs, n_warmup=5, n_runs=20):
    """
    测量推理延迟
    model_fn: 一个接受 inputs 的可调用对象
    inputs: 模型输入
    """
    # Warmup
    for _ in range(n_warmup):
        model_fn(inputs)
    
    start_time = time.time()
    for _ in range(n_runs):
        model_fn(inputs)
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / n_runs * 1000
    return avg_time_ms

def evaluate_torch_model(model, tokenizer, dataset, metric):
    model.eval()
    all_preds = []
    all_labels = []
    print("Evaluating Torch model...")
    for example in dataset:
        inputs = tokenizer(example["text"], return_tensors="pt", padding=True, truncation=True)
        # Move inputs to same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad(): # 什么是no_grad
            logits = model(**inputs).logits
        pred = logits.argmax(dim=-1).item()
        all_preds.append(pred)
        all_labels.append(example["label"])
    return metric.compute(predictions=all_preds, references=all_labels)["accuracy"]

def evaluate_onnx_model(ort_session, tokenizer, dataset, metric):
    all_preds = []
    all_labels = []
    print("Evaluating ONNX model...")
    input_name = ort_session.get_inputs()[0].name
    
    for example in dataset:
        inputs = tokenizer(example["text"], return_tensors="np", padding=True, truncation=True)
        # ONNX Runtime expects numpy arrays
        ort_inputs = {input_name: inputs["input_ids"]}
        if len(ort_session.get_inputs()) > 1:
             # Add attention mask if model expects it
             ort_inputs["attention_mask"] = inputs["attention_mask"]

        logits = ort_session.run(None, ort_inputs)[0]
        pred = np.argmax(logits, axis=-1)[0]
        all_preds.append(pred)
        all_labels.append(example["label"])
    return metric.compute(predictions=all_preds, references=all_labels)["accuracy"]

# --- Main Execution ---

# 1. Setup
model_name = "textattack/distilbert-base-uncased-AG-News"
tokenizer = AutoTokenizer.from_pretrained(model_name) # 自动加载？
dataset = load_dataset("ag_news", split="test[:200]") # 取出测试数据集 Small subset for quick testing
metric = load("accuracy") # 确定评估函数

# Prepare dummy input for latency measurement and ONNX export
dummy_text = ["This is a test sentence for latency measurement."]
dummy_inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True)

print("="*50)
print("1. Baseline FP32 Model")
print("="*50)

# Load FP32
start_mem = get_memory_usage()
model_fp32 = AutoModelForSequenceClassification.from_pretrained(model_name) # 看下这个怎么调用
mem_fp32 = get_memory_usage() - start_mem
print(f"FP32 Memory Footprint: {mem_fp32:.2f} MB")

# Latency
def run_fp32(inputs):
    with torch.no_grad():
        model_fp32(**inputs)

lat_fp32 = measure_latency(run_fp32, dummy_inputs)
print(f"FP32 Latency: {lat_fp32:.2f} ms")

# Accuracy
acc_fp32 = evaluate_torch_model(model_fp32, tokenizer, dataset, metric)
print(f"FP32 Accuracy: {acc_fp32:.4f}")


print("\n" + "="*50)
print("2. Torch Dynamic Quantization (Int8)")
print("="*50)

# Apply Dynamic Quantization
# Quantize only Linear layers (weights) to int8
model_int8 = torch.ao.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},
    dtype=torch.qint8
)

mem_int8 = get_memory_usage() - start_mem # Approximate, not perfect due to shared process
print(f"Torch Int8 Memory Footprint (Approx process diff): {mem_int8:.2f} MB")
# Check model size on disk/structure size
torch.save(model_int8.state_dict(), "temp_int8.pt")
print(f"Torch Int8 Weights Size: {os.path.getsize('temp_int8.pt') / 1024 / 1024:.2f} MB")

# Latency
def run_int8(inputs):
    with torch.no_grad():
        model_int8(**inputs)

lat_int8 = measure_latency(run_int8, dummy_inputs)
print(f"Torch Int8 Latency: {lat_int8:.2f} ms")

# Accuracy
acc_int8 = evaluate_torch_model(model_int8, tokenizer, dataset, metric)
print(f"Torch Int8 Accuracy: {acc_int8:.4f}")


print("\n" + "="*50)
print("3. Torch FX Graph Mode Quantization (Static Int8)")
print("="*50)

try:
    # Reload model to ensure clean state
    model_fx = AutoModelForSequenceClassification.from_pretrained(model_name)
    model_fx.eval()

    # FX Quantization Setup
    # qnnpack for ARM (Mac), x86 for Intel
    qconfig_mapping = get_default_qconfig_mapping("qnnpack")
    
    # Use transformers.utils.fx.symbolic_trace to handle HF model quirks (control flow, slicing)
    traced_model = symbolic_trace(
        model_fx, 
        input_names=["input_ids", "attention_mask"]
    )
    
    # Prepare
    # Example inputs must match the forward signature. 
    # DistilBert forward: (input_ids, attention_mask, ...)
    example_inputs = (dummy_inputs["input_ids"], dummy_inputs["attention_mask"])
    
    # Prepare the model for static quantization
    prepared_model = prepare_fx(traced_model, qconfig_mapping, example_inputs)
    
    # Calibrate
    print("Calibrating FX model (20 steps)...")
    with torch.no_grad():
        for i, example in enumerate(dataset):
            if i >= 20: break # Calibrate with 20 examples
            inputs = tokenizer(example["text"], return_tensors="pt", padding=True, truncation=True)
            # We must pass inputs in the same way as example_inputs if positional
            prepared_model(inputs["input_ids"], inputs["attention_mask"])
    
    # Convert
    quantized_fx_model = convert_fx(prepared_model)
    
    # Measure Performance
    # Memory
    torch.save(quantized_fx_model.state_dict(), "temp_fx_int8.pt")
    print(f"FX Int8 Weights Size: {os.path.getsize('temp_fx_int8.pt') / 1024 / 1024:.2f} MB")
    if os.path.exists("temp_fx_int8.pt"): os.remove("temp_fx_int8.pt")

    # Latency
    def run_fx(inputs):
        with torch.no_grad():
            quantized_fx_model(inputs["input_ids"], inputs["attention_mask"])
            
    lat_fx = measure_latency(run_fx, dummy_inputs)
    print(f"FX Int8 Latency: {lat_fx:.2f} ms")
    
    # Accuracy
    print("Evaluating FX model...")
    all_preds = []
    all_labels = []
    for example in dataset:
        inputs = tokenizer(example["text"], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            # GraphModule might return a tuple or the original output object depending on tracing
            outputs = quantized_fx_model(inputs["input_ids"], inputs["attention_mask"])
            # Handle potential output formats
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs # Assume tensor if neither
        
        pred = logits.argmax(dim=-1).item()
        all_preds.append(pred)
        all_labels.append(example["label"])
    
    acc_fx = metric.compute(predictions=all_preds, references=all_labels)["accuracy"]
    print(f"FX Int8 Accuracy: {acc_fx:.4f}")

except Exception as e:
    print(f"FX Quantization failed: {e}")


print("\n" + "="*50)
print("4. ONNX Runtime Quantization (Int8)")
print("="*50)

onnx_model_path = "model.onnx"
onnx_quant_path = "model.quant.onnx"

# Export to ONNX
print("Exporting to ONNX...")
try:
    torch.onnx.export(
        model_fp32,
        (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
        onnx_model_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"},
                      "attention_mask": {0: "batch_size", 1: "sequence_length"},
                      "logits": {0: "batch_size"}},
        opset_version=17
    )

    # Preprocess ONNX model: Clear shapes to avoid inference errors
    print("Preprocessing ONNX model (Clearing shapes)...")
    # quant_pre_process(onnx_model_path, onnx_model_path)
    model_onnx = onnx.load(onnx_model_path)
    # Clear shapes from inputs
    for input in model_onnx.graph.input:
        for dim in input.type.tensor_type.shape.dim:
            # Keep dim_param (dynamic) but clear dim_value if any static
            if dim.HasField("dim_value"):
                dim.ClearField("dim_value")
    # Clear shapes from outputs
    for output in model_onnx.graph.output:
        for dim in output.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dim.ClearField("dim_value")
    onnx.save(model_onnx, onnx_model_path)

    # Quantize ONNX
    print("Quantizing ONNX model...")
    quantize_dynamic(
        onnx_model_path,
        onnx_quant_path,
        weight_type=QuantType.QUInt8,
        extra_options={"DisableShapeInference": True}
    )

    print(f"ONNX FP32 File Size: {os.path.getsize(onnx_model_path) / 1024 / 1024:.2f} MB")
    print(f"ONNX Int8 File Size: {os.path.getsize(onnx_quant_path) / 1024 / 1024:.2f} MB")

    # Create Inference Session
    ort_session = ort.InferenceSession(onnx_quant_path, providers=["CPUExecutionProvider"])

    # Latency
    def run_onnx(inputs):
        ort_inputs = {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy()
        }
        ort_session.run(None, ort_inputs)

    lat_onnx = measure_latency(run_onnx, dummy_inputs)
    print(f"ONNX Int8 Latency: {lat_onnx:.2f} ms")

    # Accuracy
    acc_onnx = evaluate_onnx_model(ort_session, tokenizer, dataset, metric)
    print(f"ONNX Int8 Accuracy: {acc_onnx:.4f}")

except Exception as e:
    print(f"ONNX Quantization failed: {e}")
    print("Skipping ONNX section...")


print("\n" + "="*50)
print("5. BitsAndBytes (LLM.int8) / 8-bit Loading")
print("="*50)

if torch.cuda.is_available():
    try:
        print("Attempting to load with bitsandbytes (load_in_8bit=True)...")
        # Requires GPU
        model_bnb = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto"
        )
        
        # Latency
        def run_bnb(inputs):
            # BnB requires inputs on GPU
            inputs_cuda = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                model_bnb(**inputs_cuda)
        
        lat_bnb = measure_latency(run_bnb, dummy_inputs)
        print(f"BitsAndBytes Int8 Latency: {lat_bnb:.2f} ms")
        
        # Accuracy
        acc_bnb = evaluate_torch_model(model_bnb, tokenizer, dataset, metric)
        print(f"BitsAndBytes Int8 Accuracy: {acc_bnb:.4f}")
        
    except Exception as e:
        print(f"BitsAndBytes loading failed: {e}")
else:
    print("Skipping BitsAndBytes: CUDA not available (Required for load_in_8bit).")

# Cleanup
if os.path.exists("temp_int8.pt"): os.remove("temp_int8.pt")
if os.path.exists(onnx_model_path): os.remove(onnx_model_path)
if os.path.exists(onnx_quant_path): os.remove(onnx_quant_path)
