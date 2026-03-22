以下是这次完整的问题排查与解决过程总结。

---

## **问题背景**

在 RTX 4090（24GB VRAM）上对一个本地 14B 自定义 DeepSeek 模型（`heretic-DeepSeek`，基于 Qwen2 架构）进行 QLoRA 微调，持续出现 `torch.OutOfMemoryError: CUDA out of memory`。

---

## **第一轮：原始代码的问题**

原始代码使用标准 HuggingFace + BitsAndBytes 路线，存在以下几个叠加缺陷：

`os.environ["PYTORCH_CUDA_ALLOC_CONF"]` 被放在函数体内部，此时 CUDA allocator 已经初始化，环境变量完全不生效。`max_memory` 约束被注释掉，`device_map="auto"` 会贪婪地把所有层塞进 GPU，不给激活值和梯度留任何空间。`max_seq_length=1024` 导致 Attention 激活值显存极高（与序列长度平方成正比）。优化器使用 `paged_adamw_32bit`，状态是 `paged_adamw_8bit` 的两倍大。`target_modules` 覆盖 7 个投影层，可训练参数和梯度显存偏高。

**修复方向：** 将环境变量移到文件顶部、恢复 `max_memory` 限制、降低序列长度、换用 `paged_adamw_8bit`、精简 `target_modules`。

---

## **第二轮：换用 Unsloth 后仍然 OOM**

切换到 Unsloth 框架后，错误依旧，且 PyTorch 在训练开始前就已占用 22.90GB。这说明问题不在训练过程，而在**模型加载阶段本身**。

根本原因是：本地模型文件是 BF16 原始权重（27.51GB 单个 `.safetensors` 文件），`from_pretrained` 必须先把完整的 BF16 模型读入内存再做量化，GPU 在这个过程中就已被撑爆。Unsloth 对本地自定义模型路径的架构识别也可能失效，退化为标准 HuggingFace 加载路径，完全失去显存优化效果。

---

## **第三轮：诊断脚本定位根因**

通过诊断脚本得到了精确数据：

- 模型架构：Qwen2，48层，隐层维度 5120
- 词表大小：152064（超大词表）
- 磁盘文件：**27.51GB 单文件 BF16 权重**
- Embedding 层（BF16，不被量化）：**2.90GB**
- Transformer 层（4-bit 理论）：7.03GB
- 理论显存底线：9.93GB，但加上激活值、梯度、优化器状态，实际远超 24GB

两个核心杀手由此确认：**27.51GB BF16 文件导致加载峰值爆炸** + **2.90GB Embedding 层常驻 GPU 且无法量化**。

---

## **最终解决方案**

采用**两阶段加载策略**，彻底绕开 GPU 峰值问题：

**第一阶段**，用 `device_map="cpu"` + `low_cpu_mem_usage=True` 将模型完整加载到 CPU 内存并在 CPU 上完成 NF4 量化，GPU 在此过程中零占用，完全避免了加载峰值 OOM。

**第二阶段**，手动将 48 个 Transformer 层逐层搬到 GPU（共约 7GB），而将 `embed_tokens` 和 `lm_head` 这两个 BF16 大户（共 2.90GB）强制留在 CPU，彻底释放这部分 GPU 显存。

配合以下参数组合确保训练阶段不再 OOM：`paged_adamw_8bit` 优化器、`target_modules=["q_proj", "v_proj"]` 极简 LoRA、`max_seq_length=512`、`packing=True`、`dataloader_pin_memory=False`、梯度检查点 `use_reentrant=False`。

最终 GPU 显存分布稳定在约 **12\~14GB**，在 24GB 上有充足余量，训练可以正常启动。
