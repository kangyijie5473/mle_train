import argparse
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def init_process(rank: int, world_size: int, backend: str, master_addr: str, master_port: int):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def get_device(rank: int, backend: str) -> torch.device:
    if backend == "nccl":
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    return torch.device("cpu")


def rank_print(rank: int, msg: str):
    print(f"[rank {rank}] {msg}", flush=True)


def demo_broadcast(rank: int, device: torch.device):
    x = torch.tensor([float(rank)], device=device)
    if rank == 0:
        x.fill_(42.0)
    dist.broadcast(x, src=0)
    rank_print(rank, f"broadcast -> {x.tolist()} (src=0)")


def demo_all_reduce(rank: int, world_size: int, device: torch.device):
    x = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size + 1) / 2
    rank_print(rank, f"all_reduce(sum) -> {x.tolist()}, expected={[expected]}")


def demo_reduce_scatter(rank: int, world_size: int, device: torch.device):
    chunk = 3
    expected_val = world_size * (world_size + 1) / 2

    if hasattr(dist, "reduce_scatter_tensor"):
        inp = torch.full((world_size * chunk,), float(rank + 1), device=device)
        out = torch.empty(chunk, device=device)
        dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM)
    else:
        input_list = [torch.full((chunk,), float(rank + 1), device=device) for _ in range(world_size)]
        out = torch.empty(chunk, device=device)
        dist.reduce_scatter(out, input_list, op=dist.ReduceOp.SUM)

    rank_print(rank, f"reduce_scatter(sum) -> {out.tolist()}, expected={ [expected_val] * chunk }")


def demo_all_gather(rank: int, world_size: int, device: torch.device):
    x = torch.tensor([rank, rank + 10], dtype=torch.float32, device=device)
    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x)
    out = torch.stack(gathered, dim=0)
    rank_print(rank, f"all_gather -> {out.cpu().tolist()}")


def demo_all_to_all(rank: int, world_size: int, device: torch.device):
    chunk = 2
    send = (torch.arange(world_size * chunk, device=device).view(world_size, chunk) + rank * 100).float()
    recv = torch.empty_like(send)

    try:
        if hasattr(dist, "all_to_all_single"):
            dist.all_to_all_single(recv, send)
        else:
            send_list = [send[i].contiguous() for i in range(world_size)]
            recv_list = [recv[i].contiguous() for i in range(world_size)]
            dist.all_to_all(recv_list, send_list)
            recv = torch.stack(recv_list, dim=0)
    except RuntimeError:
        # 某些后端可能不支持 all_to_all，这里给一个点对点等价演示
        reqs = []
        tmp_recv = [torch.empty(chunk, device=device) for _ in range(world_size)]
        for peer in range(world_size):
            reqs.append(dist.isend(send[peer].contiguous(), dst=peer))
            reqs.append(dist.irecv(tmp_recv[peer], src=peer))
        for r in reqs:
            r.wait()
        recv = torch.stack(tmp_recv, dim=0)

    rank_print(rank, f"all_to_all -> {recv.cpu().tolist()}")


def run_worker(rank: int, world_size: int, backend: str, master_addr: str, master_port: int):
    init_process(rank, world_size, backend, master_addr, master_port)
    device = get_device(rank, backend)

    dist.barrier()
    if rank == 0:
        print(f"\n=== backend={backend}, world_size={world_size} ===", flush=True)
    dist.barrier()

    demo_broadcast(rank, device)
    dist.barrier()
    demo_all_reduce(rank, world_size, device)
    dist.barrier()
    demo_reduce_scatter(rank, world_size, device)
    dist.barrier()
    demo_all_gather(rank, world_size, device)
    dist.barrier()
    demo_all_to_all(rank, world_size, device)
    dist.barrier()

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="分布式原语 demo: broadcast/all_reduce/reduce_scatter/all_gather/all_to_all")
    parser.add_argument("--world_size", type=int, default=4, help="进程数")
    parser.add_argument("--backend", type=str, choices=["auto", "gloo", "nccl"], default="auto", help="通信后端")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1", help="主节点地址")
    args = parser.parse_args()

    if args.backend == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() >= args.world_size:
            backend = "nccl"
        else:
            backend = "gloo"
    else:
        backend = args.backend

    if backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("backend=nccl 需要可用 CUDA 设备")

    master_port = find_free_port()
    mp.spawn(
        run_worker,
        args=(args.world_size, backend, args.master_addr, master_port),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
