import torch
import transformers
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.flop_counter import FlopCounterMode
import torch.distributed as dist
import time
from datetime import timedelta
import os
transformers.utils.logging.set_verbosity_error()

BACKEND = 'nccl'
BATCH_SIZE = 32
NUM_STEPS = 10    # enough steps for stable timing
WARMUP_STEPS = 2  # discarded before timing to avoid cold-start bias
DEVICE = f"cuda:{os.environ['LOCAL_RANK']}"

# Peak FP32 TFLOPS for GH200 120GB: ~67 TFLOPS (FP32)
GPU_PEAK_TFLOPS = 67.0


def init_process(backend=BACKEND):
    print(f"Initializing distributed training rank {os.environ.get('RANK')} with backend: {backend} on device: {DEVICE}")
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=60*5), world_size=int(os.environ['WORLD_SIZE']))
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    world_size = dist.get_world_size()
    if dist.get_rank() == 0:
        print(f"Distributed training initialized with {world_size} processes using backend {backend}.")


def count_flops_forward(model, inputs, labels):
    """Count FLOPs for a single forward pass using FlopCounterMode."""
    flop_counter = FlopCounterMode(display=False)
    with torch.no_grad(), flop_counter:
        model(**inputs, labels=labels)
    return flop_counter.get_total_flops()


def benchmark():
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    per_gpu_batch_size = BATCH_SIZE // world_size

    if dist.get_rank() == 0:
        print(f"Running benchmark with world size: {world_size}, batch size: {BATCH_SIZE}, per GPU batch size: {per_gpu_batch_size}, training steps: {NUM_STEPS}")

    model = BertForSequenceClassification.from_pretrained(model_name).to(DEVICE)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters())
    
    # Separate data per worker
    start_idx = local_rank * per_gpu_batch_size
    end_idx = start_idx + per_gpu_batch_size
    # Create synthetic training data
    texts = [f"This is sample sentence {i} for benchmarking BERT." for i in range(start_idx, end_idx)]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, use_fast_tokenizer=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    labels = torch.ones(per_gpu_batch_size, dtype=torch.long).to(DEVICE)

    # Count forward-pass FLOPs on the unwrapped model (DDP adds no FLOPs of its own)
    # Backward pass is approximately 2x forward, so a full step ≈ 3x forward FLOPs
    flops_forward = count_flops_forward(model.module, inputs, labels)
    flops_per_step = flops_forward * 3

    if rank == 0:
        print(f"FLOPs per forward pass:       {flops_forward / 1e9:.2f} GFLOPs")
        print(f"FLOPs per step (fwd + bwd):   {flops_per_step / 1e9:.2f} GFLOPs")

    # Warmup — not timed
    dist.barrier()
    for _ in range(WARMUP_STEPS):
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        outputs.loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    dist.barrier()
    start_time = time.time()

    for _ in range(NUM_STEPS):
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    dist.barrier()
    end_time = time.time()

    elapsed = end_time - start_time
    time_per_step = elapsed / NUM_STEPS
    achieved_tflops = flops_per_step / time_per_step / 1e12
    mfu = achieved_tflops / GPU_PEAK_TFLOPS * 100

    if dist.get_rank() == 0:
        print(f"\nTotal time for {NUM_STEPS} steps:  {elapsed:.3f} s  ({time_per_step * 1000:.1f} ms/step)")
        print(f"Achieved throughput:            {achieved_tflops:.3f} TFLOPS")
        print(f"FLOPs/step:                     {flops_per_step / 1e09:.3f} TFLOP")
        print(f"GPU peak FP32:                  {GPU_PEAK_TFLOPS:.1f} TFLOPS")
        print(f"MFU:                            {mfu:.1f}%")


if __name__ == "__main__":
    init_process()
    benchmark()
    dist.destroy_process_group()