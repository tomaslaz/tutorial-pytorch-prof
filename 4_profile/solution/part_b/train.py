import torch
import torch.profiler
import transformers
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
from datetime import timedelta
import os
transformers.utils.logging.set_verbosity_error()

BACKEND = 'nccl'
BATCH_SIZE = 320
NUM_STEPS = 10    # wait=1 + warmup=2 + active=3 = 6 minimum; 10 gives headroom
DEVICE = f"cuda:{os.environ['LOCAL_RANK']}"
PROFILE_DIR_MEMORY = "./profiler_output_memory"   # Run B: memory allocation timeline


def init_process(backend=BACKEND):
    print(f"Initializing distributed training rank {os.environ.get('RANK')} with backend: {backend} on device: {DEVICE}")
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=60*5), world_size=int(os.environ['WORLD_SIZE']))
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    world_size = dist.get_world_size()
    if dist.get_rank() == 0:
        print(f"Distributed training initialized with {world_size} processes using backend {backend}.")

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

    schedule = torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1)

    trace_dir = os.path.join(PROFILE_DIR_MEMORY, f"rank_{rank}")
    os.makedirs(trace_dir, exist_ok=True)
    on_trace_ready = torch.profiler.tensorboard_trace_handler(trace_dir) if rank == 0 else None

    torch.cuda.synchronize()
    dist.barrier()
    start_time = time.time()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        on_trace_ready=on_trace_ready,
        record_shapes=False,
        with_stack=False,
        profile_memory=True,  # enables memory allocation timeline
    ) as prof:
        for _ in range(NUM_STEPS):
            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels) # forward pass
            loss = outputs.loss
            loss.backward() # backward pass
            optimizer.step()
            prof.step()

    torch.cuda.synchronize()
    dist.barrier()
    end_time = time.time()

    elapsed = end_time - start_time
    time_per_step = elapsed / NUM_STEPS

    if dist.get_rank() == 0:
        print(f"\nTotal time for {NUM_STEPS} steps:  {elapsed:.3f} s  ({time_per_step * 1000:.1f} ms/step)")
        print(f"Memory trace: {PROFILE_DIR_MEMORY}")


if __name__ == "__main__":
    init_process()
    benchmark()
    dist.destroy_process_group()