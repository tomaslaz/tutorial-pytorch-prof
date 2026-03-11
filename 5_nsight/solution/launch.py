import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def init_process(rank, world_size, fn, backend='gloo'):
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    fn(rank, world_size)

def fn(rank, world_size):
    """Example distributed function to be implemented."""
    print(f"Hello from rank {rank} out of {world_size} processes on {os.uname().nodename}")
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = int(os.environ['WORLD_SIZE'])
    mp.spawn(init_process,
             args=(world_size, fn, 'gloo'),
             nprocs=world_size,
             join=True)