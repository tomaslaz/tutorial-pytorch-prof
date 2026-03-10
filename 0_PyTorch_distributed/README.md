# Exercise 0: PyTorch Distributed (Previous Session)

This code is from the previous session on distributed training:
https://docs.isambard.ac.uk/user-documentation/tutorials/distributed-training/

It runs BERT fine-tuning with DDP across **2 nodes × 4 GPUs (8 GPUs total)** and measures total step time.

## Run

```bash
sbatch sbatch_pytorch.sh
```

## Expected output

> **Note:** Timings will vary depending on node load, NCCL initialisation, and JIT compilation. Expect minor variations (~10%) from the numbers shown here.

```
Time taken for 2 forward and backward pass(es) with BATCH_SIZE=32 on 8 workers: 0.795 seconds
```

## What the code does

- Initialises a distributed process group with NCCL backend
- Loads `bert-base-uncased` and wraps it in DDP
- Runs `TRAINING_STEPS=2` forward + backward passes with `BATCH_SIZE=32` split across all GPUs (4 samples/GPU/step)
- Reports total wall time from rank 0

The step time looks reasonable, but we have no idea how efficiently the GPUs were used, or where the time actually went. That is what the rest of the tutorial explores.

**Next:** [Exercise 1](../1_1gpu/README.md) — establish a single-GPU baseline.
