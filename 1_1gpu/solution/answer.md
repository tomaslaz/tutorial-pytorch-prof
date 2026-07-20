# Exercise 1: Answer

## Expected output

> **Note:** Timings will vary depending on node load, NCCL initialisation, JIT compilation, etc. Expect minor variations (~10%) from the numbers shown here.

```
Time taken for 2 forward and backward pass(es) with BATCH_SIZE=32 on 1 workers: 0.817 seconds
```

## Comparison

| | 1 GPU | 8 GPUs |
|---|---|---|
| Steps | 2 | 2 |
| Samples/GPU/step | 32 | 4 |
| Total samples processed | 64 | 64 |
| Time (s) | 0.817 | 0.795 |
| Wall time (s) | 39.0 | 56.0 |

The times are nearly identical — and that is the point.

## Key takeaways

**Same total work, same time.** `TRAINING_STEPS=2` is fixed and the global `BATCH_SIZE=32` is split across GPUs, as we can see from the `per_gpu_batch_size` variable on line 31 of [train.py](train.py). Each GPU in the 8-GPU run does 1/8th the arithmetic, so there is very little to speed up. Adding more GPUs only added overhead (NCCL init, all-reduce) without reducing total compute.

**Scaling only helps if the GPU is the bottleneck.** The computation finishes in under 1 second either way. The ~40 seconds of wall time visible in the job log is dominated by Python startup, model/tokeniser loading, and (for multi-GPU/multi-node runs) NCCL initialisation — all CPU- and I/O-bound, and largely independent of the *benchmarked* GPU compute time. This startup overhead does grow somewhat with GPU/node count: 39s total here vs. ~56s for the 8-GPU run, since each process loads its own model copy and multi-node NCCL setup takes longer than single-GPU init.

**Next:** check whether the GPU was even active during the job — [Exercise 2](../2_nvidia-smi/README.md).

---

The full solution code is available in the current folder when you're ready to compare your implementation.
