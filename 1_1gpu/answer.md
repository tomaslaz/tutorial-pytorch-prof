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

The times are nearly identical — and that is the point.

## Key takeaways

**Same total work, same time.** `TRAINING_STEPS=2` is fixed and the global `BATCH_SIZE=32` is split across GPUs. Each GPU in the 8-GPU run does 1/8th the arithmetic, so there is very little to speed up. Adding more GPUs only added overhead (NCCL init, all-reduce) without reducing total compute.

**Scaling only helps if the GPU is the bottleneck.** The computation finishes in under 1 second either way. The ~40 seconds of wall time visible in the job log is Python startup, model loading, and tokeniser init, all CPU-bound and not affected by GPU count.

**Next:** check whether the GPU was even active during the job — [Exercise 2](../2_nvidia-smi/README.md).

---

The job submission script containing the solution is available as [`sbatch_solution.sh`](sbatch_solution.sh).
