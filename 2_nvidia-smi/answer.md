# Exercise 2: Answer

## Expected output (nvidia-smi log)

> **Note:** Timings will vary depending on node load, NCCL initialisation, JIT compilation, etc. Expect minor variations (~10%) from the numbers shown here.

```
timestamp, index, name, utilization.gpu [%], memory.used [MiB], memory.total [MiB], power.draw [W], temperature.gpu
2026/03/06 16:45:26, 0, NVIDIA GH200 120GB,  0 %,    2 MiB, 97871 MiB,  84.33 W, 31
2026/03/06 16:45:27, 0, NVIDIA GH200 120GB,  0 %,    2 MiB, 97871 MiB,  84.25 W, 31
...  (idle for ~33 seconds)
2026/03/06 16:45:59, 0, NVIDIA GH200 120GB,  2 %, 1035 MiB, 97871 MiB,  99.02 W, 32
2026/03/06 16:46:00, 0, NVIDIA GH200 120GB,  0 %, 2589 MiB, 97871 MiB, 128.78 W, 33
2026/03/06 16:46:01, 0, NVIDIA GH200 120GB,  0 %, 3707 MiB, 97871 MiB, 136.90 W, 33
```

The GPU is at 0% for ~33 seconds, then memory jumps from 1 MiB to 3707 MiB as the model loads. The actual 0.8 s training burst never registers as a visible utilisation spike.

## Key takeaways

**Wall time ≠ compute time.** The job ran for ~40 s but the GPU computed for under 1 s. The other ~39 s were Python startup, conda env activation, tokeniser init, and model loading — all CPU-bound.

**`nvidia-smi` at 1 s resolution is too coarse for short workloads.** It is useful for sustained underutilisation (e.g. data-loading bottlenecks in long training runs) but cannot resolve sub-second bursts.

**Memory is the leading indicator.** The jump in `memory.used` marks when the model was transferred to GPU, which is a useful reference point even when utilisation reads 0%.

**Next:** measure *how efficiently* the GPU was used during that brief compute window — [Exercise 3](../3_mfu/README.md).

---

The job submission script containing the solution is available as [`sbatch_solution.sh`](sbatch_solution.sh).
