# Exercise 3: Answer

## Expected results

> **Note:** Timings will vary depending on node load, NCCL initialisation, JIT compilation, etc. Expect minor variations (~10%) from the numbers shown here.

| BATCH_SIZE | FLOPs/step (TFLOP) | ms/step | Achieved (TFLOPS) | MFU |
|---|---|---|---|---|
| 32 | 212.71 | 30.8 | 6.9 | 10.3% |
| 320 | 2127.09 | 63.7 | 33.4 | 49.8% |
| 3200 | 22911.15 | 569.9 | 40.2 | 60.0% |

## Key takeaways

**`nvidia-smi` can lie in both directions.**
- At `BATCH_SIZE=32`, each step takes only ~30 ms — far shorter than `nvidia-smi`'s 1-second polling interval. The GPU is actively computing, yet `nvidia-smi` reports ~0% utilisation because the work finishes between samples.
- At `BATCH_SIZE=3200`, steps take ~570 ms so `nvidia-smi` catches them and reports 100% utilisation — but MFU is only 60%. The GPU looks "fully busy" while still leaving 40% of its arithmetic throughput unused.

**Larger batches increase arithmetic intensity.** Each kernel has more work to amortise its launch overhead, and matrix multiplications get closer to their theoretical peak throughput. MFU rises from 10% -> 50% -> 60% as batch size grows 100 times.

**60% FP32 MFU on GH200 is healthy.** Perfect efficiency is physically impossible. 40–60% is typical for well-tuned FP32 workloads.

**MFU is a single number, not a timeline.** It tells you the average efficiency but not *which operations* are slow. For that you need a profiler — [Exercise 4](../4_profile/README.md).

---

The `train.py` script containing the solution is available as [`train_solution.py`](train_solution.py).
