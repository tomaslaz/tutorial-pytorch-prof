# Exercise 3: Answer

## Expected results

> **Note:** Timings will vary depending on node load, NCCL initialisation, JIT compilation, etc. Expect minor variations (~10%) from the numbers shown here.

| BATCH_SIZE | FLOPs/step (TFLOP) | ms/step | Achieved (TFLOPS) | MFU |
|---|---|---|---|---|
| 32 | 212.71 | 30.8 | 6.9 | 10.3% |
| 320 | 2127.09 | 63.7 | 33.4 | 49.8% |
| 3200 | 22911.15 | 569.9 | 40.2 | 60.0% |

> **Important:** `GPU_PEAK_TFLOPS=67.0` assumes true FP32 — no TF32 Tensor Cores (989 TFLOPS sparse / 494 TFLOPS dense on GH200). This holds here because `torch.backends.cuda.matmul.allow_tf32` defaults to `False` (verified on torch 2.7.0, `torch.get_float32_matmul_precision()` returns `"highest"`), and `train.py` never overrides it. BERT's FLOPs are dominated by `nn.Linear`/matmul ops, not convolutions, so cuDNN's separate `allow_tf32` flag (which *does* default to `True`) doesn't apply here. If you enable TF32 (e.g. via `torch.set_float32_matmul_precision("high")`) or run on a PyTorch version where matmul TF32 defaults on, these MFU figures would need to be recomputed against the TF32 peak instead — using 67.0 TFLOPS as the denominator in that case would overstate MFU by roughly 7-8x.

## Key takeaways

**`nvidia-smi` can lie in both directions.**
- At `BATCH_SIZE=32`, each step takes only ~30 ms — far shorter than `nvidia-smi`'s 1-second polling interval. The GPU is actively computing, yet `nvidia-smi` reports ~0% utilisation because the work finishes between samples.
- At `BATCH_SIZE=3200`, steps take ~570 ms so `nvidia-smi` catches them and reports 100% utilisation — but MFU is only 60%. The GPU looks "fully busy" while still leaving 40% of its arithmetic throughput unused.

**Larger batches improve GPU utilisation in two ways.** First, *arithmetic intensity* — the ratio of compute work (FLOPs) to memory traffic (bytes read/written) — increases. The GPU spends more time on maths and less time waiting for data. Second, each CUDA kernel launch carries a small fixed overhead; with more work per kernel, that overhead is a smaller fraction of total time. Together, these effects push matrix multiplications closer to their theoretical peak throughput. MFU rises from 10% → 50% → 60% as batch size grows 100×.

**60% FP32 MFU on GH200 is healthy.** Perfect efficiency is physically impossible. 40–60% is typical for well-tuned FP16/FP32 workloads.

**MFU is a single number, not a timeline.** It tells you the average efficiency but not *which operations* are slow. For that you need a profiler — [Exercise 4](../4_profile/README.md).

---

The `train.py` script containing the solution is available as [`train_solution.py`](train_solution.py).
