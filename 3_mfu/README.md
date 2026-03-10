# Exercise 3: MFU — Model FLOP Utilisation

`nvidia-smi` showed the GPU was active, but not *how efficiently*. **MFU** answers that:

```
MFU = achieved TFLOP/s / GPU peak TFLOP/s
```

Well-optimised large-model training reaches 40–60% MFU for FP32. Below ~20% suggests a memory-bound or kernel-launch-overhead bottleneck.

The starter `train.py` runs a basic distributed training loop but does **not** measure efficiency. Your task is to add MFU instrumentation.

## Task

Modify `train.py` to compute and print MFU. Make the following changes:

**1. Add the import**
```python
from torch.utils.flop_counter import FlopCounterMode
```

**2. Replace the constants block** — swap the sample-count approach for explicit step counts and add the GPU peak FLOP/s for your hardware:
```python
NUM_STEPS = 10       # steps to time
WARMUP_STEPS = 2     # discarded to avoid cold-start bias
GPU_PEAK_TFLOPS = 67.0   # GH200 FP32; adjust for your GPU
```

**3. Add a FLOP-counting helper** before `benchmark()`:
```python
def count_flops_forward(model, inputs, labels):
    flop_counter = FlopCounterMode(display=False)
    with torch.no_grad(), flop_counter:
        model(**inputs, labels=labels)
    return flop_counter.get_total_flops()
```

**4. Count FLOPs before the training loop** (inside `benchmark()`, after inputs are prepared):
```python
flops_forward = count_flops_forward(model.module, inputs, labels)
flops_per_step = flops_forward * 3   # backward ~ 2 x forward
```

**5. Add a warmup loop** (not timed) before timing:
```python
dist.barrier()
for _ in range(WARMUP_STEPS):
    optimizer.zero_grad()
    outputs = model(**inputs, labels=labels)
    outputs.loss.backward()
    optimizer.step()
```

**6. Wrap the timed loop with `torch.cuda.synchronize()`** so CPU-side timing captures all GPU work:
```python
torch.cuda.synchronize()
dist.barrier()
start_time = time.time()

for _ in range(NUM_STEPS):
    ...

torch.cuda.synchronize()
dist.barrier()
end_time = time.time()
```

**7. Compute and print MFU** after the loop:
```python
elapsed = end_time - start_time
time_per_step = elapsed / NUM_STEPS
achieved_tflops = flops_per_step / time_per_step / 1e12
mfu = achieved_tflops / GPU_PEAK_TFLOPS * 100

if dist.get_rank() == 0:
    print(f"Total time for {NUM_STEPS} steps:  {elapsed:.3f} s  ({time_per_step * 1000:.1f} ms/step)")
    print(f"Achieved throughput:            {achieved_tflops:.3f} TFLOPS")
    print(f"GPU peak FP32:                  {GPU_PEAK_TFLOPS:.1f} TFLOPS")
    print(f"MFU:                            {mfu:.1f}%")
```

Keep `launch.py` unchanged and `sbatch.sh` doesn't need to be changed from the previous session.

Once your changes are in place, run three times with different batch sizes:

```bash
sbatch sbatch.sh
```

**Step 1** — `BATCH_SIZE = 32`

**Step 2** — `BATCH_SIZE = 320`

**Step 3** — `BATCH_SIZE = 3200`

## Questions to consider

- How does MFU change as batch size increases?
- Why does a larger batch size improve GPU efficiency?
- At `BATCH_SIZE=32`, the GPU may show high utilisation in `nvidia-smi` yet low MFU — how is that possible?
- Is there a point of diminishing returns?

---

See [answer.md](answer.md) when you're ready.
