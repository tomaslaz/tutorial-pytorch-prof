# Exercise 4: torch.profiler

Exercise 3 timed training steps at `BATCH_SIZE=320` — that tells you *how fast* the GPU is running but not *why*. `torch.profiler` gives you the full picture: every CUDA kernel, every CPU op, memory allocations, and NCCL communication events. It is the tool you reach for when you want to know *where exactly* time is going inside a step.

## How the profiler schedule works

```
wait → warmup → active → repeat
```

- **wait** — profiler is off; normal execution
- **warmup** — profiler starts but discards results (JIT, caches settle)
- **active** — trace is captured and written to disk
- **repeat** — how many `wait→warmup→active` cycles to record; `repeat=1` stops after the first cycle

The script uses `schedule(wait=1, warmup=2, active=3, repeat=1)` so 6 steps minimum are needed; `NUM_STEPS=10` gives headroom.

## Task

The starter `train.py` already times training steps but does **not** capture a profiler trace. Your task is to add `torch.profiler` instrumentation.

> **Note:** `BATCH_SIZE = 320` was chosen to maximise GPU utilisation.

Modify `train.py` to capture and export a profiler trace. Make the following changes:

**1. Add the import**
```python
import torch.profiler
```

**2. Add the output directory constant** alongside the other constants:
```python
PROFILE_DIR_KERNELS = "./profiler_output_kernels"  # Run A: accurate kernel timings
```

**3. Set up the profiler schedule, trace handler, and barrier** inside `benchmark()`, after the `labels` tensor is created:

```python
schedule = torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1)
```

Set up the following just after the newly added schedule:
```python
trace_dir = os.path.join(PROFILE_DIR_KERNELS, f"rank_{rank}")
os.makedirs(trace_dir, exist_ok=True)

# Only write traces from rank 0 to avoid flooding the filesystem
on_trace_ready = torch.profiler.tensorboard_trace_handler(trace_dir) if rank == 0 else None
```

Then, replace the existing `dist.barrier()` before `start_time` with:
```python
torch.cuda.synchronize()
dist.barrier()
```

`torch.cuda.synchronize()` is needed because CUDA operations are asynchronous — the CPU queues them and moves on without waiting for the GPU to finish. Before `start_time`, this ensures that prior GPU work (e.g. data transfers via `.to(DEVICE)`) has actually completed before the timer starts. Before `end_time` (after the profiler block), it ensures all CUDA kernels launched during training have finished executing on the GPU before the timer stops — otherwise `end_time` reflects CPU queue flush time, not true GPU completion.

**4. Wrap the timed loop in a profiler context** and call `prof.step()` after each step.

Use `profile_memory=False` to keep all heavyweight flags off so GPU kernel durations are accurate.

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=schedule,
    on_trace_ready=on_trace_ready,
    record_shapes=False,
    with_stack=False,
    profile_memory=False,  # accurate GPU kernel timings
) as prof:
    for _ in range(NUM_STEPS):
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        prof.step()
```

Similarly, replace the existing `dist.barrier()` after the loop with:
```python
torch.cuda.synchronize()
dist.barrier()
```

This ensures all GPU work is complete before `end_time` is recorded, for the same reason as above.

**5. Print the top kernels** after the timing block:
```python
if dist.get_rank() == 0:
    print(f"\nTotal time for {NUM_STEPS} steps:  {elapsed:.3f} s  ({time_per_step * 1000:.1f} ms/step)")
    print()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(f"Kernel trace: {PROFILE_DIR_KERNELS}")
```

Keep `launch.py` unchanged and `sbatch.sh` doesn't need to be changed from the previous session.

Once your changes are in place, run the job:

```bash
sbatch sbatch.sh
```

## Analysing the traces

The TensorBoard PyTorch Profiler plugin (`torch-tb-profiler`) is deprecated and no longer available. This tutorial uses two tools instead:

| Tool | Purpose |
|---|---|
| **HTA** (Holistic Trace Analysis) | Programmatic analysis — GPU utilisation, kernel breakdown, memory, CUDA launch stats |
| **Perfetto UI** | Interactive timeline — inspect individual kernels, gaps, call stacks |


## If you prefer to do the analysis on your laptop


The job writes traces to `profiler_output_kernels/rank_0/`. Copy them back to your laptop by executing the following command from your laptop terminal if you are planning to run the analysis notebook locally:

```bash
rsync -r <host>:~/<path>/4_profile/profiler_output_kernels ./
```

### set up an isolated environment with uv

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

uv pip install HolisticTraceAnalysis jupyterlab

# Launch the notebook
jupyter lab analysis.ipynb
```

To register the environment as a named Jupyter kernel (useful if you switch kernels inside JupyterLab):
```bash
uv pip install ipykernel
python -m ipykernel install --user --name hta-env --display-name "Python (hta-env)"
```

## If you prefer to do the analysis on the cluster

Please go to https://apps.isambard.ac.uk/aip2-jupyter in a web browser. Log in with your Isambard-AI credentials, select the `BriCS-Turing Workshop 2026` project and launch a JupyterLab session. 


start with Install HTA

press on the console Python 3 (ipykernel) tab to open a python console, then run the following command to install HTA in the JupyterLab environment:

```bash
pip install HolisticTraceAnalysis
```

Once it finishes installing, you should navigate to the `4_profile` directory in the JupyterLab file browser, open `analysis.ipynb` and run all cells to perform the analysis.


### Run the analysis notebook

Open `analysis.ipynb` and run all cells. The notebook is organised into five sections:

| Section | What it covers |
|---|---|
| **1. Overview** | `get_temporal_breakdown()` — GPU idle / compute / non-compute split, per-step breakdown, idle time categories |
| **2. GPU Kernel** | `get_gpu_kernel_breakdown()` — kernel type split (compute / communication / memory) and per-kernel statistics |
| **3. Bandwidth** | `get_memory_bw_time_series()` and `get_memory_bw_summary()` — H2D, D2H, D2D bandwidth |
| **4. Trace view** | Perfetto UI — interactive timeline for inspecting individual kernels and call stacks |
| **5. Operator** | `get_cuda_kernel_launch_stats()` — CPU/GPU launch timing, short-kernel detection, launch delay outliers |

> **Perfetto tip:** CPU threads and GPU streams share the same timeline axis. Zoom in on a `ProfilerStep` block and read off the CUDA kernels that fired below it. Note: Python call stacks on click require `with_stack=True`, which is disabled in both runs above to keep overhead low — re-enable it on a one-off run if you need to trace a specific kernel back to your source code.

### Key observations from the FP32 baseline

After running the notebook against the Run A (kernel) trace you should see:

| Finding | Value | Interpretation |
|---|---|---|
| `idle_time_pctg` | ~0.7% | CPU is keeping the GPU fully fed — not a bottleneck |
| `compute_time_pctg` | ~99.2% | Almost all GPU time is spent on compute — no NCCL or D2D copy overhead |
| Dominant kernels | `simt_sgemm`, `cutlass_80_simt_sgemm` | **FP32 GEMMs on regular CUDA cores — Tensor Cores are completely idle** |
| `gpu_util_pct` per step | ~99.3% (consistent) | Stable, well-pipelined training; no stalls or JIT warm-up artifacts |

**The single highest-impact optimisation available:** switch to BF16 (`model.to(torch.bfloat16)`). The profiler confirms that every major GEMM is running `simt_sgemm` (FP32 on scalar CUDA cores) rather than a Tensor Core kernel (`tensorop_` / `ampere_`). BF16 activates Tensor Cores and is expected to roughly double throughput.

---

## Part B: Memory allocation analysis

The kernel timing trace (`profile_memory=False`) does not record allocator events, so Section 3a of the analysis notebook requires a separate run with `profile_memory=True`. The memory overhead flag adds CPU work around every op, so GPU kernel timings in this trace will be inflated — do not use it to judge kernel speed.

Start from your **Part A solution** and make three changes:

**1. Replace** `PROFILE_DIR_KERNELS` with:
```python
PROFILE_DIR_MEMORY = "./profiler_output_memory"   # Run B: memory allocation timeline
```

**2. Update** `profile_memory=False` → `profile_memory=True`, and replace all references to `PROFILE_DIR_KERNELS` with `PROFILE_DIR_MEMORY`.

**3. Replace** the print block with (no `key_averages()` — kernel timings are inflated when `profile_memory=True`):
```python
if dist.get_rank() == 0:
    print(f"\nTotal time for {NUM_STEPS} steps:  {elapsed:.3f} s  ({time_per_step * 1000:.1f} ms/step)")
    print(f"Memory trace: {PROFILE_DIR_MEMORY}")
```

**4. Rerun the job:**
```bash
sbatch sbatch.sh
```

**5. If running locally, copy the memory trace back to your laptop:**
```bash
rsync -r <host>:~/<path>/4_profile/profiler_output_memory ./
```

**6. Open `analysis_mem_usage.ipynb`** — on your laptop as described in the [laptop section above](#if-you-prefer-to-do-the-analysis-on-your-laptop), or on the cluster as described in the [cluster section above](#if-you-prefer-to-do-the-analysis-on-the-cluster). Run all cells to plot the GPU memory allocation timeline, showing peak allocated/reserved memory and per-step spike patterns.

| Finding | Value | Interpretation |
|---|---|---|
| Peak allocated memory | ~29.3 GB | Significant headroom remains before OOM on the GH200 (96 GB HBM) |

---

## Further reading

- [torch.profiler API docs](https://pytorch.org/docs/stable/profiler.html) — full reference for `profile`, `schedule`, `ProfilerActivity`, `key_averages()`, and all configuration options.
- [PyTorch Profiler with TensorBoard tutorial](https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html) — official end-to-end walkthrough (uses the deprecated `torch-tb-profiler` plugin, but the concepts still apply).
- [Profiler recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) — shorter cookbook-style tutorial focused on `key_averages()` and filtering results by op name.
- [Holistic Trace Analysis (HTA)](https://github.com/facebookresearch/HolisticTraceAnalysis) — programmatic analysis library for PyTorch profiler traces; replaces most TensorBoard views.
- https://docs.pytorch.org/tutorials/beginner/hta_intro_tutorial.html
- [Perfetto UI](https://ui.perfetto.dev) — interactive trace viewer; open `.pt.trace.json` files directly in the browser.
