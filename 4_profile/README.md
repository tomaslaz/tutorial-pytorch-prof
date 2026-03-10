# Exercise 4: torch.profiler

Exercise 3 gave you a single MFU number — BERT-base at `BATCH_SIZE=320` achieves ~X% of the GH200's peak FP32 TFLOPS. That number tells you *how well* the GPU is utilised but not *why*. `torch.profiler` gives you the full picture: every CUDA kernel, every CPU op, memory allocations, and NCCL communication events. It is the tool you reach for when you want to know *where exactly* time is going inside a step.

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

The starter `train.py` already measures MFU but does **not** capture a profiler trace. Your task is to add `torch.profiler` instrumentation.

> **Note:** `BATCH_SIZE = 320` was chosen based on the MFU experiments in Exercise 3 as close to the point of peak hardware utilisation.

Modify `train.py` to capture and export a profiler trace. Make the following changes:

**1. Add the import**
```python
import torch.profiler
```

**2. Add the output directory constants** alongside the other constants — one per run:
```python
PROFILE_DIR_KERNELS = "./profiler_output_kernels"  # Run A: accurate kernel timings
PROFILE_DIR_MEMORY  = "./profiler_output_memory"   # Run B: memory allocation timeline
```

**3. Set up the profiler schedule** inside `benchmark()`, after the warmup loop and before the timed loop:
```python
schedule = torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1)
```

For each run, set up its trace directory and handler before the `with torch.profiler.profile(...)` block:
```python
trace_dir = os.path.join(PROFILE_DIR_KERNELS, f"rank_{rank}")  # or PROFILE_DIR_MEMORY
os.makedirs(trace_dir, exist_ok=True)

# Only write traces from rank 0 to avoid flooding the filesystem
on_trace_ready = torch.profiler.tensorboard_trace_handler(trace_dir) if rank == 0 else None
```

**4. Wrap the timed loop in a profiler context** and call `prof.step()` after each step.

You will do **two runs** with different flag combinations — each answers a different question:

**Run A — kernel timing** (`profile_memory=False`): disables all heavyweight flags so GPU kernel durations are accurate. Use this run for Sections 1, 2, and 5 of the notebook. Section 3a will produce an empty plot (no memory events), but Sections 3b/3c (bandwidth) still work.

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

**Run B — memory analysis** (`profile_memory=True`): captures the full allocation timeline needed for Section 3a of the notebook. The memory overhead flag adds CPU work around every op, so GPU kernel timings will be inflated — do not use this run to judge kernel speed.

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
    profile_memory=True,  # enables Section 3a memory allocation timeline
) as prof:
    for _ in range(NUM_STEPS):
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        prof.step()
```

The two runs write to `profiler_output_kernels/` and `profiler_output_memory/` respectively, so you can point the notebook at each one independently.

**5. Print the top kernels and tensorboard hint** after the timing block:
```python
if dist.get_rank() == 0:
    ...
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(f"Kernel trace: {PROFILE_DIR_KERNELS}")
```

Keep `launch.py` unchanged and `sbatch.sh` doesn't need to be changed from the previous session.

Once your changes are in place, run the job:

```bash
sbatch sbatch.sh
```

The job writes traces to `profiler_output_kernels/rank_0/` and `profiler_output_memory/rank_0/`. Copy them back to your laptop:

```bash
rsync -r <host>:~/<path>/4_profile/profiler_output* ./
```

## Analysing the traces

The TensorBoard PyTorch Profiler plugin (`torch-tb-profiler`) is deprecated and no longer available. This tutorial uses two tools instead:

| Tool | Purpose |
|---|---|
| **HTA** (Holistic Trace Analysis) | Programmatic analysis — GPU utilisation, kernel breakdown, memory, CUDA launch stats |
| **Perfetto UI** | Interactive timeline — inspect individual kernels, gaps, call stacks |

### Install HTA

```bash
pip install HolisticTraceAnalysis
```

<details>
<summary><b>Optional: set up an isolated environment with <code>uv</code></b></summary>

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
</details>

### Run the analysis notebook

Open `analysis.ipynb` and run all cells. The notebook is organised into five sections:

| Section | What it covers | Use trace |
|---|---|---|
| **1. Overview** | `get_temporal_breakdown()` — GPU idle / compute / non-compute split, per-step breakdown, idle time categories | Run A |
| **2. GPU Kernel** | `get_gpu_kernel_breakdown()` — kernel type split (compute / communication / memory) and per-kernel statistics | Run A |
| **3a. Memory timeline** | Allocation timeline parsed from raw trace events — peak allocated/reserved, per-step spike pattern | Run B |
| **3b/3c. Bandwidth** | `get_memory_bw_time_series()` and `get_memory_bw_summary()` — H2D, D2H, D2D bandwidth | Run A |
| **4. Trace view** | Perfetto UI — interactive timeline for inspecting individual kernels and call stacks | Run A |
| **5. Operator** | `get_cuda_kernel_launch_stats()` — CPU/GPU launch timing, short-kernel detection, launch delay outliers | Run A |

> **Perfetto tip:** CPU threads and GPU streams share the same timeline axis. Zoom in on a `ProfilerStep` block and read off the CUDA kernels that fired below it. Note: Python call stacks on click require `with_stack=True`, which is disabled in both runs above to keep overhead low — re-enable it on a one-off run if you need to trace a specific kernel back to your source code.

### Key observations from the FP32 baseline

After running the notebook against the Run A (kernel) trace you should see:

| Finding | Value | Interpretation |
|---|---|---|
| `idle_time_pctg` | ~0.7% | CPU is keeping the GPU fully fed — not a bottleneck |
| `compute_time_pctg` | ~99.2% | Almost all GPU time is spent on compute — no NCCL or D2D copy overhead |
| Dominant kernels | `simt_sgemm`, `cutlass_80_simt_sgemm` | **FP32 GEMMs on regular CUDA cores — Tensor Cores are completely idle** |
| `gpu_util_pct` per step | ~99.3% (consistent) | Stable, well-pipelined training; no stalls or JIT warm-up artifacts |
| Peak allocated memory | ~29.3 GB | Significant headroom remains before OOM on the GH200 (96 GB HBM) |

**The single highest-impact optimisation available:** switch to BF16 (`model.to(torch.bfloat16)`). The profiler confirms that every major GEMM is running `simt_sgemm` (FP32 on scalar CUDA cores) rather than a Tensor Core kernel (`tensorop_` / `ampere_`). BF16 activates Tensor Cores and is expected to roughly double throughput, which you can verify by re-running Exercise 3's MFU measurement after the change.

## Further reading

- [torch.profiler API docs](https://pytorch.org/docs/stable/profiler.html) — full reference for `profile`, `schedule`, `ProfilerActivity`, `key_averages()`, and all configuration options.
- [PyTorch Profiler with TensorBoard tutorial](https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html) — official end-to-end walkthrough (uses the deprecated `torch-tb-profiler` plugin, but the concepts still apply).
- [Profiler recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) — shorter cookbook-style tutorial focused on `key_averages()` and filtering results by op name.
- [Holistic Trace Analysis (HTA)](https://github.com/facebookresearch/HolisticTraceAnalysis) — programmatic analysis library for PyTorch profiler traces; replaces most TensorBoard views.
- https://docs.pytorch.org/tutorials/beginner/hta_intro_tutorial.html
- [Perfetto UI](https://ui.perfetto.dev) — interactive trace viewer; open `.pt.trace.json` files directly in the browser.
