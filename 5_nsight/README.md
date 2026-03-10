# Exercise 5: NVIDIA Nsight Systems

Exercise 4 used `torch.profiler` to capture a Python-level timeline and confirmed that BERT-base at `BATCH_SIZE=320` spends ~99% of GPU time on FP32 GEMMs (`simt_sgemm`) with Tensor Cores idle. NVIDIA Nsight Systems (`nsys`) lets you verify the same findings at the system level — lower overhead, no code change required for the core timing, and accurate kernel durations unaffected by the profiler heavyweight flags from Exercise 4.

## torch.profiler vs Nsight Systems

| Feature | torch.profiler | Nsight Systems |
|---|---|---|
| Code change required | Yes — context manager in training loop | No — wraps the process externally |
| Profiling overhead | Moderate | Lower |
| Output format | `.pt.trace.json` (Chrome trace) | `.nsys-rep` (binary) + SQLite |
| Viewer | Perfetto UI, HTA | Nsight Systems GUI, `nsys stats` |
| CUDA kernel timeline | Yes | Yes |
| CPU ops | Yes (Python-level) | Yes (OS thread-level) |
| PyTorch memory allocations | Yes (`profile_memory=True`) | No |
| Python call stacks | Yes (`with_stack=True`) | No |
| Multi-rank on one timeline | No (one file per rank) | Yes (all ranks together) |
| NVLink / PCIe bandwidth | No | Yes |
| NVTX custom annotations | No | Yes |
| GPU power and clocks | No | Yes |

The two tools are **complementary**: torch.profiler for Python-level attribution and memory; nsys for system-level timing, hardware bandwidth, and multi-rank views.

## How to profile with nsys

### Step 1: The starter script

The starter `train.py` is already in this directory. It is a stripped-down training loop (no MFU measurement, no warmup, no profiler context) using the `BATCH_SIZE=320` established as optimal in Exercise 3. Your task is to add NVTX annotations and update `sbatch.sh` to wrap it with `nsys profile`.

> **Note on batch size:** `train_solution.py` uses `BATCH_SIZE=32` to produce shorter, easier-to-read traces in the GUI. Use either — the kernel breakdown is the same; only step duration changes.

### Step 2: Modify `sbatch.sh`

First, add `module load cuda/12.6` after the existing module loads — this makes the `nsys` binary available on the path:

```bash
module load brics/nccl brics/aws-ofi-nccl
module load cuda/12.6          # required: provides the nsys profiler
```

Then add `mkdir -p nsight_output` before the `srun`, and wrap the `srun` training command with `nsys profile`. Each MPI rank gets its own output file via `${PMI_RANK}`:

```bash
mkdir -p nsight_output

srun -N 1 \
    --gpus=1 \
    --mpi=pmi2 \
    --ntasks-per-node=1 \
    bash -c 'export WORLD_SIZE=$SLURM_GPUS; export RANK=$PMI_RANK; export LOCAL_RANK=$SLURM_LOCALID;
             nsys profile \
                 --output ./nsight_output/rank_${PMI_RANK} \
                 --trace cuda,nvtx,osrt \
                 --force-overwrite true \
                 python3 train.py'
```

Key flags:

| Flag | Meaning |
|---|---|
| `--output` | Path for the `.nsys-rep` file; use `${PMI_RANK}` to get one file per rank |
| `--trace cuda,nvtx,osrt` | Capture CUDA kernels, NVTX annotations, and OS runtime events (thread scheduling) |
| `--force-overwrite true` | Overwrite existing output files — useful when re-running |
| `--cuda-memory-usage true` | Track CUDA memory allocations (adds some overhead) |

### Step 3: Add NVTX annotations to mark training phases

Add `torch.cuda.nvtx` calls to `train.py` to label each step and its phases. This is a small code change that makes the timeline much easier to read — each range appears as a coloured bar above the kernel rows:

```python
import torch.cuda.nvtx as nvtx

# inside the training loop:
for step in range(TRAINING_STEPS):
    nvtx.range_push(f"step_{step}")

    nvtx.range_push("zero_grad")
    optimizer.zero_grad()
    nvtx.range_pop()

    nvtx.range_push("forward")
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    nvtx.range_pop()

    nvtx.range_push("backward")
    loss.backward()
    nvtx.range_pop()

    nvtx.range_push("optimizer")
    optimizer.step()
    nvtx.range_pop()

    nvtx.range_pop()  # step_N
```

The `--trace cuda,nvtx` flag in Step 2 is already set to capture these ranges.

### Step 4: Execute the job

```bash
sbatch sbatch.sh
```

### Step 5: Copy results back

```bash
rsync -r <host>:~/<path>/5_nsight/nsight_output/ ./nsight_output/
```

### Step 6: Open in the Nsight Systems GUI

Download Nsight Systems from [developer.nvidia.com/nsight-systems](https://developer.nvidia.com/nsight-systems) and open the `.nsys-rep` file. To compare multiple ranks side by side: **File → Add Report**.

To generate a text summary on the command line without the GUI:

```bash
nsys stats rank_0.nsys-rep
```

## Tasks

**Task 1: Reproduce the kernel timeline**

Open `rank_0.nsys-rep` in the Nsight Systems GUI and zoom into a training step. Compare against the Perfetto trace from Exercise 4.
- Can you locate where the first step starts? (Hint: look at the nvidia-smi output to see after how many seconds GPU memory usage starts rising, and at the NVTX `step_0` annotation in the timeline.)
- How long does the first step take according to the nsys timeline? How long do the following steps take? The first step is always longer — unlike Exercise 4 where the profiler schedule's `warmup` phase hid this, nsys captures everything from process start so the CUDA context initialisation and first-step JIT costs are visible.

**Task 2: Summary from nsys stats**

On the HPC cluster, navigate to the output directory and run:

```bash
cd <path>/5_nsight/nsight_output
module load cuda/12.6
nsys stats rank_0.nsys-rep
```

Look at the **CUDA GPU Kernel Summary** (`cuda_gpu_kern_sum`) section and compare it against `kernel_metrics_df` from Exercise 4's notebook.

### Key observations

| Finding | Expected | Interpretation |
|---|---|---|
| Dominant kernels | `simt_sgemm`, `cutlass_80_simt_sgemm` | Confirms Exercise 4: FP32 GEMMs, Tensor Cores idle — no profiler artefact |
| Step durations (steady-state) | ~575 ms at `BATCH_SIZE=320` | Should match Exercise 4's `span_ms` — validates that torch.profiler overhead was not distorting timings |
| First step duration | Significantly longer | CUDA context init + first-step allocator warm-up — visible in nsys but hidden in Exercise 4 by the profiler `warmup` phase |
| NVTX breakdown | `forward` ≈ 35%, `backward` ≈ 60%, `optimizer` small | Backward is ~2× forward, as expected; optimizer (Adam) overhead is negligible relative to compute |
| Kernel count vs Exercise 4 | Should match | If counts differ, Exercise 4's heavyweight flags (`profile_memory`, `with_stack`) were still inflating per-kernel event counts |

**The key advantage over torch.profiler here:** the `cuda_gpu_kern_sum` kernel counts and durations are unaffected by any Python-level overhead, confirming that the `simt_sgemm` dominance seen in Exercise 4 is real and not a profiler artefact. This is the ground truth against which to calibrate Exercise 4's results.

---

## Further reading

- [Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) — full CLI reference, trace types, and GUI walkthrough.
- [NVTX API](https://nvtx.readthedocs.io/en/latest/) — Python bindings for adding custom annotations.
- [nsys stats reports](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-nsys-stats) — command-line summary reports without the GUI.
- [PyTorch NVTX integration](https://pytorch.org/docs/stable/cuda.html#torch.cuda.nvtx.range_push) — `torch.cuda.nvtx` API docs.
