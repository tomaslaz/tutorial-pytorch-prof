# Tutorial: Introduction to PyTorch Profiling

Repository with hands-on exercises of the Introduction to PyTorch Profiling tutorial.

## Coordinates

* Date: 12 May 2026
* Occasion: BriCS-Turing Isambard-AI Workshop
* Location: British Library, London

## Prerequisites

To install the latest version of Conda navigate to your home directory and run the following commands:

```bash
cd $HOME
curl --location --remote-name "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
rm Miniforge3-$(uname)-$(uname -m).sh
```

Set up the conda environment:

```bash
srun -N 1 --gpus 1 bash -c "source ~/miniforge3/bin/activate && conda env create -f pytorch_conda_env.yaml -y"
```

## Exercises

| # | Folder | Topic | Goal |
|---|---|---|---|
| 0 | `0_PyTorch_distributed/` | Recap | Re-run the 8-GPU baseline from the previous session |
| 1 | `1_1gpu/` | Baseline | Measure step time on a single GPU |
| 2 | `2_nvidia-smi/` | GPU utilisation | Use `nvidia-smi` to observe idle vs active GPU |
| 3 | `3_mfu/` | MFU | Compute Model FLOP Utilisation at different batch sizes |
| 4 | `4_profile/` | torch.profiler | Capture a full CPU/CUDA timeline and analyse it with HTA and Perfetto UI |
| 5 | `5_nsight/` | Nsight Systems | Profile with `nsys`, compare against torch.profiler, and explore system-level metrics |

Work through each folder in order. Each folder contains:
- `README.md` — exercise instructions and questions to consider
- `train.py` — the training script
- `answer.md` — worked answer and key takeaways or `analysis.ipynb` (guided notebook).
- `solution` - folder containing the complete solution (e.g. `train_solution.py`, `sbatch_solution.sh`, `analysis.ipynb`)
- `sbatch.sh` — job submission script

## Extra

So far all the exercises have been run on a single GPU using the distributed setup from the previous session. For an extra challenge, try the following:

- Remove DDP from `train.py` (no `init_process_group`, no `DistributedDataParallel` wrapper, no `dist.barrier()`, etc.) and rerun the analysis of all exercises. Compare the profiler output against the DDP version and identify key differences in GPU utilisation, kernel breakdown, and memory usage.
- Rerun the analysis with an increasing number of GPUs (e.g. 4, 8, 16) and compare the profiler output across runs to identify when communication overhead starts to dominate.


## Data transfer (Isambard-AI)

```bash
# push to cluster
rsync -r pytorch_prof_tutorial/ <host>:~/<path>/pytorch_prof_tutorial/

# pull from cluster
rsync -r <host>:~/<path>/pytorch_prof_tutorial/ ./pytorch_prof_tutorial/
```
