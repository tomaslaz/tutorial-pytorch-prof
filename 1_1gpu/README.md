# Exercise 1: Baseline on 1 GPU

In the previous session you ran BERT fine-tuning across 2 nodes x 4 GPUs and measured a total step time. Before profiling a distributed run, if possible it is always best to start with the simplest setup: a single GPU.

## Task

Don't forget to start by moving into the Exercise 1 directory `1_1gpu/`.

Modify `sbatch.sh` to request only 1 GPU, then submit the job:

Make the following changes from the 8-GPU script:

```bash
#SBATCH --nodes=1            # was: 2
#SBATCH --gpus=1             # was: 8
#SBATCH --ntasks-per-node=1  # was: 4

srun -N 1 --gpus=1 --ntasks-per-node=1 ...
```

Keep `train.py` and `launch.py` unchanged from the previous session.

```bash
sbatch sbatch.sh
```

## Questions to consider

- How does the step time on 1 GPU compare to the 8-GPU run?
- What does this tell you about where the bottleneck is?

---

See [solution/answer.md](solution/answer.md) when you're ready.

The full solution code is available in [solution/](solution/) when you're ready to compare your implementation.
