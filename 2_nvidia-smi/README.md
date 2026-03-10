# Exercise 2: GPU Utilisation with `nvidia-smi`

Before reaching for a full profiler, the cheapest diagnostic is `nvidia-smi`. It tells you whether the GPU is actually busy — if utilisation is low, the bottleneck is on the CPU side (data loading, tokenisation, Python overhead) rather than in GPU compute.

## Task

Modify the job submission script from the previous exercise to add a background `nvidia-smi` process that logs GPU stats at 1-second intervals. Run the job and inspect the output.

The script launches `nvidia-smi` as a background process and writes per-second GPU stats to a `.out` file alongside the Slurm output:

```bash
srun --overlap --ntasks-per-node=1 \
    stdbuf -o0 nvidia-smi \
        --query-gpu=timestamp,index,gpu_name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv -l 1 > "${SLURM_JOB_NAME}_${SLURM_JOB_ID}-mem-$(hostname).out" &
NVIDIA_SMI_PID=$!

srun ...  # training

kill $NVIDIA_SMI_PID
```

Keep `train.py` and `launch.py` unchanged from the previous session.

```bash
sbatch sbatch.sh
```

## Questions to consider

- What GPU utilisation (%) do you observe during the job?
- Is the GPU ever at 100% utilisation? If not, why not?
- Would `nvidia-smi` catch a 0.8 s compute burst at 1-second sampling?

---

See [answer.md](answer.md) when you're ready.
