#!/bin/bash
#SBATCH --job-name=Torch_nsight
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --ntasks-per-node=1

module load brics/nccl brics/aws-ofi-nccl
module load cuda/12.6
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1) # e.g. nid001038
export MASTER_PORT=29600

echo "Job Started at $(date)"

# Start nvidia-smi monitoring in the background on the allocated node
srun --overlap --ntasks-per-node=1 \
    stdbuf -o0 nvidia-smi --query-gpu=timestamp,index,gpu_name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv -l 1 > "${SLURM_JOB_NAME}_${SLURM_JOB_ID}-mem-$(hostname).out" &
NVIDIA_SMI_PID=$!

mkdir -p nsight_output

# Run the training
srun -N 1 \
    --gpus=1 \
    --mpi=pmi2 \
    --ntasks-per-node=1 \
    bash -c 'export WORLD_SIZE=$SLURM_GPUS; export RANK=$PMI_RANK; export LOCAL_RANK=$SLURM_LOCALID; \
             nsys profile \
                 --output ./nsight_output/rank_${PMI_RANK} \
                 --trace cuda,nvtx,osrt \
                 --force-overwrite true \
                 python3 train.py'

# Stop the monitor
kill $NVIDIA_SMI_PID

echo "Job Finished at $(date)"
echo "Traces written to nsight_output/. Copy back with:"
echo "  rsync -r <host>:~/<path>/5_nsight/nsight_output/ ./nsight_output/"
