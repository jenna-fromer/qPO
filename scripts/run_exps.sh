#!/bin/bash
#SBATCH -J log
#SBATCH -o logs/%x_%a.out
#SBATCH -e logs/%x_%a.err

#SBATCH -n 20                       # number of cpus
#SBATCH --gres=gpu:volta:1

#SBATCH --array 0-9
#SBATCH --mem-per-cpu 4000                 # total memory
#SBATCH -t 1-12:00
#SBATCH -p xeon-p8    # Partition to submit to


module add anaconda/2023a
source activate torchgpu

while IFS= read -r; do
  parameters+=("$REPLY")
done < "scripts/experiments.txt"

${parameters[SLURM_ARRAY_TASK_ID]}