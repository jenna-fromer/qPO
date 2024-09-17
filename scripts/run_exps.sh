#!/bin/bash
#SBATCH -J antibiotics
#SBATCH -o logs/%x_%a.out
#SBATCH -e logs/%x_%a.err

#SBATCH -n 40                       # number of cpus
#SBATCH --gres=gpu:volta:2

#SBATCH --array 3-4
#SBATCH -t 0-2:00
#SBATCH -p debug-gpu	  # Partition to submit to


module add anaconda/2023a
source activate torchgpu

while IFS= read -r; do
  parameters+=("$REPLY")
done < "scripts/experiments_antibiotics.txt"

${parameters[SLURM_ARRAY_TASK_ID]}