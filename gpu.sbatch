#!/bin/bash
#SBATCH --mem=10g
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres gpu:1

#SBATCH --output=STD.out
#SBATCH --error=STD.err

# env setting
module load cuda
conda activate hgcn

# run
