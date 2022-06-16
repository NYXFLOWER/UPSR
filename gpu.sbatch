#!/bin/bash
# #SBATCH --account=rrg-laurence
#SBATCH --account=def-laurence
#SBATCH --gres=gpu:v100:4
#SBATCH --time=2-11:55:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --job-name=g4-eva

source /home/flower/anaconda3/bin/activate
conda activate se3
cd /home/flower/github/Protein3D/Protein3D
# python multi_run.py 2 3 4 5
# python multi_run.py 2 5 8 1
# python multi_run.py 2 4 8 1
# python multi_run.py 2 3 8 1

sleep 70000h

