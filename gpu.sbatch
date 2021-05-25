#!/bin/bash
#SBATCH --account=rrg-laurence
# #SBATCH --exclusive
#SBATCH --gres=gpu:v100:1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G

cd MIP
sleep 24h