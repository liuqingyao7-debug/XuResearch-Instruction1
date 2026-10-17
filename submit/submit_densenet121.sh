#!/bin/bash
#SBATCH --job-name=busi_densenet121
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_%j.out

cd ~/Instruction1/train
/mnt/cv_data/users/seven/envs/ml/bin/python -u train_densenet121.py