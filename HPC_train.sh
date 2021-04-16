#!/bin/bash
#SBATCH -n 4
#SBATCH --nodes=1
#SBATCH --mem=16000
#SBATCH -t 0-0:10
#SBATCH --output=larecnet.out
#SBATCH --error=larecnet.err
#SBATCH --partition=gpu
#SBATCH --constraint=P100
#SBATCH --gres=gpu:1
module purge
module load cuda/11.02
module load gcc/7.3
python3 /gpfsnyu/scratch/yw3752/fisheye_detection/LaRecNet/train_larecnet.py
