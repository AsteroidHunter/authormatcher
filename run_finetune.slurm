#!/bin/bash
#SBATCH --job-name=sample_run
#SBATCH --ntasks=6
#SBATCH --nodes=1
#SBATCH --time=100:00:00
#SBATCH --account=ling-582
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=gpu_standard

module load python/3.11/3.11.4 cuda11 cuda11-dnn cuda11-sdk ohpc
python3 -m torch.distributed.run --nproc_per_node=2 fine_tune_roberta.py