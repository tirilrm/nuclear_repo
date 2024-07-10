#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=tm3123@ic.ac.uk

#Activate virtual environment
export PATH=/vol/bitbucket/tm3123/myvenv/bin/:$PATH
source activate

#Load CUDA toolkit
source /vol/cuda/12.0.0/setup.sh
export LD_LIBRARY_PATH=/vol/cuda/11.7.0/lib64:$LD_LIBRARY_PATH

# Set Hugging Face cache directory to Bitbucket
export HF_HOME=/vol/bitbucket/tm3123/hf_cache

python llama.py