#!/bin/bash
#run this script every restart the pod to install essential package and virtual env

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo "Loading environment variables from .env file..."
    source .env
else
    echo ".env file not found. Skipping..."
fi

export HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN}"  # Default value if not set in .env
export conda_env_path="${CONDA_ENV_PATH:-/root/miniconda3}"  # Default value if not set in .env
export environment_name_1="${ENVIRONMENT_NAME_1:-Phi4}"  # Default value if not set in .env

# if HF_TOKEN is not set, exit the script
if [ "$HF_TOKEN" = "YOUR_HF_TOKEN" ]; then
    echo "HF_TOKEN is not set. Please set HF_TOKEN in .env file."
    exit 1
fi

apt update
apt upgrade -y
apt install unzip
apt install nvtop -y
apt install htop -y
apt install tmux -y
apt install git-lfs -y
apt install ffmpeg -y
ffmpeg -version

git lfs install
git lfs --version

#install credential helper
git config --global credential.helper store
git config --global user.name "$gitusername"
git config --global user.email "$gitemail"
git credential approve

# Install huggingface_hub before using huggingface-cli
pip install huggingface_hub

huggingface-cli login --token $HF_TOKEN --add-to-git-credential

# Check if Miniconda is already installed
if [ ! -d "$conda_env_path" ]; then
    echo "Miniconda not found, installing at $conda_env_path ..."
    #check if shell file is already downloaded
    if [ ! -f "Miniconda3-py311_24.5.0-0-Linux-x86_64.sh" ]; then
        echo "Downloading Miniconda3-py311_24.5.0-0-Linux-x86_64.sh ..."
        wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.5.0-0-Linux-x86_64.sh
    fi
    chmod +x Miniconda3-py311_24.5.0-0-Linux-x86_64.sh
    ./Miniconda3-py311_24.5.0-0-Linux-x86_64.sh -b -p $conda_env_path
else
    echo "Miniconda already installed at $conda_env_path"
fi

# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate Phi4
source ${conda_env_path}/etc/profile.d/conda.sh

# add condition if fail to create env, terminate the script and notify the user
echo "Creating conda environments: ${environment_name_1} ..."

conda create -y -n ${environment_name_1} python=3.11
if [ $? -ne 0 ]; then
    echo "Failed to create environment ${environment_name_1} or it already exists"
else
    conda activate ${environment_name_1}
    # python -m spacy download en_core_web_sm

    
    # install specific torch versions instead of latest nightly
    # pip install torch==2.8.0.dev20250618+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128
    # pip install "torch==2.7.0" --index-url https://download.pytorch.org/whl/cu128
    pip install torch==2.8.0 torchvision torchaudio # --index-url https://download.pytorch.org/whl/test/cu128

    pip install -r requirements.txt

    # install old torch version for compatibility with GPU
    # pip install huggingface_hub

    # huggingface-cli login --token $HF_TOKEN --add-to-git-credential

    # pip install torch==2.6.0
    # pip install -r requirementsOLD.txt
fi

# activate the virtual env for users current terminal
if [ -f "${conda_env_path}/etc/profile.d/conda.sh" ]; then
    source ${conda_env_path}/etc/profile.d/conda.sh
    conda activate ${environment_name_1}
else
    echo "Warning: conda.sh not found. Environment may not be properly activated."
fi