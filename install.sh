pip install numpy==1.26.4
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
pip install accelerate==0.30.1
pip install bitsandbytes==0.43.1
pip install moviepy==1.0.3
pip install git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
pip install timm ftfy regex einops fvcore eva-decord==0.6.1 iopath matplotlib types-regex cartopy
pip install ctranslate2==4.4.0 faster_whisper==1.0.3 neo4j hnswlib xxhash nano-vectordb
pip install transformers==4.37.1
pip install tiktoken openai tenacity
pip install ollama
pip install nvidia-cublas-cu11 nvidia-cudnn-cu11
pip install nvidia-cudnn-cu11==8.9.6.50

# Install ImageBind using the provided code in this repository, where we have removed the requirements.txt to avoid environment conflicts.
cd ImageBind
pip install .

# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

# minicpm-v
git lfs clone https://huggingface.co/openbmb/MiniCPM-V-2_6-int4

# whisper
git lfs clone https://huggingface.co/Systran/faster-distil-whisper-large-v3

# imagebind
mkdir .checkpoints
cd .checkpoints
wget https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth

ollama pull nomic-embed-text