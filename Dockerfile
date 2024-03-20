FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

### Deal with cuda keyring problem
RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

### install python3 and opencv requires
RUN DEBIAN_FRONTEND=noninteractive apt-get update --fix-missing --no-install-recommends && \
    DEBIAN_FRONTEND=noninteractive apt-get install python3.8 python3-pip nano libsm6 \
    libxext6 libxrender-dev libgl1-mesa-glx libglib2.0-0 python3-tk qt5-default curl git nano htop -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

### install common packages
RUN pip3 install -U pip && \
    pip3 install future -U

### cuda - torch - torchvision
ARG CUDA_VER="111"
ARG TORCH_VER="1.9.1"
ARG VISION_VER="0.10.1"

RUN pip3 install torch==${TORCH_VER}+cu${CUDA_VER} torchvision==${VISION_VER}+cu${CUDA_VER} -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install kornia
RUN pip3 install torch_scatter==2.0.9 -f https://data.pyg.org/whl/torch-${TORCH_VER}+cu${CUDA_VER}

WORKDIR /app
RUN git clone https://github.com/Akos0628/MonoDTR.git
WORKDIR /app/MonoDTR
RUN pip3 install -r requirement.txt

# sleep infinity

# COPY data . # recommended to use a volume to share the training data
# COPY checkpoint workdirs/MonoDTR # recommended to use a volume to share the checkpoint

