ARG PYTORCH="2.2.0"
ARG CUDA="12.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

RUN apt update
RUN apt list --upgradable

RUN apt install -y git vim libgl1-mesa-glx libglib2.0-0 ninja-build libsm6 libxrender-dev libxext6 libgl1-mesa-glx python-setuptools wget net-tools
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Install ffmpeg
RUN apt-get update
RUN apt-get upgrade -y
RUN apt install ffmpeg -y

# Install python library
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
# Install python library (System)
RUN pip install opencv-python scipy numpy==1.23.0 tqdm natsort openpyxl matplotlib chardet moviepy cython xtcocotools ez_setup pymysql ffmpegcv lap yacs faiss-gpu tensorboard pika
RUN pip install -U scikit-learn
# Install python library (MHNcity)
RUN pip install torch_geometric
# Install python library (PLASS)
RUN pip install fvcore einops
# Install python library (HRI) 
RUN pip install timm==0.6.5

# Install MMEngine and MMCV
RUN pip install -U openmim
RUN mim install mmcv==2.2.0
RUN mim install mmdet==3.3.0
RUN mim install mmpose==1.3.1
RUN mim install mmengine

# Set the default command to run when the container starts
RUN git clone https://github.com/DGU-PoliceLab/System_Integration.git
WORKDIR /System_Integration
RUN bash setting.sh
CMD ["bash"]