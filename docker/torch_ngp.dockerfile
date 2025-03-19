# Use official CUDA devel 11.3 image with Ubuntu 20.04 as Base
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# Set the working directory
WORKDIR /app

# Set system to non interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Install System Dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    curl \
    nano \
    gcc-10 \
    g++-10 \
    git \
    build-essential \
    cmake \
    ninja-build \
    gcc \
    g++ \
    libtinfo5 \
    wget \
    unzip \
    software-properties-common \
    cuda-toolkit-11-3 \
    libcudnn8=8.2.1.32-1+cuda11.3 \
    && rm -rf /var/lib/apt/lists/*

# Install GLIBC 2.4.29 or higher
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update && apt-get install -y gcc-11 g++-11 libstdc++-11-dev

# Install dependencies for GUI and GLFW (X11, GLFW, OpenGL, etc.)
RUN apt-get update && apt-get install -y \
    x11-apps \
    libglfw3 \
    libglfw3-dev \
    libx11-dev \
    libglu1-mesa \
    mesa-utils \
    python3-pyqt5 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


# Install dependencies for COLMAP
RUN apt-get update && apt-get install -y \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    && rm -rf /var/lib/apt/lists/*

# Install COLMAP
RUN git clone https://github.com/colmap/colmap.git && \
    cd colmap && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja && \
    ninja && \
    ninja install

# Install Miniconda
RUN curl -sS https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && \
    bash miniconda.sh -b -u -p /opt/conda && \
    rm miniconda.sh

# Set conda path
ENV PATH=/opt/conda/bin:$PATH

# Create and activate conda environment for Torch-NGP
RUN conda create -n torch-ngp python=3.9 pytorch=1.10 torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia

# Clone torch-ngp repo
RUN git clone https://github.com/ashawkey/torch-ngp.git /app/torch-ngp

# Install Torch-NGP Dependencies
WORKDIR /app/torch-ngp/
RUN /opt/conda/bin/conda run -n torch-ngp pip install -r requirements.txt PyYAML

# Install correct PyTorch, torchvision, and torchaudio versions, as well as other needed packages
RUN /opt/conda/bin/conda run -n torch-ngp pip uninstall torch torchvision torchaudio numpy -y
RUN /opt/conda/bin/conda run -n torch-ngp pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN /opt/conda/bin/conda run -n torch-ngp pip install numpy==1.26.4
RUN /opt/conda/bin/conda run -n torch-ngp pip install imageio[ffmpeg]

# Set GCC 10 as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100 && \
    update-alternatives --config gcc && \
    update-alternatives --config g++

# Set Environment Variables for CUDA
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV nvcc=/usr/local/cuda/bin/nvcc
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV MODEL_CONFIG_PATH=/app/torch-ngp/nerf_config/model_config.yaml
 

# Optionally, you can modify bashrc for when you enter the container interactively
RUN echo "export PATH=/usr/local/cuda/bin:$PATH" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc && \
    echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc && \
    echo "export nvcc=/usr/local/cuda/bin/nvcc" >> ~/.bashrc

# Build Torch-NGP
RUN /opt/conda/bin/conda run -n torch-ngp bash scripts/install_ext.sh