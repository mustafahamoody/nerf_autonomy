# Use the NVIDIA CUDA devel image to ensure we have the compilers for .cu files
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Declare the build argument for non-interactive installs
ARG DEBIAN_FRONTEND=noninteractive

# Set basic locale environment variables (ROS2 requires UTF-8)
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV CUDA_HOME='/usr/local/cuda'

# Install basic tools needed to add the ROS repository
RUN apt-get update && apt-get install -y --no-install-recommends \
ca-certificates \
curl \
gnupg2 \
lsb-release \
&& rm -rf /var/lib/apt/lists/*

# Add the ROS2 apt repository for Humble
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc -o /tmp/ros.asc \
&& gpg --dearmor -o /etc/apt/trusted.gpg.d/ros.gpg /tmp/ros.asc \
&& rm /tmp/ros.asc \
&& echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list

# Install required packages (including ROS2)
RUN apt-get update && \
DEBIAN_FRONTEND=$DEBIAN_FRONTEND apt-get install -y --no-install-recommends \
locales \
curl \
gnupg2 \
lsb-release \
build-essential \
cmake \
git \
wget \
python3-pip \
python3-dev \
python3-colcon-common-extensions \
python3-rosdep \
ros-humble-ros-base \
libgl1-mesa-glx \
ros-humble-foxglove-bridge \
&& rm -rf /var/lib/apt/lists/*

# Set locale
RUN locale-gen en_US en_US.UTF-8

# Install additional Python dependencies (including PyTorch with CUDA support)
RUN pip3 install --no-cache-dir \
torch==2.6.0+cu126 \
torchvision==0.21.0+cu126 \
torchaudio==2.6.0+cu126 \
torch-ema==0.3 \
ninja==1.11.1.1 \
trimesh==4.4.9 \
opencv-python==4.10.0.84 \
tensorboardX==2.6.2.2 \
numpy==2.2.2 \
pandas==2.2.3 \
tqdm==4.66.5 \
matplotlib==3.10.0 \
PyMCubes==0.1.6 \
rich==13.8.1 \
pysdf==0.1.9 \
dearpygui==2.0.0 \
packaging==23.1 \
scipy==1.13.1 \
lpips==0.1.4 \
imageio==2.35.1 \
torchmetrics==1.4.2 \
PyYAML \
--extra-index-url https://download.pytorch.org/whl/cu126

# Create the ROS2 workspace directory and its src folder
ENV WORKSPACE=/nerf_ws
RUN mkdir -p ${WORKSPACE}/src

# Copy the ROS2 package from the build context
# Assumes the build context is set to the root of nerf_ws so that src/ exists.
COPY ../nerf_ws/src/ ${WORKSPACE}/src

# (Optional) Initialize rosdep – ignore errors if already initialized
RUN rosdep init || true && rosdep update || true

# Build the workspace using colcon (sourcing ROS2 environment in the same RUN command)
WORKDIR ${WORKSPACE}
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install"

# Set environment variables for configuration file paths (if needed)
ENV MODEL_CONFIG_PATH=${WORKSPACE}/nerf_config/model_config.yaml
ENV TRAINER_CONFIG_PATH=${WORKSPACE}/nerf_config/trainer_config.yaml
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set a fixed directory for Torch extensions and create it
ENV TORCH_EXTENSIONS_DIR=${WORKSPACE}/torch_extensions
RUN mkdir -p ${TORCH_EXTENSIONS_DIR}

# **Set CUDA architecture list explicitly to avoid detection issues**
ENV TORCH_CUDA_ARCH_LIST="8.6"

# Make nerf_config Files visible to all nodes
COPY nerf_config ${WORKSPACE}/nerf_config
ENV PYTHONPATH="${WORKSPACE}/:$PYTHONPATH"

# Precompile Torch extensions by importing the modules that compile them.
# This step will compile your extensions and store the compiled artifacts in TORCH_EXTENSIONS_DIR.
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && source ${WORKSPACE}/install/setup.bash && python3 -c 'import nerf_config.libs.shencoder; import nerf_config.libs.gridencoder.grid; import nerf_config.libs.raymarching.raymarching'"

# Copy the entrypoint script (placed in nerf_ws/docker/entrypoint.sh)
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint to source ROS2 and workspace setup files before running commands
ENTRYPOINT ["/entrypoint.sh"]

# Default command: run the occupancy package node.
CMD ["ros2", "run", "occupancy_package", "occupancy_node"]
