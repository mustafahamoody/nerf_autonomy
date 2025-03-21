# nerf_autonomy
The Complete Repository for Autonomous Robot Navigation using Neural Radiance Fields (NeRFs). With quick and easy setup using Docker. 

Use ***torch-ngp-container*** to transform captured photos or a video of an environment into an accurately scaled digital model. \
Then, use ***nerf_ws*** to set up collision-free path planning and controls using ROS2 for real-time autonomous navigation in the environment, with your robot.

## Setup
This section covers how to install and set up nerf_autonomy -- To create and configure use your own environment, please refer to *create-your-environment.md*

#### Requirements:
- System with an NVIDIA GPU (CUDA Capable System)
- At Least 80 GB of free space (For Docker Containers and Data)
- Install and setup [Docker](https://www.docker.com/) 
- Add Nvidia Docker for containers to have access to the GPU


Begin by cloning this repository
```
$ git clone https://github.com/mustafahamoody/nerf_autonomy
$ cd nerf_autonomy
```

Build and run the Docker Containers **using Docker Compose**. Make sure the containers run as detached (-d flag) \
*This process may take a while (15 - 30 mins +) depending on your system* 
```
nerf_autonomy$ cd docker
nerf_autonomy/docker$ docker compose up -d # This will build and start both containers
```

*Note for Linux Systems*: If you recieve a permission denied error, you may need to add your user to the docker group to access the docker daemon. 
-- For more information see [this](https://docs.docker.com/engine/install/linux-postinstall/)
```
$ sudo groupadd docker
$ sudo usermod -aG docker $USER
# Log out and log back in for changes to take effect, then run docker compose up -d again
```

Enter the running container:
```
docker compose exec torch-ngp-container bash  # For environment creation with torch-ngp
docker compose exec nerf_ws bash  # To run ROS2 robot navigation nodes with nerf_ws
```
