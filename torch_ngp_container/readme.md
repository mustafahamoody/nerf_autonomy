## Setup and Start the Container

First, Clone this repo
```bash
git clone https://github.com/mustafahamoody/torch_ngp_container
```

Change working directory to the docker folder in this repo: 
```bash
cd torch_ngp_container/docker/
```

Build and run container (detached) with correct dependencies and mounted folder for data: 
```bash 
docker compose up -d
```

Enter running container: 
```bash
docker-compose exec torch-ngp bash
``` 


## Using env_creation.py -- Inside Container

#### To create an environment for the first time (From video or photos all the way to NeRF)
```bash
python env_create.py --run
```
- You will be prompted to name the environment you want to create, select the data type and provide the path to the data
- **Videos or photos should be added to the parent folder, torch_ngp, (from your host machine) and when prompted, for the path type the file name (eg box.mp4)**



#### To create an environment from a data folder (Photos already processed through COLMAP)
```bash
python env_create.py --train
```

#### To view NeRF environment in GUI 
```bash
python env_create.py --view
```

***To visualise the created environment using the GUI you must connect your host machine display to the docker container using X11***
To Set this up run the following commands ***On your host machine:***

** For Linux Machines: **
``` bash
# 1. Install X11 Server
sudo apt-get install xorg openbox

# 2. Allow Docker to access your X server
xhost +local:docker
```


## Citation
If you use this work, please include the following citations

Original NeRF authors:
```
@misc{mildenhall2020nerf,
    title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
    author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
    year={2020},
    eprint={2003.08934},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

Instant-NGP:

```
@article{mueller2022instant,
    title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    journal = {arXiv:2201.05989},
    year = {2022},
    month = jan
}
```

torch-ngp:
```
@misc{torch-ngp,
    Author = {Jiaxiang Tang},
    Year = {2022},
    Note = {https://github.com/ashawkey/torch-ngp},
    Title = {Torch-ngp: a PyTorch implementation of instant-ngp}
}
```

torch_ngp_container:
```
@misc{torch_ngp_container,
    Author = {Mustafa Hamoody},
    Year = {2025},
    Note = {https://github.com/mustafahamoody/torch_ngp_container},
    Title = {torch_ngp_container: Docker container of torch-ngp with easier NeRF creation}

```
