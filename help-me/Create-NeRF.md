# Turning envirionment captures into NeRF model

## Using env_creation.py -- Inside torch_ngp_container

1. Give your user access to the data folder
    ```
    nerf_autonomy$ sudo chown -R $UID $GID data
    ```
2. Put your environment captures in a folder under the data folder  
3. In the env_create.py file (under the torch_ngp_container folder), under DATA SETUP, set content_path = data/[Your capture folder name], and input_type = [ Capture Type: Image or Video]

Note: Every time you are asked for the environment name, enter the name from set up **Don't add the _nerf**

### To create an environment for the first time (From video or photos all the way to NeRF)
```bash
python env_create.py --run
```

### To create an environment from a data folder (Photos already processed through COLMAP)
```bash
python env_create.py --train
```

### To view NeRF environment in GUI 
```bash
python env_create.py --view
```