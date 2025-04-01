import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml
import os

# NeRF Functions
from nerf_config.libs.nerf.utils import *
from nerf_config.libs.nerf.provider import NeRFDataset
from nerf_config.libs.nerf.network import NeRFNetwork
from nerf_config.config.model_options import ModelOptions

# Superpoint keypoint detection
from src.localization_package.SuperPoint.keypoint_detector import get_keypoints

# Utiity to check autograd graph
from torchviz import make_dot

########################### Load NERF Model Config. ###########################

def load_config(file_path):
    """Load YAML configuration file with environment variable expansion."""
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)

        def expand_vars(item):
            if isinstance(item, str):
                return os.path.expandvars(item)
            elif isinstance(item, dict):
                return {key: expand_vars(value) for key, value in item.items()}
            elif isinstance(item, list):
                return [expand_vars(elem) for elem in item]
            else:
                return item

        config = expand_vars(config)
        return config

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration file: {e}")

########################### END Load NERF Model Config. ###########################

class Localizer():
    def __init__(self, learning_rate=0.01, max_iters=1000, batch_size=2048, kernel_size=5, dilate_iter=5, render=False, stream=False):
        """
        render_fn: function taking rays (origins, directions) and returning a dict with key 'image'
        get_rays_fn: function taking a 4x4 camera pose matrix and returning dict with 'rays_o' and 'rays_d'
        learning_rate: learning rate for optimization
        max_iters: maximum number of iterations for optimization if not converged
        kernal_size: size of the kernal used for dilation
        dilate_iter: number of iterations for dilation
        batch_size: number of pixels (from feature regions) used per iteration
        fixed_z: fixed z coordinate (camera height) -- For ground rovers
        save_render: Save intermediate renders to file
        stream: Stream intermediate renders to same file for live update
        """
        # Initialize parameters
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.dilate_iter = dilate_iter
        self.render_viz = render
        self.stream = stream
        self.device = torch.device('cuda')

        # Get config paths from environment variables with error checking
        model_config_path = os.environ.get('MODEL_CONFIG_PATH')
        trainer_config_path = os.environ.get('TRAINER_CONFIG_PATH')
        
        if not model_config_path:
            raise EnvironmentError("MODEL_CONFIG_PATH environment variable must be set")
        if not trainer_config_path:
            raise EnvironmentError("TRAINER_CONFIG_PATH environment variable must be set")
            
        # Load configurations
        try:
            self.config_model = load_config(model_config_path)
            self.config_trainer = load_config(trainer_config_path)

        except Exception as e:
            raise e
        
        # Initialize the NeRF model
        self.model = NeRFNetwork(
            encoding=self.config_model['model']['encoding'],
            bound=self.config_model['model']['bound'],
            cuda_ray=self.config_model['model']['cuda_ray'],
            density_scale=self.config_model['model']['density_scale'],
            min_near=self.config_model['model']['min_near'],
            density_thresh=self.config_model['model']['density_thresh'],
            bg_radius=self.config_model['model']['bg_radius'],
        )
        self.model.eval()  # Set the model to evaluation mode

        self.metrics = [PSNRMeter(),]
        self.criterion = torch.nn.MSELoss(reduction='none')

        # Initialize the Trainer (load weights from a checkpoint)
        self.trainer = Trainer(
            'ngp',
            opt=ModelOptions.opt(),
            model=self.model,
            device=self.device,
            workspace=self.config_trainer['trainer']['workspace'],
            criterion=self.criterion,
            fp16=self.config_model['model']['fp16'],
            metrics=self.metrics,
            use_checkpoint=self.config_trainer['trainer']['use_checkpoint'],
        )

        self.dataset = NeRFDataset(ModelOptions.opt(), device=self.device, type='test')  # Importing dataset in order to get the same camera intrinsics as training    
        self.render_fn = lambda rays_o, rays_d: self.model.render(rays_o, rays_d, staged=True, bg_color=1., perturb=False, **vars(ModelOptions.opt()))  # Function to Render Image
        self.get_rays_fn = lambda pose: get_rays(pose, self.dataset.intrinsics, self.dataset.H, self.dataset.W)  # Function to Generate Render rays


    def build_pose(self, camera_image, x=0.01, y=0.01, z=0.01, yaw=0.05): # OR call NeRF_Render
        # Preprocess Camera image 
        H, W, _ = camera_image.shape
        camera_image = (camera_image / 255).astype(np.float32)
        # Move tensor to CUDA and permute dimensions to match expected format
        camera_image_t = torch.tensor(camera_image, device='cuda').permute(2, 0, 1).unsqueeze(0)

        # Detect Image Keypoints
        keypoints, scores, descriptors = get_keypoints(camera_image, weights_path='../SuperPoint/weights/superpoint_v6_from_tf.pth')
        if keypoints.shape[0] == 0:
            print("No Features Found in Image")
            return None
        
        print(keypoints)





# ## Test
image_path = "1.png"
camera_image = cv2.imread("1.png")
# camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)
# Localizer().build_pose(camera_image)
weights_path = '../SuperPoint/weights/superpoint_v6_from_tf.pth'
keypoints, scores, descriptors = get_keypoints(image_path, weights_path)