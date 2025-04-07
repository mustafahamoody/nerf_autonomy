import torch
import yaml
import imageio.v2 as imageio

# NeRF Functions
from nerf_config.libs.nerf.utils import *
from nerf_config.libs.nerf.provider import NeRFDataset
from nerf_config.libs.nerf.network import NeRFNetwork
from nerf_config.config.model_options import ModelOptions



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

class NeRF_Renderer():
    def __init__(self):
        """
        get_rays_fn: function taking a 4x4 camera pose matrix and returning dict with 'rays_o' and 'rays_d'
        render_fn: function taking rays (origins, directions) and returning a dict with key 'image'
        """

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
        
        self.device = torch.device('cuda')
        
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
        # function taking a 4x4 camera pose matrix and returning dict with 'rays_o' and 'rays_d'
        self.get_rays_fn = lambda pose: get_rays(pose, self.dataset.intrinsics, self.dataset.H, self.dataset.W)  # Function to Generate Render rays
        # function taking rays from rays_fn (origins, directions) and returning a dict with key 'image' -- Render Image
        self.render_fn = lambda rays_o, rays_d: self.model.render(rays_o, rays_d, staged=True, bg_color=1., perturb=False, **vars(ModelOptions.opt()))


    def nerf_image(self, pose_matrix, image_height, image_width):
        # Get full (NeRF) image rays from camera position specifiled by pose matrix
        full_rays = self.get_rays_fn(pose_matrix)
        # Use full image for optimization
        output = self.render_fn(full_rays["rays_o"], full_rays["rays_d"])
        nerf_image = output["image"].reshape(image_height, image_width, 3) # Shapes NeRF image to match robot camera image

        # Create nerf_renders directory to store NeRF Renders        
        save_path = os.path.join(os.getcwd(), "localizer_images", "nerf_render.png")

        # Convert NeRF tensor to np array for saving
        # The 'render_fn' returns an image with the shape (H, W, 3) for RGB images
        nerf_image_np = nerf_image.cpu().detach().numpy()
        
        # Convert the image to [0, 255] range and uint8 format for saving
        nerf_image_np = (nerf_image_np * 255).astype(np.uint8)
        
        # Use OpenCV to save the image as PNG at the specified path
        imageio.imwrite(save_path, nerf_image_np)
        print(f"NeRF render saved at {save_path}")
