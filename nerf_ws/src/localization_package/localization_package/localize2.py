import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import yaml
import os

# NeRF Functions
from nerf_config.libs.nerf.utils import *
from nerf_config.libs.nerf.provider import NeRFDataset
from nerf_config.libs.nerf.network import NeRFNetwork
from nerf_config.config.model_options import ModelOptions


class PoseOptimizer():
    def __init__(self, pose_params, camera_image, nerf_image, max_iters=1000, learning_rate=0.01, render=False):
        self.pose_params = pose_params
        self.camera_image = camera_image
        self.nerf_image = nerf_image
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        self.render = render
        self.loss_value = np.inf

    # Estimate Pose by optimizing pixel values between camera image and rendered nerf image
    # Returns: (x, y, z) translation tuple, yaw (radians)
    def optimize_pose(self, add_noise=True):
        # Initialize pose parameters locally
        pose_params = self.pose_params
        
        # Initialize optimizer
        optimizer = torch.optim.SGD([pose_params], lr=self.learning_rate, weight_decay=1e-5) # Use Classic gradient descent with L2 Regularization (weight decay)

        # Add (cyclical) learning rate scheduler to help escape local minima
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.learning_rate, max_lr=self.learning_rate * 10, step_size_up=100, step_size_down=100)

        if self.render:
            # Create nerf_renders directory to store NeRF Renders
            render_dir = os.path.join(os.getcwd(), "nerf_renders")
            os.makedirs(render_dir, exist_ok=True)


        iter = 0
        # Optimization loop -- Keep optimizing until loss is below threshold or max iterations reached
        while self.loss_value > 0.0001:
            iter += 1

            if iter >= self.max_iters:
                print("Max iterations Reached")
                break

            # calculate loss between images pixel values
            loss = torch.nn.functional.mse_loss(self.camera_image, self.nerf_image)

            if add_noise:
                # Add random noise to parameters every 50 iterations to help escape local minima
                if iter % 50 == 0:
                    with torch.no_grad():
                        # Decrease noise over time as appoching convergence
                        noise_scale = 0.01 * (1 - iter / self.max_iters)
                        noise = torch.randn_like(pose_params) * noise_scale
                        pose_params.add_(noise)
                        print(f"Added {noise_scale*100}% noise to parameters")
            

            # Zero gradients, perform a backward pass, and update parameters.
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()

            # Update loss value
            self.loss_value = loss.item()

            # Print Progress
            if iter <= 3 or iter % 100 == 0:
                print(f"--------------------------Iteration {iter}, Loss: {self.loss_value}--------------------------")
                print(f"Current Learning rate: {scheduler.get_last_lr()[0]}")
                # print(f"Current params: {pose_params.data}")
                # print(f"Gradients: {pose_params.grad}")

            if self.render and (iter == 1 or iter % 100 == 0):
                # Render full image and save to render_viz folder for visualization
                
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(self.camera_image)
                plt.title("Camera Image")
                
                plt.subplot(1, 2, 2)
                plt.imshow(self.nerf_image)
                plt.title(f"Rendered at iter {iter}")
                
                plt.tight_layout()

                # Save to file
                viz_path = os.path.join(render_dir, f'localization_iter_{iter}.png')
                plt.savefig(viz_path)

                # For continuous update stream to same file
                viz_path = os.path.join(render_dir, f'localization_stream.png')
                plt.savefig(viz_path)

                print(f"Visualization saved to {viz_path}")
                plt.close()

        # Return optimized pose parameters
        return pose_params.detach().cpu().numpy()





# function to load nerf_config
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


# Localizer Class with NeRF Model
class Localizer():
    def __init__(self):

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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        # Import dataset to get camera intrinsics (from training)    
        self.dataset = NeRFDataset(ModelOptions.opt(), device=self.device, type='test')
        # Function to Render Image
        self.render_fn = lambda rays_o, rays_d: self.model.render(rays_o, rays_d, staged=True, bg_color=1., perturb=False, **vars(ModelOptions.opt()))  
        # Function to Generate Render rays
        self.get_rays_fn = lambda pose: get_rays(pose, self.dataset.intrinsics, self.dataset.H, self.dataset.W)

        # Set image features
        self.image_height = None
        self.image_width = None


    # Image Keypoints detector using SIFT
    def get_keypoints(self, camera_image, render=False):
        image = np.copy(camera_image)

        if image is None:
            print("No Image Recieved")
        # else:
        #     print(f'-------------------------------{image.shape, image.dtype}-------------------------------')

        if image.dtype != 'uint8':
            # Normalize to [0, 255] and convert to uint8
            image = cv2.convertScaleAbs(image, alpha=(255.0 / np.max(image)))
        
        # Convert image to grayscale -- SIFT works best with grayscale images
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Use Scale-Invariant Feature Transform (SIFT) to detect keypoints
        sift = cv2.SIFT_create()
        keypoints = sift.detect(image_gray, None)

        if render:
            # Draw keypoints on image
            feature_image = cv2.drawKeypoints(image_gray, keypoints, image)
        else:
            feature_image = None

        # Extract (x,y) coords. of keypoints on the image (as a numpy int array)
        keypoint_coords = np.array([keypoint.pt for keypoint in keypoints]).astype(int)
        
        # Remove duplicate keypoints
        keypoint_coords = np.unique(keypoint_coords, axis=0)

        extras = {'features': feature_image}

        return keypoint_coords, extras


    # Render Image from NeRF
    def nerf_image(self, camera_image, pose_matrix, only_keypoints=False, dilate_iter=2, kernel_size=3, batch_size=2048):

        # Get image dimensions
        self.image_height, self.image_width, _ = camera_image.shape

        # Get full image rays
        full_rays = self.get_rays_fn(pose_matrix)
        
        if only_keypoints:
        # Focus optimization on only keypoints

            keypoints = self.get_keypoints(camera_image, render=False)

            if keypoints.shape[0] == 0:
                print("No Features Found in Image")
                return None
        
            # Create mask from keypoints and dilate it
            interest_mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
            interest_mask[keypoints[:, 1], keypoints[:, 0]] = 1 
            
            interest_mask = cv2.dilate(interest_mask, np.ones((kernel_size, kernel_size), np.uint8),
                                        iterations=dilate_iter)
            interest_idxs = np.argwhere(interest_mask > 0)

            # Sample batch from whole interest region (if less than batch size)
            if interest_idxs.shape[0] <= batch_size:
                batch_idxs = interest_idxs
            else:
                # Randomly sample parts from interest region, until batch size is reached
                idx = np.random.choice(interest_idxs.shape[0], batch_size, replace=False)
                batch_idxs = interest_idxs[idx]

            batch_y, batch_x = batch_idxs[:, 0], batch_idxs[:, 1]
            batch_y_t = torch.tensor(batch_y, dtype=torch.long, device='cuda')
            batch_x_t = torch.tensor(batch_x, dtype=torch.long, device='cuda')

            rays_o = full_rays["rays_o"].reshape(self.image_height, self.image_width, 3)
            rays_d = full_rays["rays_d"].reshape(self.image_height, self.image_width, 3)

            # Get rays for the sampled batch
            rays_o_batch = rays_o[batch_y_t, batch_x_t].unsqueeze(0)
            rays_d_batch = rays_d[batch_y_t, batch_x_t].unsqueeze(0)

            # Render the image from the current pose
            output = self.render_fn(rays_o_batch, rays_d_batch)

        else:
            # Use full image for optimization
            output = self.render_fn(full_rays["rays_o"], full_rays["rays_d"])
        
        # Get rendered nerf image
        rendered_image = output["image"].reshape(self.image_height, self.image_width, 3).cpu().numpy()

        return rendered_image    #.squeeze().cpu().numpy()


    def localize(self, camera_image, start_pose=[0.1, 0.1, 0.1, 0.05]):
        # Build tracked parametres tensor
        pose_params = torch.tensor(start_pose, requires_grad=True, device='cuda')

        # Create pose matrix to render image from NeRF
        # Extract Parameters
        x, y, z, yaw = pose_params[0], pose_params[1], pose_params[2], pose_params[3]

        # Create rotation matrix for yaw
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)
        R = torch.zeros((3, 3), device='cuda')
        R[0, 0] = cos_y
        R[0, 1] = -sin_y
        R[1, 0] = sin_y
        R[1, 1] = cos_y
        R[2, 2] = 1.0
        
        # Create translation vector
        t = torch.zeros((3, 1), device='cuda')
        t[0, 0] = x
        t[1, 0] = y
        t[2, 0] = z

        # Build pose matrix (top 3x4 part)
        top_rows = torch.cat([R, t], dim=1)

        # Add homogeneous row
        bottom_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device='cuda')

        pose_matrix = torch.cat([top_rows, bottom_row], dim=0).unsqueeze(0) # Add batch dim


        # Render image from NeRF
        nerf_image = self.nerf_image(camera_image, pose_matrix)

        # Convert camera image and nerf image to tensor -----------------------------------------------------------------------------------
        # camera_image = torch.tensor(camera_image, device='cuda')
        nerf_image = torch.tensor(nerf_image, dtype=torch.float32, device=self.device)


        # Initialize Pose Optimizer
        pose_optimizer = PoseOptimizer(pose_params, camera_image, nerf_image, max_iters=1000, learning_rate=0.01, render=True)
        
        # optimize pose and return optimized pose
        optimized_pose = pose_optimizer.optimize_pose(add_noise=True)

        print('Optimization Complete') 
        print(f'Optimized pose: x={optimized_pose[0]}, y={optimized_pose[1]}, z={optimized_pose[2]}, yaw={optimized_pose[3]}')

        return optimized_pose



############# TEST #############
camera_image = cv2.imread("1.png")

if camera_image is None:
        print("Error: Could not read image file")
else:
    camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)

    # try:
optimized_pose = Localizer().localize(camera_image)
print(f"Final pose: {optimized_pose}")
    # except Exception as e:
        # print(f"Error during localization: {e}")
