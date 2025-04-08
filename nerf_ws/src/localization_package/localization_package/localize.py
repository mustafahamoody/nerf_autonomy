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


# ----------------------------------------------------------------------------------------

def find_keypoints(camera_image, max_keypoints=10000, render=False):
    # Extract keypoints using SIFT (Scale-Invariant Feature Transform).
    image = np.copy(camera_image)

    # Make sure image is correct datatype for processing
    if image.dtype != 'uint8':
        image = cv2.convertScaleAbs(image, alpha=(255.0 / np.max(image)))
    
    # Convert to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create SIFT detector
    orb = cv2.ORB_create(nfeatures=max_keypoints)

    # Find keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(image_gray, None)
    
    if render:
        # Draw keypoints on image
        feature_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), 
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    else:
        feature_image = None
    
    # Extract keypoint coordinates
    keypoint_coords = np.array([keypoint.pt for keypoint in keypoints]).astype(int)
    
    # Remove duplicates
    keypoint_coords = np.unique(keypoint_coords, axis=0)
    
    extras = {
        'features': feature_image,
        'descriptors': descriptors
    }
    
    return keypoint_coords, extras


# Pose Coordinate Optimizer using NeRF
class PoseOptimizer():
    def __init__(self, render_fn, get_rays_fn, learning_rate=0.01, n_iters=500, batch_size=2048, 
                 kernel_size=3, dilate_iter=2, render=False, stream=False, fixed_z=0.0):
        """
        render_fn: function taking rays (origins, directions) and returning a dict with key 'image'
        get_rays_fn: function taking a 4x4 camera pose matrix and returning dict with 'rays_o' and 'rays_d'
        learning_rate: learning rate for optimization
        n_iters: number of gradient descent iterations
        batch_size: number of pixels (from feature regions) used per iteration
        fixed_z: fixed z coordinate (camera height) -- For ground rovers
        save_render: if True, save intermediate renders to file
        """
        self.render_fn = render_fn
        self.get_rays = get_rays_fn
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.dilate_iter = dilate_iter
        self.render = render
        self.stream = stream
        # self.fixed_z = fixed_z

        self.optimizer_params = {
            'lr': learning_rate,
            'betas': (0.9, 0.999)  # Add momentum with Adam's default betas
        }


    def estimate_pose(self, camera_image, x, y, z, roll, pitch, yaw):
        """
        camera_image: RGB image as a numpy array (range 0...255)
        Returns: (x, y, z) translation tuple, yaw (radians), and the loss history.
        """
        H, W = 480, 640
        camera_image = (camera_image / 255).astype(np.float32)
        # Move tensor to CUDA and permute dimensions to match expected format
        camera_image_t = torch.tensor(camera_image, device='cuda').permute(2, 0, 1).unsqueeze(0)
        
        # Detect Keypoints
        keypoints, extras = find_keypoints(camera_image, render=self.render)
        if keypoints.shape[0] == 0:
            print("No Features Found in Image")
            return None
        
        # Filter out keypoints that are out of bounds
        valid_mask = (keypoints[:, 0] >= 0) & (keypoints[:, 0] < W) & \
                    (keypoints[:, 1] >= 0) & (keypoints[:, 1] < H)
        keypoints = keypoints[valid_mask]  # Only keep valid ones
        
        # Create mask from keypoints and dilate it
        interest_mask = np.zeros((H, W), dtype=np.uint8)
        print(f'----------------{interest_mask.shape}----------------')
        print(f'----------------{keypoints[:, 0]}----------------')
        interest_mask[keypoints[:, 1], keypoints[:, 0]] = 1 
        
        interest_mask = cv2.dilate(interest_mask, np.ones((self.kernel_size, self.kernel_size), np.uint8),
                                    iterations=self.dilate_iter)
        interest_idxs = np.argwhere(interest_mask > 0)

        # Create optimizable (trainable) pose parameters -- with small non-zero values to avoid local minima
        pose_params = torch.tensor([x, y, z, pitch, roll, yaw], device='cuda', requires_grad=True)

        # Use Adam optimizer for poses
        optimizer = torch.optim.Adam([pose_params], **self.optimizer_params)
        
        # More aggressive (cyclical learning rate scheduler to help escape local minima
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate * 10,  # Much higher peak LR
            total_steps=self.n_iters,
            pct_start=0.5,  # Spend 30% of iterations ramping up
            cycle_momentum=True,
            div_factor=25.0,  # Initial LR will be max_lr/25
            final_div_factor=1000.0  # Final LR will be max_lr/1000
        )

        # Tracking loss history
        losses = []

        if self.render:
            # Create nerf_renders directory to store NeRF Renders
            render_dir = os.path.join(os.getcwd(), "nerf_renders")
            os.makedirs(render_dir, exist_ok=True)

        
        # Pose Optimization Loop
        for iter in range(1, self.n_iters + 1):
            optimizer.zero_grad()

            # Extract Parameters
            x, y, z, roll, pitch, yaw = pose_params[0], pose_params[1], pose_params[2], pose_params[3], pose_params[4], pose_params[5]

            cos_x, sin_x = torch.cos(pitch), torch.sin(pitch)
            cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
            cos_z, sin_z = torch.cos(roll), torch.sin(roll)

           # Rotation matrix for roll
            Rx = torch.tensor([
                [1, 0, 0],
                [0, cos_x, -sin_x],
                [0, sin_x, cos_x]
            ], device='cuda', dtype=torch.float32)

           # Rotation matrix for pitch
            Ry = torch.tensor([
                [-cos_y, 0, sin_y],
                [0, -1, 0],
                [-sin_y, 0, cos_y]
            ], device='cuda', dtype=torch.float32)

           # Rotation matrix for yaw
            Rz = torch.tensor([
                [cos_z, -sin_z, 0],
                [sin_z, cos_z, 0],
                [0, 0, 1]
            ], device='cuda', dtype=torch.float32)

            # Combine Rotation Matrix
            R = torch.mm(torch.mm(Rz, Ry), Rx)

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

            # Sample batch from interest region
            if interest_idxs.shape[0] <= self.batch_size:
                batch_idxs = interest_idxs
            else:
                idx = np.random.choice(interest_idxs.shape[0], self.batch_size, replace=False)
                batch_idxs = interest_idxs[idx]

            # Get rays for entire image
            rays = self.get_rays(pose_matrix)
            rays_o = rays["rays_o"].reshape(H, W, 3)
            rays_d = rays["rays_d"].reshape(H, W, 3)

            batch_y, batch_x = batch_idxs[:, 0], batch_idxs[:, 1]
            batch_y_t = torch.tensor(batch_y, dtype=torch.long, device='cuda')
            batch_x_t = torch.tensor(batch_x, dtype=torch.long, device='cuda')

            # Get rays for the sampled batch
            rays_o_batch = rays_o[batch_y_t, batch_x_t].unsqueeze(0)
            rays_d_batch = rays_d[batch_y_t, batch_x_t].unsqueeze(0)

            # Render the image from the current pose
            output = self.render_fn(rays_o_batch, rays_d_batch)

            # Important: Match the shape exactly for loss calculation
            rendered_rgb = output["image"].reshape(-1, 3) # Shape: [N, 3]

            # Get camera RGB values at the same pixels - ensure proper shape
            camera_rgb = camera_image_t[0, :, batch_y_t, batch_x_t].permute(1, 0) # Shape: [N, 3]

            # Make sure camera_rgb has gradients to avoid backward() error
            camera_rgb = camera_rgb.detach()

            # Ensure proper shapes before computing loss
            # print(f"rendered_rgb shape: {rendered_rgb.shape}, camera_rgb shape: {camera_rgb.shape}")

            # Use MSE loss with explicit shape matching
            loss = torch.nn.functional.mse_loss(rendered_rgb, camera_rgb)

            # Add small regularization term -- Weight Decay 
            reg_factor = 0.01
            reg_loss = reg_factor * torch.sum(pose_params**2)
            total_loss = loss + reg_loss

            # Backpropagate
            total_loss.backward()

            # Apply a more aggressive update for one step -- Every 50 iterations
            if iter % 50 == 0: 
                with torch.no_grad():
                    # Scale up the gradients for one step
                    pose_params.grad *= 5.0

            # Add random noise to the parameters to escape local minima -- Every 100 iterations
            if iter % 100 == 0:
                with torch.no_grad():
                    # Add random noise to parameters
                    noise_scale = 0.1 * (1 - iter/self.n_iters)  # Decreasing noise over time
                    noise = torch.randn_like(pose_params) * noise_scale
                    pose_params.add_(noise)
                    print(f"Added noise of scale {noise_scale} to parameters")

            # # Print gradients before stepping the optimizer
            # print(f"Gradients for pose_params: {pose_params.grad}")

            # Update parameters
            optimizer.step()
            scheduler.step()

            # Store loss
            losses.append(loss.item())

            # Print Progress
            if iter:
                print(f"--------------------------Iteration {iter}, Loss: {loss.item()}--------------------------")
                print(f"Current Learning rate: {scheduler.get_last_lr()[0]}")
                print(f"Current params: {pose_params.data}")
                # print(f"Gradients: {pose_params.grad}")


            if self.render and iter % 50 == 0 or self.render and iter == 1:
                # Render full image and save to render_viz folder for visualization
                with torch.no_grad():
                    full_rays = self.get_rays(pose_matrix)
                    full_output = self.render_fn(full_rays["rays_o"], full_rays["rays_d"])
                    full_rgb = full_output["image"].reshape(H, W, 3).cpu().numpy()
                
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(camera_image)
                plt.title("Camera Image")
                
                plt.subplot(1, 2, 2)
                plt.imshow(full_rgb)
                plt.title(f"Rendered at iter {iter}")
                
                plt.tight_layout()

                # Save to file
                if self.stream:
                    viz_path = os.path.join(render_dir, f'localization_stream.png')
                else:
                    viz_path = os.path.join(render_dir, f'localization_iter_{iter}.png')
                plt.savefig(viz_path)
                print(f"Visualization saved to {viz_path}")
                plt.close()


        # Extract final values
        x, y, z, roll, pitch, yaw = [p.item() for p in pose_params]

        # # Create final pose matrix
        # cos_y = np.cos(yaw)
        # sin_y = np.sin(yaw)
        
        # final_pose = np.eye(4)
        # final_pose[0, 0] = -cos_y
        # final_pose[0, 1] = sin_y
        # final_pose[1, 0] = -sin_y
        # final_pose[1, 1] = cos_y
        # final_pose[0, 3] = x
        # final_pose[1, 3] = y
        # final_pose[2, 3] = z

        return (x, y, z), roll, pitch, yaw
    

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


class Localize():
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

        self.dataset = NeRFDataset(ModelOptions.opt(), device=self.device, type='test')  # Importing dataset in order to get the same camera intrinsics as training    
        self.render_fn = lambda rays_o, rays_d: self.model.render(rays_o, rays_d, staged=True, bg_color=1., perturb=False, **vars(ModelOptions.opt()))  # Function to Render Image
        self.get_rays_fn = lambda pose: get_rays(pose, self.dataset.intrinsics, self.dataset.H, self.dataset.W)  # Function to Generate Render rays


    def run(self, camera_image, x=1.5, y=-0.45, z=0.05, roll=0, pitch=0, yaw=3.14):
        optimizer = PoseOptimizer(self.render_fn, 
                                  self.get_rays_fn, 
                                  learning_rate=1e-2,
                                  n_iters=500,
                                  batch_size=2048,
                                  render=True,
                                  stream=False)
        
        result = optimizer.estimate_pose(camera_image, x, y, z, roll, pitch, yaw) # Run Pose Optimizer on Image
        
        if result is not None:
            final_translation, final_roll, final_pitch, final_yaw = result
            # final_translation is a tuple (x, y, z)
            print("Final translation (x, y, z):", final_translation)
            print("Final roll, pitch, yaw (radians):", final_roll, final_pitch, final_yaw)

      
################## TEST ##################
# Load your sensor image (ensure it is in RGB).
camera_image = cv2.imread("localizer_images/robot_image.png")
camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)

Localize().run(camera_image)