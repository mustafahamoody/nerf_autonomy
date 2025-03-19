import torch
from tqdm import trange
from nerf.utils import *
from nerf.provider import NeRFDataset

from torch_ngp.nerf.utils import *
from torch_ngp.nerf.provider import NeRFDataset
from torch_ngp.nerf.network import NeRFNetwork
from encoding import get_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default Model options for -O flag
class options:
    def __init__(self):
        # General options
        self.path = 'box/'  # The path to transformms
        self.workspace = 'box_nerf/'  # The nerf directory
        self.ckpt = 'box_nerf/checkpoints/ngp.pth'
        
        # Flags and booleans
        self.fp16 = True  # Use AMP mixed precision training
        self.cuda_ray = True  # Use CUDA raymarching instead of pytorch
        self.preload = True  # Preload all data into GPU
        self.gui = False  # Start a GUI

        # Dataset options
        self.color_space = 'srgb'  # Color space (supports linear, srgb)
        self.bound = 2  # Assume the scene is bounded in a box[-bound, bound]^3
        self.scale = 0.33  # Scale camera location into box[-bound, bound]^3
        self.offset = [0, 0, 0]  # Offset of camera location
        self.dt_gamma = 1/128  # Adaptive ray marching gamma
        self.min_near = 0.2  # Minimum near distance for camera
        self.density_thresh = 10  # Threshold for density grid to be occupied
        self.bg_radius = -1  # Background model at sphere (bg_radius)
        
        # Other Config
        self.rand_pose = -1  # Random pose setting
        self.patch_size = 1

opt = options()
model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    )

model.eval()
metrics = [PSNRMeter(),]
criterion = torch.nn.MSELoss(reduction='none')
trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)
dataset = NeRFDataset(opt, device=device, type='test')        #Importing dataset in order to get the same camera intrinsics as training