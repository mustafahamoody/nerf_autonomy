import torch
import torchvision
from torchvision import transforms
import numpy as np
import cv2 
import matplotlib.pyplot as plt


# Simplified keypoint detector using SIFT.
def find_keypoints(self, camera_image, render=False):
        """Finds keypoints in an image using SIFT"""
        img = np.copy(camera_image)

        # Use Scale-Invariant Feature Transform (SIFT) to detect keypoints
        sift = cv2.SIFT_create()
        keypoints = sift.detect(img, None)

        # Get coordiantes of keypoints on the image as a numpy (int) array
        keypoint_coords = [tuple(keypoint.pt) for keypoint in keypoints]
        keypoint_coords = np.array(keypoint_coords).astype(int)
        
        # Remove duplicate keypoints
        keypoint_coords = np.unique(keypoint_coords, axis=0)

        if render:
            # Draw keypoints on image
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            feature_img = cv2.drawKeypoints(img_gray, keypoints, img)
            cv2.imshow("Detected Keypoint Features", feature_img)  #  Display image
            plt.imshow(feature_img)
            plt.show()

        return keypoint_coords

class PositionEstimator():
    def init(self, filter_cfg, agent, start_state, fiter=True, get_rays_fn=None, render_fn=None) -> None:

        """
        render_fn: function taking rays (origins, directions) and returning a dict with key 'image'
        get_rays_fn: function taking a 4x4 camera pose matrix and returning dict with 'rays_o' and 'rays_d'
        lrate: learning rate for optimization
        num_iters: number of gradient descent iterations
        batch_size: number of pixels (from feature regions) used per iteration
        fixed_z: fixed z coordinate (camera height)
        render_viz: if True, display intermediate renders
        """





        self.img = cv2.imread('1.png')
        # Parameters
        ...

    

    def estimate_relative_pose(self, camera_img): #start_state, sig, camera_img_pose=None):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img_w = camera_img.shape[0]
        img_h = camera_img.shape[1]

        # find keypoints in the image
        keypoints = self.find_keypoints(camera_img, render=False)  # x, y pixel coordinates of key features in image


        if len(keypoints) < 4:
            print("Feature Detection Failed")
            return None
        
        # Normalize image and conver to tensor
        img = camera_img
        # Convert to tensor
        transform = transforms.ToTensor() # Convert to tensor
        noisey_img = transform(img).to(device)

        # Normalize image using z-score normalization
        img = img.view(3, -1)
        mean, std = img.mean(dim=1).tolist(), noisey_img.std(dim=1).tolist()
        normalize = transforms.Normalize(mean=mean,std=std)
        img = normalize(img)

        # create meshgrid from the observed image
        coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, img_h - 1, img_h), np.linspace(0, img_w - 1, img_w)), -1), dtype=int)















# Testing
camera_img = cv2.imread('1.png')
img = camera_img
transform = transforms.ToTensor() # Convert to tensor
img = transform(img)
print('image shape:', img.shape)
img_data = img.view(3, -1)
mean, std = img_data.mean(dim=1).tolist(), img_data.std(dim=1).tolist()
print(f'img mean = {mean}')
print(f'img std = {std}')
normalize = transforms.Normalize(mean=mean,std=std)
img = normalize(img)
plt.imshow(img.permute(1, 2, 0))
plt.show()

# noisey_img = noisey_img(mean,std)



