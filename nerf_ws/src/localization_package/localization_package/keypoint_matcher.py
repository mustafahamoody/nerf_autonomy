import torch
import numpy as np
import cv2
import os
import matplotlib.cm as cm

# Importing SuperPoint and SuperGlue Models
from src.localization_package.SuperGlue.models.matching import Matching

# Function to plot keypoints and matches
from src.localization_package.SuperGlue.models.utils import make_matching_plot

# ----------------- Image Processing and loading Functions -----------------
def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def read_image(path, resize, device):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)

    image = cv2.resize(image, (w_new, h_new)).astype('float32')

    image_tensor = torch.from_numpy(image/255.).float()[None, None].to(device)

    return image, image_tensor

# --------------------------- Keypoint Matcher ---------------------------
class KeypointMatcher():
    def __init__(self, robot_image_path, viz=True, keypoint_threshold=0.005, match_threshold=0.80, resize=[640, 480]):
        # Initialize Superpoint and Superglue models and Loads Robot Image
        
        self.device = torch.device('cuda')
        self.viz = viz  # Visualize the matches and dump the plots

        config = {
            'superpoint': {
                'nms_radius': 4,  # Non Maximum Suppression Radius
                'keypoint_threshold': keypoint_threshold,
                'max_keypoints': 1024
            },
            'superglue': {
                'weights': 'outdoor',
                'sinkhorn_iterations': 20, # Number of Sinkhorn iterations performed by SuperGlue
                'match_threshold': match_threshold,
            }
        }

        # Initalise Detector and Matcher
        self.matching = Matching(config).eval().to(self.device)

        # Load Robot Image Once -- Never Changes During Optimization
        if robot_image_path == None or not os.path.isfile(robot_image_path):
            print("Robot Image Not Found at robot_image_path:", robot_image_path)
            return
        
        self.robot_image, self.robot_image_tensor = read_image(robot_image_path, resize, self.device)


    def detect_and_match_keypoints(self, nerf_render_tensor):
        # Run SuperPoint, SuperGlue
        output = self.matching({'image0': self.robot_image_tensor, 'image1': nerf_render_tensor})

        # Format keypoint data into dictionary
        keypoint_data = {k: v[0] for k, v in output.items()}

        # Extract keypoints from the robot, nerf images
        keypoints0 = keypoint_data["keypoints0"]  # Shape: [N0, 2] (x, y) keypoints in image 0
        keypoints1 = keypoint_data["keypoints1"]  # Shape: [N1, 2] (x, y) keypoints in image 1

        # Extract match indices (-1 means no match)
        matches = keypoint_data["matches0"]  # Shape: [N0] - each index points to keypoints1 or -1 if no match

        # Filter out invalid matches (where matches[i] == -1)
        valid_matches = matches > -1

        if torch.sum(valid_matches) == 0:
            print("No Keypoints Matched with Enough Confidance for Localization")
            return None, None

        matched_keypoints0 = keypoints0[valid_matches]  # Get matched keypoints in image 0
        matched_keypoints1 = keypoints1[matches[valid_matches]]  # Get corresponding keypoints in image 1

        # Print Results
        print(f"Valid Keypoint Matches: {len(matched_keypoints0)}")
        # print("Image 0 Keypoints:", matched_keypoints0)
        # print("Image 1 Keypoints:", matched_keypoints1)

        if self.viz == True:
            with torch.no_grad():
                # Visualize the matches.
                match_confidance = keypoint_data['matching_scores0'][valid_matches].cpu().detach().numpy() # Get confidance of matches for visualization

                color = cm.jet(match_confidance)
                text = [
                    'SuperGlue',
                    'Keypoints: {}:{}'.format(len(keypoints0), len(keypoints1)),
                    'Matches: {}'.format(len(matches)),
                ]

                # Display extra parameter info.
                k_thresh = self.matching.superpoint.config['keypoint_threshold']
                m_thresh = self.matching.superglue.config['match_threshold']
                small_text = [
                    'Keypoint Threshold: {:.4f}'.format(k_thresh),
                    'Match Threshold: {:.2f}'.format(m_thresh),
                ]

                nerf_render = cv2.imread(str(os.path.join(os.getcwd(), "localizer_images", "nerf_render.png")), cv2.IMREAD_GRAYSCALE)
                viz_path = os.path.join(os.getcwd(), "localizer_images", "nerf_robot_keypoint_matches.png")

                make_matching_plot(
                    image0=self.robot_image, image1=nerf_render, kpts0=keypoints0.cpu().detach().numpy(), kpts1=keypoints1.cpu().detach().numpy(), 
                    mkpts0=matched_keypoints0.cpu().detach().numpy(), mkpts1=matched_keypoints1.cpu().detach().numpy(), color=color,
                    text=text, path=viz_path, show_keypoints=True, opencv_title='Matches', small_text=small_text)
        
        return matched_keypoints0, matched_keypoints1
            

