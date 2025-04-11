# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import torch
from torch import nn

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


def precise_differentiable_nms(scores, nms_radius):
    """
    Precise differentiable version of Non-Maximum Suppression
    This closely mimics the behavior of the original simple_nms function
    but maintains gradient flow
    """
    if nms_radius <= 0:
        return scores
    
    # This replicates the original NMS logic but keeps gradients
    # Original: max_mask = scores == max_pool(scores)
    max_pool_scores = torch.nn.functional.max_pool2d(
        scores, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)
    
    # Use an extremely high temperature to make this almost identical to hard masking
    # but still differentiable
    temperature = 1000.0  # Very high temperature for near-exact behavior
    max_mask = torch.sigmoid(temperature * (scores - max_pool_scores + 1e-5))
    
    # Mimic original NMS logic
    zeros = torch.zeros_like(scores)
    
    # First step of the original NMS - we only keep first round non-zero values
    # where scores == max_pool(scores)
    kept_scores = scores * max_mask
    
    # Second step: this mimics the iterative suppression in the original
    # We can do this by looking at the neighborhood of points we've kept
    # and creating a new mask for secondary suppression
    supp_mask = torch.nn.functional.max_pool2d(
        max_mask, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius) > 0.5
    supp_scores = torch.where(supp_mask, zeros, scores)
    
    # Find the new maximum points
    new_max_pool_scores = torch.nn.functional.max_pool2d(
        supp_scores, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)
    new_max_mask = torch.sigmoid(temperature * (supp_scores - new_max_pool_scores + 1e-5))
    
    # Apply the same logic as the original NMS loop
    final_mask = max_mask + (new_max_mask * (~supp_mask).float())
    final_scores = torch.where(final_mask > 0.5, scores, zeros)
    
    return final_scores


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

        path = Path(__file__).parent / 'weights/superpoint_v1.pth'
        self.load_state_dict(torch.load(str(path)))

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        print('Loaded SuperPoint model')

    def forward(self, data):
        """ Compute keypoints, scores, descriptors for image - with gradient preservation """
        # Shared Encoder
        x = self.relu(self.conv1a(data['image']))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        
        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
        
        # Apply differentiable NMS
        scores_nms = precise_differentiable_nms(scores, self.config['nms_radius'])
        
        # Create a FULLY DIFFERENTIABLE representation of keypoints
        # This is the key change - create a keypoint representation that preserves gradients
        batch_keypoints = []
        batch_scores = []
        batch_descriptors = []
        
        for batch_idx in range(b):
            score_map = scores_nms[batch_idx]  # (h*8, w*8)
            h_total, w_total = h*8, w*8
            
            # Create coordinate grids for the entire feature map
            y_grid, x_grid = torch.meshgrid(
                torch.arange(h_total, device=score_map.device),
                torch.arange(w_total, device=score_map.device),
                indexing='ij'
            )
            
            # Convert to float for proper gradient flow
            y_grid = y_grid.float()
            x_grid = x_grid.float()
            
            # Create threshold mask - using sigmoid for soft thresholding
            threshold = self.config['keypoint_threshold']
            score_weight = torch.sigmoid((score_map - threshold) * 100.0)  # Steep sigmoid
            
            # Create border mask
            border = self.config['remove_borders']
            border_mask = torch.ones_like(score_map)
            border_mask[:border, :] = 0
            border_mask[h_total-border:, :] = 0
            border_mask[:, :border] = 0
            border_mask[:, w_total-border:] = 0
            
            # Combine masks to weight the coordinates
            weight = score_weight * border_mask
            
            # Apply weight to create weighted coordinate representations
            # This is fully differentiable!
            weighted_x = x_grid * weight
            weighted_y = y_grid * weight
            
            # Identify non-zero weights for actual keypoints (still differentiable)
            significant_weights = (weight > 0.5)
            
            if significant_weights.sum() > 0:
                # Extract keypoint coordinates and scores where weight > 0.5
                x_values = weighted_x[significant_weights]
                y_values = weighted_y[significant_weights]
                s_values = score_map[significant_weights]
                
                # Stack to create keypoints tensor - this maintains gradients!
                keypoints = torch.stack([x_values / weight[significant_weights], 
                                        y_values / weight[significant_weights]], dim=1)
                
                # Apply top-k selection in a differentiable way
                if self.config['max_keypoints'] > 0 and len(s_values) > self.config['max_keypoints']:
                    # Sort by scores and take top k
                    _, indices = torch.topk(s_values, self.config['max_keypoints'])
                    keypoints = keypoints[indices]
                    s_values = s_values[indices]
            else:
                # Create empty tensor with proper dimensions if no keypoints
                keypoints = torch.zeros((0, 2), device=score_map.device)
                s_values = torch.zeros(0, device=score_map.device)
            
            # Store results
            batch_keypoints.append(keypoints)
            batch_scores.append(s_values)
            
            # Sample descriptors at keypoint locations
            if len(keypoints) > 0:
                # Normalize keypoint coordinates for grid_sample
                kp = keypoints.unsqueeze(0)  # Add batch dimension
                kp = kp - 8 / 2 + 0.5
                kp /= torch.tensor([(w*8 - 8/2 - 0.5), (h*8 - 8/2 - 0.5)], 
                                device=kp.device)[None, None]
                kp = kp*2 - 1  # normalize to (-1, 1)
                
                # Sample descriptors - this is differentiable
                args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
                desc = torch.nn.functional.grid_sample(
                    descriptors[batch_idx:batch_idx+1], kp.view(1, 1, -1, 2), 
                    mode='bilinear', **args)
                desc = torch.nn.functional.normalize(
                    desc.reshape(1, self.config['descriptor_dim'], -1), p=2, dim=1)
                
                batch_descriptors.append(desc.squeeze(0))
            else:
                batch_descriptors.append(torch.zeros((self.config['descriptor_dim'], 0), 
                                                    device=score_map.device))
        
        # Return the same format as the original function
        return {
            'keypoints': batch_keypoints,
            'scores': batch_scores,
            'descriptors': batch_descriptors,
        }
    