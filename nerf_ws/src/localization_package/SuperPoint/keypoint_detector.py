import torch
import cv2
import numpy as np
from pathlib import Path
import os


# Importing SuperPoint from superpoint_pytorch
from src.localization_package.SuperPoint.superpoint_pytorch import SuperPoint

def load_model(weights_path, device='cuda'):
    # Load superpoint model with pretrained weights
    model = SuperPoint()
    weight = torch.load(weights_path, map_location=device)

    model.load_state_dict(weight)

    # Set model to evaluation mode and move to device
    model.eval().to(device)
    
    return model


def detect_keypoints(image_path, model, device='cuda'):
    # Detect keypoints in an image using SuperPoint

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Normalize image
    image = image.astype(np.float32) / 255.0

    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image).float().to(device)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Forward pass through model
    with torch.no_grad():
        output = model({'image': image_tensor})

    print(f'----------------------------------------------{output}----------------------------------------------')

    # Extract keypoints, scores and descriptors
    keypoints = output['keypoints'][0].cpu().numpy()
    scores = output['keypoint_scores'][0].cpu().numpy()
    descriptors = output['descriptors'][0].cpu().numpy()

    return keypoints, scores, descriptors


def visualize_keypoints(image_path, keypoints, save_image=True):
    # Visualize keypoints on the image
    image = cv2.imread(image_path)
    if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

    # Draw Keypoints
    for keypoint in keypoints:
         x, y = int(keypoint[0]), int(keypoint[1])
         cv2.circle(image, (x,y), 2, (0, 255, 0), -1)

    # Save image to output path
    if save_image == True:
        os.makedirs(os.path.join(os.getcwd(), "image_keypoints"), exist_ok=True)
        output_path = os.path.join(os.getcwd(), "image_keypoints", os.path.basename(image_path))
        cv2.imwrite(output_path, image)
        
    return


def get_keypoints(image_path, weights_path='weights/superpoint_v6_from_tf.pth', device='cuda', save=True):
    
    # Check if the image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    # Check if the weights archive exists
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights archive not found at {weights_path}")

    # Load the model
    model = load_model(weights_path, device)

    # Detect keypoints
    keypoints, scores, descriptors = detect_keypoints(image_path, model, device)

    # Visualize keypoints
    visualize_keypoints(image_path, keypoints, save_image=save)

    return keypoints, scores, descriptors

# if __name__ == "__main__":
#     # Test
#     image_path = '2.png'  # Replace with your image path
#     keypoints, scores, descriptors = get_keypoints(image_path)