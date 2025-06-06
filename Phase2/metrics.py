import numpy as np
import torch
import math
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2, max_value=1.0):
    """
    Calculate PSNR between two images
    
    Args:
        img1: First image (H, W, 3) or (H*W, 3)
        img2: Second image (H, W, 3) or (H*W, 3)
        max_value: Maximum value of the images (default: 1.0)
        
    Returns:
        psnr: Peak Signal-to-Noise Ratio
    """
    # Ensure images are numpy arrays
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Reshape to 2D if needed
    if len(img1.shape) == 3:
        h, w, c = img1.shape
        img1 = img1.reshape(-1, c)
        img2 = img2.reshape(-1, c)
    
    # Calculate MSE
    mse = np.mean((img1 - img2) ** 2)
    
    # Avoid division by zero
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    return 20 * math.log10(max_value / math.sqrt(mse))

def calculate_ssim(img1, img2, max_value=1.0):
    """
    Calculate SSIM between two images
    
    Args:
        img1: First image (H, W, 3)
        img2: Second image (H, W, 3)
        max_value: Maximum value of the images (default: 1.0)
        
    Returns:
        ssim_value: Structural Similarity Index
    """
    # Ensure images are numpy arrays
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Reshape to 3D if needed
    if len(img1.shape) == 2:
        img1 = img1.reshape(int(np.sqrt(img1.shape[0])), int(np.sqrt(img1.shape[0])), -1)
        img2 = img2.reshape(int(np.sqrt(img2.shape[0])), int(np.sqrt(img2.shape[0])), -1)
    
    # Calculate SSIM for each color channel and average
    ssim_value = np.mean([
        ssim(img1[:, :, i], img2[:, :, i], data_range=max_value)
        for i in range(img1.shape[2])
    ])
    
    return ssim_value

def evaluate_metrics(rendered_images, ground_truth_images):
    """
    Evaluate PSNR and SSIM metrics between rendered and ground truth images
    
    Args:
        rendered_images: List of rendered images
        ground_truth_images: List of ground truth images
        
    Returns:
        psnr_avg: Average PSNR
        ssim_avg: Average SSIM
    """
    psnr_values = []
    ssim_values = []
    
    for rendered, gt in zip(rendered_images, ground_truth_images):
        # Calculate metrics
        psnr = calculate_psnr(rendered, gt)
        ssim_val = calculate_ssim(rendered, gt)
        
        # Store values
        psnr_values.append(psnr)
        ssim_values.append(ssim_val)
    
    # Calculate averages
    psnr_avg = np.mean(psnr_values)
    ssim_avg = np.mean(ssim_values)
    
    return psnr_avg, ssim_avg