import argparse
import glob
import json
import os
import numpy as np
import torch
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import traceback


from NeRFModel import *
from metrics import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
np.random.seed(0)
torch.manual_seed(0)



def estimate_scene_bounds(poses):

    # Extract camera positions
    cam_positions = poses[:, :3, 3]
    
    # Calculate center of the scene (average of camera positions)
    center = np.mean(cam_positions, axis=0)
    
    # Calculate distances from cameras to center
    distances = np.linalg.norm(cam_positions - center, axis=1)
    
    # Set near/far based on minimum and maximum distances
    near_factor = 0.8  # Closer than the closest camera
    far_factor = 5.0   # Further than the furthest camera
    
    near = np.min(distances) * near_factor
    far = np.max(distances) * far_factor
    
    
    return near, far

def sample_pdf(bins, weights, N_samples, det=False):

    # Get pdf
    weights = weights + 1e-5  # Prevent NaNs
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (N_rays, N_samples)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=cdf.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (N_rays, N_samples, 2)

    # Get the relevant CDF values
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    # Calculate sample positions
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def volume_rendering(model, pts, ray_directions, z_vals, args):

    # Reshape for model input
    N_rays, N_samples = pts.shape[:2]
    pts_flat = pts.reshape(-1, 3)  # [N_rays*N_samples, 3]
    
    # Debug info for point range
    if N_rays > 0 and N_samples > 0:
        pts_min = pts_flat.min(dim=0)[0]
        pts_max = pts_flat.max(dim=0)[0]
        
    
    # Expand ray directions for each sample
    dir_flat = ray_directions[:, None, :].expand(pts.shape).reshape(-1, 3)  # [N_rays*N_samples, 3]
    
    # Normalize directions
    dir_flat = dir_flat / (torch.norm(dir_flat, dim=-1, keepdim=True) + 1e-10)

    # Evaluate model in chunks to avoid OOM
    chunk_size = 32768  # Adjust based on GPU memory
    all_rgb = []
    all_sigma = []

    for i in range(0, pts_flat.shape[0], chunk_size):
        pts_chunk = pts_flat[i:i + chunk_size]
        dir_chunk = dir_flat[i:i + chunk_size]

        # Forward pass through the model
        rgb_chunk, sigma_chunk = model(pts_chunk, dir_chunk)
        
        # Ensure sigma is 1D (squeeze if necessary)
        if sigma_chunk.shape[-1] == 1:
            sigma_chunk = sigma_chunk.squeeze(-1)

        all_rgb.append(rgb_chunk)
        all_sigma.append(sigma_chunk)
        

    # Concatenate chunks
    rgb = torch.cat(all_rgb, dim=0).reshape(N_rays, N_samples, 3)  # [N_rays, N_samples, 3]
    sigma = torch.cat(all_sigma, dim=0).reshape(N_rays, N_samples)  # [N_rays, N_samples]

    # Calculate distances between samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # Add a large distance at the end
    dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e10], dim=-1)  # [N_rays, N_samples]

    # Multiply by norm of ray direction to convert to real-world distance
    dists = dists * torch.norm(ray_directions[..., None, :], dim=-1)

    # Calculate alpha values (opacity) with a small epsilon to prevent numerical instability
    alpha = 1.0 - torch.exp(-sigma * dists + 1e-10)  # [N_rays, N_samples]

    # Calculate weights (transmittance * alpha) with numerical stability
    # T_i = exp(-sum_{j=1}^{i-1} sigma_j * delta_j)
    # T = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
    weights = alpha * T  # [N_rays, N_samples]
    

    # Calculate color and depth
    exposure = 1.5  # Adjust this value (>1 = brighter)
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2) * exposure
    # rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [N_rays, 3]
    depth_map = torch.sum(weights * z_vals, dim=-1)  # [N_rays]
    


    return rgb_map, depth_map, weights


def render(model, rays_origin, rays_direction, near, far, args, use_hierarchical=True):

    # Get the number of rays
    N_rays = rays_origin.shape[0]
    device = rays_origin.device

    # Sample points along each ray (stratified sampling)
    t_vals = torch.linspace(0., 1., args.n_sample, device=device)
    z_vals = near * (1. - t_vals) + far * t_vals
    
    # Apply stratified sampling (perturb sampling points)
    if args.mode == 'train':
        # Add noise to samples for regularization
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand([N_rays, args.n_sample], device=device)
        z_vals = lower + (upper - lower) * t_rand

    # Expand z_vals to match the ray batch size
    z_vals = z_vals.expand([N_rays, args.n_sample])

    # Calculate sample points in 3D space
    pts = rays_origin[..., None, :] + rays_direction[..., None, :] * z_vals[..., :, None]  # [N_rays, n_samples, 3]

    # Process coarse model
    coarse_results = volume_rendering(model, pts, rays_direction, z_vals, args)
    rgb_map_coarse, depth_map_coarse, weights_coarse = coarse_results
    
    # Hierarchical sampling (fine model)
    if use_hierarchical and hasattr(model, 'fine_network') and model.fine_network is not None:
        # Create fine network if not already created
        if model.fine_network is None:
            model.create_fine_network()
            
        
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights_coarse[..., 1:-1], args.n_sample_fine, det=(args.mode != 'train'))
        z_samples = z_samples.detach()
        
        # Combine coarse and fine samples and sort
        z_vals_combined, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
        
        # Calculate new points
        pts_fine = rays_origin[..., None, :] + rays_direction[..., None, :] * z_vals_combined[..., :, None]
        
        # Process fine model
        fine_results = volume_rendering(model.fine_network, pts_fine, rays_direction, z_vals_combined, args)
        rgb_map_fine, depth_map_fine, weights_fine = fine_results
        
        return rgb_map_fine, depth_map_fine, weights_fine, rgb_map_coarse
    
    return rgb_map_coarse, depth_map_coarse, weights_coarse


def loadDataset(data_path, mode):

    # Load transforms.json file which contains camera info
    transforms_path = os.path.join(data_path, f"transforms_{mode}.json")
    with open(transforms_path, 'r') as f:
        meta = json.load(f)

    # Get camera info
    camera_angle_x = float(meta['camera_angle_x'])

    # Load images and poses
    images = []
    poses = []

    # For each frame in the dataset
    for frame in meta['frames']:
        # Load image
        img_path = os.path.join(data_path, frame['file_path'] + '.png')
        img = imageio.imread(img_path)

        # Convert to float and normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Extract only RGB channels if image has alpha channel
        if img.shape[-1] == 4:
            img = img[..., :3]

        # Add to list
        images.append(img)

        # Get camera pose (convert from matrix to 4x4 transform)
        pose = np.array(frame['transform_matrix'], dtype=np.float32)
        poses.append(pose)

    # Convert to numpy arrays
    images = np.stack(images, axis=0)
    poses = np.stack(poses, axis=0)

    # Get image height, width
    H, W = images[0].shape[:2]

    # Calculate focal length from camera_angle_x
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    # Create camera info dictionary
    camera_info = {
        'height': H,
        'width': W,
        'focal': focal,
        'camera_angle_x': camera_angle_x
    }

    print(f"Loaded {len(images)} {mode} images of shape {images[0].shape}")

    return images, poses, camera_info


def get_rays(H, W, focal, c2w):

    # Create a meshgrid for pixel coordinates
    i, j = torch.meshgrid(torch.arange(W, device=c2w.device),
                          torch.arange(H, device=c2w.device),
                          indexing='xy')
    
    # Convert to camera coordinates (z points in negative direction)
    dirs = torch.stack([(i - W * 0.5) / focal,
                        -(j - H * 0.5) / focal,
                        -torch.ones_like(i)], dim=-1)  # (H, W, 3)
    
    # Get rotation matrix from camera to world
    rot = c2w[:3, :3]
    
    # Transform ray directions to world space
    rays_d = torch.matmul(dirs.reshape(-1, 3), rot.T)  # (H*W, 3)
    
    # Explicitly normalize ray directions
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    # Set ray origins to camera position
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H*W, 3)
    

    
    return rays_o, rays_d


def generateBatch(images, poses, camera_info, args, is_train=True):

    H, W = camera_info['height'], camera_info['width']
    focal = camera_info['focal']
    
    if is_train:
        # Randomly select an image for training
        img_idx = np.random.randint(0, images.shape[0])
        img = torch.tensor(images[img_idx], device=device)
        pose = torch.tensor(poses[img_idx], device=device)
        
        # Generate rays for the entire image
        rays_o, rays_d = get_rays(H, W, focal, pose)
        
        # Randomly select a subset of rays
        select_inds = np.random.choice(rays_o.shape[0], size=args.n_rays_batch, replace=False)
        rays_o = rays_o[select_inds]
        rays_d = rays_d[select_inds]
        
        # Get target RGB values
        target_rgb = img.reshape(-1, 3)[select_inds]
        
    else:
        # For testing, use specific image
        img_idx = 0
        img = torch.tensor(images[img_idx], device=device)
        pose = torch.tensor(poses[img_idx], device=device)
        
        # Generate rays for the entire image
        rays_o, rays_d = get_rays(H, W, focal, pose)
        
        # Get target RGB values
        target_rgb = img.reshape(-1, 3)
    
    # Create batch dictionary
    batch = {
        'rays_o': rays_o,
        'rays_d': rays_d,
        'target_rgb': target_rgb,
        'img_idx': img_idx,
        'H': H,
        'W': W
    }
    
    return batch


def train(images, poses, camera_info, args):

    # Dynamically estimate scene bounds
    near, far = estimate_scene_bounds(poses)
    
    # Create model
    # model = NeRFmodel(embed_pos_L=args.n_pos_freq, embed_direction_L=args.n_dirc_freq).to(device)
    model = NeRFmodel(embed_pos_L=args.n_pos_freq, embed_direction_L=args.n_dirc_freq, use_positional_encoding=args.use_positional_encoding).to(device)
    # Create fine network
    model.create_fine_network()
    model.fine_network.to(device)
    
    # Initialize weights with a better scale
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    model.fine_network.apply(init_weights)
    
    # Create optimizers
    optimizer = torch.optim.Adam(params=list(model.parameters()) + list(model.fine_network.parameters()), 
                               lr=float(args.lrate), betas=(0.9, 0.999))
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1**(1/args.max_iters))
    
    # Create directories
    os.makedirs(args.logs_path, exist_ok=True)
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.images_path, exist_ok=True)
    
    # Create tensorboard writer
    writer = SummaryWriter(args.logs_path)
    
    # Load checkpoint if specified
    start_iter = 0
    if args.load_checkpoint:
        checkpoint_files = sorted(glob.glob(f"{args.checkpoint_path}/*.pth"))
        if checkpoint_files:
            try:
                checkpoint = torch.load(checkpoint_files[-1])
                model.load_state_dict(checkpoint['model_state_dict'])
                model.fine_network.load_state_dict(checkpoint['fine_network_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_iter = checkpoint['iter'] + 1
                print(f"Loaded checkpoint from {checkpoint_files[-1]}, resuming from iteration {start_iter}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                args.load_checkpoint = False
    
    # Start with low resolution during early training
    downscale_until_iter = min(500, args.max_iters // 10)
    
    # Training loop
    pbar = tqdm(range(start_iter, args.max_iters), desc="Training")
    for iter in pbar:
        try:
            # Generate batch
            batch = generateBatch(images, poses, camera_info, args, is_train=True)
            
            # Forward pass
            model.train()
            model.fine_network.train()
            
            # Render with hierarchical sampling
            rgb_fine, depth_fine, weights_fine, rgb_coarse = render(model, batch['rays_o'], batch['rays_d'], 
                                                                   near, far, args, use_hierarchical=True)
            
            # Calculate losses
            coarse_loss = torch.mean((rgb_coarse - batch['target_rgb'])**2)
            fine_loss = torch.mean((rgb_fine - batch['target_rgb'])**2)
            loss = coarse_loss + fine_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Apply gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.fine_network.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Log stats
            if iter % 100 == 0:
                print(f"\nIteration {iter}: loss={loss.item():.5f}, coarse={coarse_loss.item():.5f}, fine={fine_loss.item():.5f}")
                print(f"RGB range: {rgb_fine.min().item():.3f} to {rgb_fine.max().item():.3f}")
                print(f"Depth range: {depth_fine.min().item():.3f} to {depth_fine.max().item():.3f}")
                print(f"Weights stats: min={weights_fine.min().item():.5f}, max={weights_fine.max().item():.5f}, mean={weights_fine.mean().item():.5f}")
            
            # Log
            pbar.set_postfix({'loss': loss.item()})
            writer.add_scalar('Loss/total', loss.item(), iter)
            writer.add_scalar('Loss/coarse', coarse_loss.item(), iter)
            writer.add_scalar('Loss/fine', fine_loss.item(), iter)
            
            # Save checkpoint and render preview
            if (iter + 1) % args.save_ckpt_iter == 0 or iter == args.max_iters - 1:
                # Save checkpoint
                checkpoint_path = f"{args.checkpoint_path}/model_{iter + 1}.pth"
                torch.save({
                    'iter': iter,
                    'model_state_dict': model.state_dict(),
                    'fine_network_state_dict': model.fine_network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item()
                }, checkpoint_path)
                
                # Render and save preview at gradually increasing resolution
                with torch.no_grad():
                    model.eval()
                    model.fine_network.eval()
                    
                    # Use lower resolution for early iterations
                    preview_downscale = 8 if iter < downscale_until_iter else 4
                    renderPreview(model, poses, camera_info, near, far, args, iter, preview_downscale)
                
        except Exception as e:
            print(f"Error during iteration {iter}: {e}")
            traceback.print_exc()
            continue
    
    writer.close()


def renderPreview(model, poses, camera_info, near, far, args, iter, downscale=4):

    H, W = camera_info['height'], camera_info['width']
    focal = camera_info['focal']
    
    # Use lower resolution for preview
    H_preview, W_preview = H // downscale, W // downscale
    focal_preview = focal / downscale
    
    # Use first camera pose
    pose = torch.tensor(poses[0], device=device)
    
    # Make sure directories exist
    preview_dir = os.path.join(args.images_path, "previews")
    os.makedirs(preview_dir, exist_ok=True)
    
    # Generate rays
    rays_o, rays_d = get_rays(H_preview, W_preview, focal_preview, pose)
    
    # Render in chunks
    rgb_chunks = []
    depth_chunks = []
    chunk_size = 4096
    
    with torch.no_grad():
        for i in range(0, rays_o.shape[0], chunk_size):
            chunk_o = rays_o[i:i+chunk_size]
            chunk_d = rays_d[i:i+chunk_size]
            
            # Render with hierarchical sampling
            rgb_chunk, depth_chunk, _, _ = render(model, chunk_o, chunk_d, near, far, args, use_hierarchical=True)
            
            rgb_chunks.append(rgb_chunk.cpu())
            depth_chunks.append(depth_chunk.cpu())
    
    # Combine chunks
    rgb = torch.cat(rgb_chunks, dim=0).reshape(H_preview, W_preview, 3).numpy()
    depth = torch.cat(depth_chunks, dim=0).reshape(H_preview, W_preview).numpy()
    

    
    # Force some variation if image is completely uniform
    if np.std(rgb) < 1e-5:
        print("WARNING: Output image has no variation! Applying normalization to make preview visible.")
        # Add small noise to make image visible
        rgb = rgb + np.random.normal(0, 0.01, rgb.shape)
    
    # Normalize depth for visualization
    depth_min, depth_max = depth.min(), depth.max()
    if depth_max > depth_min:
        depth_vis = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_vis = depth
    
    # Save images
    rgb_path = os.path.join(preview_dir, f"rgb_iter_{iter + 1}.png")
    depth_path = os.path.join(preview_dir, f"depth_iter_{iter + 1}.png")
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.clip(rgb, 0, 1))
    plt.title(f"RGB - Iteration {iter+1}")
    plt.subplot(1, 2, 2)
    plt.imshow(depth_vis, cmap='viridis')
    plt.title(f"Depth - Iteration {iter+1}")
    plt.savefig(os.path.join(preview_dir, f"combined_iter_{iter + 1}.png"))
    plt.close()
    
    plt.imsave(rgb_path, np.clip(rgb, 0, 1))
    plt.imsave(depth_path, depth_vis, cmap='viridis')
    
    print(f"Saved preview to {preview_dir}")
    
    # Also save a progress image showing RGB outputs over time
    if (iter + 1) % (args.save_ckpt_iter * 2) == 0 or iter == args.max_iters - 1:
        # Find all RGB images
        rgb_files = sorted(glob.glob(os.path.join(preview_dir, "rgb_iter_*.png")))
        if len(rgb_files) > 0:
            # Load most recent images (up to 9)
            recent_files = rgb_files[-9:]
            num_images = len(recent_files)
            cols = min(3, num_images)
            rows = (num_images + cols - 1) // cols
            
            plt.figure(figsize=(4*cols, 4*rows))
            for i, img_path in enumerate(recent_files):
                iter_num = int(os.path.basename(img_path).split('_')[2].split('.')[0])
                img = plt.imread(img_path)
                plt.subplot(rows, cols, i+1)
                plt.imshow(img)
                plt.title(f"Iteration {iter_num}")
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.images_path, "training_progress.png"))
            plt.close()


def normalize(v):
    """Normalize a vector"""
    return v / np.linalg.norm(v)


def renderNovelViews(model, poses, camera_info, near, far, args):
    """Render novel views for evaluation or visualization"""
    print("Rendering novel views...")
    
    H, W = camera_info['height'], camera_info['width']
    focal = camera_info['focal']
    
    # Use lower resolution for faster rendering
    downscale = 2
    H_render = H // downscale
    W_render = W // downscale
    focal_render = focal / downscale
    
    # Generate poses for a 360° video
    render_poses = []
    for theta in np.linspace(0., 2. * np.pi, 120, endpoint=False):
        
        c = np.array([4.0 * np.cos(theta), 0., 4.0 * np.sin(theta)], dtype=np.float32)
        z = normalize(-c)  # Look at origin
        up = np.array([0., 1., 0.], dtype=np.float32)
        x = normalize(np.cross(up, z))
        y = np.cross(z, x)
        
        pose = np.stack([x, y, z, c], axis=1)
        pose = np.concatenate([pose, np.array([[0, 0, 0, 1]], dtype=np.float32)], axis=0)
        render_poses.append(pose)
    
    # Render each pose
    rendered_images = []
    
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering poses")):
        pose = torch.tensor(pose, device=device, dtype=torch.float32)  # Force float32
        
        # Generate rays
        rays_o, rays_d = get_rays(H_render, W_render, focal_render, pose)
        
        # Render in chunks
        rgb_chunks = []
        chunk_size = 4096
        
        with torch.no_grad():
            for j in range(0, rays_o.shape[0], chunk_size):
                chunk_o = rays_o[j:j+chunk_size]
                chunk_d = rays_d[j:j+chunk_size]
                
                # Render with hierarchical sampling
                rgb_chunk, _, _, _ = render(model, chunk_o, chunk_d, near, far, args, use_hierarchical=True)
                
                rgb_chunks.append(rgb_chunk.cpu())
        
        # Combine chunks
        rgb = torch.cat(rgb_chunks, dim=0).reshape(H_render, W_render, 3).numpy()
        rendered_images.append(rgb)
        
        # Save image
        img_path = f"{args.images_path}/view_{i:03d}.png"
        plt.imsave(img_path, np.clip(rgb, 0, 1))
    
    # Create GIF
    if rendered_images:
        try:
            rendered_images = [np.clip(img * 255, 0, 255).astype(np.uint8) for img in rendered_images]
            gif_path = f"{args.images_path}/NeRF.gif"
            imageio.mimsave(gif_path, rendered_images, fps=15)
            print(f"Saved GIF to {gif_path}")
        except Exception as e:
            print(f"Error creating GIF: {e}")


def test(images, poses, camera_info, args):
    """Test the NeRF model by rendering novel views"""
    # Calculate scene bounds
    near, far = -10.0, 10.0  # Fixed bounds for synthetic dataset
    
    # Create model
    # model = NeRFmodel(embed_pos_L=args.n_pos_freq, embed_direction_L=args.n_dirc_freq).to(device)
    model = NeRFmodel(embed_pos_L=args.n_pos_freq,embed_direction_L=args.n_dirc_freq, use_positional_encoding=args.use_positional_encoding).to(device)
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=0.5)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    model.apply(init_weights)
    
    model.create_fine_network()
    model.fine_network.apply(init_weights)
    model.fine_network.to(device)
    
    # Load checkpoint
    checkpoint_files = glob.glob(f"{args.checkpoint_path}/*.pth")
    checkpoint_files.sort(key=os.path.getmtime)  

    if not checkpoint_files:
        print("No checkpoint found. Exiting.")
        return

    print(f"Loading checkpoint from {checkpoint_files[-1]}")
    checkpoint = torch.load(checkpoint_files[-1], map_location=device) 
    model.load_state_dict(checkpoint['model_state_dict'])
    model.fine_network.load_state_dict(checkpoint['fine_network_state_dict'])
    
    
    poses = poses.astype(np.float32) if isinstance(poses, np.ndarray) else poses
    
    
    os.makedirs(args.images_path, exist_ok=True)
    
    # Render novel views
    model.eval()
    model.fine_network.eval()
    with torch.no_grad():
        renderNovelViews(model, poses, camera_info, near, far, args)



def compare_models(args):
    """Compare NeRF with and without positional encoding"""    
    # Load test dataset
    images, poses, camera_info = loadDataset(args.data_path, "test")
    
    # Calculate scene bounds
    near, far = estimate_scene_bounds(poses)
    
    # Create output directory
    os.makedirs(args.images_path, exist_ok=True)
    comparison_dir = os.path.join(args.images_path, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Parameters
    H, W = camera_info['height'], camera_info['width']
    focal = camera_info['focal']
    
    # Use lower resolution for faster evaluation
    downscale = 4
    H_render = H // downscale
    W_render = W // downscale
    focal_render = focal / downscale
    
    # Initialize models
    # Model with positional encoding
    model_pe = NeRFmodel(embed_pos_L=args.n_pos_freq, 
                         embed_direction_L=args.n_dirc_freq, 
                         use_positional_encoding=True).to(device)
    model_pe.create_fine_network()
    model_pe.fine_network.to(device)
    
    # Model without positional encoding
    model_no_pe = NeRFmodel(embed_pos_L=args.n_pos_freq, 
                            embed_direction_L=args.n_dirc_freq, 
                            use_positional_encoding=False).to(device)
    model_no_pe.create_fine_network()
    model_no_pe.fine_network.to(device)
    
    # Load checkpoints
    # For model with PE
    checkpoint_files = sorted(glob.glob(f"{args.checkpoint_path}/*.pth"))
    if checkpoint_files:
        print(f"Loading checkpoint for model with PE from {checkpoint_files[-1]}")
        checkpoint_pe = torch.load(checkpoint_files[-1], map_location=device)
        model_pe.load_state_dict(checkpoint_pe['model_state_dict'])
        model_pe.fine_network.load_state_dict(checkpoint_pe['fine_network_state_dict'])
    
    # For model without PE
    no_pe_checkpoint_path = args.checkpoint_path.rstrip('/') + '_no_pe'
    no_pe_checkpoint_files = sorted(glob.glob(f"{no_pe_checkpoint_path}/*.pth"))
    if no_pe_checkpoint_files:
        print(f"Loading checkpoint for model without PE from {no_pe_checkpoint_files[-1]}")
        checkpoint_no_pe = torch.load(no_pe_checkpoint_files[-1], map_location=device)
        model_no_pe.load_state_dict(checkpoint_no_pe['model_state_dict'])
        model_no_pe.fine_network.load_state_dict(checkpoint_no_pe['fine_network_state_dict'])
    
    # Put models in evaluation mode
    model_pe.eval()
    model_pe.fine_network.eval()
    model_no_pe.eval()
    model_no_pe.fine_network.eval()
    
    # Lists to store rendered images
    rendered_images_pe = []
    rendered_images_no_pe = []
    ground_truth_images = []
    
    # Render test images
    num_test_images = min(10, len(poses))
    with torch.no_grad():
        for i in range(num_test_images):
            pose = torch.tensor(poses[i], device=device)
            
            # Generate rays
            rays_o, rays_d = get_rays(H_render, W_render, focal_render, pose)
            
            # Load ground truth image
            gt_img = images[i]
            from PIL import Image
            gt_img_pil = Image.fromarray((gt_img * 255).astype(np.uint8))
            gt_img_resized = np.array(gt_img_pil.resize((W_render, H_render))) / 255.0
            ground_truth_images.append(gt_img_resized)
            
            # Render with positional encoding
            rgb_pe = render_full_image(model_pe, rays_o, rays_d, H_render, W_render, near, far, args)
            rendered_images_pe.append(rgb_pe)
            
            # Render without positional encoding
            rgb_no_pe = render_full_image(model_no_pe, rays_o, rays_d, H_render, W_render, near, far, args)
            rendered_images_no_pe.append(rgb_no_pe)
            
            # Save comparison image
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(np.clip(gt_img_resized, 0, 1))
            plt.title("Ground Truth")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(np.clip(rgb_pe, 0, 1))
            plt.title("With Positional Encoding")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(np.clip(rgb_no_pe, 0, 1))
            plt.title("Without Positional Encoding")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{comparison_dir}/comparison_{i:03d}.png")
            plt.close()
    
    # Evaluate metrics
    psnr_pe, ssim_pe = evaluate_metrics(rendered_images_pe, ground_truth_images)
    psnr_no_pe, ssim_no_pe = evaluate_metrics(rendered_images_no_pe, ground_truth_images)
    
    # Print and save results
    results = {
        "with_positional_encoding": {
            "PSNR": psnr_pe,
            "SSIM": ssim_pe
        },
        "without_positional_encoding": {
            "PSNR": psnr_no_pe,
            "SSIM": ssim_no_pe
        }
    }
    
    print("\nEvaluation Results:")
    print(f"With Positional Encoding    - PSNR: {psnr_pe:.4f}, SSIM: {ssim_pe:.4f}")
    print(f"Without Positional Encoding - PSNR: {psnr_no_pe:.4f}, SSIM: {ssim_no_pe:.4f}")
    
    with open(f"{comparison_dir}/metrics.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create visual comparison of metrics
    plt.figure(figsize=(10, 6))
    metrics = ["PSNR", "SSIM"]
    values_pe = [psnr_pe, ssim_pe]
    values_no_pe = [psnr_no_pe, ssim_no_pe]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, values_pe, width, label='With Positional Encoding')
    plt.bar(x + width/2, values_no_pe, width, label='Without Positional Encoding')
    
    plt.ylabel('Value')
    plt.title('Comparison of Metrics')
    plt.xticks(x, metrics)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{comparison_dir}/metrics_comparison.png")
    plt.close()
    
    return results

def render_full_image(model, rays_o, rays_d, H, W, near, far, args):
    """Render a full image in chunks"""
    rgb_chunks = []
    chunk_size = 4096
    
    for j in range(0, rays_o.shape[0], chunk_size):
        chunk_o = rays_o[j:j+chunk_size]
        chunk_d = rays_d[j:j+chunk_size]
        
        # Render with hierarchical sampling
        rgb_chunk, _, _, _ = render(model, chunk_o, chunk_d, near, far, args, use_hierarchical=True)
        
        rgb_chunks.append(rgb_chunk.cpu())
    
    # Combine chunks
    rgb = torch.cat(rgb_chunks, dim=0).reshape(H, W, 3).numpy()
    return rgb

def main(args):
    print(f"Running NeRF in {args.mode} mode...")
    
    if args.mode == 'compare':
        # Compare models with and without positional encoding
        compare_models(args)
    else:
        # Load dataset
        images, poses, camera_info = loadDataset(args.data_path, args.mode)
        
        if args.mode == 'train':
            train(images, poses, camera_info, args)
        else:
            test(images, poses, camera_info, args)


def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="./lego/", help="dataset path")
    parser.add_argument('--mode', default='train', help="train/test/val")
    parser.add_argument('--lrate', default=1e-4, help="learning rate")
    parser.add_argument('--n_pos_freq', default=10, type=int, help="number of positional encoding frequencies for position")
    parser.add_argument('--n_dirc_freq', default=4, type=int, help="number of positional encoding frequencies for viewing direction")
    parser.add_argument('--n_rays_batch', default=1024, type=int, help="number of rays per batch")
    parser.add_argument('--n_sample', default=64, type=int, help="number of samples per ray (coarse)")
    parser.add_argument('--n_sample_fine', default=128, type=int, help="number of additional samples per ray (fine)")
    parser.add_argument('--max_iters', default=200000, type=int, help="maximum number of training iterations")
    parser.add_argument('--logs_path', default="./logs/", help="logs path")
    parser.add_argument('--checkpoint_path', default="./checkpoint/", help="checkpoints path")
    parser.add_argument('--load_checkpoint', default=False, type=bool, help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_iter', default=1000, type=int, help="save checkpoint every N iterations")
    parser.add_argument('--images_path', default="./images/", help="folder to store rendered images")

    parser.add_argument('--use_positional_encoding', default=True, type=bool, help="whether to use positional encoding")

    return parser


if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)