# RBE549 - Building Reconstruction with SfM and NeRF

***By: Divam Trivedi and Shreya Boyane***

This project implements the paper NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis to synthesize novel views of 3D scenes. This repository contains the implementation for Phase 2 of the RBE549 project, focusing on **Neural Radiance Fields (NeRF)**.

## Project Overview

NeRF represents a 3D scene as a continuous 5D function that outputs RGB color and volume density at any given 3D location and 2D viewing direction. By training on a set of input views, NeRF can render photorealistic novel views of complex scenes.

## Dependencies

The code requires the following packages:

- PyTorch
- NumPy
- Matplotlib
- tqdm
- imageio
- skimage

To install dependencies:

```bash
pip install torch numpy matplotlib tqdm imageio scikit-image
```

## Code Structure

The codebase consists of three main files:

1. **NeRFModel.py**: Implements the neural network architecture for NeRF, including positional encoding and hierarchical volume sampling.
2. **metrics.py**: Contains functions to calculate PSNR and SSIM metrics for evaluating rendering quality.
3. **Wrapper.py**: The main script that handles training, testing, and rendering. It includes functions for:

   - Loading datasets
   - Generating rays
   - Volume rendering
   - Training the model
   - Rendering novel views
   - Evaluating results

## Usage

### Training

To train the model on a dataset:

```bash
python Wrapper.py --data_path ./lego/ --mode train --n_pos_freq 10 --n_dirc_freq 4 --n_rays_batch 1024 --max_iters 200000
```

Key parameters:

- `--data_path`: Path to the dataset
- `--mode`: 'train', 'test', or 'val'
- `--n_pos_freq`: Number of frequency bands for position encoding (default: 10)
- `--n_dirc_freq`: Number of frequency bands for direction encoding (default: 4)
- `--n_rays_batch`: Number of rays per batch during training (default: 1024)
- `--n_sample`: Number of coarse samples per ray (default: 64)
- `--n_sample_fine`: Number of fine samples per ray (default: 128)
- `--max_iters`: Maximum number of training iterations (default: 200000)
- `--load_checkpoint`: Whether to load from existing checkpoint (default: False)

### Testing and Rendering Novel Views

To generate novel views from a trained model:

```bash
python Wrapper.py --data_path ./lego/ --mode test --checkpoint_path ./checkpoint/
```

This will render a 360° orbit of the scene and save the images and a GIF animation.

### Comparing Models With and Without Positional Encoding

To compare the effect of positional encoding:

```bash
python Wrapper.py --data_path ./lego/ --mode compare --checkpoint_path ./checkpoint/
```

This will render the same views with and without positional encoding and compute quality metrics.

## Implementation Details

The implementation follows the original NeRF paper with:

1. **Positional Encoding**: Transforms input 5D coordinates into a higher-dimensional space to better represent high-frequency details.
2. **Hierarchical Volume Sampling**: Uses a coarse-to-fine approach with two networks:

   - Coarse network: Provides an initial estimate of the scene
   - Fine network: Focuses sampling on relevant parts of the scene
3. **Volume Rendering**: Accumulates color and density along rays using classical volume rendering techniques.

## Expected Outputs

During training, the code generates:

- Checkpoints of the model saved in `./checkpoint/`
- Preview renderings in `./images/previews/`
- Training logs in `./logs/`

After testing, the code produces:

- Novel view renderings in `./images/`
- A GIF animation showing a 360° orbit of the scene (`./images/NeRF.gif`)
- Metrics for comparison with and without positional encoding (if requested)
