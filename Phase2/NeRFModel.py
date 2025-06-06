import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NeRFmodel(nn.Module):
    def __init__(self, embed_pos_L=10, embed_direction_L=4, use_positional_encoding=True):
        super(NeRFmodel, self).__init__()

        # Save encoding parameters
        self.embed_pos_L = embed_pos_L
        self.embed_direction_L = embed_direction_L
        self.use_positional_encoding = use_positional_encoding

        # Calculate the dimension of the encoded inputs
        if use_positional_encoding:
            self.embed_pos_dims = 3 + 3 * 2 * embed_pos_L
            self.embed_direction_dims = 3 + 3 * 2 * embed_direction_L
        else:
            self.embed_pos_dims = 3  # Just x, y, z without encoding
            self.embed_direction_dims = 3  # Just dx, dy, dz without encoding

        # Coarse network
        # Position MLP - first 8 layers with skip connection at layer 4
        self.pos_layers = nn.ModuleList([
            nn.Linear(self.embed_pos_dims, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(self.embed_pos_dims + 256, 256),  # Skip connection
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
        ])
        
        # Output density (sigma) - one layer
        self.density_layer = nn.Linear(256, 1)
        
        # Output feature vector - one layer
        self.feature_layer = nn.Linear(256, 256)
        
        # RGB output - two layers with view direction as additional input
        self.rgb_layers = nn.ModuleList([
            nn.Linear(256 + self.embed_direction_dims, 128),
            nn.Linear(128, 3)
        ])
        
        # Fine network (optional, can be enabled later)
        self.fine_network = None

    def position_encoding(self, x, L):

        if not self.use_positional_encoding:
            return x  # Return input directly if not using positional encoding
        
        # Create output tensor
        y = torch.zeros((*x.shape[:-1], x.shape[-1] + x.shape[-1] * 2 * L),
                        device=x.device)
        
        # Copy original coordinates
        y[..., :x.shape[-1]] = x
        
        # Apply sin and cos encoding for each frequency
        for i in range(L):
            freq = 2.0 ** i
            y[..., x.shape[-1] + i * 2 * x.shape[-1]:x.shape[-1] + (i + 1) * 2 * x.shape[-1]] = \
                torch.cat([torch.sin(freq * np.pi * x), torch.cos(freq * np.pi * x)], dim=-1)
        
        return y

    def forward(self, pos, direction):

        # Apply positional encoding to inputs
        pos_encoded = self.position_encoding(pos, self.embed_pos_L)
        direction_encoded = self.position_encoding(direction, self.embed_direction_L)
        
        # Process through the network layers with ReLU activation and skip connection
        h = pos_encoded
        for i, layer in enumerate(self.pos_layers):
            if i == 4:  # Apply skip connection at 4th layer (as in the original paper)
                h = torch.cat([pos_encoded, h], dim=-1)
            h = layer(h)
            h = torch.relu(h)
        
        # Get density output
        sigma = self.density_layer(h)
        #sigma = torch.relu(sigma)  # Ensure density is non-negative
        sigma = F.softplus(sigma)  # Import torch.nn.functional as F
        
        # Get feature vector
        feature = self.feature_layer(h)
        
        # Concatenate feature with viewing direction
        h = torch.cat([feature, direction_encoded], dim=-1)
        
        # Get RGB output
        for i, layer in enumerate(self.rgb_layers):
            h = layer(h)
            if i < len(self.rgb_layers) - 1:
                h = torch.relu(h)
        
        # Apply sigmoid to RGB to ensure values are between 0 and 1
        rgb = torch.sigmoid(h)
        
        return rgb, sigma
    
    def create_fine_network(self):

        self.fine_network = NeRFmodel(self.embed_pos_L, self.embed_direction_L, self.use_positional_encoding)



