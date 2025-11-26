"""
Wrapper for PointTransformerV3 to match MinkUNet interface
PointTransformerV3 works with point clouds directly, not voxels
"""

import torch
import torch.nn as nn
import sys
import os

# Add PointTransformerV3 to path
ptv3_path = os.path.join(os.path.dirname(__file__), '../../PointTransformerV3')
if os.path.exists(ptv3_path):
    sys.path.insert(0, ptv3_path)
    try:
        from model import PointTransformerV3, Point
    except ImportError:
        # Try alternative import path
        sys.path.insert(0, os.path.join(ptv3_path, 'Pointcept'))
        from model import PointTransformerV3, Point
else:
    raise ImportError(f"PointTransformerV3 not found at {ptv3_path}. Please ensure PointTransformerV3 is installed.")


class PointTransformerV3Wrapper(nn.Module):
    """
    Wrapper for PointTransformerV3 that matches MinkUNet's interface.
    
    MinkUNet expects:
    - Input: ME.SparseTensor (voxelized data)
    - Output: ME.SparseTensor with .F attribute for features
    
    PointTransformerV3 expects:
    - Input: Point dict with 'feat', 'coord' (or 'grid_coord'), 'offset' (or 'batch')
    - Output: Point dict with updated 'feat'
    
    This wrapper converts between the two formats.
    """
    
    def __init__(self, in_channels, out_channels, D=3, arch="PointTransformerV3", 
                 voxel_size=0.04, **kwargs):
        super().__init__()
        self.voxel_size = voxel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Default PointTransformerV3 config - can be customized via kwargs
        ptv3_config = {
            'in_channels': in_channels,
            'order': ("z", "z-trans", "hilbert", "hilbert-trans"),
            'stride': (2, 2, 2, 2),
            'enc_depths': (2, 2, 2, 6, 2),
            'enc_channels': (32, 64, 128, 256, 512),
            'enc_num_head': (2, 4, 8, 16, 32),
            'enc_patch_size': (1024, 1024, 1024, 1024, 1024),
            'dec_depths': (2, 2, 2, 2),
            'dec_channels': (64, 64, 128, 256),
            'dec_num_head': (4, 4, 8, 16),
            'dec_patch_size': (1024, 1024, 1024, 1024),
            'mlp_ratio': 4,
            'qkv_bias': True,
            'drop_path': 0.3,
            'enable_flash': True,
            'enable_rpe': False,
            'cls_mode': False,
        }
        
        # Override with any provided kwargs
        ptv3_config.update(kwargs)
        
        # Create PointTransformerV3 model
        self.model = PointTransformerV3(**ptv3_config)
        
        # Add a final projection layer to match output channels if needed
        if ptv3_config['dec_channels'][-1] != out_channels:
            self.final_proj = nn.Linear(ptv3_config['dec_channels'][-1], out_channels)
        else:
            self.final_proj = nn.Identity()
    
    def forward(self, point_dict):
        """
        Forward pass that works with point cloud data.
        
        Args:
            point_dict: Dictionary with:
                - 'feat': (N, in_channels) tensor of features
                - 'coord': (N, 3) tensor of point coordinates (original, not voxelized)
                - 'offset': (B,) tensor indicating batch boundaries, or
                - 'batch': (N,) tensor indicating batch id for each point
                - 'grid_size': (optional) grid size for grid_coord computation
        
        Returns:
            Dictionary with 'feat': (N, out_channels) tensor of output features
        """
        # Convert to Point object
        point = Point(point_dict)
        
        # If grid_coord not provided, compute from coord and grid_size
        if 'grid_coord' not in point.keys():
            if 'grid_size' not in point.keys():
                point['grid_size'] = self.voxel_size
            # Compute grid_coord from coord
            point['grid_coord'] = torch.div(
                point.coord - point.coord.min(0)[0], 
                point.grid_size, 
                rounding_mode="trunc"
            ).int()
        
        # Forward through PointTransformerV3
        point = self.model(point)
        
        # Apply final projection if needed
        point.feat = self.final_proj(point.feat)
        
        return point


class PointTransformerV3FromVoxels(nn.Module):
    """
    Alternative wrapper that converts voxelized data back to point cloud format.
    This allows using the existing voxelization pipeline.
    """
    
    def __init__(self, in_channels, out_channels, D=3, arch="PointTransformerV3", 
                 voxel_size=0.04, **kwargs):
        super().__init__()
        self.voxel_size = voxel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Default PointTransformerV3 config
        ptv3_config = {
            'in_channels': in_channels,
            'order': ("z", "z-trans", "hilbert", "hilbert-trans"),
            'stride': (2, 2, 2, 2),
            'enc_depths': (2, 2, 2, 6, 2),
            'enc_channels': (32, 64, 128, 256, 512),
            'enc_num_head': (2, 4, 8, 16, 32),
            'enc_patch_size': (1024, 1024, 1024, 1024, 1024),
            'dec_depths': (2, 2, 2, 2),
            'dec_channels': (64, 64, 128, 256),
            'dec_num_head': (4, 4, 8, 16),
            'dec_patch_size': (1024, 1024, 1024, 1024),
            'mlp_ratio': 4,
            'qkv_bias': True,
            'drop_path': 0.3,
            'enable_flash': True,
            'enable_rpe': False,
            'cls_mode': False,
        }
        
        ptv3_config.update(kwargs)
        
        self.model = PointTransformerV3(**ptv3_config)
        
        if ptv3_config['dec_channels'][-1] != out_channels:
            self.final_proj = nn.Linear(ptv3_config['dec_channels'][-1], out_channels)
        else:
            self.final_proj = nn.Identity()
    
    def forward(self, locs, features):
        """
        Forward pass that converts voxel coordinates back to point coordinates.
        
        Args:
            locs: (N, 4) tensor where first column is batch_id, rest are voxel coordinates
            features: (N, in_channels) tensor of features
        
        Returns:
            features: (N, out_channels) tensor of output features
        """
        # Extract batch and voxel coordinates
        batch_ids = locs[:, 0].long()
        voxel_coords = locs[:, 1:4].float()
        
        # Convert voxel coordinates back to point coordinates
        # Multiply by voxel_size to get approximate point positions
        point_coords = voxel_coords * self.voxel_size
        
        # Create offset from batch_ids
        # batch_ids should be 0-indexed and consecutive
        batch_size = int(batch_ids.max().item() + 1)
        offset_list = []
        for b in range(batch_size):
            mask = (batch_ids == b)
            offset_list.append(mask.sum().item())
        offset = torch.tensor([0] + offset_list, device=locs.device, dtype=torch.long).cumsum(0)
        
        # Create Point dict
        point_dict = {
            'feat': features,
            'coord': point_coords,
            'offset': offset,
            'grid_size': self.voxel_size,
        }
        
        point = Point(point_dict)
        point['grid_coord'] = voxel_coords.int()
        
        # Forward through model
        point = self.model(point)
        
        # Apply final projection
        point.feat = self.final_proj(point.feat)
        
        return point.feat

