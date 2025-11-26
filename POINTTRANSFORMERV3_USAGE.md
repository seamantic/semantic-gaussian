# Using PointTransformerV3 Instead of MinkUNet

This document explains how to use PointTransformerV3 instead of MinkUNet for generating 3D embeddings in the semantic-gaussian pipeline.

## Overview

The semantic-gaussian repository uses MinkUNet to generate 3D embeddings from voxelized point cloud data. PointTransformerV3 is an alternative model that works directly with point clouds (not voxels), offering potentially better performance.

## Key Differences

### MinkUNet
- **Input**: Voxelized data (MinkowskiEngine SparseTensor)
- **Process**: Works on sparse voxel grids
- **Output**: Features per voxel

### PointTransformerV3
- **Input**: Point cloud data (Point dict with coordinates and features)
- **Process**: Works directly on point coordinates
- **Output**: Features per point

## Implementation Details

### Wrapper Architecture

A wrapper class `PointTransformerV3FromVoxels` has been created to bridge the gap between the existing voxelization pipeline and PointTransformerV3's point cloud requirements:

1. **Input Conversion**: The wrapper takes voxelized data (locs, features) and converts it back to point cloud format
2. **Coordinate Conversion**: Voxel coordinates are multiplied by `voxel_size` to approximate original point positions
3. **Batch Handling**: Creates proper offset/batch information from batch IDs
4. **Output**: Returns features in the same format as MinkUNet

### Files Modified

1. **`model/point_transformer_v3_wrapper.py`**: New wrapper classes
   - `PointTransformerV3Wrapper`: For direct point cloud input
   - `PointTransformerV3FromVoxels`: For voxelized input (used in current pipeline)

2. **`distill.py`**: Updated to support both models
   - Detects PointTransformerV3 by checking if `model_3d` starts with "PointTransformerV3" or "PTV3"
   - Routes data through appropriate forward pass

3. **`eval_segmentation.py`**: Updated evaluation functions
   - `eval_mink()`: Updated for PointTransformerV3
   - `eval_mink_and_fusion()`: Updated for PointTransformerV3

## Usage

### Configuration

To use PointTransformerV3, update your config file (e.g., `config/distill_scannet.yaml`):

```yaml
distill:
  model_3d: PointTransformerV3  # or PTV3
  voxel_size: 0.04
  # ... other settings
```

### Example Config

```yaml
distill:
  exp_name: ptv3_experiment
  model_3d: PointTransformerV3  # Changed from MinkUNet14A
  voxel_size: 0.04
  aug: True
  feature_type: all
  lr: 0.001
  epochs: 100
  # ... rest of config
```

### Running

The code will automatically detect PointTransformerV3 and use the appropriate wrapper:

```bash
python distill.py --config config/distill_scannet.yaml
```

## How It Works

### Data Flow

1. **Original Pipeline (MinkUNet)**:
   ```
   Point Cloud → Voxelization → MinkowskiEngine SparseTensor → MinkUNet → Embeddings
   ```

2. **New Pipeline (PointTransformerV3)**:
   ```
   Point Cloud → Voxelization → Convert to Point Format → PointTransformerV3 → Embeddings
   ```

### Conversion Process

The wrapper performs the following conversions:

1. **Voxel to Point Coordinates**:
   ```python
   point_coords = voxel_coords * voxel_size
   ```

2. **Batch to Offset**:
   - Extracts batch IDs from the first column of `locs`
   - Creates offset tensor for proper batching

3. **Grid Coordinates**:
   - Uses voxel coordinates as `grid_coord` for PointTransformerV3's serialization

## Model Configuration

PointTransformerV3 uses default hyperparameters that can be customized. The current implementation uses:

- **Encoder**: 5 stages with depths [2, 2, 2, 6, 2]
- **Decoder**: 4 stages with depths [2, 2, 2, 2]
- **Channels**: [32, 64, 128, 256, 512] (encoder), [64, 64, 128, 256] (decoder)
- **Attention Heads**: [2, 4, 8, 16, 32] (encoder), [4, 4, 8, 16] (decoder)
- **Patch Size**: 1024 for all stages
- **Flash Attention**: Enabled by default

To customize, modify the wrapper initialization in `distill.py` or pass kwargs through the config.

## Requirements

1. **PointTransformerV3 Installation**:
   - Ensure PointTransformerV3 is installed at `/ocean/projects/cis250226p/shared/PointTransformerV3`
   - Or update the import path in `point_transformer_v3_wrapper.py`

2. **Dependencies**:
   - Flash Attention (recommended for performance)
   - spconv (for sparse convolutions)
   - torch_scatter

## Notes

- The wrapper converts voxelized data back to point format, which is an approximation
- For best results, consider modifying the pipeline to work directly with point clouds (skip voxelization)
- The current implementation maintains compatibility with the existing voxelization pipeline
- PointTransformerV3 may require different hyperparameters (learning rate, batch size) compared to MinkUNet

## Troubleshooting

### Import Errors

If you see import errors for PointTransformerV3:
1. Check that PointTransformerV3 is installed at the expected path
2. Verify the model.py file exists in the PointTransformerV3 directory
3. Check that all dependencies (spconv, torch_scatter, flash_attn) are installed

### Performance Issues

- PointTransformerV3 may use more memory than MinkUNet
- Consider reducing batch size or patch size if you encounter OOM errors
- Flash Attention significantly improves performance - ensure it's installed

### Model Compatibility

- Saved MinkUNet checkpoints cannot be loaded into PointTransformerV3
- You'll need to retrain from scratch when switching models
- The output dimensions should match (768 for embeddings)

