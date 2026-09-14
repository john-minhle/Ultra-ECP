"""
Simplified script to run UltraSam inference using mmdet test command.
This script creates a temporary config and runs inference.
"""

import subprocess
import sys
import argparse
from pathlib import Path
import json


def create_temp_config(
    base_config: str,
    data_root: str,
    ann_file: str,
    image_prefix: str,
    output_config: str
):
    """
    Create a temporary config file for inference.
    
    Args:
        base_config: Path to base UltraSam config
        data_root: Data root directory
        ann_file: Annotation file path (COCO JSON)
        image_prefix: Image prefix path
        output_config: Output config file path
    """
    import os
    
    # Convert all paths to use forward slashes (works on both Windows and Linux)
    def normalize_path(path_str):
        """Normalize path to use forward slashes for Python config files."""
        return str(path_str).replace('\\', '/')
    
    # Convert base_config to absolute path
    # mmengine resolves _base_ paths relative to the config file location
    # So we need to use absolute path or calculate correct relative path
    base_config_path = Path(base_config)
    project_root = Path(__file__).parent.parent
    output_config_path = Path(output_config)
    work_dir = output_config_path.parent
    
    # If relative path, make it absolute from project root
    if not base_config_path.is_absolute():
        base_config_path = project_root / base_config_path
    
    # Verify the file exists
    if not base_config_path.exists():
        raise FileNotFoundError(
            f"Base config not found: {base_config_path}\n"
            f"Please check the path: {base_config}"
        )
    
    # mmengine resolves _base_ paths relative to the config file location
    # Since temp_config.py is in work_dir/ultrasam/, we need to calculate
    # relative path from there, or use absolute path
    
    # Try to calculate relative path from work_dir to base_config
    # work_dir is: D:/Khanh/Fetal-2/work_dir/ultrasam
    # base_config is: D:/Khanh/Fetal-2/UltraSam/configs/...
    # Need: ../../UltraSam/configs/...
    try:
        # Resolve both paths to absolute
        work_dir_abs = work_dir.resolve()
        base_config_abs = base_config_path.resolve()
        
        # Calculate relative path
        base_config_relative = base_config_abs.relative_to(work_dir_abs)
        base_config_relative = normalize_path(base_config_relative)
        
        # Verify this path would work
        test_path = (work_dir_abs / base_config_relative).resolve()
        if test_path.exists() and test_path == base_config_abs:
            # Use relative path (e.g., ../../UltraSam/configs/...)
            base_config = base_config_relative
            print(f"Using relative path from work_dir: {base_config}")
        else:
            # Fall back to absolute path
            base_config = normalize_path(base_config_abs)
            print(f"Using absolute path: {base_config}")
    except (ValueError, RuntimeError) as e:
        # If can't make relative (e.g., different drives on Windows), use absolute
        base_config = normalize_path(base_config_path.resolve())
        print(f"Using absolute path (relative failed: {e}): {base_config}")
    
    # Normalize all paths to use forward slashes
    data_root = normalize_path(data_root)
    ann_file = normalize_path(ann_file)
    image_prefix = normalize_path(image_prefix)
    
    # Escape single quotes in paths if any
    def escape_path(path_str):
        """Escape single quotes in path string."""
        return path_str.replace("'", "\\'")
    
    # Create config content with normalized paths and proper escaping
    # Use raw strings (r'...') to avoid backslash issues
    config_content = f"""_base_ = [r'{escape_path(base_config)}']

data_root = r'{escape_path(data_root)}'

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img=r'{escape_path(image_prefix)}'),
        ann_file=r'{escape_path(ann_file)}',
    ),
)

test_evaluator = dict(
    ann_file=rf'{{data_root}}/{ann_file}',
    format_only=False,
    outfile_prefix='./work_dir/ultrasam_predictions'
)
"""
    
    # Ensure output directory exists
    output_config_path = Path(output_config)
    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"Created temporary config: {output_config}")


def run_ultrasam_inference(
    checkpoint: str,
    base_config: str,
    data_root: str,
    ann_file: str,
    image_prefix: str,
    work_dir: str,
    show_dir: str = None
):
    """
    Run UltraSam inference using mmdet test command.
    
    Args:
        checkpoint: Path to UltraSam checkpoint
        base_config: Path to base UltraSam config
        data_root: Data root directory
        ann_file: Annotation file (COCO JSON)
        image_prefix: Image prefix path
        work_dir: Work directory for outputs
        show_dir: Directory to save visualization (optional)
    """
    # Create temporary config
    temp_config = Path(work_dir) / 'temp_config.py'
    temp_config.parent.mkdir(parents=True, exist_ok=True)
    
    create_temp_config(
        base_config,
        data_root,
        ann_file,
        image_prefix,
        str(temp_config)
    )
    
    # Build mmdet test command
    import os
    cmd = [
        'mim', 'test', 'mmdet',
        str(temp_config),
        '--checkpoint', checkpoint,
        '--work-dir', work_dir
    ]
    
    if show_dir:
        cmd.extend(['--show-dir', show_dir])
    
    # Add PYTHONPATH - use absolute path and proper separator
    env = dict(os.environ)
    ultrasam_path = Path(__file__).parent.parent / 'UltraSam'
    ultrasam_abs_path = str(ultrasam_path.resolve())
    
    # Use os.pathsep for cross-platform compatibility (; on Windows, : on Linux)
    path_sep = os.pathsep
    if 'PYTHONPATH' in env:
        # Avoid duplicates
        existing_paths = env['PYTHONPATH'].split(path_sep)
        if ultrasam_abs_path not in existing_paths:
            env['PYTHONPATH'] = f"{ultrasam_abs_path}{path_sep}{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = ultrasam_abs_path
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"PYTHONPATH: {env.get('PYTHONPATH', 'not set')}")
    
    # Run command
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running inference:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True


def convert_coco_to_masks(
    coco_results_file: str,
    output_dir: str,
    image_dir: str
):
    """
    Convert COCO format results to individual mask files.
    
    Args:
        coco_results_file: Path to COCO results JSON
        output_dir: Output directory for masks
        image_dir: Directory containing images
    """
    import cv2
    from pycocotools import mask as mask_util
    
    with open(coco_results_file, 'r') as f:
        results = json.load(f)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cardiac_dir = output_path / 'cardiac'
    thoracic_dir = output_path / 'thoracic'
    cardiac_dir.mkdir(exist_ok=True)
    thoracic_dir.mkdir(exist_ok=True)
    
    # Group results by image_id
    image_results = {}
    for result in results:
        image_id = result['image_id']
        if image_id not in image_results:
            image_results[image_id] = []
        image_results[image_id].append(result)
    
    # Load image to get dimensions
    image_path = Path(image_dir)
    image_files = {f.stem: f for f in image_path.glob('*.png')}
    
    for image_id, results_list in image_results.items():
        # Find corresponding image
        img_file = None
        for img_stem, img_path in image_files.items():
            if str(image_id) in img_stem or img_stem == str(image_id):
                img_file = img_path
                break
        
        if img_file is None:
            print(f"Warning: Image not found for image_id {image_id}")
            continue
        
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        
        height, width = img.shape[:2]
        
        cardiac_mask = np.zeros((height, width), dtype=np.uint8)
        thoracic_mask = np.zeros((height, width), dtype=np.uint8)
        
        for result in results_list:
            category_id = result.get('category_id', 1)
            segmentation = result.get('segmentation', {})
            
            if isinstance(segmentation, dict):
                # RLE format
                mask = mask_util.decode(segmentation)
            elif isinstance(segmentation, list):
                # Polygon format - convert to mask
                from pycocotools import mask as mask_util
                rle = mask_util.frPyObjects(segmentation, height, width)
                mask = mask_util.decode(rle)
            else:
                continue
            
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
            mask_binary = (mask > 0).astype(np.uint8) * 255
            
            if category_id == 1:  # cardiac
                cardiac_mask = np.maximum(cardiac_mask, mask_binary)
            elif category_id == 2:  # thoracic
                thoracic_mask = np.maximum(thoracic_mask, mask_binary)
        
        # Save masks
        img_id = img_file.stem
        cv2.imwrite(str(cardiac_dir / f'{img_id}-cardiac.png'), cardiac_mask)
        cv2.imwrite(str(thoracic_dir / f'{img_id}-thorax.png'), thoracic_mask)


def main():
    parser = argparse.ArgumentParser(
        description='Run UltraSam inference on FOCUS dataset using mmdet'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='UltraSam/UltraSam.pth',
        help='Path to UltraSam checkpoint'
    )
    parser.add_argument(
        '--base-config',
        type=str,
        default='UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py',
        help='Path to base UltraSam config'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='Root directory containing images and COCO JSON'
    )
    parser.add_argument(
        '--ann-file',
        type=str,
        required=True,
        help='COCO annotation file (relative to data-root)'
    )
    parser.add_argument(
        '--image-prefix',
        type=str,
        default='images',
        help='Image prefix path (relative to data-root)'
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        default='work_dir/ultrasam_inference',
        help='Work directory for outputs'
    )
    parser.add_argument(
        '--show-dir',
        type=str,
        default=None,
        help='Directory to save visualization images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for converted masks (optional)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).parent.parent / checkpoint_path
    
    base_config_path = Path(args.base_config)
    if not base_config_path.is_absolute():
        base_config_path = Path(__file__).parent.parent / base_config_path
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if not base_config_path.exists():
        raise FileNotFoundError(f"Config not found: {base_config_path}")
    
    # Run inference
    success = run_ultrasam_inference(
        str(checkpoint_path),
        str(base_config_path),
        args.data_root,
        args.ann_file,
        args.image_prefix,
        args.work_dir,
        args.show_dir
    )
    
    if success and args.output_dir:
        # Convert results to mask files
        results_file = Path(args.work_dir) / 'ultrasam_predictions.bbox.json'
        if results_file.exists():
            import numpy as np
            import cv2
            os.environ['PYTHONPATH'] = str(Path(__file__).parent.parent / 'UltraSam')
            convert_coco_to_masks(
                str(results_file),
                args.output_dir,
                str(Path(args.data_root) / args.image_prefix)
            )
        else:
            print(f"Results file not found: {results_file}")


if __name__ == '__main__':
    import os
    import numpy as np
    main()

