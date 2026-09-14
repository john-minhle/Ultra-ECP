"""
Convert FOCUS dataset masks to COCO format for UltraSam inference.
This script converts mask annotations to COCO JSON format that UltraSam expects.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import sys

try:
    from pycocotools import mask as mask_util
except ImportError:
    print("Warning: pycocotools not installed. Please install with: pip install pycocotools")
    sys.exit(1)


def mask_to_rle(mask: np.ndarray) -> dict:
    """Convert binary mask to RLE format."""
    mask = mask.astype(np.uint8)
    rle = mask_util.encode(np.asfortranarray(mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def convert_masks_to_coco(
    image_dir: str,
    mask_dir: str,
    output_file: str,
    split: str = 'testing',
    target_classes: list = ['cardiac', 'thoracic']
):
    """
    Convert FOCUS dataset masks to COCO format.
    
    Args:
        image_dir: Directory containing images
        mask_dir: Directory containing mask files
        output_file: Output COCO JSON file path
        split: Dataset split name
        target_classes: List of classes to include
    """
    image_path = Path(image_dir)
    mask_path = Path(mask_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = sorted(image_path.glob('*.png'))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    # COCO format structure
    coco_data = {
        'info': {
            'description': f'FOCUS {split} dataset for UltraSam',
            'version': '1.0',
            'year': 2024
        },
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': [
            {'id': 1, 'name': 'cardiac', 'supercategory': 'anatomy'},
            {'id': 2, 'name': 'thoracic', 'supercategory': 'anatomy'}
        ]
    }
    
    class_to_id = {'cardiac': 1, 'thoracic': 2}
    ann_id = 0
    
    print(f"Converting {len(image_files)} images...")
    
    for img_file in tqdm(image_files):
        img_id_str = img_file.stem
        
        # Load image to get dimensions
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Warning: Could not load image {img_file}")
            continue
        
        height, width = img.shape[:2]
        
        # Add image info
        image_info = {
            'id': len(coco_data['images']),
            'width': width,
            'height': height,
            'file_name': img_file.name
        }
        coco_data['images'].append(image_info)
        image_id = image_info['id']
        
        # Load masks for each class
        for class_name in target_classes:
            if class_name not in class_to_id:
                continue
            
            # Try different mask file naming conventions
            mask_file = None
            possible_names = [
                f'{img_id_str}-{class_name}.png',
                f'{img_id_str}_{class_name}.png',
                f'{class_name}_{img_id_str}.png'
            ]
            
            for name in possible_names:
                potential_file = mask_path / name
                if potential_file.exists():
                    mask_file = potential_file
                    break
            
            if mask_file is None:
                # Try looking for combined mask files
                combined_mask_file = mask_path / f'{img_id_str}.png'
                if combined_mask_file.exists():
                    # Load combined mask and extract class
                    combined_mask = cv2.imread(str(combined_mask_file), cv2.IMREAD_GRAYSCALE)
                    if combined_mask is not None:
                        # Assume class 1 = cardiac, class 2 = thoracic in combined mask
                        class_value = class_to_id[class_name]
                        mask = (combined_mask == class_value).astype(np.uint8)
                    else:
                        continue
                else:
                    continue
            else:
                # Load binary mask
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                mask = (mask > 0).astype(np.uint8)
            
            # Resize mask if needed (should match image size)
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Skip if mask is empty
            if mask.sum() == 0:
                continue
            
            # Convert mask to RLE
            rle = mask_to_rle(mask)
            area = float(mask_util.area(rle))
            
            # Get bounding box
            bbox = mask_util.toBbox(rle).tolist()
            
            # Add annotation
            annotation = {
                'id': ann_id,
                'image_id': image_id,
                'category_id': class_to_id[class_name],
                'segmentation': rle,
                'area': area,
                'bbox': bbox,  # [x, y, width, height]
                'iscrowd': 0
            }
            coco_data['annotations'].append(annotation)
            ann_id += 1
    
    # Filter out images without annotations (UltraSam requires at least one annotation per image)
    image_ids_with_anns = set(ann['image_id'] for ann in coco_data['annotations'])
    filtered_images = [img for img in coco_data['images'] if img['id'] in image_ids_with_anns]
    
    if len(filtered_images) < len(coco_data['images']):
        skipped = len(coco_data['images']) - len(filtered_images)
        print(f"\nWarning: {skipped} images without annotations will be skipped")
        print(f"  (UltraSam requires at least one annotation per image)")
        coco_data['images'] = filtered_images
    
    # Save COCO JSON
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"  Images: {len(coco_data['images'])} (after filtering)")
    print(f"  Annotations: {len(coco_data['annotations'])}")
    print(f"  Saved to: {output_path}")
    
    if len(coco_data['images']) == 0:
        raise ValueError("No images with annotations found! Please check your mask files.")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert FOCUS dataset masks to COCO format for UltraSam'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='Root directory containing dataset splits'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['training', 'validation', 'testing', 'train', 'val', 'test'],
        default='testing',
        help='Dataset split to convert'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='ultrasam_data',
        help='Output directory for COCO JSON files'
    )
    parser.add_argument(
        '--target-classes',
        type=str,
        nargs='+',
        default=['cardiac', 'thoracic'],
        help='Target classes to include'
    )
    
    args = parser.parse_args()
    
    # Normalize split name
    split_map = {'train': 'training', 'val': 'validation', 'test': 'testing'}
    split = split_map.get(args.split, args.split)
    
    # Determine paths
    data_root = Path(args.data_root)
    image_dir = data_root / split / 'images'
    
    # Try different mask directory names
    mask_dir = None
    for mask_dir_name in ['annfiles_mask', 'masks']:
        potential_mask_dir = data_root / split / mask_dir_name
        if potential_mask_dir.exists():
            mask_dir = potential_mask_dir
            break
    
    if mask_dir is None:
        raise ValueError(f"Mask directory not found for split {split}")
    
    # Output file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{split}_coco.json'
    
    convert_masks_to_coco(
        str(image_dir),
        str(mask_dir),
        str(output_file),
        split,
        args.target_classes
    )


if __name__ == '__main__':
    main()

