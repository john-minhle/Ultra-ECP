"""
Evaluation script for fine-tuned UltraSAM.
Computes metrics including robustness to point jittering.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import sys
from typing import Dict, List, Tuple


from .inference_ultrasam_lora import load_finetuned_ultrasam, predict_with_point_prompt
from .data_loader_ultrasam_finetune import compute_mask_centroid, jitter_point


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Dice score between binary masks."""
    pred_binary = (pred > 0).astype(np.float32)
    target_binary = (target > 0).astype(np.float32)
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2.0 * intersection) / union


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute IoU score between binary masks."""
    pred_binary = (pred > 0).astype(np.float32)
    target_binary = (target > 0).astype(np.float32)
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute 95th percentile Hausdorff distance.
    Simplified version - for full implementation, use scipy or similar.
    """
    from scipy.spatial.distance import directed_hausdorff
    
    pred_binary = (pred > 0).astype(np.uint8)
    target_binary = (target > 0).astype(np.uint8)
    
    # Get contours
    pred_contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    target_contours, _ = cv2.findContours(target_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(pred_contours) == 0 or len(target_contours) == 0:
        return float('inf')
    
    # Get largest contours
    pred_contour = max(pred_contours, key=cv2.contourArea)
    target_contour = max(target_contours, key=cv2.contourArea)
    
    if len(pred_contour) == 0 or len(target_contour) == 0:
        return float('inf')
    
    # Reshape contours
    pred_points = pred_contour.reshape(-1, 2)
    target_points = target_contour.reshape(-1, 2)
    
    try:
        # Compute directed Hausdorff distances
        d1 = directed_hausdorff(pred_points, target_points)[0]
        d2 = directed_hausdorff(target_points, pred_points)[0]
        
        # 95th percentile approximation (using max for simplicity)
        hd95 = max(d1, d2)
        
        return hd95
    except:
        return float('inf')


def evaluate_robustness(
    model,
    image: np.ndarray,
    mask_gt: np.ndarray,
    true_centroid: Tuple[float, float],
    jitter_radii: List[float],
    image_size: int = 512,
    device: str = 'cuda',
    num_samples: int = 5
) -> Dict[str, List[float]]:
    """
    Evaluate model robustness to point jittering.
    
    Args:
        model: Fine-tuned model
        image: Input image
        mask_gt: Ground truth mask
        true_centroid: True centroid coordinates
        jitter_radii: List of jitter radii to test
        image_size: Image size
        device: Device
        num_samples: Number of random samples per radius
        
    Returns:
        Dictionary with metrics for each jitter radius
    """
    results = {
        'dice': {r: [] for r in jitter_radii},
        'iou': {r: [] for r in jitter_radii}
    }
    
    # Test with no jitter (baseline)
    pred_mask = predict_with_point_prompt(
        model, image, true_centroid, point_label=1, image_size=image_size, device=device
    )
    dice_base = dice_score(pred_mask, mask_gt)
    iou_base = iou_score(pred_mask, mask_gt)
    
    results['dice'][0.0] = [dice_base]
    results['iou'][0.0] = [iou_base]
    
    # Test with different jitter radii
    for radius in jitter_radii:
        for _ in range(num_samples):
            # Jitter point
            jittered_point = jitter_point(
                true_centroid,
                max_radius=radius,
                image_shape=image.shape[:2]
            )
            
            # Predict
            pred_mask = predict_with_point_prompt(
                model, image, jittered_point, point_label=1, image_size=image_size, device=device
            )
            
            # Compute metrics
            dice = dice_score(pred_mask, mask_gt)
            iou = iou_score(pred_mask, mask_gt)
            
            results['dice'][radius].append(dice)
            results['iou'][radius].append(iou)
    
    return results


def evaluate_dataset(
    model_path: str,
    config_path: str,
    data_root: str,
    split: str,
    class_name: str = 'cardiac',
    jitter_radii: List[float] = [0.0, 2.0, 5.0, 10.0],
    device: str = 'cuda',
    image_size: int = 512
) -> Dict:
    """
    Evaluate model on entire dataset.
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = load_finetuned_ultrasam(model_path, config_path, device)
    print("Model loaded!")
    
    # Setup directories
    data_path = Path(data_root) / split
    image_dir = data_path / 'images'
    mask_dir = data_path / 'annfiles_mask'
    
    # Get all images
    image_files = sorted(image_dir.glob('*.png'))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    print(f"Evaluating on {len(image_files)} images...")
    
    # Metrics
    all_dice = []
    all_iou = []
    all_hd95 = []
    
    robustness_results = {r: {'dice': [], 'iou': []} for r in jitter_radii}
    
    for img_file in tqdm(image_files):
        img_id = img_file.stem
        
        # Load image
        image = cv2.imread(str(img_file))
        if image is None:
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load ground truth mask
        mask_path = mask_dir / f'{img_id}-{class_name}.png'
        if not mask_path.exists():
            mask_path = mask_dir / f'{img_id}_{class_name}.png'
        
        if not mask_path.exists():
            continue
        
        mask_gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gt is None:
            continue
        
        mask_gt = (mask_gt > 0).astype(np.uint8) * 255
        
        # Compute true centroid
        mask_gt_binary = (mask_gt > 0).astype(np.float32)
        true_centroid = compute_mask_centroid(mask_gt_binary)
        
        # Predict with true centroid (baseline)
        try:
            pred_mask = predict_with_point_prompt(
                model, image, true_centroid, point_label=1, image_size=image_size, device=device
            )
            
            # Compute metrics
            dice = dice_score(pred_mask, mask_gt)
            iou = iou_score(pred_mask, mask_gt)
            hd95 = hausdorff_distance_95(pred_mask, mask_gt)
            
            all_dice.append(dice)
            all_iou.append(iou)
            if not np.isinf(hd95):
                all_hd95.append(hd95)
            
            # Evaluate robustness
            img_robustness = evaluate_robustness(
                model, image, mask_gt, true_centroid, jitter_radii, image_size, device, num_samples=3
            )
            
            for radius in jitter_radii:
                if radius in img_robustness['dice']:
                    robustness_results[radius]['dice'].extend(img_robustness['dice'][radius])
                    robustness_results[radius]['iou'].extend(img_robustness['iou'][radius])
            
        except Exception as e:
            print(f"Error processing {img_id}: {e}")
            continue
    
    # Compute summary statistics
    results = {
        'baseline': {
            'dice_mean': np.mean(all_dice) if all_dice else 0.0,
            'dice_std': np.std(all_dice) if all_dice else 0.0,
            'iou_mean': np.mean(all_iou) if all_iou else 0.0,
            'iou_std': np.std(all_iou) if all_iou else 0.0,
            'hd95_mean': np.mean(all_hd95) if all_hd95 else float('inf'),
            'hd95_std': np.std(all_hd95) if all_hd95 else 0.0,
            'num_samples': len(all_dice)
        },
        'robustness': {}
    }
    
    for radius in jitter_radii:
        if robustness_results[radius]['dice']:
            results['robustness'][f'radius_{radius}'] = {
                'dice_mean': np.mean(robustness_results[radius]['dice']),
                'dice_std': np.std(robustness_results[radius]['dice']),
                'iou_mean': np.mean(robustness_results[radius]['iou']),
                'iou_std': np.std(robustness_results[radius]['iou']),
                'num_samples': len(robustness_results[radius]['dice'])
            }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate fine-tuned UltraSAM'
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to fine-tuned checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to UltraSAM config file')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root directory of dataset')
    parser.add_argument('--split', type=str, default='testing',
                       choices=['training', 'validation', 'testing'],
                       help='Dataset split')
    parser.add_argument('--class-name', type=str, default='cardiac',
                       choices=['cardiac', 'thoracic'],
                       help='Class name')
    parser.add_argument('--jitter-radii', type=float, nargs='+',
                       default=[0.0, 2.0, 5.0, 10.0],
                       help='Jitter radii to test for robustness')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    parser.add_argument('--image-size', type=int, default=512,
                       help='Image size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Evaluate
    results = evaluate_dataset(
        args.checkpoint,
        args.config,
        args.data_root,
        args.split,
        args.class_name,
        args.jitter_radii,
        args.device,
        args.image_size
    )
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    print("\nBaseline (no jitter):")
    baseline = results['baseline']
    print(f"  Dice: {baseline['dice_mean']:.4f} ± {baseline['dice_std']:.4f}")
    print(f"  IoU:  {baseline['iou_mean']:.4f} ± {baseline['iou_std']:.4f}")
    print(f"  HD95: {baseline['hd95_mean']:.2f} ± {baseline['hd95_std']:.2f}")
    print(f"  Samples: {baseline['num_samples']}")
    
    print("\nRobustness (point jittering):")
    for radius_key, metrics in results['robustness'].items():
        radius = radius_key.replace('radius_', '')
        print(f"\n  Jitter radius: {radius}px")
        print(f"    Dice: {metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f}")
        print(f"    IoU:  {metrics['iou_mean']:.4f} ± {metrics['iou_std']:.4f}")
        print(f"    Samples: {metrics['num_samples']}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()



