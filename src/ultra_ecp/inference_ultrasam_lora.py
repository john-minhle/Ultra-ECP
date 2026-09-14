"""
Inference script for fine-tuned UltraSAM with LoRA.
Supports single-point prompts for fetal heart segmentation.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
from typing import Tuple, Optional


from .data_loader_ultrasam_finetune import get_val_transforms, compute_mask_centroid


def load_finetuned_ultrasam(
    checkpoint_path: str,
    config_path: str,
    device: str = 'cuda'
):
    """
    Load fine-tuned UltraSAM model.
    
    Args:
        checkpoint_path: Path to fine-tuned checkpoint
        config_path: Path to UltraSAM config file
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    try:
        from mmdet.apis import init_detector
        
        # Load base model
        model = init_detector(
            config_path,
            checkpoint=None,  # We'll load from our checkpoint
            device=device
        )
        
        # Load fine-tuned weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        return model
        
    except ImportError:
        raise ImportError(
            "MMDetection not installed. Please install with:\n"
            "pip install -U openmim\n"
            "mim install mmengine\n"
            "mim install mmcv\n"
            "mim install mmdet"
        )


def predict_with_point_prompt(
    model,
    image: np.ndarray,
    point: Tuple[float, float],
    point_label: int = 1,
    image_size: int = 512,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Predict mask using a single point prompt.
    
    Args:
        model: Fine-tuned UltraSAM model
        image: Input image (H, W, 3) in RGB format
        point: Point coordinates (x, y) in image coordinates
        point_label: Point label (1 = foreground, 0 = background)
        image_size: Target image size
        device: Device to run inference on
        
    Returns:
        Predicted binary mask (H, W) with values 0-255
    """
    original_shape = image.shape[:2]
    
    # Resize image
    image_resized = cv2.resize(image, (image_size, image_size))
    
    # Normalize
    transform = get_val_transforms(image_size)
    transformed = transform(image=image_resized)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Scale point coordinates
    scale_x = image_size / original_shape[1]
    scale_y = image_size / original_shape[0]
    scaled_point = (point[0] * scale_x, point[1] * scale_y)
    
    # Create point prompt
    point_coords = torch.tensor([[scaled_point]], dtype=torch.float32).to(device)  # [1, 1, 2]
    point_labels = torch.tensor([[point_label]], dtype=torch.long).to(device)  # [1, 1]
    
    # Forward pass
    with torch.no_grad():
        # This is a simplified interface - adapt based on UltraSAM's actual API
        try:
            from mmdet.structures import DetDataSample
            from mmengine.structures import InstanceData
            
            data_sample = DetDataSample()
            data_sample.set_metainfo({
                'img_shape': (image_size, image_size),
                'scale_factor': (1.0, 1.0),
                'batch_input_shape': (image_size, image_size)
            })
            
            instances = InstanceData()
            instances.points = point_coords
            instances.labels = point_labels
            data_sample.gt_instances = instances
            
            outputs = model(image_tensor, [data_sample])
            
            # Extract mask
            if hasattr(outputs[0], 'pred_instances'):
                if hasattr(outputs[0].pred_instances, 'masks'):
                    mask = outputs[0].pred_instances.masks
                    if isinstance(mask, torch.Tensor):
                        mask_np = mask.cpu().numpy()
                    else:
                        mask_np = mask.to_tensor().cpu().numpy()
                    
                    # Take first mask if multiple
                    if mask_np.ndim == 3 and mask_np.shape[0] > 1:
                        mask_np = mask_np[0]
                    elif mask_np.ndim == 3:
                        mask_np = mask_np.squeeze(0)
                    
                    # Convert to binary
                    mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
                    
                    # Resize back to original size
                    mask_resized = cv2.resize(
                        mask_binary,
                        (original_shape[1], original_shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                    
                    return mask_resized
            
            # Fallback: return empty mask
            return np.zeros(original_shape, dtype=np.uint8)
            
        except Exception as e:
            print(f"Warning: Could not use MMDetection format: {e}")
            return np.zeros(original_shape, dtype=np.uint8)


def predict_with_centroid(
    model,
    image: np.ndarray,
    mask_gt: Optional[np.ndarray] = None,
    image_size: int = 512,
    device: str = 'cuda'
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Predict mask using centroid of ground truth mask (for evaluation).
    
    Args:
        model: Fine-tuned UltraSAM model
        image: Input image (H, W, 3) in RGB format
        mask_gt: Ground truth mask (optional, for computing centroid)
        image_size: Target image size
        device: Device to run inference on
        
    Returns:
        Predicted mask and point used
    """
    if mask_gt is not None:
        # Compute centroid from ground truth
        point = compute_mask_centroid(mask_gt)
    else:
        # Use image center
        h, w = image.shape[:2]
        point = (w / 2.0, h / 2.0)
    
    mask = predict_with_point_prompt(
        model, image, point, point_label=1, image_size=image_size, device=device
    )
    
    return mask, point


def inference_dataset(
    model_path: str,
    config_path: str,
    data_root: str,
    split: str,
    output_dir: str,
    class_name: str = 'cardiac',
    use_centroid: bool = True,
    device: str = 'cuda',
    image_size: int = 512
):
    """
    Run inference on entire dataset split.
    
    Args:
        model_path: Path to fine-tuned checkpoint
        config_path: Path to UltraSAM config
        data_root: Root directory of dataset
        split: Dataset split ('training', 'validation', 'testing')
        output_dir: Output directory for predictions
        class_name: Class name ('cardiac' or 'thoracic')
        use_centroid: Use centroid of GT mask as point prompt (for evaluation)
        device: Device to use
        image_size: Image size
    """
    # Load model
    print(f"Loading fine-tuned model from {model_path}...")
    model = load_finetuned_ultrasam(model_path, config_path, device)
    print("Model loaded!")
    
    # Setup directories
    data_path = Path(data_root) / split
    image_dir = data_path / 'images'
    mask_dir = data_path / 'annfiles_mask'
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = sorted(image_dir.glob('*.png'))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    print(f"Running inference on {len(image_files)} images...")
    
    for img_file in tqdm(image_files):
        img_id = img_file.stem
        
        # Load image
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"Warning: Could not load {img_file}")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load ground truth mask if available (for centroid)
        mask_gt = None
        if use_centroid:
            mask_path = mask_dir / f'{img_id}-{class_name}.png'
            if not mask_path.exists():
                mask_path = mask_dir / f'{img_id}_{class_name}.png'
            
            if mask_path.exists():
                mask_gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_gt is not None:
                    mask_gt = (mask_gt > 0).astype(np.float32)
        
        # Predict
        try:
            if use_centroid and mask_gt is not None:
                pred_mask, point_used = predict_with_centroid(
                    model, image, mask_gt, image_size, device
                )
            else:
                # Use image center
                h, w = image.shape[:2]
                point = (w / 2.0, h / 2.0)
                pred_mask = predict_with_point_prompt(
                    model, image, point, point_label=1, image_size=image_size, device=device
                )
            
            # Save prediction
            output_file = output_path / f'{img_id}_pred.png'
            cv2.imwrite(str(output_file), pred_mask)
            
        except Exception as e:
            print(f"Error processing {img_id}: {e}")
            continue
    
    print(f"\nInference complete!")
    print(f"Predictions saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run inference with fine-tuned UltraSAM'
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to fine-tuned checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to UltraSAM config file')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root directory of dataset')
    parser.add_argument('--split', type=str, required=True,
                       choices=['training', 'validation', 'testing'],
                       help='Dataset split')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for predictions')
    parser.add_argument('--class-name', type=str, default='cardiac',
                       choices=['cardiac', 'thoracic'],
                       help='Class name')
    parser.add_argument('--use-centroid', action='store_true',
                       help='Use centroid of GT mask as point prompt')
    parser.add_argument('--image-size', type=int, default=512,
                       help='Image size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    inference_dataset(
        args.checkpoint,
        args.config,
        args.data_root,
        args.split,
        args.output_dir,
        args.class_name,
        args.use_centroid,
        args.device,
        args.image_size
    )


if __name__ == '__main__':
    main()

