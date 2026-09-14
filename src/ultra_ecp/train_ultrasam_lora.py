"""
Training script for UltraSAM fine-tuning with LoRA, Point-Robust Augmentation, and Ellipse-Aware Loss.

This script implements:
1. Parameter-efficient fine-tuning with LoRA (only prompt encoder + mask decoder)
2. Point-Robust Augmentation (jittering point prompts)
3. Ellipse-Aware Loss (encouraging ellipse-shaped masks)

Note: This script is designed to work with UltraSAM's MMDetection framework.
You may need to adapt it based on your specific UltraSAM model structure.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import numpy as np
import sys
from datetime import datetime


from .data_loader_ultrasam_finetune import get_data_loaders_ultrasam
from .ellipse_aware_loss import EllipseAwareLoss
from .lora_sam import (
    apply_lora_to_module,
    freeze_module,
    get_trainable_parameters,
    count_parameters
)


def load_ultrasam_for_training(
    checkpoint_path: str,
    config_path: str,
    device: str = 'cuda',
    apply_lora: bool = True,
    lora_rank: int = 8,
    lora_alpha: float = 16.0
):
    """
    Load UltraSAM model and apply LoRA for fine-tuning.
    
    Args:
        checkpoint_path: Path to UltraSAM checkpoint
        config_path: Path to UltraSAM config file
        device: Device to load model on
        apply_lora: Whether to apply LoRA
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        
    Returns:
        Model with LoRA applied (if enabled)
    """
    try:
        from mmdet.apis import init_detector
        
        # Load model
        model = init_detector(
            config_path,
            checkpoint_path,
            device=device
        )
        
        if apply_lora:
            # Freeze image encoder (backbone)
            if hasattr(model, 'backbone'):
                print("Freezing image encoder (backbone)...")
                freeze_module(model.backbone)
            
            # Apply LoRA to prompt encoder and mask decoder
            print("Applying LoRA to prompt encoder and mask decoder...")
            
            # Target modules for LoRA (adjust based on actual SAM structure)
            target_modules = ['qkv', 'proj', 'fc1', 'fc2', 'q_proj', 'k_proj', 'v_proj']
            
            # Apply LoRA to prompt encoder
            if hasattr(model, 'prompt_encoder'):
                lora_modules_pe = apply_lora_to_module(
                    model.prompt_encoder,
                    target_modules=target_modules,
                    rank=lora_rank,
                    alpha=lora_alpha
                )
                print(f"Applied LoRA to {len(lora_modules_pe)} modules in prompt encoder")
            
            # Apply LoRA to mask decoder (decoder)
            if hasattr(model, 'decoder'):
                lora_modules_dec = apply_lora_to_module(
                    model.decoder,
                    target_modules=target_modules,
                    rank=lora_rank,
                    alpha=lora_alpha
                )
                print(f"Applied LoRA to {len(lora_modules_dec)} modules in mask decoder")
            
            # Count parameters
            total_params = count_parameters(model, trainable_only=False)
            trainable_params = count_parameters(model, trainable_only=True)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        model.train()
        return model
        
    except ImportError:
        raise ImportError(
            "MMDetection not installed. Please install with:\n"
            "pip install -U openmim\n"
            "mim install mmengine\n"
            "mim install mmcv\n"
            "mim install mmdet"
        )


def forward_ultrasam_with_point_prompt(
    model,
    images: torch.Tensor,
    point_coords: torch.Tensor,
    point_labels: torch.Tensor,
    image_size: int = 512
):
    """
    Forward pass through UltraSAM with point prompts.
    
    This is a simplified interface - you may need to adapt this based on
    UltraSAM's actual forward signature.
    
    Args:
        model: UltraSAM model
        images: Input images [B, 3, H, W]
        point_coords: Point coordinates [B, N, 2]
        point_labels: Point labels [B, N] (1 = foreground, 0 = background)
        image_size: Image size
        
    Returns:
        Predicted masks [B, H, W]
    """
    # This is a placeholder - adapt based on UltraSAM's actual API
    # UltraSAM uses MMDetection's data format, so you'll need to create
    # proper data samples
    
    # For now, we'll use a simplified approach
    # In practice, you'll need to:
    # 1. Create proper MMDetection data samples
    # 2. Call model's forward method with proper format
    
    batch_size = images.shape[0]
    
    # Create data samples (simplified - adapt to actual format)
    try:
        from mmdet.structures import DetDataSample
        from mmengine.structures import InstanceData
        
        data_samples = []
        for i in range(batch_size):
            data_sample = DetDataSample()
            data_sample.set_metainfo({
                'img_shape': (image_size, image_size),
                'scale_factor': (1.0, 1.0),
                'batch_input_shape': (image_size, image_size)
            })
            
            # Create instance data with point prompts
            instances = InstanceData()
            instances.points = point_coords[i].unsqueeze(0)  # [1, N, 2]
            instances.labels = point_labels[i].unsqueeze(0)  # [1, N]
            
            data_sample.gt_instances = instances
            data_samples.append(data_sample)
        
        # Forward pass
        # Note: This may need adjustment based on UltraSAM's actual forward signature
        outputs = model(images, data_samples)
        
        # Extract masks from outputs
        # This depends on UltraSAM's output format
        if hasattr(outputs[0], 'pred_instances'):
            masks = []
            for output in outputs:
                if hasattr(output.pred_instances, 'masks'):
                    mask = output.pred_instances.masks
                    if isinstance(mask, torch.Tensor):
                        masks.append(mask)
                    else:
                        # Convert mask format if needed
                        masks.append(mask.to_tensor())
            
            if len(masks) > 0:
                masks = torch.stack(masks, dim=0)
                # Take first mask if multiple
                if masks.dim() == 4 and masks.shape[1] > 1:
                    masks = masks[:, 0]
                return masks.sigmoid()  # Convert logits to probabilities
        
        # Fallback: return dummy masks
        return torch.zeros(batch_size, image_size, image_size, device=images.device)
        
    except Exception as e:
        print(f"Warning: Could not use MMDetection format: {e}")
        print("Returning dummy masks - please adapt forward_ultrasam_with_point_prompt")
        return torch.zeros(batch_size, image_size, image_size, device=images.device)


def train_epoch(
    model,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    image_size: int = 512
):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_dice = 0.0
    total_bce = 0.0
    total_ellipse = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        point_coords = batch['point_coords'].to(device)
        point_labels = batch['point_labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Get predictions
        pred_masks = forward_ultrasam_with_point_prompt(
            model,
            images,
            point_coords,
            point_labels,
            image_size
        )
        
        # Resize predictions to match target masks if needed
        if pred_masks.shape[-2:] != masks.shape[-2:]:
            pred_masks = torch.nn.functional.interpolate(
                pred_masks.unsqueeze(1) if pred_masks.dim() == 3 else pred_masks,
                size=masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            if pred_masks.dim() == 4:
                pred_masks = pred_masks.squeeze(1)
        
        # Compute loss
        loss_dict = criterion(pred_masks, masks)
        loss = loss_dict['total']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        total_dice += loss_dict['dice'].item()
        total_bce += loss_dict['bce'].item()
        total_ellipse += loss_dict['ellipse'].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{loss_dict["dice"].item():.4f}',
            'ellipse': f'{loss_dict["ellipse"].item():.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_dice = total_dice / len(train_loader)
    avg_bce = total_bce / len(train_loader)
    avg_ellipse = total_ellipse / len(train_loader)
    
    return {
        'loss': avg_loss,
        'dice': avg_dice,
        'bce': avg_bce,
        'ellipse': avg_ellipse
    }


def validate(
    model,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    image_size: int = 512
):
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    total_dice = 0.0
    total_bce = 0.0
    total_ellipse = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            point_coords = batch['point_coords'].to(device)
            point_labels = batch['point_labels'].to(device)
            
            # Forward pass
            pred_masks = forward_ultrasam_with_point_prompt(
                model,
                images,
                point_coords,
                point_labels,
                image_size
            )
            
            # Resize if needed
            if pred_masks.shape[-2:] != masks.shape[-2:]:
                pred_masks = torch.nn.functional.interpolate(
                    pred_masks.unsqueeze(1) if pred_masks.dim() == 3 else pred_masks,
                    size=masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                if pred_masks.dim() == 4:
                    pred_masks = pred_masks.squeeze(1)
            
            # Compute loss
            loss_dict = criterion(pred_masks, masks)
            loss = loss_dict['total']
            
            total_loss += loss.item()
            total_dice += loss_dict['dice'].item()
            total_bce += loss_dict['bce'].item()
            total_ellipse += loss_dict['ellipse'].item()
    
    avg_loss = total_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    avg_bce = total_bce / len(val_loader)
    avg_ellipse = total_ellipse / len(val_loader)
    
    return {
        'loss': avg_loss,
        'dice': avg_dice,
        'bce': avg_bce,
        'ellipse': avg_ellipse
    }


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune UltraSAM with LoRA, Point-Robust Augmentation, and Ellipse-Aware Loss'
    )
    
    # Data arguments
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root directory of processed data')
    parser.add_argument('--class-name', type=str, default='cardiac',
                       choices=['cardiac', 'thoracic'],
                       help='Class to segment')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to UltraSAM checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to UltraSAM config file')
    
    # LoRA arguments
    parser.add_argument('--lora-rank', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora-alpha', type=float, default=16.0,
                       help='LoRA alpha scaling factor')
    parser.add_argument('--no-lora', action='store_true',
                       help='Disable LoRA (fine-tune all parameters)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--early-stop-patience', type=int, default=15,
                       help='Stop if validation loss does not improve for N epochs (0=disabled)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--image-size', type=int, default=512,
                       help='Image size')
    
    # Point augmentation arguments
    parser.add_argument('--point-jitter-radius', type=float, default=10.0,
                       help='Maximum radius for point jittering (pixels)')
    parser.add_argument('--no-point-jitter', action='store_true',
                       help='Disable point jittering')
    
    # Loss arguments
    parser.add_argument('--ellipse-weight', type=float, default=1.0,
                       help='Weight for ellipse-aware loss')
    parser.add_argument('--dice-weight', type=float, default=1.0,
                       help='Weight for Dice loss')
    parser.add_argument('--bce-weight', type=float, default=1.0,
                       help='Weight for BCE loss')
    parser.add_argument('--no-ellipse-loss', action='store_true',
                       help='Disable ellipse-aware loss')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='checkpoints/ultrasam_lora',
                       help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / 'training_log.txt'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("UltraSAM Fine-tuning with LoRA")
    logger.info("="*60)
    logger.info(f"Arguments: {args}")
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders_ultrasam(
        args.data_root,
        class_name=args.class_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        point_jitter_radius=args.point_jitter_radius,
        use_point_jitter_train=not args.no_point_jitter,
        use_point_jitter_val=False
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Load model
    logger.info("Loading UltraSAM model...")
    model = load_ultrasam_for_training(
        args.checkpoint,
        args.config,
        device=device,
        apply_lora=not args.no_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha
    )
    
    # Loss function
    criterion = EllipseAwareLoss(
        ellipse_weight=args.ellipse_weight,
        dice_weight=args.dice_weight,
        bce_weight=args.bce_weight,
        use_ellipse_loss=not args.no_ellipse_loss
    )
    
    # Optimizer (only trainable parameters)
    trainable_params = get_trainable_parameters(model)
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1, args.image_size
        )
        
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Dice: {train_metrics['dice']:.4f}, "
                   f"BCE: {train_metrics['bce']:.4f}, "
                   f"Ellipse: {train_metrics['ellipse']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, args.image_size)
        
        logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                   f"Dice: {val_metrics['dice']:.4f}, "
                   f"BCE: {val_metrics['bce']:.4f}, "
                   f"Ellipse: {val_metrics['ellipse']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'args': args
        }
        
        # Save latest
        torch.save(checkpoint, output_dir / 'latest.pth')
        
        # Save best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            torch.save(checkpoint, output_dir / 'best.pth')
            logger.info(f"Saved best model (val_loss: {best_val_loss:.4f})")
        elif args.early_stop_patience > 0:
            patience_counter += 1
            logger.info(
                f"No val loss improvement ({patience_counter}/{args.early_stop_patience})"
            )
            if patience_counter >= args.early_stop_patience:
                logger.info("Early stopping triggered.")
                break

    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to: {output_dir}")


if __name__ == '__main__':
    main()



