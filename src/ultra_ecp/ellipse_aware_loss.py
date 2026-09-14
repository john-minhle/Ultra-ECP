"""
Ellipse-Aware Loss for fetal heart segmentation.
Encourages masks to have smooth, ellipse-like shapes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Optional, Tuple, Dict


def fit_ellipse_to_mask_tensor(mask: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
    """
    Fit an ellipse to a binary mask tensor.
    
    Args:
        mask: Binary mask tensor [H, W] or [B, H, W] with values 0-1
        
    Returns:
        Dictionary with ellipse parameters:
        - center: [x, y] or [B, 2]
        - axes: [width, height] or [B, 2]
        - angle: rotation angle in degrees or [B]
        Returns None if ellipse fitting fails
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size = mask.shape[0]
    device = mask.device
    
    # Convert to numpy for OpenCV
    mask_np = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
    
    centers = []
    axes_list = []
    angles = []
    
    for b in range(batch_size):
        mask_b = mask_np[b]
        
        # Find contours
        contours, _ = cv2.findContours(
            mask_b,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )
        
        if len(contours) == 0:
            # Empty mask - return default ellipse (center of image)
            h, w = mask_b.shape
            centers.append([w / 2.0, h / 2.0])
            axes_list.append([w / 4.0, h / 4.0])
            angles.append(0.0)
            continue
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Need at least 5 points to fit ellipse
        if len(largest_contour) < 5:
            h, w = mask_b.shape
            centers.append([w / 2.0, h / 2.0])
            axes_list.append([w / 4.0, h / 4.0])
            angles.append(0.0)
            continue
        
        # Fit ellipse
        try:
            ellipse = cv2.fitEllipse(largest_contour)
            center, axes, angle = ellipse
            
            centers.append([center[0], center[1]])
            axes_list.append([axes[0], axes[1]])
            angles.append(angle)
        except:
            # Fitting failed - return default
            h, w = mask_b.shape
            centers.append([w / 2.0, h / 2.0])
            axes_list.append([w / 4.0, h / 4.0])
            angles.append(0.0)
    
    # Convert to tensors
    center_tensor = torch.tensor(centers, dtype=torch.float32, device=device)
    axes_tensor = torch.tensor(axes_list, dtype=torch.float32, device=device)
    angle_tensor = torch.tensor(angles, dtype=torch.float32, device=device)
    
    if squeeze_output:
        return {
            'center': center_tensor.squeeze(0),
            'axes': axes_tensor.squeeze(0),
            'angle': angle_tensor.squeeze(0)
        }
    
    return {
        'center': center_tensor,
        'axes': axes_tensor,
        'angle': angle_tensor
    }


def create_ellipse_mask(
    shape: Tuple[int, int],
    center: torch.Tensor,
    axes: torch.Tensor,
    angle: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Create a binary ellipse mask.
    
    Args:
        shape: (H, W) or (B, H, W) output shape
        center: [x, y] or [B, 2] center coordinates
        axes: [width, height] or [B, 2] ellipse axes
        angle: rotation angle in degrees or [B]
        device: Device to create tensor on
        
    Returns:
        Binary mask tensor [H, W] or [B, H, W]
    """
    if len(shape) == 2:
        h, w = shape
        batch_size = 1
        squeeze_output = True
    else:
        batch_size, h, w = shape
        squeeze_output = False
    
    if center.dim() == 1:
        center = center.unsqueeze(0)
    if axes.dim() == 1:
        axes = axes.unsqueeze(0)
    if angle.dim() == 0:
        angle = angle.unsqueeze(0)
    
    # Create coordinate grids
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    # Expand for batch
    if batch_size > 1:
        y_coords = y_coords.unsqueeze(0).expand(batch_size, -1, -1)
        x_coords = x_coords.unsqueeze(0).expand(batch_size, -1, -1)
    else:
        y_coords = y_coords.unsqueeze(0)
        x_coords = x_coords.unsqueeze(0)
    
    # Translate to center
    x_centered = x_coords - center[:, 0:1, None]
    y_centered = y_coords - center[:, 1:2, None]
    
    # Rotate coordinates
    angle_rad = torch.deg2rad(angle)
    cos_a = torch.cos(angle_rad).view(-1, 1, 1)
    sin_a = torch.sin(angle_rad).view(-1, 1, 1)
    
    x_rot = x_centered * cos_a + y_centered * sin_a
    y_rot = -x_centered * sin_a + y_centered * cos_a
    
    # Check if point is inside ellipse: (x/a)^2 + (y/b)^2 <= 1
    a = axes[:, 0:1, None] / 2.0  # half-width
    b = axes[:, 1:2, None] / 2.0  # half-height
    
    ellipse_eq = (x_rot / a) ** 2 + (y_rot / b) ** 2
    
    mask = (ellipse_eq <= 1.0).float()
    
    if squeeze_output:
        mask = mask.squeeze(0)
    
    return mask


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Dice loss between predicted and target masks.
    
    Args:
        pred: Predicted mask [B, H, W] or [H, W]
        target: Target mask [B, H, W] or [H, W]
        smooth: Smoothing factor
        
    Returns:
        Dice loss (1 - Dice coefficient)
    """
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return 1.0 - dice


def binary_cross_entropy_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Binary cross-entropy loss.
    
    Args:
        pred: Predicted mask [B, H, W] or [H, W]
        target: Target mask [B, H, W] or [H, W]
        smooth: Smoothing factor
        
    Returns:
        BCE loss
    """
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    # Clamp predictions to avoid numerical issues
    pred_flat = torch.clamp(pred_flat, smooth, 1.0 - smooth)
    
    bce = F.binary_cross_entropy(pred_flat, target_flat, reduction='mean')
    
    return bce


class EllipseAwareLoss(nn.Module):
    """
    Combined loss with segmentation loss and ellipse-aware regularization.
    
    L_total = L_seg(P, G) + alpha * L_ellipse(P, M_e)
    
    where:
    - L_seg: Segmentation loss (Dice + BCE)
    - L_ellipse: Ellipse loss (1 - IoU(P, M_e))
    - P: Predicted mask
    - G: Ground truth mask
    - M_e: Ellipse mask fitted to P
    """
    
    def __init__(
        self,
        ellipse_weight: float = 1.0,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        use_ellipse_loss: bool = True
    ):
        """
        Args:
            ellipse_weight: Weight for ellipse-aware loss (alpha)
            dice_weight: Weight for Dice loss
            bce_weight: Weight for BCE loss
            use_ellipse_loss: Whether to apply ellipse loss
        """
        super().__init__()
        self.ellipse_weight = ellipse_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.use_ellipse_loss = use_ellipse_loss
    
    def forward(
        self,
        pred_mask: torch.Tensor,
        target_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            pred_mask: Predicted mask [B, H, W] or [H, W], values in [0, 1]
            target_mask: Ground truth mask [B, H, W] or [H, W], values in [0, 1]
            
        Returns:
            Dictionary with loss components:
            - 'total': Total loss
            - 'dice': Dice loss
            - 'bce': BCE loss
            - 'ellipse': Ellipse loss (if enabled)
        """
        # Ensure same shape
        if pred_mask.shape != target_mask.shape:
            # Resize pred_mask to match target
            pred_mask = F.interpolate(
                pred_mask.unsqueeze(0) if pred_mask.dim() == 2 else pred_mask.unsqueeze(1),
                size=target_mask.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            if pred_mask.dim() == 3:
                pred_mask = pred_mask.squeeze(1)
            else:
                pred_mask = pred_mask.squeeze(0)
        
        # Clamp predictions
        pred_mask = torch.clamp(pred_mask, 0.0, 1.0)
        
        # Segmentation losses
        dice_loss_val = dice_loss(pred_mask, target_mask)
        bce_loss_val = binary_cross_entropy_loss(pred_mask, target_mask)
        
        seg_loss = self.dice_weight * dice_loss_val + self.bce_weight * bce_loss_val
        
        losses = {
            'dice': dice_loss_val,
            'bce': bce_loss_val,
            'segmentation': seg_loss
        }
        
        # Ellipse-aware loss
        if self.use_ellipse_loss:
            # Fit ellipse to predicted mask
            ellipse_params = fit_ellipse_to_mask_tensor(pred_mask)
            
            if ellipse_params is not None:
                # Create ellipse mask
                ellipse_mask = create_ellipse_mask(
                    pred_mask.shape,
                    ellipse_params['center'],
                    ellipse_params['axes'],
                    ellipse_params['angle'],
                    pred_mask.device
                )
                
                # Compute IoU between predicted mask and ellipse mask
                pred_flat = pred_mask.contiguous().view(-1)
                ellipse_flat = ellipse_mask.contiguous().view(-1)
                
                intersection = (pred_flat * ellipse_flat).sum()
                union = pred_flat.sum() + ellipse_flat.sum() - intersection
                
                iou = (intersection + 1e-6) / (union + 1e-6)
                
                # Ellipse loss: encourage mask to be ellipse-like
                ellipse_loss_val = 1.0 - iou
                
                losses['ellipse'] = ellipse_loss_val
                losses['total'] = seg_loss + self.ellipse_weight * ellipse_loss_val
            else:
                # Ellipse fitting failed, use only segmentation loss
                losses['ellipse'] = torch.tensor(0.0, device=pred_mask.device)
                losses['total'] = seg_loss
        else:
            losses['ellipse'] = torch.tensor(0.0, device=pred_mask.device)
            losses['total'] = seg_loss
        
        return losses



