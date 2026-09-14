"""
Data loader for UltraSAM fine-tuning with point-robust augmentation.
Generates point prompts with jittering for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Callable, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


def compute_mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """
    Compute centroid (center of mass) of a binary mask.
    
    Args:
        mask: Binary mask (H, W) with values 0 or 1
        
    Returns:
        (cx, cy): Centroid coordinates
    """
    mask_binary = (mask > 0).astype(np.float32)
    
    if mask_binary.sum() == 0:
        # If mask is empty, return center of image
        h, w = mask.shape
        return w / 2.0, h / 2.0
    
    # Compute moments
    moments = cv2.moments(mask_binary)
    
    if moments['m00'] == 0:
        h, w = mask.shape
        return w / 2.0, h / 2.0
    
    cx = moments['m10'] / moments['m00']
    cy = moments['m01'] / moments['m00']
    
    return float(cx), float(cy)


def jitter_point(
    point: Tuple[float, float],
    max_radius: float = 10.0,
    image_shape: Optional[Tuple[int, int]] = None
) -> Tuple[float, float]:
    """
    Add random jitter to a point within a specified radius.
    
    Args:
        point: Original point (x, y)
        max_radius: Maximum jitter radius in pixels
        image_shape: (height, width) to clamp point within bounds
        
    Returns:
        Jittered point (x, y)
    """
    x, y = point
    
    # Generate random offset
    angle = random.uniform(0, 2 * np.pi)
    radius = random.uniform(0, max_radius)
    
    dx = radius * np.cos(angle)
    dy = radius * np.sin(angle)
    
    jittered_x = x + dx
    jittered_y = y + dy
    
    # Clamp to image bounds if provided
    if image_shape is not None:
        h, w = image_shape
        jittered_x = np.clip(jittered_x, 0, w - 1)
        jittered_y = np.clip(jittered_y, 0, h - 1)
    
    return float(jittered_x), float(jittered_y)


class UltraSAMFineTuneDataset(Dataset):
    """
    Dataset for UltraSAM fine-tuning with point prompts.
    Supports point-robust augmentation by jittering point coordinates.
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        class_name: str = 'cardiac',
        transform: Optional[Callable] = None,
        point_jitter_radius: float = 10.0,
        use_point_jitter: bool = True,
        image_size: int = 512
    ):
        """
        Args:
            image_dir: Directory containing images
            mask_dir: Directory containing masks
            class_name: 'cardiac' or 'thoracic'
            transform: Optional augmentation transforms
            point_jitter_radius: Maximum radius for point jittering (pixels)
            use_point_jitter: Whether to apply point jittering during training
            image_size: Target image size (assumes square)
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.class_name = class_name
        self.point_jitter_radius = point_jitter_radius
        self.use_point_jitter = use_point_jitter
        self.image_size = image_size
        
        # Get all image files
        self.image_files = sorted(self.image_dir.glob('*.png'))
        
        print(f"Loaded {len(self.image_files)} images for {class_name} class from {image_dir}")
        print(f"Point jitter: {'enabled' if use_point_jitter else 'disabled'} (radius={point_jitter_radius})")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_id = img_file.stem
        
        # Load image
        image = cv2.imread(str(img_file))
        if image is None:
            raise ValueError(f"Could not load image: {img_file}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]  # (H, W)
        
        # Load mask
        mask = self._load_binary_mask(img_id, original_shape)
        
        # Compute true centroid (before any transforms)
        true_centroid = compute_mask_centroid(mask)
        
        # Apply transforms (image and mask together)
        if self.transform:
            transformed = self.transform(
                image=image,
                mask=(mask * 255).astype(np.uint8)
            )
            image = transformed['image']
            mask = (np.array(transformed['mask']) > 0).astype(np.float32)
        else:
            # Basic resize and normalize if no transform
            image = cv2.resize(image, (self.image_size, self.image_size))
            mask = cv2.resize(mask.astype(np.uint8), (self.image_size, self.image_size), 
                            interpolation=cv2.INTER_NEAREST).astype(np.float32)
            
            # Normalize image
            image = image.astype(np.float32) / 255.0
        
        # Compute centroid after resize (scale coordinates)
        scale_x = self.image_size / original_shape[1]
        scale_y = self.image_size / original_shape[0]
        scaled_centroid = (true_centroid[0] * scale_x, true_centroid[1] * scale_y)
        
        # Apply point jittering if enabled (only during training)
        if self.use_point_jitter and self.transform is not None:
            point = jitter_point(
                scaled_centroid,
                max_radius=self.point_jitter_radius,
                image_shape=(self.image_size, self.image_size)
            )
        else:
            point = scaled_centroid
        
        # Convert to tensors
        if not isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = torch.from_numpy(image).permute(2, 0, 1).float()
            else:
                image = torch.from_numpy(image).unsqueeze(0).float()
        
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()
        
        # Point prompt: (x, y) coordinates and label (1 = foreground)
        point_coords = torch.tensor([point], dtype=torch.float32)  # [1, 2]
        point_labels = torch.tensor([1], dtype=torch.long)  # [1] - 1 means foreground
        
        return {
            'image': image,
            'mask': mask,
            'point_coords': point_coords,
            'point_labels': point_labels,
            'image_id': img_id,
            'original_shape': original_shape
        }
    
    def _load_binary_mask(self, img_id: str, image_shape: Tuple[int, int]) -> np.ndarray:
        """Load binary mask for the specified class."""
        candidate_paths = [
            self.mask_dir / f'{img_id}_{self.class_name}.png',
            self.mask_dir / f'{img_id}-{self.class_name}.png'
        ]
        
        if self.class_name == 'thoracic':
            candidate_paths.extend([
                self.mask_dir / f'{img_id}_thorax.png',
                self.mask_dir / f'{img_id}-thorax.png'
            ])
        
        for path in candidate_paths:
            if path.exists():
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Resize to match image if needed
                    if mask.shape[:2] != image_shape:
                        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
                    return (mask > 0).astype(np.float32)
        
        # Return empty mask if not found
        return np.zeros(image_shape, dtype=np.float32)


def get_train_transforms(image_size: int = 512):
    """Get training augmentations (preserves point-mask correspondence)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,  # Smaller shifts to preserve point accuracy
            scale_limit=0.1,
            rotate_limit=5,  # Small rotation
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transforms(image_size: int = 512):
    """Get validation transforms (no augmentation)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_data_loaders_ultrasam(
    data_root: str,
    class_name: str = 'cardiac',
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: int = 512,
    point_jitter_radius: float = 10.0,
    use_point_jitter_train: bool = True,
    use_point_jitter_val: bool = False
):
    """
    Get train and validation data loaders for UltraSAM fine-tuning.
    
    Args:
        data_root: Root directory containing processed data
        class_name: 'cardiac' or 'thoracic'
        batch_size: Batch size
        num_workers: Number of worker processes
        image_size: Target image size
        point_jitter_radius: Maximum jitter radius for training
        use_point_jitter_train: Enable point jittering in training
        use_point_jitter_val: Enable point jittering in validation (usually False)
        
    Returns:
        train_loader, val_loader
    """
    data_path = Path(data_root)
    
    # Training dataset
    train_dataset = UltraSAMFineTuneDataset(
        image_dir=str(data_path / 'training' / 'images'),
        mask_dir=str(data_path / 'training' / 'annfiles_mask'),
        class_name=class_name,
        transform=get_train_transforms(image_size),
        point_jitter_radius=point_jitter_radius,
        use_point_jitter=use_point_jitter_train,
        image_size=image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Validation dataset
    val_dataset = UltraSAMFineTuneDataset(
        image_dir=str(data_path / 'validation' / 'images'),
        mask_dir=str(data_path / 'validation' / 'annfiles_mask'),
        class_name=class_name,
        transform=get_val_transforms(image_size),
        point_jitter_radius=0.0,  # No jitter in validation
        use_point_jitter=use_point_jitter_val,
        image_size=image_size
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader



