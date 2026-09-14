"""
LoRA (Low-Rank Adaptation) implementation for UltraSAM.
Applies LoRA to prompt encoder and mask decoder, freezes image encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import math


class LoRALinear(nn.Module):
    """
    LoRA wrapper for a linear layer.
    Original weight W is frozen, only A and B are trainable.
    Output = W(x) + (B @ A)(x) * (alpha / rank)
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze original layer
        for param in original_layer.parameters():
            param.requires_grad = False
        
        self.original_layer = original_layer
        
        # LoRA matrices
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output
        original_out = self.original_layer(x)
        
        # LoRA output
        x_drop = self.dropout(x)
        lora_out = F.linear(x_drop, self.lora_A.t())  # [..., rank]
        lora_out = F.linear(lora_out, self.lora_B)  # [..., out_features]
        lora_out = lora_out * self.scaling
        
        return original_out + lora_out
    
    def merge_weights(self):
        """Merge LoRA weights into original layer (for inference)."""
        with torch.no_grad():
            merged_weight = self.original_layer.weight + (
                self.lora_B @ self.lora_A
            ) * self.scaling
            self.original_layer.weight.data = merged_weight


class LoRAAttention(nn.Module):
    """
    LoRA wrapper for multi-head attention.
    Applies LoRA to query, key, value projections.
    """
    
    def __init__(
        self,
        original_attention: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Freeze original attention
        for param in original_attention.parameters():
            param.requires_grad = False
        
        self.original_attention = original_attention
        
        # Find qkv projection (could be separate q, k, v or combined qkv)
        # This is a simplified version - may need adjustment based on actual SAM structure
        if hasattr(original_attention, 'qkv'):
            self.has_combined_qkv = True
            self.lora_qkv = LoRALinear(original_attention.qkv, rank, alpha, dropout)
        elif hasattr(original_attention, 'q_proj') and hasattr(original_attention, 'k_proj') and hasattr(original_attention, 'v_proj'):
            self.has_combined_qkv = False
            self.lora_q = LoRALinear(original_attention.q_proj, rank, alpha, dropout)
            self.lora_k = LoRALinear(original_attention.k_proj, rank, alpha, dropout)
            self.lora_v = LoRALinear(original_attention.v_proj, rank, alpha, dropout)
        else:
            raise ValueError("Could not find qkv projections in attention module")
    
    def forward(self, *args, **kwargs):
        # For now, just call original attention
        # In a full implementation, we'd intercept and modify qkv computation
        return self.original_attention(*args, **kwargs)


def apply_lora_to_module(
    module: nn.Module,
    target_modules: List[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0
) -> Dict[str, LoRALinear]:
    """
    Apply LoRA to specified modules.
    
    Args:
        module: The module to apply LoRA to
        target_modules: List of module names to target (e.g., ['qkv', 'proj'])
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        dropout: Dropout rate for LoRA
        
    Returns:
        Dictionary mapping module names to LoRA wrappers
    """
    lora_modules = {}
    
    def _apply_lora_recursive(name, mod):
        for child_name, child_mod in mod.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            # Check if this is a target module
            if any(target in child_name for target in target_modules):
                if isinstance(child_mod, nn.Linear):
                    lora_wrapper = LoRALinear(child_mod, rank, alpha, dropout)
                    setattr(mod, child_name, lora_wrapper)
                    lora_modules[full_name] = lora_wrapper
            
            # Recursively apply to children
            _apply_lora_recursive(full_name, child_mod)
    
    _apply_lora_recursive("", module)
    
    return lora_modules


def freeze_module(module: nn.Module):
    """Freeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = False


def get_trainable_parameters(model: nn.Module) -> List[torch.nn.Parameter]:
    """Get all trainable parameters (LoRA parameters only)."""
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count number of parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

