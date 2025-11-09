import torch
import triton
import triton.language as tl

@triton.jit
def _vision_language_fusion_forward_kernel(
    # Pointers to matrices
    visual_feat_ptr, text_feat_ptr, output_ptr,
    # Matrix dimensions
    B, T, D, C,
    # Strides
    stride_vb, stride_vt, stride_vd,
    stride_tb, stride_tt, stride_td,
    stride_ob, stride_ot, stride_od,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for fusing visual and language features using cross-attention.
    
    Args:
        visual_feat_ptr: [B, T_v, D] Visual features
        text_feat_ptr: [B, T_t, D] Text features
        output_ptr: [B, T_t, D] Output fused features
    """
    # Parallelize over batch and sequence dimensions
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    
    # Create ranges for the feature dimension
    offsets = pid_b * stride_ob + pid_t * stride_ot + tl.arange(0, BLOCK_SIZE)
    mask = offsets < D
    
    # Load visual features (pooled over time)
    visual_offsets = pid_b * stride_vb + tl.arange(0, T) * stride_vt + tl.arange(0, BLOCK_SIZE) * stride_vd
    visual_mask = (tl.arange(0, T) < T) & (tl.arange(0, BLOCK_SIZE) < D)[None, :]
    visual = tl.load(visual_feat_ptr + visual_offsets, mask=visual_mask, other=0.0)
    
    # Load text features
    text_offsets = pid_b * stride_tb + pid_t * stride_tt + tl.arange(0, BLOCK_SIZE) * stride_td
    text = tl.load(text_feat_ptr + text_offsets, mask=mask, other=0.0)
    
    # Simple fusion: weighted sum (can be replaced with more sophisticated attention)
    alpha = 0.5  # Learnable parameter in practice
    fused = alpha * tl.sum(visual, axis=0) / T + (1 - alpha) * text
    
    # Store result
    tl.store(output_ptr + offsets, fused, mask=mask)

@triton.jit
def _vision_language_fusion_backward_kernel(
    grad_output_ptr, visual_feat_ptr, text_feat_ptr,
    grad_visual_ptr, grad_text_ptr,
    B, T_v, T_t, D,
    stride_gout_b, stride_gout_t, stride_gout_d,
    stride_vb, stride_vt, stride_vd,
    stride_tb, stride_tt, stride_td,
    stride_gv_b, stride_gv_t, stride_gv_d,
    stride_gt_b, stride_gt_t, stride_gt_d,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward pass for vision-language fusion."""
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    
    # Create ranges for the feature dimension
    offsets = pid_b * stride_gout_b + pid_t * stride_gout_t + tl.arange(0, BLOCK_SIZE)
    mask = offsets < D
    
    # Load gradient output
    grad_out = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
    
    # Gradient for text features: (1 - alpha) * grad_output
    alpha = 0.5
    grad_text = (1 - alpha) * grad_out
    text_offsets = pid_b * stride_gt_b + pid_t * stride_gt_t + tl.arange(0, BLOCK_SIZE)
    tl.store(grad_text_ptr + text_offsets, grad_text, mask=mask)
    
    # Gradient for visual features: alpha * grad_output / T_v (averaged over time)
    grad_visual_sum = alpha * grad_out / T_v
    
    # Accumulate gradients for visual features (sum over time dimension)
    for t in range(T_v):
        visual_offsets = pid_b * stride_gv_b + t * stride_gv_t + tl.arange(0, BLOCK_SIZE)
        tl.store(grad_visual_ptr + visual_offsets, grad_visual_sum, mask=mask)

# Python wrappers
def vision_language_fusion_forward(visual_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
    """Fuse visual and language features using cross-attention."""
    B, T_v, D = visual_feat.shape
    B_t, T_t, D_t = text_feat.shape
    assert B == B_t and D == D_t, "Batch and feature dimensions must match"
    
    # Allocate output
    output = torch.empty_like(text_feat)
    
    # Launch kernel
    grid = lambda meta: (B, T_t)
    _vision_language_fusion_forward_kernel[grid](
        visual_feat, text_feat, output,
        B, T_v, D, 1,  # C=1 for now, can be extended to multi-head
        visual_feat.stride(0), visual_feat.stride(1), visual_feat.stride(2),
        text_feat.stride(0), text_feat.stride(1), text_feat.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_SIZE=min(1024, triton.next_power_of_2(D)),
    )
    return output

def vision_language_fusion_backward(
    grad_output: torch.Tensor, 
    visual_feat: torch.Tensor, 
    text_feat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gradients for the fusion operation."""
    B, T_v, D = visual_feat.shape
    B_t, T_t, D_t = text_feat.shape
    assert B == B_t and D == D_t, "Batch and feature dimensions must match"
    
    # Allocate gradient tensors
    grad_visual = torch.zeros_like(visual_feat)
    grad_text = torch.zeros_like(text_feat)
    
    # Launch kernel
    grid = lambda meta: (B, T_t)
    _vision_language_fusion_backward_kernel[grid](
        grad_output, visual_feat, text_feat, grad_visual, grad_text,
        B, T_v, T_t, D,
        grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
        visual_feat.stride(0), visual_feat.stride(1), visual_feat.stride(2),
        text_feat.stride(0), text_feat.stride(1), text_feat.stride(2),
        grad_visual.stride(0), grad_visual.stride(1), grad_visual.stride(2),
        grad_text.stride(0), grad_text.stride(1), grad_text.stride(2),
        BLOCK_SIZE=min(1024, triton.next_power_of_2(D)),
    )
    return grad_visual, grad_text
