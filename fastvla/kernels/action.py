import torch
import triton
import triton.language as tl

@triton.jit
def _action_decode_forward_kernel(
    # Pointers to input and weights
    hidden_ptr, weight1_ptr, bias1_ptr, weight2_ptr, bias2_ptr, output_ptr,
    # Matrix dimensions
    B, D, H, A,
    # Strides
    stride_hb, stride_hd,
    stride_w1i, stride_w1o,
    stride_w2i, stride_w2o,
    stride_ob, stride_oa,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for action decoding with two-layer MLP.
    
    Args:
        hidden_ptr: [B, D] Input hidden states
        weight1_ptr: [D, H] First layer weights
        bias1_ptr: [H] First layer bias
        weight2_ptr: [H, A] Second layer weights
        bias2_ptr: [A] Second layer bias
        output_ptr: [B, A] Output actions
    """
    # Parallelize over batch and action dimensions
    pid_b = tl.program_id(0)
    pid_a = tl.program_id(1)
    
    # Create ranges for the hidden dimension
    offsets_h = tl.arange(0, BLOCK_SIZE)
    mask_h = offsets_h < H
    
    # Load hidden state for this batch
    hidden_offsets = pid_b * stride_hb + offsets_h * stride_hd
    hidden = tl.load(hidden_ptr + hidden_offsets, mask=mask_h, other=0.0)
    
    # First layer: hidden = ReLU(x @ W1 + b1)
    w1_offsets = offsets_h * stride_w1o + tl.arange(0, BLOCK_SIZE)[:, None] * stride_w1i
    w1 = tl.load(weight1_ptr + w1_offsets, mask=mask_h[:, None] & (tl.arange(0, BLOCK_SIZE)[None, :] < D), other=0.0)
    b1 = tl.load(bias1_ptr + offsets_h, mask=mask_h, other=0.0)
    
    # Compute first layer output
    h = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, D, BLOCK_SIZE):
        mask = (i + tl.arange(0, BLOCK_SIZE)) < D
        h += tl.dot(
            tl.load(hidden_ptr + pid_b * stride_hb + (i + tl.arange(0, BLOCK_SIZE)) * stride_hd, mask=mask, other=0.0),
            tl.load(weight1_ptr + (i + tl.arange(0, BLOCK_SIZE)) * stride_w1i + offsets_h * stride_w1o, 
                   mask=mask[:, None] & mask_h[None, :], other=0.0)
        )
    h = tl.maximum(h + b1, 0)  # ReLU
    
    # Second layer: output = tanh(h @ W2 + b2)
    w2 = tl.load(weight2_ptr + offsets_h * stride_w2o + pid_a * stride_w2i, mask=mask_h, other=0.0)
    b2 = tl.load(bias2_ptr + pid_a, mask=pid_a < A, other=0.0)
    
    # Compute final output
    out = tl.sum(h * w2, axis=0) + b2
    out = tl.tanh(out)  # Clip to [-1, 1] for actions
    
    # Store result
    if pid_a < A:
        tl.store(output_ptr + pid_b * stride_ob + pid_a * stride_oa, out)

# Python wrappers
def action_decode_forward(
    hidden: torch.Tensor,
    weight1: torch.Tensor,
    bias1: torch.Tensor,
    weight2: torch.Tensor,
    bias2: torch.Tensor,
) -> torch.Tensor:
    """Forward pass for action decoding."""
    B, D = hidden.shape
    H, A = weight2.shape
    
    # Allocate output
    output = torch.empty((B, A), device=hidden.device, dtype=hidden.dtype)
    
    # Launch kernel
    grid = lambda meta: (B, A)
    _action_decode_forward_kernel[grid](
        hidden, weight1, bias1, weight2, bias2, output,
        B, D, H, A,
        hidden.stride(0), hidden.stride(1),
        weight1.stride(0), weight1.stride(1),
        weight2.stride(0), weight2.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE=triton.next_power_of_2(H),
    )
    return output

def action_decode_backward(
    grad_output: torch.Tensor,
    hidden: torch.Tensor,
    weight1: torch.Tensor,
    bias1: torch.Tensor,
    weight2: torch.Tensor,
    bias2: torch.Tensor,
):
    """Backward pass for action decoding."""
    # Implementation depends on the forward pass
    raise NotImplementedError("Backward pass not implemented")
