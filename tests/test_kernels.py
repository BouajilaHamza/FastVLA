"""Unit tests for custom Triton kernels."""
import torch
import pytest
import numpy as np
from fastvla.kernels import (
    multi_cam_pack_forward,
    vision_language_fusion_forward,
    action_decode_forward,
)

# Test parameters
BATCH_SIZE = 2
NUM_CAMS = 3
C, H, W = 3, 224, 224
SEQ_LENGTH = 32
HIDDEN_DIM = 64
ACTION_DIM = 7

class TestMultiCamPackKernel:
    """Tests for multi-camera packing kernel."""
    
    def test_forward(self):
        """Test forward pass of multi-camera packing."""
        # Create test input
        x = torch.randn(BATCH_SIZE, NUM_CAMS, C, H, W, device='cuda')
        
        # Run kernel
        packed = multi_cam_pack_forward(x)
        
        # Check output shape
        assert packed.shape == (BATCH_SIZE, NUM_CAMS * C, H, W)
        
        # Check values
        for b in range(BATCH_SIZE):
            for c in range(NUM_CAMS):
                assert torch.allclose(
                    x[b, c],
                    packed[b, c*C:(c+1)*C],
                    atol=1e-6
                )

class TestVisionLanguageFusionKernel:
    """Tests for vision-language fusion kernel."""
    
    def test_forward(self):
        """Test forward pass of vision-language fusion."""
        # Create test inputs
        visual = torch.randn(BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM, device='cuda')
        text = torch.randn(BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM, device='cuda')
        
        # Run kernel
        fused = vision_language_fusion_forward(visual, text)
        
        # Check output shape
        assert fused.shape == (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
        
        # Check that output is not all zeros
        assert not torch.allclose(fused, torch.zeros_like(fused), atol=1e-6)

class TestActionDecodeKernel:
    """Tests for action decoding kernel."""
    
    def test_forward(self):
        """Test forward pass of action decoding."""
        # Create test inputs
        hidden = torch.randn(BATCH_SIZE, HIDDEN_DIM, device='cuda')
        weight1 = torch.randn(HIDDEN_DIM, 2*HIDDEN_DIM, device='cuda')
        bias1 = torch.randn(2*HIDDEN_DIM, device='cuda')
        weight2 = torch.randn(2*HIDDEN_DIM, ACTION_DIM, device='cuda')
        bias2 = torch.randn(ACTION_DIM, device='cuda')
        
        # Run kernel
        actions = action_decode_forward(hidden, weight1, bias1, weight2, bias2)
        
        # Check output shape
        assert actions.shape == (BATCH_SIZE, ACTION_DIM)
        
        # Check that actions are in [-1, 1] range due to tanh
        assert torch.all(actions >= -1.0 - 1e-6) and torch.all(actions <= 1.0 + 1e-6)

if __name__ == "__main__":
    pytest.main([__file__])
