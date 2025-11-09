"""Pytest configuration and fixtures for FastVLA tests."""
import pytest
import torch
import numpy as np
from fastvla import FastVLAConfig

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Test configurations
TEST_BATCH_SIZE = 2
TEST_IMAGE_SIZE = 224
TEST_NUM_CAMS = 3
TEST_SEQ_LENGTH = 32
TEST_HIDDEN_DIM = 64
TEST_ACTION_DIM = 7

@pytest.fixture
def test_config():
    """Create a test configuration."""
    config = FastVLAConfig(
        vision_encoder_name="google/vit-base-patch16-224",
        llm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        image_size=TEST_IMAGE_SIZE,
        max_sequence_length=TEST_SEQ_LENGTH,
        hidden_size=TEST_HIDDEN_DIM,
        action_dim=TEST_ACTION_DIM,
        load_in_4bit=False,  # Disable for testing
        use_peft=False,      # Disable for testing
    )
    # Add batch_size attribute for compatibility
    config.batch_size = TEST_BATCH_SIZE
    return config

@pytest.fixture
def test_batch():
    """Create a test batch of data."""
    # Generate random images [B, num_cams, C, H, W]
    images = torch.randn(
        TEST_BATCH_SIZE,
        TEST_NUM_CAMS,
        3,  # RGB
        TEST_IMAGE_SIZE,
        TEST_IMAGE_SIZE,
    )
    
    # Generate random input IDs and attention masks
    input_ids = torch.randint(0, 1000, (TEST_BATCH_SIZE, TEST_SEQ_LENGTH))
    attention_mask = torch.ones_like(input_ids)
    
    # Generate random actions
    actions = torch.randn(TEST_BATCH_SIZE, TEST_ACTION_DIM)
    
    return {
        "pixel_values": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": actions,
    }
