"""Integration tests for the FastVLA model."""
import torch
import pytest
from fastvla import FastVLAModel, FastVLAConfig

class TestFastVLAModel:
    """Tests for the FastVLA model."""
    
    def test_forward_pass(self, test_config, test_batch):
        """Test forward pass through the model."""
        # Move batch to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_batch = {k: v.to(device) for k, v in test_batch.items()}
        
        # Initialize model
        model = FastVLAModel(test_config).to(device)
        
        # Forward pass
        outputs = model(
            pixel_values=test_batch["pixel_values"],
            input_ids=test_batch["input_ids"],
            attention_mask=test_batch["attention_mask"],
            labels=test_batch["labels"]
        )
        
        # Check outputs
        action_preds, loss = outputs
        assert action_preds.shape == (test_config.batch_size, test_config.action_dim)
        assert loss is not None
        assert not torch.isnan(loss)
    
    def test_training_step(self, test_config, test_batch):
        """Test a single training step."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_batch = {k: v.to(device) for k, v in test_batch.items()}
        
        # Initialize model and optimizer
        model = FastVLAModel(test_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Training step
        optimizer.zero_grad()
        _, loss = model(
            pixel_values=test_batch["pixel_values"],
            input_ids=test_batch["input_ids"],
            attention_mask=test_batch["attention_mask"],
            labels=test_batch["labels"]
        )
        loss.backward()
        optimizer.step()
        
        # Check that parameters were updated
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.all(param.grad == 0)

if __name__ == "__main__":
    pytest.main([__file__])
