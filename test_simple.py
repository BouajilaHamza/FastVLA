"""
Simple test script to verify FastVLA implementation.
This script tests the core functionality without requiring actual models or data.
"""
import torch
import sys
import traceback

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from fastvla import (
            FastVLAModel,
            FastVLAConfig,
            UnslothVLACollator,
            FastVLATrainer,
            get_quantization_config,
            get_8bit_optimizer,
            PerformanceProfiler,
        )
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_config():
    """Test configuration creation."""
    print("\nTesting configuration...")
    try:
        from fastvla import FastVLAConfig
        
        config = FastVLAConfig(
            vision_encoder_name="google/vit-base-patch16-224",
            llm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            action_dim=7,
            load_in_4bit=False,  # Disable for testing
            use_peft=False,
        )
        
        assert config.action_dim == 7
        assert config.load_in_4bit == False
        print("✅ Configuration test passed")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_kernels():
    """Test custom kernels."""
    print("\nTesting kernels...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        from fastvla.kernels import (
            vision_language_fusion_forward,
            multi_cam_pack_forward,
        )
        
        # Test fusion kernel
        batch_size = 2
        seq_length = 32
        hidden_dim = 64
        
        visual = torch.randn(batch_size, seq_length, hidden_dim, device=device)
        text = torch.randn(batch_size, seq_length, hidden_dim, device=device)
        
        fused = vision_language_fusion_forward(visual, text)
        assert fused.shape == (batch_size, seq_length, hidden_dim)
        print("✅ Fusion kernel test passed")
        
        # Test multi-cam kernel
        num_cams = 3
        c, h, w = 3, 224, 224
        cams = torch.randn(batch_size, num_cams, c, h, w, device=device)
        
        packed = multi_cam_pack_forward(cams)
        assert packed.shape == (batch_size, num_cams * c, h, w)
        print("✅ Multi-cam kernel test passed")
        
        return True
    except Exception as e:
        print(f"❌ Kernel test failed: {e}")
        traceback.print_exc()
        return False


def test_optimization_utils():
    """Test optimization utilities."""
    print("\nTesting optimization utilities...")
    try:
        from fastvla.optimization import (
            get_quantization_config,
            get_peft_config,
            estimate_memory_usage,
        )
        
        # Test quantization config
        qconfig = get_quantization_config(load_in_4bit=True)
        assert qconfig is not None
        print("✅ Quantization config test passed")
        
        # Test PEFT config
        peft_config = get_peft_config(r=16, lora_alpha=32)
        assert peft_config.r == 16
        assert peft_config.lora_alpha == 32
        print("✅ PEFT config test passed")
        
        # Test memory estimation (with dummy model)
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(100, 10)
                self.config = type('Config', (), {'hidden_size': 100, 'num_hidden_layers': 2})()
        
        dummy_model = DummyModel()
        memory_est = estimate_memory_usage(dummy_model, batch_size=2, seq_length=32)
        assert "total_memory_gb" in memory_est
        print("✅ Memory estimation test passed")
        
        return True
    except Exception as e:
        print(f"❌ Optimization utilities test failed: {e}")
        traceback.print_exc()
        return False


def test_collator():
    """Test data collator."""
    print("\nTesting data collator...")
    try:
        from fastvla import UnslothVLACollator
        from transformers import AutoTokenizer
        
        # Create a simple tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        collator = UnslothVLACollator(
            tokenizer=tokenizer,
            max_length=128,
            padding=True,
        )
        
        # Create dummy features
        features = [
            {
                "images": {"rgb": torch.randn(3, 224, 224)},
                "states": torch.randn(7),
                "actions": torch.randn(7),
                "instructions": "Pick up the red block",
            }
        ]
        
        batch = collator(features)
        assert "input_ids" in batch
        assert "attention_mask" in batch
        print("✅ Collator test passed")
        
        return True
    except Exception as e:
        print(f"❌ Collator test failed: {e}")
        traceback.print_exc()
        return False


def test_benchmarking():
    """Test benchmarking utilities."""
    print("\nTesting benchmarking utilities...")
    try:
        from fastvla import PerformanceProfiler
        
        profiler = PerformanceProfiler(device="cpu")
        
        # Test profiling
        with profiler.profile("test_operation"):
            # Simulate some work
            x = torch.randn(100, 100)
            y = torch.randn(100, 100)
            z = x @ y
        
        summary = profiler.get_summary()
        assert len(summary["operations"]) > 0
        print("✅ Benchmarking test passed")
        
        return True
    except Exception as e:
        print(f"❌ Benchmarking test failed: {e}")
        traceback.print_exc()
        return False


def test_training_utils():
    """Test training utilities."""
    print("\nTesting training utilities...")
    try:
        from fastvla.training import FastVLATrainer
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dummy model
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 7)
            
            def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
                batch_size = pixel_values.size(0)
                x = pixel_values.mean(dim=(1, 2, 3, 4))  # Average all dimensions
                x = x[:, :10]  # Take first 10 features
                preds = self.linear(x)
                loss = torch.nn.functional.mse_loss(preds, labels) if labels is not None else None
                return preds, loss
        
        model = DummyModel()
        
        # Create dummy dataset
        dataset = TensorDataset(
            torch.randn(10, 3, 3, 224, 224),  # pixel_values
            torch.randint(0, 1000, (10, 32)),  # input_ids
            torch.ones(10, 32),  # attention_mask
            torch.randn(10, 7),  # labels
        )
        
        def collate_fn(batch):
            pixel_values = torch.stack([b[0] for b in batch])
            input_ids = torch.stack([b[1] for b in batch])
            attention_mask = torch.stack([b[2] for b in batch])
            labels = torch.stack([b[3] for b in batch])
            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        
        train_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        
        # Create trainer
        trainer = FastVLATrainer(
            model=model,
            train_dataloader=train_loader,
            use_8bit_optimizer=False,  # Disable for testing
            use_mixed_precision=False,
            device="cpu",
        )
        
        # Test a single training step
        batch = next(iter(train_loader))
        metrics = trainer.train_step(batch)
        assert "loss" in metrics
        print("✅ Training utilities test passed")
        
        return True
    except Exception as e:
        print(f"❌ Training utilities test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("FastVLA Simple Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Kernels", test_kernels),
        ("Optimization Utils", test_optimization_utils),
        ("Collator", test_collator),
        ("Benchmarking", test_benchmarking),
        ("Training Utils", test_training_utils),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:20s} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

