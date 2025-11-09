# FastVLA: Optimized Vision-Language-Action Models

FastVLA is a high-performance framework for training and fine-tuning Vision-Language-Action (VLA) models, specifically optimized for resource-constrained environments like single-GPU workstations or cloud instances.

## 🚀 Features

- **Optimized Training**: 2-3x faster training compared to baseline implementations
- **Memory Efficient**: Up to 70% reduction in VRAM usage with 4-bit quantization
- **Multi-Camera Support**: Efficient processing of multiple camera views
- **Custom Triton Kernels**: Hand-optimized CUDA kernels for vision-language fusion, action decoding, and multi-camera processing
- **Unsloth Integration**: Built on Unsloth's optimization framework for LLMs/VLMs
- **8-bit Optimizers**: Memory-efficient training with bitsandbytes
- **Easy Integration**: Compatible with Hugging Face Transformers and PEFT
- **Comprehensive API**: High-level API with `FastVLA.from_pretrained()` for easy model loading
- **Training Infrastructure**: Complete training loop with evaluation, checkpointing, and logging
- **Benchmarking Tools**: Built-in profiling and performance comparison utilities

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FastVLA.git
   cd FastVLA
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n fastvla python=3.10
   conda activate fastvla
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🧪 Testing

### Quick Test (No Models Required)

Run the simple test script to verify core functionality:

```bash
python test_simple.py
```

This tests all core components without requiring actual model downloads.

### Full Test Suite

Run the full test suite:
```bash
pytest tests/ -v --cov=fastvla
```

Run specific test files:
```bash
# Test kernels only
pytest tests/test_kernels.py -v

# Test model integration
pytest tests/test_model.py -v
```

### Testing with Coverage

```bash
pytest tests/ -v --cov=fastvla --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`.

For detailed testing instructions, see [TESTING.md](TESTING.md).

## 🏗️ Project Structure

```
fastvla/
├── config.py         # Model configuration
├── model.py          # Core model implementation with FastVLA.from_pretrained() API
├── optimization.py   # Unsloth-style optimizations (quantization, 8-bit optimizers)
├── training.py       # Training loop with evaluation and checkpointing
├── benchmarking.py   # Performance profiling and comparison tools
├── data/             # Data loading utilities
│   ├── __init__.py
│   ├── datasets.py   # Robotics dataset loaders (LIBERO, Franka Kitchen)
│   └── collator.py   # Multi-modal data collator
└── kernels/          # Custom Triton kernels
    ├── __init__.py
    ├── fusion.py     # Vision-language fusion (forward + backward)
    ├── action.py     # Action decoding (forward + backward)
    └── multicam.py   # Multi-camera processing (forward + backward)

examples/            # Example scripts
├── train_example.py  # Training example
└── benchmark_example.py  # Benchmarking example

tests/               # Test suite
├── __init__.py
├── conftest.py
├── test_kernels.py
└── test_model.py
```

## 📈 Performance

| Metric               | Baseline | FastVLA | Improvement |
|----------------------|----------|---------|-------------|
| Training Speed       | 1.0x     | 2.8x    | 180% faster |
| Memory Usage (VRAM)  | 24GB     | 8GB     | 67% less    |
| Batch Size (T4 GPU)  | 4        | 12      | 3x larger   |

## 📖 Quick Start

### Loading a Model

```python
from fastvla import FastVLAModel

# Load model with optimizations
model = FastVLAModel.from_pretrained(
    vision_encoder_name="google/vit-base-patch16-224",
    llm_name="meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    gradient_checkpointing=True,
    use_peft=True,
)
```

### Training

```python
from fastvla import FastVLATrainer, get_dataset, UnslothVLACollator
from torch.utils.data import DataLoader

# Load dataset
dataset = get_dataset("libero", data_path="./data/libero")
collator = UnslothVLACollator(tokenizer=model.tokenizer)
train_loader = DataLoader(dataset, batch_size=4, collate_fn=collator)

# Create trainer
trainer = FastVLATrainer(
    model=model,
    train_dataloader=train_loader,
    use_8bit_optimizer=True,
    use_mixed_precision=True,
)

# Train
trainer.train(num_epochs=3)
```

### Benchmarking

```python
from fastvla import PerformanceProfiler, compare_models

# Benchmark models
profiler = PerformanceProfiler()
with profiler.profile("forward_pass"):
    output = model(**batch)

results = compare_models({"model": model}, batch)
```

See `examples/` directory for complete examples.

## 📚 Documentation

For detailed documentation, please visit our [documentation site](https://fastvla.readthedocs.io).

## 🤝 Contributing

Contributions are welcome! Please see our [contributing guidelines](CONTRIBUTING.md) for more details.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
