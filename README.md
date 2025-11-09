# FastVLA: Optimized Vision-Language-Action Models

FastVLA is a high-performance framework for training and fine-tuning Vision-Language-Action (VLA) models, specifically optimized for resource-constrained environments like single-GPU workstations or cloud instances.

## 🚀 Features

- **Optimized Training**: 2-3x faster training compared to baseline implementations
- **Memory Efficient**: Up to 70% reduction in VRAM usage with 4-bit quantization
- **Multi-Camera Support**: Efficient processing of multiple camera views
- **Custom Triton Kernels**: Hand-optimized CUDA kernels for critical operations
- **Easy Integration**: Compatible with Hugging Face Transformers and PEFT

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

Run benchmarks:
```bash
pytest tests/ -k "benchmark" --benchmark-only
```

## 🏗️ Project Structure

```
fastvla/
├── config.py         # Model configuration
├── model.py          # Core model implementation
├── data/             # Data loading utilities
│   ├── __init__.py
│   ├── datasets.py
│   └── collator.py
└── kernels/          # Custom Triton kernels
    ├── __init__.py
    ├── fusion.py     # Vision-language fusion
    ├── action.py     # Action decoding
    └── multicam.py   # Multi-camera processing

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

## 📚 Documentation

For detailed documentation, please visit our [documentation site](https://fastvla.readthedocs.io).

## 🤝 Contributing

Contributions are welcome! Please see our [contributing guidelines](CONTRIBUTING.md) for more details.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
