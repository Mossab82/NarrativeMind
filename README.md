# NarrativeMind: A Hybrid Neural-Symbolic System for Arabic Story Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

Official implementation of "NarrativeMind: A Hybrid Neural-Symbolic System for Structure-Aware Arabic Story Generation" (2024).

## Features

- Cultural-aware neural generation using enhanced AraBERT
- Pattern-based narrative structure preservation
- Adaptive dialectal processing
- Comprehensive evaluation framework

## Key Results

- BLEU Score: 29.8 ± 0.4
- Cultural Preservation: 75.2% (CI: [72.8%, 77.6%])
- Pattern Preservation (F1): 0.76 ± 0.03
- Dialectal Accuracy (κ): 0.72

## Quick Start

```bash
# Clone repository
git clone https://github.com/NarrativeMind/arabic-generation.git
cd arabic-generation

# Install dependencies
pip install -e .

# Download MADAR corpus
python scripts/download_data.py

# Preprocess data
python scripts/preprocess_data.py

# Train model
python scripts/train.py --config configs/training_config.yaml

# Evaluate model
python scripts/evaluate.py --model_path outputs/best_model
```

## Repository Structure

```
NarrativeMind/
├── narrative_mind/          # Main package
│   ├── models/             # Model implementations
│   ├── data/              # Data processing
│   ├── evaluation/        # Evaluation metrics
│   └── utils/             # Utilities
├── scripts/               # Training scripts
├── notebooks/            # Analysis notebooks
├── tests/               # Test suite
├── configs/            # Configurations
├── data/              # Data directory
└── docs/             # Documentation
```

## Citation

```bibtex
@article{ibrahim2024narrativemind,
  title={NarrativeMind: A Hybrid Neural-Symbolic System for Structure-Aware Arabic Story Generation},
  author={Ibrahim, Mossab and Gervás, Pablo and Méndez, Gonzalo},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
