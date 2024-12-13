# Installation Guide

## Prerequisites

- Python 3.7+
- PyTorch 1.8+
- CUDA (optional, for GPU support)

## Step-by-Step Installation

1. Clone the repository:
```bash
git clone https://github.com/NarrativeMind/arabic-generation.git
cd arabic-generation
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

4. Download required resources:
```bash
python scripts/download_data.py
```

## Verify Installation

Run tests to verify the installation:
```bash
pytest tests/
```

## Common Issues

1. CUDA/GPU Issues:
   - Ensure CUDA toolkit matches PyTorch version
   - Set CUDA_VISIBLE_DEVICES if needed

2. Dependencies:
   - If experiencing conflicts, try: `pip install -r requirements.txt --no-deps`

3. Resource Downloads:
   - If data download fails, manually download from provided links
