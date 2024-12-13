# Guide to Reproduce Results

This guide details how to reproduce the results from our paper.

## 1. Data Preparation

1. Download MADAR corpus:
```bash
python scripts/download_data.py --dataset madar
```

2. Preprocess data:
```bash
python scripts/preprocess_data.py \
    --madar_path data/raw/madar/parallel-corpus.txt \
    --stories_path data/raw/stories.json
```

## 2. Training

1. Train the model:
```bash
python scripts/train.py \
    --config configs/training_config.yaml \
    --output_dir outputs/experiment1
```

Expected training metrics:
- Training time: ~8 hours on V100 GPU
- Final loss: ~2.3
- Validation BLEU: ~29.8

## 3. Evaluation

1. Run full evaluation:
```bash
python scripts/evaluate.py \
    --model_path outputs/experiment1/best_model \
    --output_dir results/experiment1
```

Expected results:
- BLEU: 29.8 ± 0.4
- Cultural Preservation: 75.2%
- Dialectal Accuracy (κ): 0.72

## 4. Analysis

Run analysis notebooks:
```bash
jupyter notebook notebooks/3_results_analysis.ipynb
```

## Troubleshooting

1. Memory Issues:
   - Reduce batch size in configs/training_config.yaml
   - Use gradient accumulation

2. Reproducibility:
   - Set random seeds: `--seed 42`
   - Use same hardware setup if possible
