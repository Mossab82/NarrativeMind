import argparse
import torch
import json
from pathlib import Path
import logging
from narrative_mind.models import NarrativeMind
from narrative_mind.data import DataLoader
from narrative_mind.evaluation import NarrativeEvaluator
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load model
    model = NarrativeMind.load(args.model_path)
    model.to(args.device)
    model.eval()
    
    # Initialize data loader and evaluator
    data_loader = DataLoader(
        data_dir=args.data_dir,
        tokenizer=model.tokenizer
    )
    evaluator = NarrativeEvaluator()
    
    # Load test data
    test_loader = data_loader.load_madar(split='test')
    
    # Evaluate
    all_metrics = []
    generated_samples = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            # Generate story
            text = model.generate_story(
                model.tokenizer.decode(batch['input_ids'][0]),
                target_dialect=batch.get('dialect', [None])[0]
            )
            
            # Evaluate
            metrics = evaluator.evaluate(
                text,
                target_dialect=batch.get('dialect', [None])[0]
            )
            all_metrics.append(metrics)
            
            # Save generated sample
            generated_samples.append({
                'generated_text': text,
                'metrics': metrics
            })
    
    # Calculate average metrics
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) 
                  for k in all_metrics[0].keys()}
    
    # Log results
    logger.info("Test Results:")
    for k, v in avg_metrics.items():
        logger.info(f"{k}: {v:.4f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump({
            'average_metrics': avg_metrics,
            'generated_samples': generated_samples
        }, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
