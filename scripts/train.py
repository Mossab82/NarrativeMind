import argparse
import torch
from pathlib import Path
import yaml
import logging
from narrative_mind.models import NarrativeMind
from narrative_mind.data import DataLoader, MADARProcessor
from narrative_mind.evaluation import NarrativeEvaluator
from torch.optim import AdamW
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = NarrativeMind(
        model_name=config['model']['base_model'],
        device=args.device
    )
    
    # Initialize data loader
    data_loader = DataLoader(
        data_dir=args.data_dir,
        tokenizer=model.tokenizer,
        batch_size=config['training']['batch_size']
    )
    
    # Get training data
    train_loader = data_loader.load_madar(split='train')
    val_loader = data_loader.load_madar(split='val')
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Initialize evaluator
    evaluator = NarrativeEvaluator()
    
    # Training loop
    best_val_loss = float('inf')
    patience = config['training']['patience']
    patience_counter = 0
    
    for epoch in range(config['training']['max_epochs']):
        logger.info(f"Starting epoch {epoch+1}")
        
        # Training
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc='Training'):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'].to(args.device),
                attention_mask=batch['attention_mask'].to(args.device),
                labels=batch['input_ids'].to(args.device)
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                outputs = model(
                    input_ids=batch['input_ids'].to(args.device),
                    attention_mask=batch['attention_mask'].to(args.device),
                    labels=batch['input_ids'].to(args.device)
                )
                
                val_loss += outputs.loss.item()
                
                # Generate and evaluate sample
                if batch.get('dialect'):
                    text = model.generate_story(
                        model.tokenizer.decode(batch['input_ids'][0]),
                        target_dialect=batch['dialect'][0]
                    )
                    metrics = evaluator.evaluate(text)
                    val_metrics.append(metrics)
                    
        avg_val_loss = val_loss / len(val_loader)
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}")
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Val Loss: {avg_val_loss:.4f}")
        
        if val_metrics:
            avg_metrics = {k: np.mean([m[k] for m in val_metrics]) 
                         for k in val_metrics[0].keys()}
            logger.info("Validation Metrics:")
            for k, v in avg_metrics.items():
                logger.info(f"{k}: {v:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            model_path = Path(args.output_dir) / 'best_model'
            model_path.mkdir(parents=True, exist_ok=True)
            model.save(model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break

if __name__ == '__main__':
    main()
