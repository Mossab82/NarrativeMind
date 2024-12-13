import argparse
from pathlib import Path
import logging
from narrative_mind.data import MADARProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--madar_path', type=str, required=True)
    parser.add_argument('--stories_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='data/processed')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize processor
    processor = MADARProcessor()
    
    # Process MADAR corpus
    logger.info("Processing MADAR corpus...")
    processor.process_parallel_corpus(
        args.madar_path,
        save=True
    )
    
    # Process story dataset
    logger.info("Processing story dataset...")
    processor.create_story_dataset(
        args.stories_path,
        save=True
    )

if __name__ == '__main__':
    main()
