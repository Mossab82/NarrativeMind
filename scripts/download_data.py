import argparse
import os
import requests
import hashlib
import tarfile
import logging
from pathlib import Path
from tqdm import tqdm
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data source configurations
DATA_SOURCES = {
    'madar': {
        'url': 'https://github.com/camel-lab/MADAR/releases/download/v2.0/MADAR-Corpus-v2.0.tar.gz',
        'md5': 'a7b12d43fb069d2c47f13eb6e4fc717b',  # Example MD5, would need real hash
        'target_dir': 'data/raw/madar'
    },
    'stories': {
        'url': 'https://github.com/NarrativeMind/arabic-generation/releases/download/v1.0/stories.json',
        'md5': 'b8f2d43fb069d2c47f13eb6e4fc818c',  # Example MD5, would need real hash
        'target_dir': 'data/raw/stories'
    }
}

def download_file(url: str, target_path: Path, chunk_size: int = 8192):
    """Download file with progress bar.
    
    Args:
        url: URL to download from
        target_path: Path to save file
        chunk_size: Download chunk size
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(target_path, 'wb') as f:
        with tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            desc=f"Downloading {target_path.name}"
        ) as pbar:
            for data in response.iter_content(chunk_size=chunk_size):
                size = f.write(data)
                pbar.update(size)

def verify_md5(file_path: Path, expected_md5: str) -> bool:
    """Verify file MD5 hash.
    
    Args:
        file_path: Path to file
        expected_md5: Expected MD5 hash
        
    Returns:
        True if hash matches
    """
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5.update(chunk)
    return md5.hexdigest() == expected_md5

def extract_tar(file_path: Path, target_dir: Path):
    """Extract tar archive.
    
    Args:
        file_path: Path to tar file
        target_dir: Extraction directory
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=target_dir)

def download_dataset(name: str, force: bool = False):
    """Download and prepare dataset.
    
    Args:
        name: Dataset name ('madar' or 'stories')
        force: Force download even if files exist
    """
    if name not in DATA_SOURCES:
        raise ValueError(f"Unknown dataset: {name}")
        
    config = DATA_SOURCES[name]
    target_dir = Path(config['target_dir'])
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Download file
    file_name = Path(config['url']).name
    target_path = target_dir / file_name
    
    if not target_path.exists() or force:
        logger.info(f"Downloading {name} dataset...")
        download_file(config['url'], target_path)
    
    # Verify download
    logger.info("Verifying download...")
    if not verify_md5(target_path, config['md5']):
        raise ValueError(f"MD5 verification failed for {name}")
    
    # Extract if needed
    if file_name.endswith('.tar.gz'):
        logger.info("Extracting archive...")
        extract_tar(target_path, target_dir)
        
    logger.info(f"Successfully prepared {name} dataset")

def main():
    parser = argparse.ArgumentParser(description="Download datasets for NarrativeMind")
    parser.add_argument(
        '--dataset',
        choices=['madar', 'stories', 'all'],
        default='all',
        help='Dataset to download'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force download even if files exist'
    )
    
    args = parser.parse_args()
    
    try:
        if args.dataset == 'all':
            for dataset in DATA_SOURCES:
                download_dataset(dataset, args.force)
        else:
            download_dataset(args.dataset, args.force)
            
        logger.info("All downloads completed successfully")
        
    except Exception as e:
        logger.error(f"Error downloading data: {str(e)}")
        raise

if __name__ == '__main__':
    main()
