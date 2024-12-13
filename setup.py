from setuptools import setup, find_packages

setup(
    name="narrative_mind",
    version="1.0.0",
    description="Hybrid Neural-Symbolic System for Arabic Story Generation",
    author="Mossab Ibrahim, Pablo Gervás, Gonzalo Méndez",
    author_email="mibrahim@ucm.es",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "transformers>=4.5.0",
        "camel-tools>=1.0.0",
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "pyyaml>=5.4.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
