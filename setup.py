from setuptools import setup, find_packages

setup(
    name="hot",
    version="0.1.0",
    description="Harmonic Oscillator Transformer (HOT)",
    author="HOT Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "tokenizers>=0.15.0",
        "wandb>=0.16.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
    },
)
