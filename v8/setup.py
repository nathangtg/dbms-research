#!/usr/bin/env python3
"""
ZGQ Package Setup

Installation script for the Zone-Guided Quantization (ZGQ) package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="zgq",
    version="8.0.0",
    author="Nathan Aldyth Prananta Ginting",
    author_email="nathan.n@imail.sunway.edu.my",
    description="Zone-Guided Quantization for Fast Approximate Nearest Neighbor Search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nathangtg/dbms-research",
    project_urls={
        "Bug Tracker": "https://github.com/nathangtg/dbms-research/issues",
        "Documentation": "https://github.com/nathangtg/dbms-research/docs",
        "Source": "https://github.com/nathangtg/dbms-research",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Database :: Database Engines/Servers",
    ],
    keywords=[
        "nearest-neighbor-search",
        "approximate-nearest-neighbors",
        "ann",
        "vector-search",
        "similarity-search",
        "hnsw",
        "quantization",
        "machine-learning",
        "embeddings",
    ],
    packages=find_packages(exclude=["tests*", "benchmarks*", "examples*", "docs*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "hnswlib>=0.7.0",
        "numba>=0.55.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
        "benchmark": [
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
            "pandas>=1.4.0",
            "tqdm>=4.64.0",
            "faiss-cpu>=1.7.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
            "pandas>=1.4.0",
            "tqdm>=4.64.0",
            "faiss-cpu>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "zgq-benchmark=benchmarks.run_benchmarks:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
