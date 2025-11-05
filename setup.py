"""Setup script for MoR package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mor",
    version="0.1.0",
    author="Jushaan Kalra",
    author_email="jkalra@andrew.cmu.edu",
    description="MoR: Mixture of Retrievers - A package for combining multiple retrieval methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jushaan/MixtureRetrievers",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Indexing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "beir>=2.2.0",
        "pandas>=2.3.0",
        "datasets>=2.0.0",
        "huggingface-hub>=0.15.0",
        "sentence-transformers>=2.2.0",
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "pyserini>=0.20.0",
        "faiss-cpu>=1.7.0",  # or faiss-gpu if you have GPU
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mor=mor.main:main",
        ],
    },
)
