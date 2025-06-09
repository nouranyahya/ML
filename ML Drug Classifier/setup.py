"""
Setup script for the Drug Classification Project.

This makes the project installable as a Python package.
"""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README for long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="drug-classification",
    version="1.0.0",
    author="Drug Classification Team",
    author_email="your.email@example.com",
    description="A comprehensive machine learning project for drug classification based on patient characteristics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/drug-classification-project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800"
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=5.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "drug-classify-train=scripts.train_models:main",
            "drug-classify-evaluate=scripts.evaluate_models:main",
            "drug-classify-visualize=scripts.generate_plots:main",
            "drug-classify-pipeline=scripts.run_pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
    },
    zip_safe=False,
    keywords="machine-learning, drug-classification, healthcare, ml, classification, medical",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/drug-classification-project/issues",
        "Source": "https://github.com/yourusername/drug-classification-project",
        "Documentation": "https://github.com/yourusername/drug-classification-project/wiki",
    },
)