"""Setup script for the QuantumDiffusion package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="quantum-diffusion",
    version="0.1.0",
    author="Quantum Diffusion Team",
    author_email="your.email@example.com",
    description="A comprehensive framework for quantum circuit generation using diffusion models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quantum-diffusion",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
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
            "pytest>=7.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "jupyter",
            "notebook"
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "myst-parser"
        ]
    },
    entry_points={
        "console_scripts": [
            "qd-generate=quantum_diffusion.scripts.generate_dataset:main",
            "qd-train=quantum_diffusion.scripts.train_model:main", 
            "qd-evaluate=quantum_diffusion.scripts.evaluate_model:main",
        ],
    },
    include_package_data=True,
    package_data={
        "quantum_diffusion": [
            "configs/**/*.yaml",
            "configs/**/*.yml"
        ]
    },
    zip_safe=False,
    keywords="quantum computing diffusion models machine learning circuits",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/quantum-diffusion/issues",
        "Source": "https://github.com/yourusername/quantum-diffusion",
        "Documentation": "https://quantum-diffusion.readthedocs.io/",
    },
)