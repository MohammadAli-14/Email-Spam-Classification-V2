"""
setup.py - Package installation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="email-spam-classifier",
    version="1.0.0",
    author="Email Spam Classifier Team",
    author_email="",
    description="Machine learning model to classify emails as spam or not spam",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/email-spam-classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.28.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "plotly>=5.17.0",
        "xgboost>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pylint>=2.17.0",
        ],
        "api": [
            "fastapi",
            "uvicorn[standard]",
            "httpx",
        ]
    },
    entry_points={
        "console_scripts": [
            "spam-classifier=src.predictor:test_predictor",
            "spam-api=api:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["models/*.pkl", "models/*.json", "data/*.json"],
    },
)