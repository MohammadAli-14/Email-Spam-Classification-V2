"""
Email Spam Classifier Package
"""

__version__ = "1.0.0"
__author__ = "Email Spam Classifier"
__email__ = ""

from .predictor import SpamPredictor
from .preprocessing import TextPreprocessor

__all__ = ["SpamPredictor", "TextPreprocessor"]