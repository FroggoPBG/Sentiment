# advanced_sentiment.py - Custom module for advanced sentiment analysis

import spacy
from transformers import pipeline
import os

# Download spacy model if not present (for Streamlit Cloud)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_advanced_sentiment(text):
    """Advanced sentiment with nuance detection"""
    if not text or not isinstance(text, str) or len(text.strip()) < 3:
        return 0.0, "Neutral", []
    
    # Get polarity and label
    result = sentiment_classifier(text)[0]
    polarity = result['score'] if result['label'] == 'POSITIVE' else -result['score']
    
    # Detect entities/themes
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PRODUCT', 'GPE']]
    
    # Simple nuance (e.g., detect negation for sarcasm-like flips)
    category = 'Positive' if polarity > 0.1 else ('Negative' if polarity < -0.1 else 'Neutral')
    if "not" in text.lower() and polarity > 0:
        category = 'Mixed (Possible Sarcasm)'
    
    return polarity, category, entities

def batch_analyze(texts):
    """Batch process multiple texts"""
    return [analyze_advanced_sentiment(text) for text in texts]
