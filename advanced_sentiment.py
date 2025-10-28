# advanced_sentiment.py - Lightweight sentiment analysis

import spacy
import os

# Try to load spacy model, fallback if not available
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except OSError:
    SPACY_AVAILABLE = False
    nlp = None

def analyze_advanced_sentiment(text):
    """Advanced sentiment with fallback to simple analysis"""
    if not text or not isinstance(text, str) or len(text.strip()) < 3:
        return 0.0, "Neutral", []
    
    try:
        # Simple sentiment analysis as fallback
        from textblob import TextBlob
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        # Determine category
        if polarity > 0.1:
            category = 'Positive'
        elif polarity < -0.1:
            category = 'Negative'
        else:
            category = 'Neutral'
        
        # Extract entities if spacy is available
        entities = []
        if SPACY_AVAILABLE and nlp:
            try:
                doc = nlp(text)
                entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PRODUCT', 'GPE']]
            except:
                entities = []
        
        # Detect potential sarcasm
        if "not" in text.lower() and polarity > 0:
            category = 'Mixed (Possible Sarcasm)'
        
        return polarity, category, entities
        
    except Exception as e:
        # Ultimate fallback
        return 0.0, "Neutral", []

def batch_analyze(texts):
    """Batch process multiple texts with error handling"""
    results = []
    for text in texts:
        try:
            result = analyze_advanced_sentiment(text)
            results.append(result)
        except:
            results.append((0.0, "Neutral", []))
    return results
