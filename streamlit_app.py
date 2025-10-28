import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from datetime import datetime, timedelta
import re
from collections import Counter
import advanced_sentiment  # Import the new custom module
from sklearn.linear_model import LinearRegression
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pytesseract
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="NPS Analysis Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

class NPSAnalyzer:
    def __init__(self):
        self.df = None
        
    def load_data(self, uploaded_file):
        """Load and validate CSV data"""
        try:
            self.df = pd.read_csv(uploaded_file)
            return True
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return False
    
    def validate_data(self, df):
        """Validate that required columns exist"""
        required_columns = ['score', 'feedback']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, missing_columns
        return True, []
    
    def calculate_nps_from_categories(self, score_column):
        """Calculate NPS from pre-categorized data"""
        normalized_scores = score_column.str.lower().str.strip()
        
        promoters_count = normalized_scores.str.contains('promoter', na=False).sum()
        passives_count = normalized_scores.str.contains('passive', na=False).sum()
        detractors_count = normalized_scores.str.contains('detractor', na=False).sum()
        
        total_responses = len(score_column.dropna())
        
        if total_responses == 0:
            return 0, 0, 0, 0, 0, 0, 0
        
        promoters_pct = (promoters_count / total_responses) * 100
        passives_pct = (passives_count / total_responses) * 100
        detractors_pct = (detractors_count / total_responses) * 100
        
        nps_score = promoters_pct - detractors_pct
        
        if total_responses > 1:
            margin_error = 1.96 * np.sqrt((promoters_pct * (100 - promoters_pct)) / total_responses)
            conf_lower = max(-100, nps_score - margin_error)
            conf_upper = min(100, nps_score + margin_error)
        else:
            margin_error = 0
            conf_lower = nps_score
            conf_upper = nps_score
        
        return nps_score, promoters_count, passives_count, detractors_count, conf_lower, conf_upper, margin_error
    
    def analyze_sentiment(self, text_column):
        """Analyze sentiment using the custom advanced module"""
        results = advanced_sentiment.batch_analyze(text_column)
        polarities = [r[0] for r in results]
        categories = [r[1] for r in results]
        all_entities = [ent for r in results for ent in r[2]]  # Flatten
        df['sentiment_score'] = polarities
        df['sentiment_category'] = categories
        return polarities, categories, Counter(all_entities).most_common(10)
    
    def get_key_themes(self, text_column, max_features=20):
        """Extract key themes using TF-IDF"""
        # [Existing code - omitted for brevity]
        texts = [str(text).lower() for text in text_column if pd.notna(text) and len(str(text).strip()) > 3]
        
        if len(texts) < 2:
            return []
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.sum(axis=0).A1
        
        theme_scores = list(zip(feature_names, scores))
        theme_scores.sort(key=lambda x: x[1], reverse=True)
        
        return theme_scores[:15]

def create_nps_gauge(nps_score):
    """Create an NPS gauge chart"""
    # [Existing code - omitted for brevity]
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = nps_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "NPS Score"},
        gauge = {
            'axis': {'range': [-100, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-100, 0], 'color': "lightgray"},
                {'range': [0, 50], 'color': "gray"},
                {'range': [50, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def create_distribution_chart(promoters, passives, detractors):
    """Create distribution pie chart"""
    # [Existing code - omitted for brevity]
    labels = ['Promoters', 'Passives', 'Detractors']
    values = [promoters, passives, detractors]
    colors = ['#2E8B57', '#FFD700', '#DC143C']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        marker_colors=colors,
        textinfo='label+percent+value',
        textfont_size=12
    )])
    
    fig.update_layout(
        title="NPS Distribution",
        height=400
    )
    return fig

def create_sentiment_analysis_chart(df):
    """Create sentiment analysis visualization"""
    # [Existing code - omitted for brevity]
    if 'sentiment_score' not in df.columns:
        return None
    
    df['sentiment_category'] = df['sentiment_score'].apply(
        lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral')
    )
    
    sentiment_counts = df['sentiment_category'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            marker_color=['green', 'gray', 'red']
        )
    ])
    
    fig.update_layout(
        title="Sentiment Analysis of Feedback",
        xaxis_title="Sentiment",
        yaxis_title="Count",
        height=400
    )
    
    return fig

def generate_wordcloud(text_data):
    """Generate word cloud from text data"""
    # [Existing code - omitted for brevity]
    try:
        all_text = ' '.join(text_data.dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400).generate(all_text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        return fig
    except:
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Advanced NPS Analysis Platform</h1>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = NPSAnalyzer()
    
    # Sidebar
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        if analyzer.load_data(uploaded_file):
            df = analyzer.df
            is_valid, missing = analyzer.validate_data(df)
            if not is_valid:
                st.error(f"Missing columns: {missing}")
                return
            
            # Calculate NPS
            nps_score, promoters, passives, detractors, _, _, _ = analyzer.calculate_nps_from_categories(df['score'])
            
            # Sentiment analysis using custom module
            polarities, categories, entities = analyzer.analyze_sentiment(df['feedback'])
            df['sentiment_score'] = polarities
            df['sentiment_category'] = categories
            
            # Themes
            themes = analyzer.get_key_themes(df['feedback'])
            
            # Display metrics and charts
            st.write("NPS Score:", nps_score)
            # Add more UI elements as in original
            
            # Word cloud
            wc_fig = generate_wordcloud(df['feedback'])
            if wc_fig:
                st.pyplot(wc_fig)

if __name__ == "__main__":
    main()
