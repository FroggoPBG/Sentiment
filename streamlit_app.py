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

# Safe import of custom module
try:
    import advanced_sentiment
    ADVANCED_SENTIMENT_AVAILABLE = True
except ImportError as e:
    st.error(f"Advanced sentiment module not available: {e}")
    ADVANCED_SENTIMENT_AVAILABLE = False

# Remove these problematic imports for now:
# from sklearn.linear_model import LinearRegression
# import io
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# import pytesseract
# from PIL import Image

# Page configuration
st.set_page_config(
    page_title="NPS Analysis Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rest of your CSS and NPSAnalyzer class stays the same...

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
        """Analyze sentiment using available methods"""
        if ADVANCED_SENTIMENT_AVAILABLE:
            try:
                results = advanced_sentiment.batch_analyze(text_column)
                polarities = [r[0] for r in results]
                categories = [r[1] for r in results]
                all_entities = [ent for r in results for ent in r[2]]
                return polarities, categories, Counter(all_entities).most_common(10)
            except Exception as e:
                st.warning(f"Advanced sentiment failed, using fallback: {e}")
        
        # Fallback to simple sentiment
        from textblob import TextBlob
        polarities = []
        categories = []
        
        for text in text_column:
            if pd.isna(text) or str(text).strip() == '':
                polarities.append(0)
                categories.append('Neutral')
            else:
                try:
                    blob = TextBlob(str(text))
                    polarity = blob.sentiment.polarity
                    polarities.append(polarity)
                    
                    if polarity > 0.1:
                        categories.append('Positive')
                    elif polarity < -0.1:
                        categories.append('Negative')
                    else:
                        categories.append('Neutral')
                except:
                    polarities.append(0)
                    categories.append('Neutral')
        
        return polarities, categories, []
    
    def get_key_themes(self, text_column, max_features=20):
        """Extract key themes using TF-IDF"""
        texts = [str(text).lower() for text in text_column if pd.notna(text) and len(str(text).strip()) > 3]
        
        if len(texts) < 2:
            return []
        
        try:
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
        except Exception as e:
            st.warning(f"Theme extraction failed: {e}")
            return []

# Keep all your chart functions the same...

def main():
    st.markdown('<h1 class="main-header">🚀 Advanced NPS Analysis Platform</h1>', unsafe_allow_html=True)
    
    analyzer = NPSAnalyzer()
    
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        if analyzer.load_data(uploaded_file):
            df = analyzer.df
            is_valid, missing = analyzer.validate_data(df)
            if not is_valid:
                st.error(f"Missing columns: {missing}")
                return
            
            st.success("✅ Data loaded successfully!")
            
            # Calculate NPS
            nps_score, promoters, passives, detractors, conf_lower, conf_upper, margin_error = analyzer.calculate_nps_from_categories(df['score'])
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("NPS Score", f"{nps_score:.1f}")
            with col2:
                st.metric("Promoters", promoters)
            with col3:
                st.metric("Passives", passives)
            with col4:
                st.metric("Detractors", detractors)
            
            # Advanced sentiment analysis
            try:
                polarities, categories, entities = analyzer.analyze_sentiment(df['feedback'])
                df['sentiment_score'] = polarities
                df['sentiment_category'] = categories
                
                st.subheader("💭 Sentiment Analysis")
                sentiment_counts = pd.Series(categories).value_counts()
                
                fig = go.Figure(data=[
                    go.Bar(x=sentiment_counts.index, y=sentiment_counts.values)
                ])
                fig.update_layout(title="Sentiment Distribution")
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Sentiment analysis failed: {e}")
            
            # Themes
            try:
                themes = analyzer.get_key_themes(df['feedback'])
                if themes:
                    st.subheader("🔍 Key Themes")
                    for i, (theme, score) in enumerate(themes[:10], 1):
                        st.write(f"{i}. {theme} (Score: {score:.3f})")
            except Exception as e:
                st.error(f"Theme extraction failed: {e}")
            
            # Word cloud (simplified)
            try:
                st.subheader("☁️ Feedback Word Cloud")
                valid_texts = [str(text) for text in df['feedback'] if pd.notna(text) and len(str(text).strip()) > 2]
                if valid_texts:
                    all_text = ' '.join(valid_texts)
                    if len(all_text.strip()) > 10:
                        wordcloud = WordCloud(width=800, height=400).generate(all_text)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                    else:
                        st.info("Not enough text for word cloud")
                else:
                    st.info("No valid text data for word cloud")
            except Exception as e:
                st.warning(f"Word cloud generation failed: {e}")
    
    else:
        st.info("👆 Please upload a CSV file to get started!")

if __name__ == "__main__":
    main()
