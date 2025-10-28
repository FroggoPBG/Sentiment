import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from datetime import datetime, timedelta
import re
from collections import Counter

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
        # Normalize the categories (handle different capitalizations)
        normalized_scores = score_column.astype(str).str.lower().str.strip()
        
        # Count each category
        promoters_count = normalized_scores.str.contains('promoter', na=False).sum()
        passives_count = normalized_scores.str.contains('passive', na=False).sum()
        detractors_count = normalized_scores.str.contains('detractor', na=False).sum()
        
        total_responses = len(score_column.dropna())
        
        if total_responses == 0:
            return 0, 0, 0, 0, 0, 0, 0
        
        # Calculate percentages
        promoters_pct = (promoters_count / total_responses) * 100
        passives_pct = (passives_count / total_responses) * 100
        detractors_pct = (detractors_count / total_responses) * 100
        
        # Calculate NPS
        nps_score = promoters_pct - detractors_pct
        
        # Simple confidence interval
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
        """Analyze sentiment of feedback text using TextBlob"""
        sentiments = []
        
        for text in text_column:
            if pd.isna(text) or str(text).strip() == '':
                sentiments.append(0)
            else:
                try:
                    blob = TextBlob(str(text))
                    sentiments.append(blob.sentiment.polarity)
                except:
                    sentiments.append(0)
        
        return sentiments
    
    def get_key_themes(self, text_column, max_features=20):
        """Extract key themes from feedback using TF-IDF"""
        # Clean and filter text
        texts = []
        for text in text_column:
            if pd.isna(text) or str(text).strip() == '':
                continue
            # Basic text cleaning
            clean_text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
            if len(clean_text.strip()) > 3:
                texts.append(clean_text)
        
        if len(texts) < 2:
            return []
        
        try:
            # TF-IDF Analysis
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.sum(axis=0).A1
            
            # Create theme scores
            theme_scores = list(zip(feature_names, scores))
            theme_scores.sort(key=lambda x: x[1], reverse=True)
            
            return theme_scores[:15]
        except Exception as e:
            st.warning(f"Theme extraction failed: {e}")
            return []

def create_nps_gauge(nps_score):
    """Create an NPS gauge chart"""
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
    if 'sentiment_score' not in df.columns:
        return None
    
    # Categorize sentiments
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
    try:
        # Filter out empty/null values
        valid_texts = []
        for text in text_data:
            if pd.notna(text) and str(text).strip() != '' and len(str(text).strip()) > 2:
                valid_texts.append(str(text))
        
        if len(valid_texts) == 0:
            return None
        
        # Combine all text
        all_text = ' '.join(valid_texts)
        
        if len(all_text.strip()) < 10:
            return None
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis',
            collocations=False,
            relative_scaling=0.5
        ).generate(all_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        return fig
    
    except Exception as e:
        st.warning(f"Word cloud generation failed: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Advanced NPS Analysis Platform</h1>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = NPSAnalyzer()
    
    # Sidebar
    st.sidebar.title("📊 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your NPS data with 'score' and 'feedback' columns"
    )
    
    if uploaded_file is not None:
        # Load data
        if analyzer.load_data(uploaded_file):
            df = analyzer.df
            
            # Validate data
            is_valid, missing_columns = analyzer.validate_data(df)
            
            if not is_valid:
                st.error(f"❌ Missing required columns: {missing_columns}")
                st.info("📋 Your CSV should have 'score' and 'feedback' columns")
                return
            
            st.success("✅ Data loaded successfully!")
            
            # Show data preview
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head(10))
                st.write(f"**Total responses:** {len(df)}")
            
            # Calculate NPS from categories
            try:
                nps_score, promoters_count, passives_count, detractors_count, conf_lower, conf_upper, margin_error = analyzer.calculate_nps_from_categories(df['score'])
                
                # Main metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>NPS Score</h3>
                        <h2>{nps_score:.1f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Promoters</h3>
                        <h2>{promoters_count}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Passives</h3>
                        <h2>{passives_count}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Detractors</h3>
                        <h2>{detractors_count}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(create_nps_gauge(nps_score), use_container_width=True)
                
                with col2:
                    st.plotly_chart(create_distribution_chart(promoters_count, passives_count, detractors_count), use_container_width=True)
                
                # Confidence interval
                st.markdown(f"""
                <div class="insight-box">
                    <h4>📊 Statistical Confidence</h4>
                    <p>NPS Score: <strong>{nps_score:.1f}</strong></p>
                    <p>95% Confidence Interval: <strong>{conf_lower:.1f} to {conf_upper:.1f}</strong></p>
                    <p>Margin of Error: <strong>±{margin_error:.1f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Sentiment Analysis
                st.subheader("💭 Sentiment Analysis")
                
                with st.spinner("Analyzing sentiment..."):
                    sentiments = analyzer.analyze_sentiment(df['feedback'])
                    df['sentiment_score'] = sentiments
                
                sentiment_chart = create_sentiment_analysis_chart(df)
                if sentiment_chart:
                    st.plotly_chart(sentiment_chart, use_container_width=True)
                
                # Key Themes Analysis
                st.subheader("🔍 Key Themes in Feedback")
                
                with st.spinner("Extracting key themes..."):
                    themes = analyzer.get_key_themes(df['feedback'])
                
                if themes:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Top Themes:**")
                        for i, (theme, score) in enumerate(themes[:10], 1):
                            st.write(f"{i}. {theme} (Score: {score:.3f})")
                    
                    with col2:
                        # Theme visualization
                        theme_names = [theme[0] for theme in themes[:10]]
                        theme_scores = [theme[1] for theme in themes[:10]]
                        
                        fig = go.Figure(data=[
                            go.Bar(x=theme_scores, y=theme_names, orientation='h')
                        ])
                        fig.update_layout(
                            title="Top Themes by Importance",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Word Cloud
                st.subheader("☁️ Feedback Word Cloud")
                wordcloud_fig = generate_wordcloud(df['feedback'])
                if wordcloud_fig:
                    st.pyplot(wordcloud_fig)
                else:
                    st.info("Not enough text data to generate word cloud")
                
                # Detailed Feedback Analysis
                st.subheader("📝 Detailed Feedback Analysis")
                
                # Filter by category
                category_filter = st.selectbox(
                    "Filter by NPS Category:",
                    ["All", "Promoters", "Passives", "Detractors"]
                )
                
                if category_filter != "All":
                    filtered_df = df[df['score'].astype(str).str.lower().str.contains(category_filter.lower()[:-1], na=False)]
                else:
                    filtered_df = df
                
                # Show filtered feedback
                if len(filtered_df) > 0:
                    st.write(f"**Showing {len(filtered_df)} responses**")
                    
                    for idx, row in filtered_df.head(10).iterrows():
                        sentiment_emoji = "😊" if row.get('sentiment_score', 0) > 0.1 else ("😔" if row.get('sentiment_score', 0) < -0.1 else "😐")
                        
                        st.markdown(f"""
                        <div style="border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px;">
                            <strong>{row['score']}</strong> {sentiment_emoji}
                            <p>{row['feedback']}</p>
                            <small>Sentiment Score: {row.get('sentiment_score', 0):.3f}</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No responses found for the selected category.")
                
                # Download processed data
                st.subheader("💾 Download Processed Data")
                
                processed_df = df.copy()
                if 'sentiment_score' in processed_df.columns:
                    processed_df['sentiment_category'] = processed_df['sentiment_score'].apply(
                        lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral')
                    )
                
                csv = processed_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Analysis Results",
                    data=csv,
                    file_name=f"nps_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.info("Please check your data format. The 'score' column should contain: 'promoters', 'passive', or 'detractors'")
    
    else:
        # Instructions
        st.markdown("""
        ## 📋 How to Use This Platform
        
        1. **Prepare your CSV file** with these columns:
           - `score`: Contains "promoters", "passive", or "detractors"
           - `feedback`: Customer feedback text
        
        2. **Upload your file** using the sidebar
        
        3. **View comprehensive analysis** including:
           - NPS Score calculation
           - Distribution charts
           - Sentiment analysis
           - Key themes extraction
           - Word clouds
           - Detailed feedback review
        
        ### 📊 Sample Data Format:
        ```
        score,feedback
        promoters,Great service! Very satisfied
        passive,It was okay, nothing special
        detractors,Poor experience, disappointed
        ```
        
        ### 🎯 Features:
        - **Real-time NPS calculation** from categorized data
        - **Sentiment analysis** of feedback text
        - **Key themes extraction** using TF-IDF
        - **Interactive visualizations**
        - **Statistical confidence intervals**
        - **Downloadable results**
        """)

if __name__ == "__main__":
    main()
