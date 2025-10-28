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
from transformers import pipeline  # For advanced sentiment
import spacy  # For theme/entity extraction
from sklearn.linear_model import LinearRegression  # For predictions
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas  # For PDF exports
import pytesseract  # For OCR on images (multimodal)
from PIL import Image  # For image processing

# Page configuration
st.set_page_config(
    page_title="NPS Analysis Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (enhanced for better visuals)
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
    .fallback-message {
        color: red;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load advanced models (cached for performance)
@st.cache_resource
def load_models():
    sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    nlp = spacy.load("en_core_web_sm")  # For entities/themes
    reg_model = LinearRegression()  # For predictions
    # Train simple model (mock data; replace with real)
    X = np.array([[1, 2, 3], [4, 5, 6]])  # Features: negatives, neutrals, positives
    y = np.array([10, -5])  # NPS changes
    reg_model.fit(X, y)
    return sentiment_classifier, nlp, reg_model

sentiment_classifier, nlp, reg_model = load_models()

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
        """Analyze sentiment using advanced model"""
        sentiments = []
        
        for text in text_column:
            if pd.isna(text) or str(text).strip() == '':
                sentiments.append(0)
            else:
                try:
                    result = sentiment_classifier(text)[0]
                    polarity = result['score'] if result['label'] == 'POSITIVE' else -result['score']
                    sentiments.append(polarity)
                except:
                    sentiments.append(0)
        
        return sentiments
    
    def get_key_themes(self, text_column, max_features=20, custom_themes=""):
        """Extract key themes using TF-IDF and entities"""
        texts = [str(text).lower() for text in text_column if pd.notna(text) and len(str(text).strip()) > 3]
        
        if len(texts) < 2:
            return {'tfidf_themes': [], 'entities': []}
        
        # TF-IDF
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2), min_df=2)
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.sum(axis=0).A1
        theme_scores = sorted(list(zip(feature_names, scores)), key=lambda x: x[1], reverse=True)[:15]
        
        # Entity extraction
        entities = []
        for text in texts:
            doc = nlp(text)
            entities.extend([ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PRODUCT', 'GPE']])
        
        entity_counts = Counter(entities).most_common(10)
        
        # Custom themes if provided
        if custom_themes:
            custom_list = [t.strip() for t in custom_themes.split(',')]
            custom_scores = [(t, sum(1 for txt in texts if t in txt)) for t in custom_list]
            theme_scores.extend(custom_scores)
        
        return {'tfidf_themes': theme_scores, 'entities': entity_counts}
    
    def predict_nps_change(self, negatives, neutrals, positives):
        """Simple prediction of NPS change"""
        input_data = np.array([[negatives, neutrals, positives]])
        predicted_change = reg_model.predict(input_data)[0]
        return predicted_change
    
    def generate_followup(self, sentiment_score, themes):
        """Generate follow-up template"""
        top_theme = themes['tfidf_themes'][0][0] if themes['tfidf_themes'] else "general"
        if sentiment_score < -0.1:
            return f"Dear [Client], We're sorry to hear about {top_theme}. We'd like to schedule a call to address this."
        elif sentiment_score > 0.1:
            return f"Dear [Client], Thank you for your positive feedback on {top_theme}. How can we support you further?"
        else:
            return f"Dear [Client], Thanks for your input. We're working on improvements to {top_theme}."

def extract_text_from_image(image):
    """OCR for multimodal image inputs"""
    try:
        return pytesseract.image_to_string(Image.open(image))
    except Exception as e:
        st.error(f"OCR error: {str(e)}")
        return ""

def generate_pdf_report(df, nps_score, themes):
    """Generate PDF report"""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, f"NPS Report - {datetime.now()}")
    c.drawString(100, 700, f"NPS Score: {nps_score}")
    y = 650
    for theme, score in themes['tfidf_themes'][:5]:
        c.drawString(100, y, f"{theme}: {score:.2f}")
        y -= 20
    c.save()
    buffer.seek(0)
    return buffer

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
    valid_texts = [str(text) for text in text_data if pd.notna(text) and len(str(text).strip()) > 2]
    if not valid_texts:
        return None
    
    all_text = ' '.join(valid_texts)
    if len(all_text.strip()) < 10:
        return None
    
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100, colormap='viridis').generate(all_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def main():
    # Fallback for no JS (new)
    st.markdown('<p class="fallback-message">Note: JavaScript is required for full interactivity. If disabled, basic results will show below.</p>', unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Advanced NPS Analysis Platform</h1>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = NPSAnalyzer()
    
    # Sidebar (enhanced)
    st.sidebar.title("📊 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your NPS data with 'score' and 'feedback' columns"
    )
    uploaded_image = st.sidebar.file_uploader("Upload Image for OCR (Optional)", type=["png", "jpg", "jpeg"])
    anonymize = st.sidebar.checkbox("Anonymize Data (Redact PII)", value=False)
    custom_themes = st.sidebar.text_input("Custom Themes (comma-separated)", "")
    
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
            
            # New: Process image if uploaded (multimodal)
            if uploaded_image:
                image_text = extract_text_from_image(uploaded_image)
                if image_text.strip():
                    new_row = pd.DataFrame({'score': ['unknown'], 'feedback': [image_text]})
                    df = pd.concat([df, new_row], ignore_index=True)
                    st.success("✅ Image text extracted and added to dataset!")
            
            # New: Anonymize if selected
            if anonymize:
                df['feedback'] = df['feedback'].apply(lambda x: re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', str(x)))
                st.success("✅ Data anonymized!")
            
            st.success("✅ Data loaded successfully!")
            
            # Show data preview
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head(10))
                st.write(f"**Total responses:** {len(df)}")
            
            # Calculate NPS
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
            # [Similar for other metrics - omitted for brevity]
            
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
            
            # Sentiment Analysis (advanced)
            st.subheader("💭 Sentiment Analysis")
            with st.spinner("Analyzing sentiment..."):
                sentiments = analyzer.analyze_sentiment(df['feedback'])
                df['sentiment_score'] = sentiments
            
            sentiment_chart = create_sentiment_analysis_chart(df)
            if sentiment_chart:
                st.plotly_chart(sentiment_chart, use_container_width=True)
            
            # Key Themes Analysis (enhanced)
            st.subheader("🔍 Key Themes in Feedback")
            with st.spinner("Extracting key themes..."):
                themes = analyzer.get_key_themes(df['feedback'], custom_themes=custom_themes)
            
            if themes['tfidf_themes']:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Top TF-IDF Themes:**")
                    for i, (theme, score) in enumerate(themes['tfidf_themes'][:10], 1):
                        st.write(f"{i}. {theme} (Score: {score:.3f})")
                with col2:
                    st.write("**Extracted Entities:**")
                    for entity, count in themes['entities']:
                        st.write(f"{entity}: {count}")
                    # Theme visualization
                    theme_names = [t[0] for t in themes['tfidf_themes'][:10]]
                    theme_scores = [t[1] for t in themes['tfidf_themes'][:10]]
                    fig = go.Figure(data=[go.Bar(x=theme_scores, y=theme_names, orientation='h')])
                    fig.update_layout(title="Top Themes by Importance", height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Word Cloud
            st.subheader("☁️ Feedback Word Cloud")
            wordcloud_fig = generate_wordcloud(df['feedback'])
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)
            else:
                st.info("Not enough text data to generate word cloud")
            
            # New: Predictive What-If
            st.subheader("📈 Predictive What-If Simulation")
            negatives = (df['sentiment_score'] < -0.1).sum()
            neutrals = ((df['sentiment_score'] >= -0.1) & (df['sentiment_score'] <= 0.1)).sum()
            positives = (df['sentiment_score'] > 0.1).sum()
            predicted_change = analyzer.predict_nps_change(negatives, neutrals, positives)
            st.write(f"Current Predicted NPS Change: {predicted_change:.1f}")
            fix_negatives = st.slider("Simulate fixing negatives", 0, negatives, 0)
            what_if_change = analyzer.predict_nps_change(negatives - fix_negatives, neutrals, positives + fix_negatives)
            st.write(f"What-If Change: {what_if_change:.1f}")
            
            # Detailed Feedback Analysis
            st.subheader("📝 Detailed Feedback Analysis")
            category_filter = st.selectbox("Filter by NPS Category:", ["All", "Promoters", "Passives", "Detractors"])
            if category_filter != "All":
                filtered_df = df[df['score'].str.lower().str.contains(category_filter.lower()[:-1], na=False)]
            else:
                filtered_df = df
            if len(filtered_df) > 0:
                for _, row in filtered_df.head(10).iterrows():
                    sentiment_emoji = "😊" if row.get('sentiment_score', 0) > 0.1 else ("😔" if row.get('sentiment_score', 0) < -0.1 else "😐")
                    st.markdown(f"""
                    <div style="border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px;">
                        <strong>{row['score']}</strong> {sentiment_emoji}
                        <p>{row['feedback']}</p>
                        <small>Sentiment Score: {row.get('sentiment_score', 0):.3f}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # New: Auto-Generated Follow-Up
            st.subheader("✉️ Auto-Generated Follow-Up Template")
            template = analyzer.generate_followup(df['sentiment_score'].mean(), themes)
            st.text_area("Template", template, height=100)
            
            # Downloads (enhanced with PDF)
            st.subheader("💾 Download Processed Data")
            csv = df.to_csv(index=False)
            st.download_button(label="📥 Download CSV", data=csv, file_name=f"nps_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
            pdf_buffer = generate_pdf_report(df, nps_score, themes)
            st.download_button(label="📥 Download PDF Report", data=pdf_buffer, file_name="nps_report.pdf", mime="application/pdf")
    
    else:
        # Instructions (enhanced with tips)
        st.markdown("""
        ## 📋 How to Use This Platform
        
        1. **Prepare your CSV file** with these columns:
           - `score`: Contains "promoters", "passive", or "detractors"
           - `feedback`: Customer feedback text
        
        2. **Upload your file** using the sidebar. Optionally upload images for OCR.
        
        3. **View comprehensive analysis** including advanced sentiment, predictions, and more.
        """)

if __name__ == "__main__":
    main()
