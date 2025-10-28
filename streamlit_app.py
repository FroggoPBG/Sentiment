import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Legal Feedback Intelligence Hub",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: -1rem -1rem 2rem -1rem;
        color: white;
        border-radius: 10px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .insight-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: #333;
        margin: 1rem 0;
        border-left: 5px solid #28a745;
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Legal domain-specific aspects and keywords
LEGAL_ASPECTS = {
    'search': {
        'name': 'Case Search & Research',
        'keywords': ['search', 'case law', 'research', 'precedent', 'citation', 'find', 'query', 'boolean', 'westlaw', 'lexis'],
        'icon': '🔍'
    },
    'citation': {
        'name': 'Citation Management', 
        'keywords': ['citation', 'bluebook', 'shepard', 'cite', 'reference', 'footnote', 'bibliography'],
        'icon': '📚'
    },
    'document': {
        'name': 'Document Management',
        'keywords': ['document', 'pdf', 'file', 'upload', 'storage', 'organize', 'folder', 'brief', 'contract'],
        'icon': '📄'
    },
    'billing': {
        'name': 'Billing & Time Tracking',
        'keywords': ['billing', 'time', 'hours', 'invoice', 'payment', 'rates', 'expense', 'timesheet'],
        'icon': '💰'
    },
    'ui': {
        'name': 'User Interface',
        'keywords': ['interface', 'design', 'layout', 'navigation', 'menu', 'button', 'screen', 'usability'],
        'icon': '🖥️'
    },
    'integration': {
        'name': 'Integrations',
        'keywords': ['integration', 'api', 'connect', 'sync', 'export', 'import', 'outlook', 'office'],
        'icon': '🔗'
    },
    'compliance': {
        'name': 'Compliance & Ethics',
        'keywords': ['compliance', 'ethics', 'gdpr', 'privacy', 'security', 'confidential', 'ethical'],
        'icon': '⚖️'
    },
    'performance': {
        'name': 'System Performance',
        'keywords': ['speed', 'slow', 'fast', 'performance', 'load', 'crash', 'bug', 'error'],
        'icon': '⚡'
    }
}

class LegalSentimentAnalyzer:
    def __init__(self):
        # Legal-specific positive terms
        self.legal_positive = [
            'efficient', 'accurate', 'comprehensive', 'reliable', 'professional',
            'streamlined', 'intuitive', 'thorough', 'precise', 'excellent',
            'outstanding', 'impressed', 'helpful', 'valuable', 'essential',
            'game-changer', 'revolutionary', 'innovative', 'cutting-edge',
            'fantastic', 'amazing', 'perfect', 'love', 'great', 'wonderful'
        ]
        
        # Legal-specific negative terms
        self.legal_negative = [
            'outdated', 'inaccurate', 'slow', 'confusing', 'unreliable',
            'cumbersome', 'inadequate', 'frustrating', 'disappointing', 'useless',
            'terrible', 'awful', 'horrible', 'broken', 'failed', 'error',
            'bug', 'crash', 'unethical', 'non-compliant', 'biased', 'hate',
            'worst', 'bad', 'poor', 'difficult', 'complicated'
        ]
        
        # Intensifiers
        self.intensifiers = ['very', 'extremely', 'absolutely', 'completely', 'totally', 'really']
    
    def analyze_sentiment(self, text):
        if not text or pd.isna(text):
            return {'score': 0, 'confidence': 0, 'label': 'neutral'}
        
        text = str(text).lower()
        words = re.findall(r'\b\w+\b', text)
        
        positive_count = 0
        negative_count = 0
        confidence = 0
        intensifier_multiplier = 1
        
        for i, word in enumerate(words):
            # Check for intensifiers
            if word in self.intensifiers and i < len(words) - 1:
                intensifier_multiplier = 1.5
                continue
            
            if word in self.legal_positive:
                positive_count += intensifier_multiplier
                confidence += 0.1
            
            if word in self.legal_negative:
                negative_count += intensifier_multiplier
                confidence += 0.1
            
            intensifier_multiplier = 1
        
        # Calculate final score
        total_words = len(words)
        if total_words == 0:
            return {'score': 0, 'confidence': 0, 'label': 'neutral'}
        
        score = (positive_count - negative_count) / max(total_words, 1)
        score = max(-1, min(1, score))
        confidence = min(confidence, 1)
        
        # Determine label
        if score > 0.1:
            label = 'positive'
        elif score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'score': score,
            'confidence': confidence,
            'label': label,
            'positive_words': positive_count,
            'negative_words': negative_count
        }
    
    def extract_aspects(self, text):
        if not text or pd.isna(text):
            return []
        
        text = str(text).lower()
        detected_aspects = []
        
        for aspect_id, aspect in LEGAL_ASPECTS.items():
            matches = [keyword for keyword in aspect['keywords'] if keyword in text]
            
            if matches:
                # Get sentences containing the aspect
                sentences = re.split(r'[.!?]+', str(text))
                relevant_sentences = [s for s in sentences if any(match in s.lower() for match in matches)]
                
                if relevant_sentences:
                    aspect_sentiment = self.analyze_sentiment(' '.join(relevant_sentences))
                    detected_aspects.append({
                        'id': aspect_id,
                        'name': aspect['name'],
                        'icon': aspect['icon'],
                        'sentiment': aspect_sentiment['score'],
                        'confidence': aspect_sentiment['confidence'],
                        'matches': matches,
                        'text': relevant_sentences[0][:100] + '...' if len(relevant_sentences[0]) > 100 else relevant_sentences[0]
                    })
        
        return detected_aspects

def calculate_nps(data, score_column):
    """Calculate NPS from feedback data"""
    if data.empty or score_column not in data.columns:
        return {'score': 0, 'distribution': {'promoters': 0, 'passives': 0, 'detractors': 0}, 'total': 0}
    
    promoters = passives = detractors = 0
    
    for score in data[score_column]:
        if pd.isna(score):
            continue
            
        # Handle string scores
        if isinstance(score, str):
            score_lower = score.lower().strip()
            if 'promoter' in score_lower or any(word in score_lower for word in ['excellent', 'outstanding', 'love']):
                promoters += 1
            elif 'passive' in score_lower or any(word in score_lower for word in ['okay', 'fine', 'average']):
                passives += 1
            elif 'detractor' in score_lower or any(word in score_lower for word in ['poor', 'bad', 'terrible']):
                detractors += 1
        # Handle numeric scores
        else:
            try:
                score_num = float(score)
                if score_num >= 9:
                    promoters += 1
                elif score_num >= 7:
                    passives += 1
                elif score_num >= 0:
                    detractors += 1
            except (ValueError, TypeError):
                continue
    
    total = promoters + passives + detractors
    nps_score = ((promoters - detractors) / total * 100) if total > 0 else 0
    
    return {
        'score': round(nps_score, 1),
        'distribution': {'promoters': promoters, 'passives': passives, 'detractors': detractors},
        'total': total
    }

def generate_recommendations(analysis_data):
    """Generate AI-powered recommendations"""
    recommendations = []
    nps = analysis_data['nps']
    aspects = analysis_data['aspects']
    
    # NPS-based recommendations
    if nps['score'] < 0:
        recommendations.append({
            'title': "🚨 Critical: Address Detractor Concerns",
            'priority': "HIGH",
            'content': f"Your NPS of {nps['score']} indicates significant client dissatisfaction. Immediate action required to prevent churn.",
            'impact': "Improving to positive NPS could reduce churn by up to 40% and increase referrals.",
            'actions': [
                "Schedule immediate calls with recent detractors",
                "Implement emergency fixes for top 3 pain points",
                "Create dedicated support channel for at-risk clients"
            ]
        })
    elif nps['score'] < 30:
        recommendations.append({
            'title': "📈 Improve Client Satisfaction",
            'priority': "MEDIUM",
            'content': f"Your NPS of {nps['score']} is below industry average. Focus on converting passives to promoters.",
            'impact': "Reaching NPS 50+ could increase client lifetime value by 25%.",
            'actions': [
                "Analyze passive feedback for improvement opportunities",
                "Implement client success program",
                "Regular check-ins with passive clients"
            ]
        })
    
    # Aspect-based recommendations
    if aspects:
        negative_aspects = [(k, v) for k, v in aspects.items() if v['average_sentiment'] < -0.2 and v['count'] > 0]
        negative_aspects.sort(key=lambda x: x[1]['average_sentiment'])
        
        if negative_aspects:
            aspect_id, aspect_data = negative_aspects[0]
            aspect = LEGAL_ASPECTS[aspect_id]
            
            recommendations.append({
                'title': f"🔧 Fix {aspect['name']} Issues",
                'priority': "HIGH" if aspect_data['average_sentiment'] < -0.5 else "MEDIUM",
                'content': f"{aspect['name']} has the most negative sentiment ({aspect_data['average_sentiment']*100:.1f}%). This is your biggest opportunity for improvement.",
                'impact': f"Fixing {aspect['name']} issues could improve overall NPS by 5-10 points.",
                'actions': [
                    f"Conduct user research on {aspect['name'].lower()}",
                    f"Prioritize {aspect['name'].lower()} in next sprint",
                    f"Create targeted communication about {aspect['name'].lower()} improvements"
                ]
            })
    
    # Add general recommendations if none found
    if not recommendations:
        recommendations.append({
            'title': "✅ Maintain Excellence",
            'priority': "LOW",
            'content': "Your feedback analysis shows generally positive sentiment. Focus on maintaining current quality.",
            'impact': "Continued excellence can lead to increased referrals and client retention.",
            'actions': [
                "Monitor feedback trends for early warning signs",
                "Continue current best practices",
                "Seek opportunities for incremental improvements"
            ]
        })
    
    return recommendations

def create_sample_data():
    """Create sample legal feedback data for demonstration"""
    sample_feedback = [
        {"feedback": "The case search functionality is excellent and very intuitive. Love the Boolean search capabilities.", "score": 9, "client_type": "small"},
        {"feedback": "Citation management is terrible. The Bluebook formatting is completely wrong and unreliable.", "score": 3, "client_type": "large"},
        {"feedback": "Document upload is slow and crashes frequently. Very frustrating for our daily workflow.", "score": 4, "client_type": "medium"},
        {"feedback": "The user interface is clean and professional. Easy to navigate and find what we need.", "score": 8, "client_type": "solo"},
        {"feedback": "Billing integration with our time tracking is a game-changer. Saves hours every week.", "score": 10, "client_type": "large"},
        {"feedback": "Search results are often outdated and inaccurate. Missing recent case law updates.", "score": 2, "client_type": "medium"},
        {"feedback": "Overall good product but the performance is slow during peak hours.", "score": 7, "client_type": "small"},
        {"feedback": "Compliance features are comprehensive and help with GDPR requirements. Very valuable.", "score": 9, "client_type": "large"},
        {"feedback": "The API integration is broken and doesn't sync properly with Outlook.", "score": 3, "client_type": "medium"},
        {"feedback": "Great tool for legal research. The AI-powered summaries are particularly helpful.", "score": 9, "client_type": "solo"},
        {"feedback": "Document management is confusing and not intuitive. Need better organization.", "score": 5, "client_type": "small"},
        {"feedback": "Citation tracking is fantastic. Automatically updates and very accurate.", "score": 8, "client_type": "large"},
        {"feedback": "System crashes during important presentations. Completely unreliable.", "score": 1, "client_type": "medium"},
        {"feedback": "User interface could be more modern but functionality is solid.", "score": 7, "client_type": "solo"},
        {"feedback": "The search feature is revolutionary for our practice. Saves tremendous time.", "score": 10, "client_type": "large"}
    ]
    
    return pd.DataFrame(sample_feedback)

# Main Streamlit App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ Legal Feedback Intelligence Hub</h1>
        <p>AI-Powered Client Sentiment Analysis & Predictive Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    
    # File upload or sample data
    data_option = st.sidebar.radio(
        "Choose Data Source:",
        ["Upload CSV File", "Use Sample Data"]
    )
    
    df = None
    
    if data_option == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload your feedback CSV",
            type=['csv'],
            help="CSV should contain columns: 'feedback' (text) and 'score' (number or text)"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success(f"✅ Loaded {len(df)} feedback entries")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {str(e)}")
    else:
        df = create_sample_data()
        st.sidebar.info("📝 Using sample legal feedback data")
    
    if df is not None and not df.empty:
        # Ensure required columns exist
        feedback_col = None
        score_col = None
        
        # Try to find feedback column
        for col in df.columns:
            if any(term in col.lower() for term in ['feedback', 'comment', 'text', 'review']):
                feedback_col = col
                break
        
        # Try to find score column
        for col in df.columns:
            if any(term in col.lower() for term in ['score', 'rating', 'nps']):
                score_col = col
                break
        
        if feedback_col is None:
            feedback_col = st.sidebar.selectbox("Select feedback column:", df.columns)
        
        if score_col is None:
            score_col = st.sidebar.selectbox("Select score column:", df.columns)
        
        # Filters
        st.sidebar.subheader("🔍 Filters")
        
        client_filter = "All"
        if 'client_type' in df.columns:
            client_types = ['All'] + list(df['client_type'].unique())
            client_filter = st.sidebar.selectbox("Client Type:", client_types)
        
        # Process data
        analyzer = LegalSentimentAnalyzer()
        
        # Filter data
        filtered_df = df.copy()
        if client_filter != "All" and 'client_type' in df.columns:
            filtered_df = filtered_df[filtered_df['client_type'] == client_filter]
        
        # Analyze sentiment and aspects
        with st.spinner("🤖 Analyzing feedback with AI..."):
            sentiment_results = []
            aspect_results = []
            
            for _, row in filtered_df.iterrows():
                sentiment = analyzer.analyze_sentiment(row[feedback_col])
                aspects = analyzer.extract_aspects(row[feedback_col])
                
                sentiment_results.append(sentiment)
                aspect_results.append(aspects)
            
            filtered_df['sentiment_score'] = [r['score'] for r in sentiment_results]
            filtered_df['sentiment_label'] = [r['label'] for r in sentiment_results]
            filtered_df['sentiment_confidence'] = [r['confidence'] for r in sentiment_results]
            filtered_df['aspects'] = aspect_results
        
        # Calculate metrics
        nps_result = calculate_nps(filtered_df, score_col)
        avg_sentiment = filtered_df['sentiment_score'].mean()
        
        # Aggregate aspect data
        aspect_data = {}
        for aspect_id in LEGAL_ASPECTS.keys():
            aspect_data[aspect_id] = {
                'count': 0,
                'total_sentiment': 0,
                'average_sentiment': 0,
                'feedback_items': []
            }
        
        for _, row in filtered_df.iterrows():
            for aspect in row['aspects']:
                if aspect['id'] in aspect_data:
                    aspect_data[aspect['id']]['count'] += 1
                    aspect_data[aspect['id']]['total_sentiment'] += aspect['sentiment']
                    aspect_data[aspect['id']]['feedback_items'].append(row)
        
        # Calculate averages
        for aspect_id in aspect_data:
            data = aspect_data[aspect_id]
            data['average_sentiment'] = data['total_sentiment'] / data['count'] if data['count'] > 0 else 0
        
        # Create analysis object
        analysis_data = {
            'nps': nps_result,
            'aspects': aspect_data,
            'processed_feedback': filtered_df
        }
        
        # Display Dashboard
        st.header("📊 Executive Dashboard")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{nps_result['score']}</h2>
                <p>Net Promoter Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{avg_sentiment*100:.1f}%</h2>
                <p>Avg Sentiment Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_feedback = len(filtered_df)
            st.markdown(f"""
            <div class="metric-card">
                <h2>{total_feedback}</h2>
                <p>Total Feedback</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            urgent_count = len(filtered_df[filtered_df['sentiment_score'] < -0.5])
            st.markdown(f"""
            <div class="metric-card">
                <h2>{urgent_count}</h2>
                <p>Urgent Issues</p>
            </div>
            """, unsafe_allow_html=True)
        
        # NPS Gauge Chart
        st.subheader("🎯 NPS Performance")
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = nps_result['score'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Net Promoter Score"},
            delta = {'reference': 0},
            gauge = {
                'axis': {'range': [-100, 100]},
                'bar': {'color': "darkgreen" if nps_result['score'] > 0 else "darkred"},
                'steps': [
                    {'range': [-100, 0], 'color': "lightcoral"},
                    {'range': [0, 50], 'color': "lightyellow"},
                    {'range': [50, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig_gauge.update_layout(height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Tabs for detailed analysis
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Aspect Analysis", "📈 Sentiment Trends", "💡 AI Recommendations", "📝 Feedback Details"])
        
        with tab1:
            st.subheader("Legal Aspect-Based Sentiment Analysis")
            
            # Aspect sentiment chart
            aspects_with_data = [(k, v) for k, v in aspect_data.items() if v['count'] > 0]
            
            if aspects_with_data:
                aspect_names = [LEGAL_ASPECTS[k]['name'] for k, _ in aspects_with_data]
                aspect_sentiments = [v['average_sentiment'] * 100 for _, v in aspects_with_data]
                aspect_counts = [v['count'] for _, v in aspects_with_data]
                
                fig_aspects = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Bar chart for sentiment
                fig_aspects.add_trace(
                    go.Bar(
                        x=aspect_names,
                        y=aspect_sentiments,
                        name="Sentiment Score (%)",
                        marker_color=['green' if s > 0 else 'red' if s < -20 else 'orange' for s in aspect_sentiments]
                    ),
                    secondary_y=False
                )
                
                # Line chart for volume
                fig_aspects.add_trace(
                    go.Scatter(
                        x=aspect_names,
                        y=aspect_counts,
                        mode='lines+markers',
                        name="Mention Count",
                        line=dict(color='blue', width=3),
                        marker=dict(size=8)
                    ),
                    secondary_y=True
                )
                
                fig_aspects.update_xaxes(title_text="Legal Aspects")
                fig_aspects.update_yaxes(title_text="Sentiment Score (%)", secondary_y=False)
                fig_aspects.update_yaxes(title_text="Mention Count", secondary_y=True)
                fig_aspects.update_layout(title="Aspect Sentiment vs Volume Analysis", height=500)
                
                st.plotly_chart(fig_aspects, use_container_width=True)
                
                # Aspect details
                st.subheader("📋 Aspect Details")
                for aspect_id, aspect_data_item in aspects_with_data:
                    aspect = LEGAL_ASPECTS[aspect_id]
                    sentiment_color = "🟢" if aspect_data_item['average_sentiment'] > 0.1 else "🔴" if aspect_data_item['average_sentiment'] < -0.1 else "🟡"
                    
                    with st.expander(f"{sentiment_color} {aspect['icon']} {aspect['name']} - {aspect_data_item['average_sentiment']*100:.1f}% sentiment"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Mentions", aspect_data_item['count'])
                        with col2:
                            st.metric("Avg Sentiment", f"{aspect_data_item['average_sentiment']*100:.1f}%")
                        
                        # Show sample feedback for this aspect
                        if aspect_data_item['feedback_items']:
                            st.write("**Sample Feedback:**")
                            sample_feedback = aspect_data_item['feedback_items'][:3]
                            for item in sample_feedback:
                                sentiment_emoji = "😊" if item['sentiment_score'] > 0.1 else "😞" if item['sentiment_score'] < -0.1 else "😐"
                                st.write(f"{sentiment_emoji} *{item[feedback_col][:200]}...*")
        
        with tab2:
            st.subheader("📈 Sentiment Distribution")
            
            # Sentiment distribution
            sentiment_counts = filtered_df['sentiment_label'].value_counts()
            
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                color_discrete_map={
                    'positive': '#28a745',
                    'negative': '#dc3545',
                    'neutral': '#ffc107'
                }
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Sentiment vs Score correlation
            if score_col in filtered_df.columns:
                fig_scatter = px.scatter(
                    filtered_df,
                    x='sentiment_score',
                    y=score_col,
                    color='sentiment_label',
                    title="Sentiment Score vs NPS Rating",
                    labels={'sentiment_score': 'AI Sentiment Score', score_col: 'NPS Score'},
                    color_discrete_map={
                        'positive': '#28a745',
                        'negative': '#dc3545',
                        'neutral': '#ffc107'
                    }
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        with tab3:
            st.subheader("💡 AI-Generated Recommendations")
            
            recommendations = generate_recommendations(analysis_data)
            
            for i, rec in enumerate(recommendations):
                priority_color = "#dc3545" if rec['priority'] == "HIGH" else "#ffc107" if rec['priority'] == "MEDIUM" else "#28a745"
                
                st.markdown(f"""
                <div class="recommendation-card">
                    <h3>{rec['title']} <span style="background: {priority_color}; color: white; padding: 3px 8px; border-radius: 10px; font-size: 0.8em;">{rec['priority']} PRIORITY</span></h3>
                    <p><strong>Analysis:</strong> {rec['content']}</p>
                    <p><strong>Expected Impact:</strong> {rec['impact']}</p>
                    <p><strong>Recommended Actions:</strong></p>
                    <ul>
                        {''.join([f'<li>{action}</li>' for action in rec['actions']])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        with tab4:
            st.subheader("📝 Detailed Feedback Analysis")
            
            # Filters for feedback
            col1, col2 = st.columns(2)
            with col1:
                sentiment_filter = st.selectbox(
                    "Filter by Sentiment:",
                    ["All", "Positive", "Negative", "Neutral"]
                )
            
            with col2:
                sort_by = st.selectbox(
                    "Sort by:",
                    ["Sentiment Score", "Confidence", "Original Order"]
                )
            
            # Apply filters
            display_df = filtered_df.copy()
            
            if sentiment_filter != "All":
                display_df = display_df[display_df['sentiment_label'] == sentiment_filter.lower()]
            
            # Sort
            if sort_by == "Sentiment Score":
                display_df = display_df.sort_values('sentiment_score')
            elif sort_by == "Confidence":
                display_df = display_df.sort_values('sentiment_confidence', ascending=False)
            
            # Display feedback items
            for _, row in display_df.head(20).iterrows():
                sentiment_emoji = "😊" if row['sentiment_score'] > 0.1 else "😞" if row['sentiment_score'] < -0.1 else "😐"
                confidence_stars = "⭐" * min(5, int(row['sentiment_confidence'] * 5))
                
                with st.expander(f"{sentiment_emoji} Score: {row[score_col]} | Sentiment: {row['sentiment_score']*100:.1f}% | Confidence: {confidence_stars}"):
                    st.write(f"**Feedback:** {row[feedback_col]}")
                    
                    if row['aspects']:
                        st.write("**Detected Aspects:**")
                        for aspect in row['aspects']:
                            aspect_sentiment_emoji = "😊" if aspect['sentiment'] > 0.1 else "😞" if aspect['sentiment'] < -0.1 else "😐"
                            st.write(f"- {aspect['icon']} {aspect['name']}: {aspect_sentiment_emoji} {aspect['sentiment']*100:.1f}%")
                    
                    if 'client_type' in row:
                        st.write(f"**Client Type:** {row['client_type']}")
    
    else:
        st.info("👆 Please upload a CSV file or select 'Use Sample Data' to begin analysis.")
        
        st.markdown("""
        ### 📋 CSV Format Requirements
        
        Your CSV file should contain at least these columns:
        - **feedback/comment/text**: The actual feedback text
        - **score/rating/nps**: Numeric score (0-10) or text (promoter/passive/detractor)
        
        Optional columns:
        - **client_type**: Type of client (solo, small, medium, large)
        - **date**: Date of feedback
        
        ### 🎯 Features
        
        This Legal Feedback Intelligence Hub provides:
        
        - **🔍 Aspect-Based Analysis**: Identifies sentiment for specific legal product areas
        - **⚖️ Legal Domain Intelligence**: Understands legal terminology and context  
        - **📊 Predictive Insights**: NPS calculation and trend analysis
        - **💡 AI Recommendations**: Actionable suggestions based on data patterns
        - **📈 Interactive Visualizations**: Charts and graphs for easy interpretation
        - **🎯 Client Segmentation**: Analysis by firm size and client type
        """)

if __name__ == "__main__":
    main()
