import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime, timedelta
import warnings
import json
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Legal Feedback Intelligence Hub",
    page_icon="⚖️",
    layout="wide"
)

class LegalSentimentAnalyzer:
    """Advanced sentiment analyzer specifically designed for legal feedback analysis"""
    
    def __init__(self):
        # Legal-specific positive terms
        self.positive_terms = {
            'excellent', 'outstanding', 'exceptional', 'superior', 'impressive',
            'efficient', 'streamlined', 'intuitive', 'user-friendly', 'comprehensive',
            'accurate', 'precise', 'reliable', 'responsive', 'helpful', 'professional',
            'satisfied', 'pleased', 'happy', 'love', 'amazing', 'fantastic',
            'recommend', 'valuable', 'useful', 'effective', 'seamless', 'smooth',
            'fast', 'quick', 'timely', 'organized', 'clear', 'detailed'
        }
        
        # Legal-specific negative terms
        self.negative_terms = {
            'terrible', 'awful', 'horrible', 'disappointing', 'frustrating',
            'confusing', 'complicated', 'difficult', 'slow', 'buggy', 'broken',
            'unreliable', 'inaccurate', 'outdated', 'missing', 'incomplete',
            'useless', 'waste', 'annoying', 'clunky', 'cumbersome', 'tedious',
            'hate', 'dislike', 'poor', 'bad', 'worst', 'lacking', 'inadequate',
            'unresponsive', 'crashes', 'errors', 'issues', 'problems'
        }
        
        # Intensifiers that modify sentiment
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'incredibly': 1.8, 'absolutely': 1.7,
            'really': 1.3, 'quite': 1.2, 'somewhat': 0.8, 'slightly': 0.7,
            'barely': 0.5, 'hardly': 0.4, 'completely': 1.9, 'totally': 1.8
        }
        
        # Legal aspect categories with keywords
        self.legal_aspects = {
            'case_search': {
                'name': 'Case Search & Research',
                'icon': '🔍',
                'keywords': ['case search', 'research', 'find cases', 'legal research', 
                           'search results', 'search function', 'database search', 'case law',
                           'precedent', 'citation', 'westlaw', 'lexis']
            },
            'document_management': {
                'name': 'Document Management',
                'icon': '📄',
                'keywords': ['document', 'filing', 'upload', 'download', 'pdf', 
                           'brief', 'pleading', 'contract', 'agreement', 'template',
                           'document review', 'version control']
            },
            'billing': {
                'name': 'Billing & Time Tracking',
                'icon': '💰',
                'keywords': ['billing', 'invoice', 'time tracking', 'hours', 'rates',
                           'expense', 'cost', 'payment', 'fee', 'pricing', 'timesheet']
            },
            'client_portal': {
                'name': 'Client Portal',
                'icon': '👥',
                'keywords': ['client portal', 'client access', 'communication', 
                           'client login', 'sharing', 'collaboration', 'messages']
            },
            'calendar': {
                'name': 'Calendar & Scheduling',
                'icon': '📅',
                'keywords': ['calendar', 'schedule', 'appointment', 'deadline', 
                           'court date', 'meeting', 'reminder', 'docket']
            },
            'compliance': {
                'name': 'Compliance & Ethics',
                'icon': '⚖️',
                'keywords': ['compliance', 'ethics', 'rules', 'bar requirements',
                           'professional responsibility', 'conflict check', 'audit trail']
            },
            'reporting': {
                'name': 'Reporting & Analytics',
                'icon': '📊',
                'keywords': ['report', 'analytics', 'dashboard', 'metrics', 'statistics',
                           'performance', 'insights', 'data', 'trends']
            },
            'performance': {
                'name': 'System Performance',
                'icon': '⚡',
                'keywords': ['speed', 'performance', 'slow', 'fast', 'loading', 'lag',
                           'responsive', 'crash', 'freeze', 'timeout', 'error', 'bug']
            },
            'training': {
                'name': 'Training & Support',
                'icon': '🎓',
                'keywords': ['training', 'support', 'help', 'tutorial', 'documentation',
                           'onboarding', 'learning', 'guidance', 'assistance']
            },
            'integration': {
                'name': 'Integration & Compatibility',
                'icon': '🔗',
                'keywords': ['integration', 'compatibility', 'sync', 'import', 'export',
                           'api', 'third party', 'outlook', 'office', 'accounting software']
            }
        }
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of legal feedback text"""
        if not text or not text.strip():
            return {'score': 0, 'label': 'neutral', 'confidence': 0}
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_score = 0
        negative_score = 0
        total_words = len(words)
        
        for i, word in enumerate(words):
            # Check for intensifiers before sentiment words
            intensifier = 1.0
            if i > 0 and words[i-1] in self.intensifiers:
                intensifier = self.intensifiers[words[i-1]]
            
            # Calculate sentiment
            if word in self.positive_terms:
                positive_score += intensifier
            elif word in self.negative_terms:
                negative_score += intensifier
        
        # Normalize scores
        if total_words > 0:
            net_score = (positive_score - negative_score) / total_words
        else:
            net_score = 0
        
        # Determine label and confidence
        if net_score > 0.02:
            label = 'positive'
            confidence = min(abs(net_score) * 10, 1.0)
        elif net_score < -0.02:
            label = 'negative'
            confidence = min(abs(net_score) * 10, 1.0)
        else:
            label = 'neutral'
            confidence = 1.0 - min(abs(net_score) * 5, 0.5)
        
        return {
            'score': net_score,
            'label': label,
            'confidence': confidence
        }
    
    def extract_aspects(self, text):
        """Extract legal aspects mentioned in the text"""
        text_lower = text.lower()
        found_aspects = []
        
        for aspect_key, aspect_info in self.legal_aspects.items():
            for keyword in aspect_info['keywords']:
                if keyword in text_lower:
                    # Find the sentence containing this keyword for context
                    sentences = re.split(r'[.!?]+', text)
                    context_sentence = ""
                    
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            context_sentence = sentence.strip()
                            break
                    
                    # Analyze sentiment of the context sentence
                    sentence_sentiment = self.analyze_sentiment(context_sentence)
                    
                    found_aspects.append({
                        'name': aspect_info['name'],
                        'icon': aspect_info['icon'],
                        'keyword': keyword,
                        'text': context_sentence[:100] + "..." if len(context_sentence) > 100 else context_sentence,
                        'sentiment': sentence_sentiment['score']
                    })
                    break  # Only count each aspect once
        
        return found_aspects

def generate_sample_data():
    """Generate realistic legal software feedback data"""
    np.random.seed(42)  # For reproducible results
    
    # Sample responses that reflect real legal software feedback
    sample_responses = [
        "The case search functionality is excellent and saves us hours of research time daily.",
        "Document management system is intuitive, but billing integration needs improvement.",
        "Loving the new client portal! Communication with clients has never been easier.",
        "Calendar scheduling works well, though deadline reminders could be more prominent.",
        "System performance is generally good, but search can be slow during peak hours.",
        "Training materials are comprehensive and onboarding was smooth for our team.",
        "Integration with Outlook works perfectly, but accounting software sync has issues.",
        "Compliance tracking is thorough and helps us stay on top of bar requirements.",
        "The research database is comprehensive but the interface feels outdated.",
        "Time tracking is accurate and billing reports are detailed and professional.",
        "Client communication features are great, but document sharing could be more streamlined.",
        "Love the analytics dashboard - gives great insights into our practice metrics.",
        "System occasionally crashes during large document uploads, quite frustrating.",
        "Customer support is responsive and knowledgeable about legal industry needs.",
        "The mobile app works well for basic functions but lacks advanced features.",
        "Conflict checking is thorough and integration with client intake is seamless.",
        "Document templates save significant time, though more customization options needed.",
        "Reporting functionality is powerful but could benefit from more visual charts.",
        "Security features are robust and give us confidence in client data protection.",
        "User interface is clean and modern, much better than our previous software."
    ]
    
    # Generate data
    dates = pd.date_range(start='2024-01-01', end='2024-10-28', freq='D')
    data = []
    
    for i in range(200):  # Generate 200 feedback entries
        date = np.random.choice(dates)
        nps_score = np.random.randint(0, 11)
        
        # Select response based on NPS score tendency
        if nps_score >= 9:
            response = np.random.choice([r for r in sample_responses if any(word in r.lower() for word in ['excellent', 'love', 'great', 'perfect'])])
        elif nps_score <= 6:
            response = np.random.choice([r for r in sample_responses if any(word in r.lower() for word in ['issues', 'slow', 'needs', 'crashes', 'frustrating'])])
        else:
            response = np.random.choice(sample_responses)
        
        client_type = np.random.choice(['Small Firm', 'Mid-size Firm', 'Large Firm', 'Solo Practice'], 
                                     p=[0.3, 0.3, 0.2, 0.2])
        
        practice_area = np.random.choice(['Corporate Law', 'Litigation', 'Family Law', 'Criminal Law', 
                                        'Real Estate', 'Personal Injury', 'Employment Law'])
        
        data.append({
            'Date': date,
            'NPS_Score': nps_score,
            'Feedback_Text': response,
            'Client_Type': client_type,
            'Practice_Area': practice_area,
            'Respondent_ID': f"Client_{i+1:03d}"
        })
    
    return pd.DataFrame(data)

def calculate_nps_metrics(df):
    """Calculate NPS score and categorize respondents"""
    promoters = len(df[df['NPS_Score'] >= 9])
    detractors = len(df[df['NPS_Score'] <= 6])
    passives = len(df[(df['NPS_Score'] >= 7) & (df['NPS_Score'] <= 8)])
    total = len(df)
    
    nps_score = ((promoters - detractors) / total) * 100 if total > 0 else 0
    
    return {
        'nps_score': nps_score,
        'promoters': promoters,
        'detractors': detractors,
        'passives': passives,
        'total_responses': total,
        'promoter_rate': (promoters / total) * 100 if total > 0 else 0,
        'detractor_rate': (detractors / total) * 100 if total > 0 else 0
    }

def create_nps_gauge(nps_score):
    """Create NPS gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = nps_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Net Promoter Score"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [-100, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-100, 0], 'color': "lightgray"},
                {'range': [0, 50], 'color': "yellow"},
                {'range': [50, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def analyze_feedback_aspects(df, analyzer):
    """Analyze feedback text for legal aspects and sentiment"""
    aspect_data = []
    
    for _, row in df.iterrows():
        aspects = analyzer.extract_aspects(row['Feedback_Text'])
        sentiment = analyzer.analyze_sentiment(row['Feedback_Text'])
        
        for aspect in aspects:
            aspect_data.append({
                'Date': row['Date'],
                'Aspect': aspect['name'],
                'Sentiment_Score': aspect['sentiment'],
                'Sentiment_Label': 'Positive' if aspect['sentiment'] > 0.1 else 'Negative' if aspect['sentiment'] < -0.1 else 'Neutral',
                'Client_Type': row['Client_Type'],
                'Practice_Area': row['Practice_Area'],
                'NPS_Score': row['NPS_Score'],
                'Context': aspect['text']
            })
    
    return pd.DataFrame(aspect_data)

def main():
    # Header
    st.title("⚖️ Legal Feedback Intelligence Hub")
    st.markdown("**Advanced NPS Analysis & Sentiment Intelligence for Legal Software**")
    
    # Initialize analyzer
    analyzer = LegalSentimentAnalyzer()
    
    # Sidebar for data upload
    st.sidebar.header("📊 Data Management")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload your feedback data (CSV)", 
        type=['csv'],
        help="Upload a CSV file with columns: Date, NPS_Score, Feedback_Text, Client_Type, Practice_Area"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df['Date'] = pd.to_datetime(df['Date'])
            st.sidebar.success("✅ Data uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
            df = generate_sample_data()
            st.sidebar.info("Using sample data instead.")
    else:
        df = generate_sample_data()
        st.sidebar.info("📋 Using sample data. Upload your own CSV to analyze real feedback.")
        
        if st.sidebar.button("📥 Download Sample Data Template"):
            sample_csv = df.head(10).to_csv(index=False)
            st.sidebar.download_button(
                label="Download CSV Template",
                data=sample_csv,
                file_name="legal_feedback_template.csv",
                mime="text/csv"
            )
    
    # Date filtering
    st.sidebar.subheader("📅 Date Range Filter")
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    
    start_date = st.sidebar.date_input("Start Date", min_date)
    end_date = st.sidebar.date_input("End Date", max_date)
    
    # Filter data
    mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
    filtered_df = df.loc[mask]
    
    if len(filtered_df) == 0:
        st.warning("No data available for the selected date range.")
        return
    
    # Calculate metrics
    nps_metrics = calculate_nps_metrics(filtered_df)
    
    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 NPS Score",
            value=f"{nps_metrics['nps_score']:.1f}",
            delta=f"Target: 50+"
        )
    
    with col2:
        st.metric(
            label="🎯 Promoters",
            value=f"{nps_metrics['promoters']}",
            delta=f"{nps_metrics['promoter_rate']:.1f}%"
        )
    
    with col3:
        st.metric(
            label="⚠️ Detractors", 
            value=f"{nps_metrics['detractors']}",
            delta=f"{nps_metrics['detractor_rate']:.1f}%"
        )
    
    with col4:
        st.metric(
            label="📝 Total Responses",
            value=f"{nps_metrics['total_responses']}",
            delta=f"Last 30 days"
        )
    
    # Create tabs - Updated with the new fifth tab
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Aspect Analysis", "📈 Sentiment Trends", 
    "💡 AI Recommendations", "📝 Feedback Details",
    "✉️ Email/Transcript Analysis"  # <-- This should be the new 5th tab
])
    
    # Tab 1: Aspect Analysis
    with tab1:
        st.subheader("🔍 Legal Aspect Analysis")
        
        # Analyze aspects
        aspect_df = analyze_feedback_aspects(filtered_df, analyzer)
        
        if len(aspect_df) > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Aspect sentiment heatmap
                aspect_summary = aspect_df.groupby(['Aspect', 'Sentiment_Label']).size().unstack(fill_value=0)
                
                fig = px.imshow(
                    aspect_summary.values,
                    x=aspect_summary.columns,
                    y=aspect_summary.index,
                    aspect="auto",
                    color_continuous_scale="RdYlGn",
                    title="Legal Aspect Sentiment Heatmap"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 Aspect Mentions")
                aspect_counts = aspect_df['Aspect'].value_counts().head(8)
                
                for aspect, count in aspect_counts.items():
                    avg_sentiment = aspect_df[aspect_df['Aspect'] == aspect]['Sentiment_Score'].mean()
                    sentiment_emoji = "😊" if avg_sentiment > 0.1 else "😞" if avg_sentiment < -0.1 else "😐"
                    
                    st.metric(
                        label=f"{sentiment_emoji} {aspect}",
                        value=f"{count} mentions",
                        delta=f"Avg: {avg_sentiment:.2f}"
                    )
        else:
            st.info("No specific legal aspects detected in the current data range.")
    
    # Tab 2: Sentiment Trends
    with tab2:
        st.subheader("📈 Sentiment Trends Over Time")
        
        # Calculate daily sentiment
        daily_sentiment = []
        for date in filtered_df['Date'].dt.date.unique():
            day_data = filtered_df[filtered_df['Date'].dt.date == date]
            avg_sentiment = np.mean([
                analyzer.analyze_sentiment(text)['score'] 
                for text in day_data['Feedback_Text']
            ])
            daily_sentiment.append({
                'Date': date,
                'Avg_Sentiment': avg_sentiment,
                'Response_Count': len(day_data)
            })
        
        sentiment_df = pd.DataFrame(daily_sentiment)
        
        # Sentiment trend chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Average Daily Sentiment', 'Daily Response Volume'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(
                x=sentiment_df['Date'],
                y=sentiment_df['Avg_Sentiment'],
                mode='lines+markers',
                name='Sentiment Score',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=sentiment_df['Date'],
                y=sentiment_df['Response_Count'],
                name='Response Count',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment distribution
        col1, col2 = st.columns(2)
        
        with col1:
            sentiments = [analyzer.analyze_sentiment(text)['label'] for text in filtered_df['Feedback_Text']]
            sentiment_counts = pd.Series(sentiments).value_counts()
            
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Overall Sentiment Distribution",
                color_discrete_map={
                    'positive': '#2E8B57',
                    'neutral': '#DAA520', 
                    'negative': '#DC143C'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # NPS by sentiment
            nps_sentiment = []
            for _, row in filtered_df.iterrows():
                sentiment = analyzer.analyze_sentiment(row['Feedback_Text'])['label']
                nps_sentiment.append({
                    'NPS_Score': row['NPS_Score'],
                    'Sentiment': sentiment
                })
            
            nps_sent_df = pd.DataFrame(nps_sentiment)
            fig = px.box(
                nps_sent_df,
                x='Sentiment',
                y='NPS_Score',
                title="NPS Score by Sentiment Category",
                color='Sentiment',
                color_discrete_map={
                    'positive': '#2E8B57',
                    'neutral': '#DAA520',
                    'negative': '#DC143C'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: AI Recommendations
    with tab3:
        st.subheader("💡 AI-Powered Recommendations")
        
        # Analyze for recommendations
        aspect_df = analyze_feedback_aspects(filtered_df, analyzer)
        
        if len(aspect_df) > 0:
            # Priority issues
            negative_aspects = aspect_df[aspect_df['Sentiment_Label'] == 'Negative']
            priority_issues = negative_aspects['Aspect'].value_counts().head(5)
            
            st.subheader("🚨 Priority Issues to Address")
            for i, (aspect, count) in enumerate(priority_issues.items(), 1):
                with st.expander(f"{i}. {aspect} ({count} negative mentions)"):
                    aspect_feedback = negative_aspects[negative_aspects['Aspect'] == aspect]
                    
                    st.write("**Sample feedback:**")
                    for _, row in aspect_feedback.head(3).iterrows():
                        st.write(f"- \"{row['Context']}\"")
                    
                    # Generate recommendations based on aspect
                    recommendations = {
                        'Case Search & Research': [
                            "Optimize search algorithms for faster results",
                            "Improve search result relevance ranking",
                            "Add advanced filtering options",
                            "Implement search result caching"
                        ],
                        'Document Management': [
                            "Streamline document upload process",
                            "Add bulk document operations",
                            "Improve version control interface",
                            "Add document preview functionality"
                        ],
                        'System Performance': [
                            "Conduct performance optimization review",
                            "Implement server scaling solutions",
                            "Add system monitoring alerts",
                            "Optimize database queries"
                        ],
                        'Billing & Time Tracking': [
                            "Simplify time entry interface",
                            "Add automated time tracking features",
                            "Improve billing report customization",
                            "Integrate with popular accounting software"
                        ]
                    }
                    
                    if aspect in recommendations:
                        st.write("**Recommended actions:**")
                        for rec in recommendations[aspect]:
                            st.write(f"✅ {rec}")
            
            # Positive highlights
            st.subheader("🌟 Strengths to Leverage")
            positive_aspects = aspect_df[aspect_df['Sentiment_Label'] == 'Positive']
            strengths = positive_aspects['Aspect'].value_counts().head(3)
            
            for aspect, count in strengths.items():
                st.success(f"**{aspect}**: {count} positive mentions - Continue excellence and consider expanding features")
        
        else:
            st.info("Generate aspect analysis first to see AI recommendations.")
    
    # Tab 4: Feedback Details
    with tab4:
        st.subheader("📝 Detailed Feedback Analysis")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            client_filter = st.selectbox(
                "Filter by Client Type",
                ['All'] + list(filtered_df['Client_Type'].unique())
            )
        
        with col2:
            practice_filter = st.selectbox(
                "Filter by Practice Area", 
                ['All'] + list(filtered_df['Practice_Area'].unique())
            )
        
        with col3:
            nps_filter = st.selectbox(
                "Filter by NPS Category",
                ['All', 'Promoters (9-10)', 'Passives (7-8)', 'Detractors (0-6)']
            )
        
        # Apply filters
        display_df = filtered_df.copy()
        
        if client_filter != 'All':
            display_df = display_df[display_df['Client_Type'] == client_filter]
        
        if practice_filter != 'All':
            display_df = display_df[display_df['Practice_Area'] == practice_filter]
        
        if nps_filter != 'All':
            if nps_filter == 'Promoters (9-10)':
                display_df = display_df[display_df['NPS_Score'] >= 9]
            elif nps_filter == 'Passives (7-8)':
                display_df = display_df[(display_df['NPS_Score'] >= 7) & (display_df['NPS_Score'] <= 8)]
            elif nps_filter == 'Detractors (0-6)':
                display_df = display_df[display_df['NPS_Score'] <= 6]
        
        # Display filtered results
        st.write(f"Showing {len(display_df)} feedback entries")
        
        for _, row in display_df.iterrows():
            sentiment_analysis = analyzer.analyze_sentiment(row['Feedback_Text'])
            sentiment_emoji = "😊" if sentiment_analysis['label'] == 'positive' else "😞" if sentiment_analysis['label'] == 'negative' else "😐"
            
            with st.expander(f"{sentiment_emoji} NPS: {row['NPS_Score']} | {row['Client_Type']} | {row['Date'].strftime('%Y-%m-%d')}"):
                st.write(f"**Feedback:** {row['Feedback_Text']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Practice Area:** {row['Practice_Area']}")
                    st.write(f"**Sentiment:** {sentiment_analysis['label'].title()} ({sentiment_analysis['confidence']:.1%} confidence)")
                
                with col2:
                    aspects = analyzer.extract_aspects(row['Feedback_Text'])
                    if aspects:
                        st.write("**Mentioned aspects:**")
                        for aspect in aspects:
                            st.write(f"- {aspect['icon']} {aspect['name']}")
    
    # Tab 5: Email/Transcript Analysis - NEW TAB
    with tab5:
        st.subheader("✉️ Email/Transcript Analysis")
        st.write("Paste the text of a client email or conversation transcript below to analyze its sentiment and key aspects:")
        
        user_text = st.text_area("Enter email or transcript text:", height=150, 
                                placeholder="Paste your email content or call transcript here...")
        
        if st.button("Analyze Text", type="primary"):
            if not user_text.strip():
                st.warning("Please enter some text to analyze.")
            else:
                with st.spinner("Analyzing text..."):
                    # Perform analysis on the input text
                    result = analyzer.analyze_sentiment(user_text)
                    aspects_found = analyzer.extract_aspects(user_text)
                    
                    # Display overall sentiment
                    st.subheader("📊 Analysis Results")
                    
                    sentiment_label = result['label']
                    sentiment_score = result['score']
                    confidence_pct = result['confidence'] * 100
                    emoji = "😊" if sentiment_label == "positive" else "😞" if sentiment_label == "negative" else "😐"
                    
                    # Overall sentiment in a nice container
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div style="padding: 20px; border-radius: 10px; background-color: {'#d4edda' if sentiment_label == 'positive' else '#f8d7da' if sentiment_label == 'negative' else '#fff3cd'}; border: 1px solid {'#c3e6cb' if sentiment_label == 'positive' else '#f5c6cb' if sentiment_label == 'negative' else '#ffeaa7'};">
                            <h3>{emoji} Overall Sentiment: {sentiment_label.capitalize()}</h3>
                            <p><strong>Score:</strong> {sentiment_score:.3f} | <strong>Confidence:</strong> {confidence_pct:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Create a mini gauge for the sentiment score
                        fig_mini = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = sentiment_score,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Sentiment Score"},
                            gauge = {
                                'axis': {'range': [-1, 1]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [-1, -0.1], 'color': "lightcoral"},
                                    {'range': [-0.1, 0.1], 'color': "lightyellow"},
                                    {'range': [0.1, 1], 'color': "lightgreen"}
                                ],
                            }
                        ))
                        fig_mini.update_layout(height=200, margin=dict(t=50, b=0, l=0, r=0))
                        st.plotly_chart(fig_mini, use_container_width=True)
                    
                    # Display detected aspects
                    st.subheader("🔍 Detected Legal Aspects")
                    
                    if aspects_found:
                        st.write("**Detected Aspects & Sentiments:**")
                        
                        # Create columns for better layout
                        for i, asp in enumerate(aspects_found):
                            asp_score = asp['sentiment']
                            asp_label = "positive" if asp_score > 0.1 else "negative" if asp_score < -0.1 else "neutral"
                            asp_emoji = "😊" if asp_label == "positive" else "😞" if asp_label == "negative" else "😐"
                            
                            # Create an expandable section for each aspect
                            with st.expander(f"{asp['icon']} {asp['name']} - {asp_emoji} {asp_label.capitalize()} ({asp_score*100:.1f}%)"):
                                st.write(f"**Context:** \"{asp['text']}\"")
                                st.write(f"**Detected keyword:** {asp['keyword']}")
                                
                                # Mini sentiment bar
                                sentiment_color = "#28a745" if asp_score > 0.1 else "#dc3545" if asp_score < -0.1 else "#ffc107"
                                st.markdown(f"""
                                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0;">
                                    <div style="background-color: {sentiment_color}; height: 10px; width: {abs(asp_score)*100}%; border-radius: 5px;"></div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("No specific legal aspect keywords were detected in this text.")
                        st.write("The text may be too general or may not contain legal software-specific terminology.")
                    
                    # Additional insights
                    st.subheader("💡 Quick Insights")
                    
                    # Word count and reading time
                    word_count = len(user_text.split())
                    reading_time = max(1, word_count // 200)  # Assume 200 words per minute
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Word Count", word_count)
                    with col2:
                        st.metric("Reading Time", f"{reading_time} min")
                    with col3:
                        st.metric("Aspects Found", len(aspects_found))
                    
                    # Action recommendations based on sentiment
                    if sentiment_label == "negative":
                        st.warning("**Recommended Action:** This communication shows negative sentiment. Consider prioritizing a response and addressing the specific concerns mentioned.")
                    elif sentiment_label == "positive":
                        st.success("**Recommended Action:** This is positive feedback! Consider following up to gather more details or asking for a testimonial.")
                    else:
                        st.info("**Recommended Action:** This communication is neutral. Monitor for any specific requests or questions that need addressing.")
        
        # Add some example texts for testing
        st.subheader("📝 Example Texts to Try")
        
        examples = {
            "Positive Client Email": "Hi team, I wanted to reach out and thank you for the excellent case search functionality. Our research time has been cut in half since implementing your system. The document management is also very intuitive and our whole firm has adapted quickly. Great work!",
            "Negative Support Email": "I'm having serious issues with the billing module. It's been crashing repeatedly when I try to generate invoices, and the time tracking interface is confusing. This is affecting our ability to bill clients properly. Please help urgently.",
            "Mixed Feedback": "The legal research database is comprehensive and I love the integration with our case management. However, the system performance has been quite slow lately, especially during peak hours. The client portal works well though."
        }
        
        for example_name, example_text in examples.items():
            if st.button(f"Load: {example_name}"):
                st.text_area("", value=example_text, height=100, key=f"example_{example_name}")

if __name__ == "__main__":
    main()
