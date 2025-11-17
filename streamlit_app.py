import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Legal Feedback Intelligence Hub",
    page_icon="⚖️",
    layout="wide"
)

class LegalSentimentAnalyzer:
    """Advanced sentiment analyzer specifically designed for legal feedback analysis."""
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
                'keywords': [
                    'case search', 'research', 'find cases', 'legal research',
                    'search results', 'search function', 'database search', 'case law',
                    'precedent', 'citation', 'westlaw', 'lexis'
                ]
            },
            'document_management': {
                'name': 'Document Management',
                'icon': '📄',
                'keywords': [
                    'document', 'filing', 'upload', 'download', 'pdf', 'brief',
                    'pleading', 'contract', 'agreement', 'template', 'document review',
                    'version control'
                ]
            },
            'billing': {
                'name': 'Billing & Time Tracking',
                'icon': '💰',
                'keywords': [
                    'billing', 'invoice', 'time tracking', 'hours', 'rates', 'expense',
                    'cost', 'payment', 'fee', 'pricing', 'timesheet'
                ]
            },
            'client_portal': {
                'name': 'Client Portal',
                'icon': '👥',
                'keywords': [
                    'client portal', 'client access', 'communication', 'client login',
                    'sharing', 'collaboration', 'messages'
                ]
            },
            'calendar': {
                'name': 'Calendar & Scheduling',
                'icon': '📅',
                'keywords': [
                    'calendar', 'schedule', 'appointment', 'deadline', 'court date',
                    'meeting', 'reminder', 'docket'
                ]
            },
            'compliance': {
                'name': 'Compliance & Ethics',
                'icon': '⚖️',
                'keywords': [
                    'compliance', 'ethics', 'rules', 'bar requirements',
                    'professional responsibility', 'conflict check', 'audit trail'
                ]
            },
            'reporting': {
                'name': 'Reporting & Analytics',
                'icon': '📊',
                'keywords': [
                    'report', 'analytics', 'dashboard', 'metrics', 'statistics',
                    'performance', 'insights', 'data', 'trends'
                ]
            },
            'performance': {
                'name': 'System Performance',
                'icon': '⚡',
                'keywords': [
                    'speed', 'performance', 'slow', 'fast', 'loading', 'lag',
                    'responsive', 'crash', 'freeze', 'timeout', 'error', 'bug'
                ]
            },
            'training': {
                'name': 'Training & Support',
                'icon': '🎓',
                'keywords': [
                    'training', 'support', 'help', 'tutorial', 'documentation',
                    'onboarding', 'learning', 'guidance', 'assistance'
                ]
            },
            'integration': {
                'name': 'Integration & Compatibility',
                'icon': '🔗',
                'keywords': [
                    'integration', 'compatibility', 'sync', 'import', 'export',
                    'api', 'third party', 'outlook', 'office', 'accounting software'
                ]
            }
        }

    def analyze_sentiment(self, text):
        """Analyze sentiment of legal feedback text and return score, label, confidence."""
        if not text or not text.strip():
            return {'score': 0.0, 'label': 'neutral', 'confidence': 0.0}
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        positive_score = 0.0
        negative_score = 0.0
        total_words = len(words)

        for i, word in enumerate(words):
            # Check if the previous word is an intensifier
            intensifier = 1.0
            if i > 0 and words[i - 1] in self.intensifiers:
                intensifier = self.intensifiers[words[i - 1]]
            # Increment positive or negative score based on word sentiment
            if word in self.positive_terms:
                positive_score += intensifier
            elif word in self.negative_terms:
                negative_score += intensifier

        # Calculate net sentiment score (normalized by length)
        net_score = (positive_score - negative_score) / total_words if total_words > 0 else 0.0

        # Determine sentiment label and confidence
        if net_score > 0.02:
            label = 'positive'
            confidence = min(abs(net_score) * 10, 1.0)  # up to 1.0
        elif net_score < -0.02:
            label = 'negative'
            confidence = min(abs(net_score) * 10, 1.0)
        else:
            label = 'neutral'
            # Higher confidence for net_score near zero (no strong sentiment words)
            confidence = 1.0 - min(abs(net_score) * 5, 0.5)

        return {'score': net_score, 'label': label, 'confidence': confidence}

    def extract_aspects(self, text):
        """Extract legal aspects mentioned in the text, with context and sentiment for each aspect."""
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
                    # Analyze sentiment of that context sentence (to gauge aspect sentiment)
                    sentence_sentiment = self.analyze_sentiment(context_sentence)
                    found_aspects.append({
                        'name': aspect_info['name'],
                        'icon': aspect_info['icon'],
                        'keyword': keyword,
                        'text': context_sentence[:100] + "..." if len(context_sentence) > 100 else context_sentence,
                        'sentiment': sentence_sentiment['score']
                    })
                    break  # Only record the first occurrence per aspect category
        return found_aspects

def generate_sample_data():
    """Generate a sample dataframe of legal software feedback with realistic content."""
    np.random.seed(42)
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
        "Document templates save significant time, though more customization options are needed.",
        "Reporting functionality is powerful but could benefit from more visual charts.",
        "Security features are robust and give us confidence in client data protection.",
        "User interface is clean and modern, much better than our previous software."
    ]
    
    data = []
    for i in range(200):  # 200 feedback entries
        nps_score = np.random.randint(0, 11)  # 0 to 10 inclusive
        # Choose response tendency based on NPS score
        if nps_score >= 9:
            # likely positive feedback
            candidate_responses = [r for r in sample_responses if any(word in r.lower() for word in ['excellent', 'love', 'great', 'amazing', 'perfect'])]
            response = np.random.choice(candidate_responses) if candidate_responses else np.random.choice(sample_responses)
        elif nps_score <= 6:
            # likely negative feedback
            candidate_responses = [r for r in sample_responses if any(word in r.lower() for word in ['issues', 'slow', 'needs', 'crashes', 'frustrating'])]
            response = np.random.choice(candidate_responses) if candidate_responses else np.random.choice(sample_responses)
        else:
            # neutral or mixed
            response = np.random.choice(sample_responses)
        client_type = np.random.choice(
            ['Small Firm', 'Mid-size Firm', 'Large Firm', 'Solo Practice'],
            p=[0.3, 0.3, 0.2, 0.2]
        )
        practice_area = np.random.choice([
            'Corporate Law', 'Litigation', 'Family Law', 'Criminal Law',
            'Real Estate', 'Personal Injury', 'Employment Law'
        ])
        data.append({
            'NPS_Score': nps_score,
            'Feedback_Text': response,
            'Client_Type': client_type,
            'Practice_Area': practice_area,
            'Respondent_ID': f"Client_{i+1:03d}"
        })
    df_sample = pd.DataFrame(data)
    return df_sample

def calculate_nps_metrics(df):
    """Calculate NPS and related metrics from the dataframe."""
    total = len(df)
    promoters = len(df[df['NPS_Score'] >= 9])
    detractors = len(df[df['NPS_Score'] <= 6])
    passives = len(df[(df['NPS_Score'] >= 7) & (df['NPS_Score'] <= 8)])
    nps_score = ((promoters - detractors) / total * 100) if total > 0 else 0.0
    return {
        'nps_score': nps_score,
        'promoters': promoters,
        'detractors': detractors,
        'passives': passives,
        'total_responses': total,
        'promoter_rate': (promoters / total * 100) if total > 0 else 0.0,
        'detractor_rate': (detractors / total * 100) if total > 0 else 0.0
    }

def create_nps_gauge(nps_score):
    """Create a Plotly gauge chart for NPS score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=nps_score,
        number={'suffix': "%"},
        title={'text': "Net Promoter Score"},
        delta={'reference': 50, 'relative': False, 'position': "top"},
        gauge={
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
    fig.update_layout(height=250, margin=dict(t=20, b=20, l=40, r=40))
    return fig

def analyze_feedback_aspects(df, analyzer):
    """Analyze each feedback entry for aspect mentions and sentiment."""
    aspect_data = []
    for _, row in df.iterrows():
        aspects = analyzer.extract_aspects(row['Feedback_Text'])
        for asp in aspects:
            sentiment_label = 'Positive' if asp['sentiment'] > 0.1 else 'Negative' if asp['sentiment'] < -0.1 else 'Neutral'
            aspect_data.append({
                'Aspect': asp['name'],
                'Sentiment_Score': asp['sentiment'],
                'Sentiment_Label': sentiment_label,
                'Client_Type': row['Client_Type'],
                'Practice_Area': row['Practice_Area'],
                'NPS_Score': row['NPS_Score'],
                'Context': asp['text']
            })
    return pd.DataFrame(aspect_data)

def main():
    # Title and description
    st.title("⚖️ Legal Feedback Intelligence Hub")
    st.markdown("**Advanced NPS Analysis & Sentiment Intelligence for Legal Software**")

    # Initialize the sentiment analyzer
    analyzer = LegalSentimentAnalyzer()

    # Sidebar - Data upload and filtering
    st.sidebar.header("📊 Data Management")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your feedback data (CSV)",
        type=['csv'],
        help="Expecting columns: NPS_Score, Feedback_Text, Client_Type, Practice_Area"
    )
    # Load data
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ Data uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
            df = generate_sample_data()
            st.sidebar.info("Using sample data instead.")
    else:
        df = generate_sample_data()
        st.sidebar.info("📋 Using sample data (randomly generated). Upload a CSV to analyze your own feedback.")
        # Offer sample template download
        sample_csv = df.head(10).to_csv(index=False)
        st.sidebar.download_button(
            label="📥 Download Sample Data Template",
            data=sample_csv,
            file_name="legal_feedback_template.csv",
            mime="text/csv"
        )

    # Use the full dataframe (no date filtering)
    filtered_df = df.copy()
    
    if filtered_df.empty:
        st.warning("No feedback data available.")
        return

    # Calculate NPS metrics for filtered data
    nps_metrics = calculate_nps_metrics(filtered_df)

    # Top-level KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="📊 NPS Score", value=f"{nps_metrics['nps_score']:.1f}", delta="Target: 50+")
    col2.metric(label="🎯 Promoters", value=nps_metrics['promoters'], delta=f"{nps_metrics['promoter_rate']:.1f}%")
    col3.metric(label="⚠️ Detractors", value=nps_metrics['detractors'], delta=f"{nps_metrics['detractor_rate']:.1f}%")
    col4.metric(label="📝 Total Responses", value=nps_metrics['total_responses'])

    # Create tabs for different analysis sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Aspect Analysis", "📈 Sentiment Analysis",
        "💡 AI Recommendations", "📝 Feedback Details",
        "✉️ Email/Transcript Analysis"
    ])

    # Tab 1: Aspect Analysis
    with tab1:
        st.subheader("🔍 Legal Aspect Analysis")
        aspect_df = analyze_feedback_aspects(filtered_df, analyzer)
        if not aspect_df.empty:
            # Layout: heatmap on left, top aspects on right
            left_col, right_col = st.columns([2, 1])
            with left_col:
                aspect_summary = aspect_df.groupby(['Aspect', 'Sentiment_Label']).size().unstack(fill_value=0)
                fig = px.imshow(
                    aspect_summary,
                    color_continuous_scale="RdYlGn",
                    labels={'x': 'Sentiment', 'y': 'Aspect', 'color': 'Count'},
                    title="Aspect Sentiment Heatmap"
                )
                fig.update_layout(height=500, margin=dict(t=50, b=50))
                st.plotly_chart(fig, use_container_width=True)
            with right_col:
                st.subheader("📊 Top Aspect Mentions")
                top_aspects = aspect_df['Aspect'].value_counts().head(8)
                for aspect, count in top_aspects.items():
                    avg_sentiment = aspect_df[aspect_df['Aspect'] == aspect]['Sentiment_Score'].mean()
                    sentiment_emoji = "😊" if avg_sentiment > 0.1 else "😞" if avg_sentiment < -0.1 else "😐"
                    st.metric(label=f"{sentiment_emoji} {aspect}", value=f"{count} mentions", delta=f"Avg: {avg_sentiment:.2f}")
        else:
            st.info("No specific legal aspects were mentioned in the selected feedback data.")

    # Tab 2: Sentiment Analysis
    with tab2:
        st.subheader("📈 Sentiment Distribution Analysis")
        
        # Distribution of sentiment labels
        col_left, col_right = st.columns(2)
        with col_left:
            sentiment_labels = [analyzer.analyze_sentiment(text)['label'] for text in filtered_df['Feedback_Text']]
            sentiment_counts = pd.Series(sentiment_labels).value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Overall Sentiment Distribution",
                color_discrete_map={'positive': '#2E8B57', 'neutral': '#DAA520', 'negative': '#DC143C'}
            )
            fig_pie.update_traces(textinfo='label+percent', showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_right:
            nps_sentiment = pd.DataFrame([
                {'NPS_Score': row['NPS_Score'], 'Sentiment': analyzer.analyze_sentiment(row['Feedback_Text'])['label']}
                for _, row in filtered_df.iterrows()
            ])
            fig_box = px.box(
                nps_sentiment, x='Sentiment', y='NPS_Score', color='Sentiment',
                title="NPS Score by Sentiment Category",
                color_discrete_map={'positive': '#2E8B57', 'neutral': '#DAA520', 'negative': '#DC143C'}
            )
            fig_box.update_traces(quartilemethod="inclusive")
            st.plotly_chart(fig_box, use_container_width=True)
            
        # Sentiment by Client Type and Practice Area
        st.subheader("📊 Sentiment by Demographics")
        col_demo1, col_demo2 = st.columns(2)
        
        with col_demo1:
            client_sentiment = pd.DataFrame([
                {
                    'Client_Type': row['Client_Type'], 
                    'Sentiment': analyzer.analyze_sentiment(row['Feedback_Text'])['label']
                }
                for _, row in filtered_df.iterrows()
            ])
            client_sent_counts = client_sentiment.groupby(['Client_Type', 'Sentiment']).size().unstack(fill_value=0)
            fig_client = px.bar(
                client_sent_counts, 
                title="Sentiment by Client Type",
                color_discrete_map={'positive': '#2E8B57', 'neutral': '#DAA520', 'negative': '#DC143C'}
            )
            st.plotly_chart(fig_client, use_container_width=True)
            
        with col_demo2:
            practice_sentiment = pd.DataFrame([
                {
                    'Practice_Area': row['Practice_Area'], 
                    'Sentiment': analyzer.analyze_sentiment(row['Feedback_Text'])['label']
                }
                for _, row in filtered_df.iterrows()
            ])
            practice_sent_counts = practice_sentiment.groupby(['Practice_Area', 'Sentiment']).size().unstack(fill_value=0)
            fig_practice = px.bar(
                practice_sent_counts, 
                title="Sentiment by Practice Area",
                color_discrete_map={'positive': '#2E8B57', 'neutral': '#DAA520', 'negative': '#DC143C'}
            )
            st.plotly_chart(fig_practice, use_container_width=True)

    # Tab 3: AI Recommendations
    with tab3:
        st.subheader("💡 AI-Powered Recommendations")
        aspect_df = analyze_feedback_aspects(filtered_df, analyzer)
        if not aspect_df.empty:
            # Identify top negative aspects to prioritize
            neg_aspects = aspect_df[aspect_df['Sentiment_Label'] == 'Negative']
            priority_issues = neg_aspects['Aspect'].value_counts().head(5)
            st.subheader("🚨 Priority Issues to Address")
            for i, (aspect, count) in enumerate(priority_issues.items(), start=1):
                with st.expander(f"{i}. {aspect} ({count} negative mentions)"):
                    # Show some sample negative feedback for this aspect
                    aspect_feedback = neg_aspects[neg_aspects['Aspect'] == aspect]
                    st.write("**Sample Feedback Snippets:**")
                    for _, row in aspect_feedback.head(3).iterrows():
                        st.write(f"- \"{row['Context']}\"")
                    # Suggest recommendations (if available for this aspect)
                    recommendations = {
                        'Case Search & Research': [
                            "Optimize the search algorithm for faster results.",
                            "Improve search result relevance and filtering options.",
                            "Implement caching to speed up repeated searches."
                        ],
                        'Document Management': [
                            "Streamline the document upload process.",
                            "Introduce bulk document operations (upload/download).",
                            "Enhance version control with clearer change tracking."
                        ],
                        'System Performance': [
                            "Conduct a performance audit and optimize slow processes.",
                            "Upgrade server infrastructure or enable auto-scaling.",
                            "Implement real-time monitoring to catch issues early."
                        ],
                        'Billing & Time Tracking': [
                            "Simplify the time entry interface for users.",
                            "Introduce automatic time tracking suggestions.",
                            "Allow more customization in billing reports."
                        ],
                        'Integration & Compatibility': [
                            "Improve sync stability with accounting software.",
                            "Expand integration documentation for third-party apps.",
                            "Add alerts for any integration failures."
                        ]
                    }
                    if aspect in recommendations:
                        st.write("**Recommended Actions:**")
                        for rec in recommendations[aspect]:
                            st.write(f"✅ {rec}")
            # Highlight top positive aspects
            st.subheader("🌟 Strengths to Leverage")
            pos_aspects = aspect_df[aspect_df['Sentiment_Label'] == 'Positive']
            top_strengths = pos_aspects['Aspect'].value_counts().head(3)
            for aspect, count in top_strengths.items():
                st.success(f"**{aspect}** — {count} positive mentions. Continue excelling here and consider expanding these features.")
        else:
            st.info("No aspect analysis available. Once feedback is provided, recommendations will appear here.")

    # Tab 4: Feedback Details
    with tab4:
        st.subheader("📝 Detailed Feedback Analysis")
        # Filters for detailed view
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            client_filter = st.selectbox("Filter by Client Type", ['All'] + sorted(filtered_df['Client_Type'].unique().tolist()))
        with filter_col2:
            practice_filter = st.selectbox("Filter by Practice Area", ['All'] + sorted(filtered_df['Practice_Area'].unique().tolist()))
        with filter_col3:
            nps_filter = st.selectbox("Filter by NPS Category", ['All', 'Promoters (9-10)', 'Passives (7-8)', 'Detractors (0-6)'])
        # Apply selected filters
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
        st.write(f"Showing **{len(display_df)}** feedback entries after filtering:")
        # Iterate through filtered feedback for detailed display
        for _, row in display_df.iterrows():
            sentiment_result = analyzer.analyze_sentiment(row['Feedback_Text'])
            sentiment_label = sentiment_result['label']
            sentiment_conf = sentiment_result['confidence']
            sentiment_emoji = "😊" if sentiment_label == 'positive' else "😞" if sentiment_label == 'negative' else "😐"
            expander_title = f"{sentiment_emoji} NPS: {row['NPS_Score']} | {row['Client_Type']}"
            with st.expander(expander_title):
                st.write(f"**Feedback:** {row['Feedback_Text']}")
                col_det1, col_det2 = st.columns(2)
                with col_det1:
                    st.write(f"**Practice Area:** {row['Practice_Area']}")
                    st.write(f"**Sentiment:** {sentiment_label.capitalize()} ({sentiment_conf:.1%} confidence)")
                with col_det2:
                    aspects = analyzer.extract_aspects(row['Feedback_Text'])
                    if aspects:
                        st.write("**Mentioned Aspects:**")
                        for asp in aspects:
                            st.write(f"- {asp['icon']} {asp['name']}")
                    else:
                        st.write("**Mentioned Aspects:** None")

    # Tab 5: Email/Transcript Analysis
    with tab5:
        st.subheader("✉️ Email/Transcript Analysis")
        st.write("Paste the text of a client email or conversation transcript below to analyze its sentiment and key aspects:")
        
        # Use session state to persist input text
        if "input_text" not in st.session_state:
            st.session_state.input_text = ""
        
        user_text = st.text_area(
            "Enter email or transcript text:",
            value=st.session_state.input_text,
            placeholder="Paste your email content or call transcript here...",
            height=150
        )
        
        analyze_button = st.button("Analyze Text", type="primary")
        
        if analyze_button:
            if not user_text or not user_text.strip():
                st.warning("Please enter some text to analyze.")
            else:
                # Save current input to session state
                st.session_state.input_text = user_text
                
                with st.spinner("Analyzing text..."):
                    result = analyzer.analyze_sentiment(user_text)
                    aspects_found = analyzer.extract_aspects(user_text)
                
                # Display Overall Sentiment
                st.subheader("📊 Analysis Results")
                sentiment_label = result['label']
                sentiment_score = result['score']
                confidence_pct = result['confidence'] * 100
                emoji = "😊" if sentiment_label == "positive" else "😞" if sentiment_label == "negative" else "😐"
                
                # Colored container for overall sentiment
                bg_color = '#d4edda' if sentiment_label == 'positive' else '#f8d7da' if sentiment_label == 'negative' else '#fff3cd'
                border_color = '#c3e6cb' if sentiment_label == 'positive' else '#f5c6cb' if sentiment_label == 'negative' else '#ffeeba'
                
                container_html = f"""
                <div style="padding: 15px; border-radius: 8px; background-color: {bg_color}; 
                            border: 1px solid {border_color}; margin-bottom: 1rem;">
                    <h4>{emoji} Overall Sentiment: {sentiment_label.capitalize()}</h4>
                    <p><strong>Score:</strong> {sentiment_score:.3f} &nbsp;|&nbsp; 
                       <strong>Confidence:</strong> {confidence_pct:.1f}%</p>
                </div>
                """
                st.markdown(container_html, unsafe_allow_html=True)
                
                # Mini gauge chart for sentiment score
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=sentiment_score,
                    number={'font': {'size': 48}, 'suffix': ""},
                    gauge={
                        'axis': {'range': [-1, 1], 'tickwidth': 1},
                        'bar': {'color': "black"},
                        'steps': [
                            {'range': [-1, -0.1], 'color': "#F8D7DA"},
                            {'range': [-0.1, 0.1], 'color': "#FFF3CD"},
                            {'range': [0.1, 1], 'color': "#D4EDDA"}
                        ],
                    }
                ))
                gauge_fig.update_layout(height=200, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Display detected aspects
                st.subheader("🔍 Detected Legal Aspects")
                
                if aspects_found:
                    st.write("The following aspects were mentioned and their sentiment context:")
                    for asp in aspects_found:
                        asp_name = asp['name']
                        asp_icon = asp['icon']
                        asp_keyword = asp['keyword']
                        asp_text = asp['text']
                        asp_score = asp['sentiment']
                        asp_label = "positive" if asp_score > 0.1 else "negative" if asp_score < -0.1 else "neutral"
                        asp_emoji = "😊" if asp_label == "positive" else "😞" if asp_label == "negative" else "😐"
                        
                        expander_label = f"{asp_icon} {asp_name} — {asp_emoji} {asp_label.capitalize()} ({asp_score*100:.1f}%)"
                        with st.expander(expander_label):
                            st.write(f"**Context Snippet:** \"{asp_text}\"")
                            st.write(f"**Detected Keyword:** *{asp_keyword}*")
                            
                            # Sentiment bar indicator
                            sentiment_color = "#28a745" if asp_score > 0.1 else "#dc3545" if asp_score < -0.1 else "#ffc107"
                            bar_width = min(100, abs(asp_score) * 100)
                            bar_html = f"""
                            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 2px; margin-top: 5px; width: 100%;">
                                <div style="background-color: {sentiment_color}; width: {bar_width}%; height: 8px; border-radius: 5px;"></div>
                            </div>
                            """
                            st.markdown(bar_html, unsafe_allow_html=True)
                else:
                    st.info("No specific legal aspect keywords were detected in the text.")
                    st.write("The content might be general or not related to key product features.")
                
                # Quick insights metrics
                st.subheader("💡 Quick Insights")
                word_count = len(user_text.split())
                reading_time = max(1, word_count // 200)
                
                insight_col1, insight_col2, insight_col3 = st.columns(3)
                insight_col1.metric("Word Count", word_count)
                insight_col2.metric("Estimated Read Time", f"{reading_time} min")
                insight_col3.metric("Aspects Detected", len(aspects_found))
                
                # Recommended action based on overall sentiment
                if sentiment_label == "negative":
                    st.warning("**Recommended Action:** The sentiment is negative. Prioritize a prompt response and address the concerns mentioned.")
                elif sentiment_label == "positive":
                    st.success("**Recommended Action:** The sentiment is positive. Consider following up for more details or a potential testimonial.")
                else:
                    st.info("**Recommended Action:** The sentiment is neutral. Monitor for any specific questions or requests to address.")
        
        # Example texts for quick testing
        st.subheader("📝 Example Texts to Try")
        
        examples = {
            "Positive Client Email": "Hi team, I wanted to reach out and thank you for the excellent case search functionality. Our research time has been cut in half since implementing your system. The document management is also very intuitive and our whole firm adapted quickly. Great work!",
            "Negative Support Email": "I'm having serious issues with the billing module. It has been crashing repeatedly when I try to generate invoices, and the time tracking interface is confusing. This is affecting our ability to bill clients properly. Please help urgently.",
            "Mixed Feedback": "The legal research database is comprehensive and I love the integration with our case management. However, the system performance has been quite slow lately, especially during peak hours. The client portal works well though."
        }
        
        for example_name, example_text in examples.items():
            if st.button(f"Load: {example_name}"):
                st.session_state.input_text = example_text

if __name__ == "__main__":
    main()
