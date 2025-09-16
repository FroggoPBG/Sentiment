import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import re
from collections import defaultdict, Counter
import json

# Page config
st.set_page_config(
    page_title="Legal Feedback Intelligence Hub",
    page_icon="⚖️",
    layout="wide"
)

# Legal-specific aspect categories
LEGAL_ASPECTS = {
    'search_accuracy': {
        'keywords': ['search', 'find', 'results', 'relevant', 'accuracy', 'precision', 'case law', 'precedent', 'citations'],
        'phrases': ['can\'t find', 'irrelevant results', 'outdated cases', 'missing precedents', 'search fails']
    },
    'ai_features': {
        'keywords': ['ai', 'artificial intelligence', 'summarization', 'analysis', 'prediction', 'automation'],
        'phrases': ['ai summary', 'case analysis', 'predictive analytics', 'automated research']
    },
    'compliance': {
        'keywords': ['compliance', 'gdpr', 'privacy', 'security', 'audit', 'regulatory', 'ethics'],
        'phrases': ['data privacy', 'compliance issues', 'security concerns', 'ethical ai']
    },
    'workflow_integration': {
        'keywords': ['integration', 'workflow', 'sync', 'export', 'api', 'compatibility'],
        'phrases': ['doesn\'t sync', 'workflow issues', 'integration problems', 'export failed']
    },
    'ui_navigation': {
        'keywords': ['interface', 'navigation', 'ui', 'ux', 'usability', 'design', 'layout'],
        'phrases': ['hard to navigate', 'confusing interface', 'user-friendly', 'intuitive design']
    },
    'pricing_value': {
        'keywords': ['price', 'cost', 'expensive', 'value', 'roi', 'budget', 'subscription'],
        'phrases': ['too expensive', 'good value', 'cost-effective', 'pricing tier']
    },
    'performance': {
        'keywords': ['speed', 'slow', 'fast', 'performance', 'loading', 'response time', 'latency'],
        'phrases': ['loads slowly', 'fast response', 'system lag', 'quick results']
    },
    'support_quality': {
        'keywords': ['support', 'help', 'response', 'customer service', 'training', 'documentation'],
        'phrases': ['poor support', 'helpful team', 'slow response', 'great documentation']
    }
}

# Firm type classifications
FIRM_TYPES = {
    'solo': ['solo', 'individual', 'freelance', 'independent'],
    'small_firm': ['small firm', 'boutique', '2-10 attorneys', 'small practice'],
    'mid_size': ['mid-size', 'medium', '10-50 lawyers', 'regional'],
    'big_law': ['big law', 'large firm', 'am law', '100+', 'major firm', 'global'],
    'in_house': ['in-house', 'corporate', 'company legal', 'internal'],
    'government': ['government', 'public sector', 'state', 'federal', 'municipal']
}

# Legal practice areas
PRACTICE_AREAS = {
    'litigation': ['litigation', 'trial', 'court', 'dispute', 'lawsuit'],
    'corporate': ['corporate', 'm&a', 'transaction', 'contract', 'commercial'],
    'ip': ['intellectual property', 'patent', 'trademark', 'copyright', 'ip'],
    'employment': ['employment', 'labor', 'hr', 'workplace'],
    'real_estate': ['real estate', 'property', 'land', 'construction'],
    'criminal': ['criminal', 'defense', 'prosecution', 'dui'],
    'family': ['family', 'divorce', 'custody', 'adoption'],
    'tax': ['tax', 'irs', 'revenue', 'taxation']
}

class LegalFeedbackAnalyzer:
    def __init__(self):
        self.sentiment_scores = {}
        self.aspect_sentiments = {}
        
    def extract_legal_aspects(self, text):
        """Extract sentiment for each legal-specific aspect"""
        text_lower = text.lower()
        aspect_sentiments = {}
        
        for aspect, data in LEGAL_ASPECTS.items():
            # Check for keywords and phrases
            keyword_matches = sum(1 for keyword in data['keywords'] if keyword in text_lower)
            phrase_matches = sum(1 for phrase in data['phrases'] if phrase in text_lower)
            
            if keyword_matches > 0 or phrase_matches > 0:
                # Simple sentiment scoring for the aspect
                positive_words = ['good', 'great', 'excellent', 'love', 'perfect', 'amazing', 'helpful']
                negative_words = ['bad', 'terrible', 'awful', 'hate', 'poor', 'slow', 'confusing', 'expensive']
                
                # Context window around aspect mentions
                sentences = text_lower.split('.')
                relevant_sentences = [s for s in sentences if any(kw in s for kw in data['keywords']) or any(ph in s for ph in data['phrases'])]
                
                if relevant_sentences:
                    pos_score = sum(1 for sentence in relevant_sentences for word in positive_words if word in sentence)
                    neg_score = sum(1 for sentence in relevant_sentences for word in negative_words if word in sentence)
                    
                    if pos_score > neg_score:
                        sentiment = 'positive'
                        score = (pos_score - neg_score) / len(relevant_sentences)
                    elif neg_score > pos_score:
                        sentiment = 'negative'
                        score = -(neg_score - pos_score) / len(relevant_sentences)
                    else:
                        sentiment = 'neutral'
                        score = 0
                    
                    aspect_sentiments[aspect] = {
                        'sentiment': sentiment,
                        'score': score,
                        'mentions': keyword_matches + phrase_matches,
                        'context': relevant_sentences[:2]  # First 2 relevant sentences
                    }
        
        return aspect_sentiments
    
    def classify_firm_type(self, text):
        """Classify the type of law firm based on feedback content"""
        text_lower = text.lower()
        
        for firm_type, indicators in FIRM_TYPES.items():
            if any(indicator in text_lower for indicator in indicators):
                return firm_type
        
        return 'unknown'
    
    def identify_practice_area(self, text):
        """Identify practice area from feedback"""
        text_lower = text.lower()
        areas = []
        
        for area, keywords in PRACTICE_AREAS.items():
            if any(keyword in text_lower for keyword in keywords):
                areas.append(area)
        
        return areas if areas else ['general']
    
    def predict_nps_trend(self, historical_data):
        """Predict NPS trend based on aspect sentiment changes"""
        # Simplified trend prediction
        recent_scores = historical_data[-30:]  # Last 30 feedback items
        older_scores = historical_data[-60:-30] if len(historical_data) >= 60 else []
        
        if not older_scores:
            return "Insufficient data for trend prediction"
        
        recent_avg = np.mean([item['compound_score'] for item in recent_scores])
        older_avg = np.mean([item['compound_score'] for item in older_scores])
        
        trend = recent_avg - older_avg
        
        if trend > 0.1:
            return f"📈 Positive trend: +{trend:.2f} score improvement"
        elif trend < -0.1:
            return f"📉 Negative trend: {trend:.2f} score decline"
        else:
            return f"➡️ Stable trend: {trend:.2f} score change"
    
    def generate_recommendations(self, aspect_sentiments, firm_type, practice_areas):
        """Generate AI-powered recommendations"""
        recommendations = []
        
        # Aspect-based recommendations
        negative_aspects = [aspect for aspect, data in aspect_sentiments.items() 
                          if data['sentiment'] == 'negative']
        
        if 'search_accuracy' in negative_aspects:
            recommendations.append({
                'priority': 'High',
                'category': 'Product',
                'action': 'Improve search algorithm relevance',
                'rationale': 'Search accuracy issues directly impact user productivity',
                'estimated_impact': '+2-3 NPS points'
            })
        
        if 'compliance' in negative_aspects:
            recommendations.append({
                'priority': 'Critical',
                'category': 'Legal/Security',
                'action': 'Review compliance documentation and security measures',
                'rationale': 'Compliance concerns can lead to client churn in legal industry',
                'estimated_impact': 'Risk mitigation'
            })
        
        # Firm-type specific recommendations
        if firm_type == 'solo' and 'pricing_value' in negative_aspects:
            recommendations.append({
                'priority': 'Medium',
                'category': 'Commercial',
                'action': 'Consider solo practitioner pricing tier',
                'rationale': 'Solo practitioners have different budget constraints',
                'estimated_impact': '+1-2 NPS points for solo segment'
            })
        
        return recommendations

def create_sample_data():
    """Create sample historical data for demonstration"""
    dates = [datetime.now() - timedelta(days=x) for x in range(90, 0, -1)]
    
    sample_feedback = [
        {
            'date': dates[0],
            'feedback': "The AI case summarization is fantastic, but the search results for precedent cases are often outdated. Our litigation team struggles with relevance.",
            'source': 'NPS Survey',
            'firm_type': 'mid_size',
            'practice_area': ['litigation'],
            'compound_score': -0.2
        },
        {
            'date': dates[10],
            'feedback': "Love the new interface! Much more intuitive for our corporate team. The contract analysis features are game-changing.",
            'source': 'Email',
            'firm_type': 'big_law',
            'practice_area': ['corporate'],
            'compound_score': 0.7
        },
        {
            'date': dates[20],
            'feedback': "Pricing is getting expensive for a solo practice. The features are good but I need better value for money.",
            'source': 'Support Ticket',
            'firm_type': 'solo',
            'practice_area': ['general'],
            'compound_score': -0.3
        },
        {
            'date': dates[30],
            'feedback': "Integration with our case management system failed multiple times. Very frustrating for workflow.",
            'source': 'Email',
            'firm_type': 'small_firm',
            'practice_area': ['general'],
            'compound_score': -0.5
        }
    ]
    
    return sample_feedback

# Main App
st.title("⚖️ Legal Feedback Intelligence Hub")
st.markdown("### Transform client feedback into strategic legal tech insights")

# Initialize analyzer
analyzer = LegalFeedbackAnalyzer()

# Sidebar navigation
st.sidebar.title("🎯 Intelligence Dashboard")
analysis_mode = st.sidebar.selectbox(
    "Select Analysis Mode:",
    ["Single Feedback Analysis", "Batch Processing", "Trend Analytics", "Predictive Insights"]
)

if analysis_mode == "Single Feedback Analysis":
    st.header("📋 Advanced Feedback Analysis")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        feedback_text = st.text_area(
            "Client Feedback:",
            height=150,
            placeholder="Enter NPS comment, email, or support ticket content..."
        )
    
    with col2:
        st.write("**Context Information:**")
        source = st.selectbox("Source", ["NPS Survey", "Email", "Support Ticket", "Phone Call"])
        
        # Sample feedback buttons
        if st.button("🏢 Corporate Law Firm"):
            feedback_text = "The AI contract analysis is revolutionary for our M&A practice, but the integration with our document management system keeps failing. Our associates love the research capabilities though."
        
        if st.button("👤 Solo Practitioner"):
            feedback_text = "Great tool for case research but honestly too expensive for a solo practice like mine. The search accuracy for local cases could be better too."
        
        if st.button("⚖️ Litigation Focus"):
            feedback_text = "The precedent search is usually good but we found several outdated citations in a recent case. The AI summarization of depositions is fantastic though."
    
    if st.button("🔍 Analyze Feedback", type="primary"):
        if feedback_text.strip():
            # Perform analysis
            aspect_sentiments = analyzer.extract_legal_aspects(feedback_text)
            firm_type = analyzer.classify_firm_type(feedback_text)
            practice_areas = analyzer.identify_practice_area(feedback_text)
            recommendations = analyzer.generate_recommendations(aspect_sentiments, firm_type, practice_areas)
            
            # Display results
            st.markdown("---")
            st.header("🧠 AI Analysis Results")
            
            # Key insights cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🏢 Firm Type", firm_type.replace('_', ' ').title())
            
            with col2:
                st.metric("⚖️ Practice Areas", len(practice_areas))
            
            with col3:
                st.metric("🎯 Aspects Mentioned", len(aspect_sentiments))
            
            with col4:
                negative_aspects = sum(1 for data in aspect_sentiments.values() if data['sentiment'] == 'negative')
                st.metric("⚠️ Pain Points", negative_aspects)
            
            # Aspect-based sentiment analysis
            if aspect_sentiments:
                st.subheader("📊 Aspect-Based Sentiment Analysis")
                
                # Create visualization
                aspects = list(aspect_sentiments.keys())
                scores = [data['score'] for data in aspect_sentiments.values()]
                sentiments = [data['sentiment'] for data in aspect_sentiments.values()]
                
                # Color mapping
                colors = ['red' if s == 'negative' else 'green' if s == 'positive' else 'gray' for s in sentiments]
                
                fig = go.Figure(data=go.Bar(
                    x=[aspect.replace('_', ' ').title() for aspect in aspects],
                    y=scores,
                    marker_color=colors,
                    text=[f"{data['mentions']} mentions" for data in aspect_sentiments.values()],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="Legal Aspect Sentiment Scores",
                    yaxis_title="Sentiment Score",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed aspect breakdown
                st.subheader("🔍 Aspect Details")
                for aspect, data in aspect_sentiments.items():
                    with st.expander(f"{aspect.replace('_', ' ').title()} - {data['sentiment'].title()}"):
                        st.write(f"**Sentiment Score:** {data['score']:.2f}")
                        st.write(f"**Mentions:** {data['mentions']}")
                        st.write("**Context:**")
                        for sentence in data['context']:
                            st.write(f"• {sentence.strip()}")
            
            # Recommendations
            if recommendations:
                st.subheader("💡 AI-Generated Recommendations")
                
                for i, rec in enumerate(recommendations):
                    priority_colors = {'Critical': '🔴', 'High': '🟠', 'Medium': '🟡', 'Low': '🟢'}
                    
                    st.write(f"**{priority_colors[rec['priority']]} {rec['priority']} Priority - {rec['category']}**")
                    st.write(f"**Action:** {rec['action']}")
                    st.write(f"**Rationale:** {rec['rationale']}")
                    st.write(f"**Estimated Impact:** {rec['estimated_impact']}")
                    st.write("---")
            
            # Practice area insights
            if practice_areas:
                st.subheader("⚖️ Practice Area Context")
                st.write(f"**Identified Areas:** {', '.join([area.replace('_', ' ').title() for area in practice_areas])}")
                
                if 'litigation' in practice_areas and 'search_accuracy' in aspect_sentiments:
                    st.info("💡 **Litigation Insight:** Search accuracy issues are critical for litigation practices where precedent research is essential.")
                
                if 'corporate' in practice_areas and 'workflow_integration' in aspect_sentiments:
                    st.info("💡 **Corporate Insight:** Workflow integration is key for corporate practices handling high-volume transactions.")

elif analysis_mode == "Trend Analytics":
    st.header("📈 Legal Tech Trend Analytics")
    
    # Sample data for demonstration
    sample_data = create_sample_data()
    
    # Time series analysis
    df = pd.DataFrame(sample_data)
    
    # Trend visualization
    fig = px.line(df, x='date', y='compound_score', 
                  title='Client Sentiment Trend Over Time',
                  color='firm_type')
    st.plotly_chart(fig, use_container_width=True)
    
    # Firm type breakdown
    firm_sentiment = df.groupby('firm_type')['compound_score'].mean().reset_index()
    
    fig2 = px.bar(firm_sentiment, x='firm_type', y='compound_score',
                  title='Average Sentiment by Firm Type')
    st.plotly_chart(fig2, use_container_width=True)
    
    # Predictive insights
    st.subheader("🔮 Predictive Analytics")
    trend_prediction = analyzer.predict_nps_trend(sample_data)
    st.info(f"**Trend Prediction:** {trend_prediction}")

elif analysis_mode == "Predictive Insights":
    st.header("🎯 Predictive Legal Tech Insights")
    
    st.subheader("🚨 Early Warning System")
    
    # Simulated alerts
    alerts = [
        {
            'type': 'Critical',
            'message': 'Compliance sentiment dropping 25% among Big Law firms',
            'action': 'Review security documentation and schedule client calls',
            'impact': 'Potential 15% NPS drop if unaddressed'
        },
        {
            'type': 'Opportunity',
            'message': 'AI features receiving 90% positive sentiment',
            'action': 'Expand AI marketing to similar firm segments',
            'impact': 'Potential 10% market share increase'
        }
    ]
    
    for alert in alerts:
        if alert['type'] == 'Critical':
            st.error(f"🚨 **{alert['type']}:** {alert['message']}")
        else:
            st.success(f"💡 **{alert['type']}:** {alert['message']}")
        
        st.write(f"**Recommended Action:** {alert['action']}")
        st.write(f"**Projected Impact:** {alert['impact']}")
        st.write("---")

# Footer
st.markdown("---")
st.markdown("### 🚀 Intelligence Hub Capabilities:")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🎯 Aspect-Based Analysis**
    - Search accuracy sentiment
    - AI feature perception
    - Compliance concerns
    - Workflow integration issues
    """)

with col2:
    st.markdown("""
    **📊 Legal-Specific Insights**
    - Firm type classification
    - Practice area analysis
    - Industry benchmarking
    - Competitive positioning
    """)

with col3:
    st.markdown("""
    **🔮 Predictive Intelligence**
    - NPS trend forecasting
    - Churn risk assessment
    - Feature demand prediction
    - Market opportunity identification
    """)
