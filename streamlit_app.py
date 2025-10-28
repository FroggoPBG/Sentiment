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
    
    .alert-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        margin: 1rem 0;
        border-left: 5px solid #dc3545;
    }
    
    .success-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
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

# Enhanced Legal domain-specific aspects and keywords
LEGAL_ASPECTS = {
    'search': {
        'name': 'Case Search & Research',
        'keywords': ['search', 'case law', 'research', 'precedent', 'citation', 'find', 'query', 'boolean', 'westlaw', 'lexis', 'ai', 'artificial intelligence'],
        'icon': '🔍'
    },
    'citation': {
        'name': 'Citation Management', 
        'keywords': ['citation', 'bluebook', 'shepard', 'cite', 'reference', 'footnote', 'bibliography', 'formatting'],
        'icon': '📚'
    },
    'document': {
        'name': 'Document Management',
        'keywords': ['document', 'pdf', 'file', 'upload', 'storage', 'organize', 'folder', 'brief', 'contract', 'template'],
        'icon': '📄'
    },
    'billing': {
        'name': 'Billing & Time Tracking',
        'keywords': ['billing', 'time', 'hours', 'invoice', 'payment', 'rates', 'expense', 'timesheet', 'accounting'],
        'icon': '💰'
    },
    'ui': {
        'name': 'User Interface',
        'keywords': ['interface', 'design', 'layout', 'navigation', 'menu', 'button', 'screen', 'usability', 'dashboard'],
        'icon': '🖥️'
    },
    'integration': {
        'name': 'Integrations',
        'keywords': ['integration', 'api', 'connect', 'sync', 'export', 'import', 'outlook', 'office', 'calendar'],
        'icon': '🔗'
    },
    'compliance': {
        'name': 'Compliance & Ethics',
        'keywords': ['compliance', 'ethics', 'gdpr', 'privacy', 'security', 'confidential', 'ethical', 'regulation'],
        'icon': '⚖️'
    },
    'performance': {
        'name': 'System Performance',
        'keywords': ['speed', 'slow', 'fast', 'performance', 'load', 'crash', 'bug', 'error', 'lag', 'timeout'],
        'icon': '⚡'
    },
    'pricing': {
        'name': 'Pricing & Value',
        'keywords': ['price', 'pricing', 'cost', 'expensive', 'costly', 'affordable', 'value', 'subscription', 'plan'],
        'icon': '💵'
    },
    'support': {
        'name': 'Customer Support',
        'keywords': ['support', 'help', 'customer service', 'response', 'assistance', 'training', 'onboarding'],
        'icon': '🎧'
    }
}

# Sample improvement milestones for demonstration
IMPROVEMENT_MILESTONES = [
    {"date": "2024-10-01", "description": "Search Performance Upgrade", "target_aspect": "performance"},
    {"date": "2024-09-15", "description": "UI Redesign Rollout", "target_aspect": "ui"},
    {"date": "2024-08-01", "description": "Citation Engine Update", "target_aspect": "citation"},
    {"date": "2024-07-15", "description": "Billing Integration Enhancement", "target_aspect": "billing"}
]

class LegalSentimentAnalyzer:
    def __init__(self):
        # Legal-specific positive terms
        self.legal_positive = [
            'efficient', 'accurate', 'comprehensive', 'reliable', 'professional',
            'streamlined', 'intuitive', 'thorough', 'precise', 'excellent',
            'outstanding', 'impressed', 'helpful', 'valuable', 'essential',
            'game-changer', 'revolutionary', 'innovative', 'cutting-edge',
            'fantastic', 'amazing', 'perfect', 'love', 'great', 'wonderful',
            'brilliant', 'superb', 'exceptional', 'flawless', 'powerful'
        ]
        
        # Legal-specific negative terms
        self.legal_negative = [
            'outdated', 'inaccurate', 'slow', 'confusing', 'unreliable',
            'cumbersome', 'inadequate', 'frustrating', 'disappointing', 'useless',
            'terrible', 'awful', 'horrible', 'broken', 'failed', 'error',
            'bug', 'crash', 'unethical', 'non-compliant', 'biased', 'hate',
            'worst', 'bad', 'poor', 'difficult', 'complicated', 'annoying',
            'laggy', 'glitchy', 'unstable', 'problematic'
        ]
        
        # Intensifiers
        self.intensifiers = ['very', 'extremely', 'absolutely', 'completely', 'totally', 'really', 'incredibly', 'exceptionally']
    
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

def analyze_trends(df, feedback_col, score_col):
    """Analyze trends over time and detect alerts"""
    if 'date' not in df.columns or df.empty:
        return None, []
    
    try:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')
        
        # Monthly trends
        trend = df.groupby('month').agg({
            'sentiment_score': 'mean',
            feedback_col: 'count',
            score_col: lambda x: calculate_nps(pd.DataFrame({score_col: x}), score_col)['score']
        }).reset_index()
        
        trend.columns = ['month', 'avg_sentiment', 'feedback_count', 'nps']
        trend['month'] = trend['month'].dt.to_timestamp()
        
        # Generate alerts
        alerts = []
        if len(trend) >= 2:
            latest = trend.iloc[-1]
            prev = trend.iloc[-2]
            
            # Sentiment drop alert
            if (prev['avg_sentiment'] - latest['avg_sentiment']) > 0.15:
                alerts.append({
                    'type': 'warning',
                    'title': '📉 Sentiment Drop Alert',
                    'message': f"Sentiment dropped significantly in {latest['month'].strftime('%b %Y')} compared to {prev['month'].strftime('%b %Y')} ({prev['avg_sentiment']*100:.1f}% → {latest['avg_sentiment']*100:.1f}%)"
                })
            
            # Feedback volume spike
            if latest['feedback_count'] > prev['feedback_count'] * 1.5:
                alerts.append({
                    'type': 'info',
                    'title': '📈 Feedback Volume Spike',
                    'message': f"Feedback volume increased {((latest['feedback_count']/prev['feedback_count']-1)*100):.0f}% this month. Monitor for emerging issues."
                })
            
            # NPS improvement
            if (latest['nps'] - prev['nps']) > 10:
                alerts.append({
                    'type': 'success',
                    'title': '🎉 NPS Improvement',
                    'message': f"NPS improved by {latest['nps'] - prev['nps']:.1f} points this month!"
                })
        
        return trend, alerts
        
    except Exception as e:
        return None, []

def generate_recommendations(analysis_data, trend_data=None):
    """Generate enhanced AI-powered recommendations"""
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
        
        positive_aspects = [(k, v) for k, v in aspects.items() if v['average_sentiment'] > 0.5 and v['count'] > 3]
        
        # Negative aspect recommendations
        if negative_aspects:
            aspect_id, aspect_data = negative_aspects[0]
            aspect = LEGAL_ASPECTS[aspect_id]
            
            recommendations.append({
                'title': f"🔧 Fix {aspect['name']} Issues",
                'priority': "HIGH" if aspect_data['average_sentiment'] < -0.5 else "MEDIUM",
                'content': f"{aspect['name']} has the most negative sentiment ({aspect_data['average_sentiment']*100:.1f}%). {aspect_data['detractor_count']} detractors mentioned this area.",
                'impact': f"Fixing {aspect['name']} issues could convert up to {aspect_data['detractor_count']} detractors and improve overall NPS by 5-10 points.",
                'actions': [
                    f"Conduct user research on {aspect['name'].lower()}",
                    f"Prioritize {aspect['name'].lower()} in next sprint",
                    f"Create targeted communication about {aspect['name'].lower()} improvements"
                ]
            })
        
        # Positive aspect recommendations
        if positive_aspects:
            aspect_id, aspect_data = positive_aspects[0]
            aspect = LEGAL_ASPECTS[aspect_id]
            
            recommendations.append({
                'title': f"👏 Leverage {aspect['name']} as a Selling Point",
                'priority': "LOW",
                'content': f"{aspect['name']} is consistently praised by users. (Avg sentiment {aspect_data['average_sentiment']*100:.0f}% positive, {aspect_data['promoter_count']} promoter mentions)",
                'impact': "Highlighting this strength can attract new users and reassure existing ones.",
                'actions': [
                    f"Showcase {aspect['name']} in next product newsletter",
                    f"Encourage promoters to give testimonials about {aspect['name']}",
                    f"Create case studies highlighting {aspect['name']} success stories"
                ]
            })
    
    # Trend-based recommendations
    if trend_data and len(trend_data) >= 2:
        latest = trend_data.iloc[-1]
        prev = trend_data.iloc[-2]
        
        if (prev['avg_sentiment'] - latest['avg_sentiment']) > 0.1:
            recommendations.append({
                'title': "⏰ Investigate Recent Sentiment Decline",
                'priority': "HIGH",
                'content': f"Sentiment has dropped {(prev['avg_sentiment'] - latest['avg_sentiment'])*100:.1f} percentage points since last period. This trend needs immediate attention.",
                'impact': "Stopping negative sentiment trends early prevents larger satisfaction issues.",
                'actions': [
                    "Review recent product changes or incidents",
                    "Conduct emergency survey with recent detractors",
                    "Implement immediate fixes for any identified issues"
                ]
            })
    
    # Segment-based recommendations (if available)
    processed_df = analysis_data.get('processed_feedback')
    if processed_df is not None and 'client_type' in processed_df.columns:
        segment_nps = {}
        for ctype, group in processed_df.groupby('client_type'):
            segment_nps[ctype] = calculate_nps(group, processed_df.columns[1])['score']  # Assuming score is second column
        
        if len(segment_nps) > 1:
            worst_segment = min(segment_nps, key=segment_nps.get)
            best_segment = max(segment_nps, key=segment_nps.get)
            
            if segment_nps[worst_segment] < segment_nps[best_segment] - 20:
                recommendations.append({
                    'title': f"🎯 Focus on {worst_segment.capitalize()} Clients",
                    'priority': "MEDIUM",
                    'content': f"{worst_segment.capitalize()} clients have much lower NPS ({segment_nps[worst_segment]:.1f}) vs {best_segment} clients ({segment_nps[best_segment]:.1f}).",
                    'impact': f"Closing the satisfaction gap for {worst_segment} clients could significantly improve overall NPS.",
                    'actions': [
                        f"Review all feedback from {worst_segment} segment",
                        f"Schedule interviews with {worst_segment} clients",
                        f"Develop targeted improvements for {worst_segment} needs"
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
    """Create enhanced sample legal feedback data with dates and segments"""
    base_date = datetime(2024, 6, 1)
    sample_feedback = []
    
    # Generate sample data over 6 months
    feedback_items = [
        {"feedback": "The case search functionality is excellent and very intuitive. Love the Boolean search capabilities and the AI-powered summaries.", "score": 9, "client_type": "small", "plan_tier": "Professional", "industry": "Corporate"},
        {"feedback": "Citation management is terrible. The Bluebook formatting is completely wrong and unreliable. Costs us hours.", "score": 3, "client_type": "large", "plan_tier": "Enterprise", "industry": "Litigation"},
        {"feedback": "Document upload is slow and crashes frequently. Very frustrating for our daily workflow. Performance is awful.", "score": 4, "client_type": "medium", "plan_tier": "Professional", "industry": "Real Estate"},
        {"feedback": "The user interface is clean and professional. Easy to navigate and find what we need. Great design.", "score": 8, "client_type": "solo", "plan_tier": "Solo", "industry": "Family Law"},
        {"feedback": "Billing integration with our time tracking is a game-changer. Saves hours every week. Absolutely brilliant.", "score": 10, "client_type": "large", "plan_tier": "Enterprise", "industry": "Corporate"},
        {"feedback": "Search results are often outdated and inaccurate. Missing recent case law updates. Very disappointing.", "score": 2, "client_type": "medium", "plan_tier": "Professional", "industry": "Criminal"},
        {"feedback": "Overall good product but the performance is slow during peak hours. Could be faster.", "score": 7, "client_type": "small", "plan_tier": "Professional", "industry": "Personal Injury"},
        {"feedback": "Compliance features are comprehensive and help with GDPR requirements. Very valuable for our practice.", "score": 9, "client_type": "large", "plan_tier": "Enterprise", "industry": "Corporate"},
        {"feedback": "The API integration is broken and doesn't sync properly with Outlook. Completely unreliable.", "score": 3, "client_type": "medium", "plan_tier": "Professional", "industry": "Employment"},
        {"feedback": "Great tool for legal research. The AI features are particularly helpful for case analysis.", "score": 9, "client_type": "solo", "plan_tier": "Solo", "industry": "Immigration"},
        {"feedback": "Document management is confusing and not intuitive. Need better organization and folder structure.", "score": 5, "client_type": "small", "plan_tier": "Solo", "industry": "Family Law"},
        {"feedback": "Citation tracking is fantastic. Automatically updates and very accurate. Love this feature.", "score": 8, "client_type": "large", "plan_tier": "Enterprise", "industry": "Litigation"},
        {"feedback": "System crashes during important presentations. Completely unreliable. Performance issues are unacceptable.", "score": 1, "client_type": "medium", "plan_tier": "Professional", "industry": "Corporate"},
        {"feedback": "User interface could be more modern but functionality is solid. Does what we need.", "score": 7, "client_type": "solo", "plan_tier": "Solo", "industry": "Criminal"},
        {"feedback": "The search feature is revolutionary for our practice. Saves tremendous time on research.", "score": 10, "client_type": "large", "plan_tier": "Enterprise", "industry": "Litigation"},
        {"feedback": "Pricing is too expensive for small firms. Need more affordable options for solo practitioners.", "score": 4, "client_type": "solo", "plan_tier": "Solo", "industry": "Family Law"},
        {"feedback": "Customer support is excellent. Quick responses and very helpful training sessions.", "score": 9, "client_type": "medium", "plan_tier": "Professional", "industry": "Real Estate"},
        {"feedback": "AI features are impressive but sometimes give irrelevant results. Needs improvement.", "score": 6, "client_type": "small", "plan_tier": "Professional", "industry": "Personal Injury"},
        {"feedback": "Interface crashes when uploading large documents. Very annoying bug that needs fixing.", "score": 3, "client_type": "large", "plan_tier": "Enterprise", "industry": "Corporate"},
        {"feedback": "Love the comprehensive citation database. Makes research so much easier and faster.", "score": 9, "client_type": "medium", "plan_tier": "Professional", "industry": "Immigration"}
    ]
    
    # Distribute feedback over time with some trending
    for i, item in enumerate(feedback_items):
        # Add random days to base date
        days_offset = (i * 7) + np.random.randint(0, 7)  # Roughly weekly with variation
        item['date'] = base_date + timedelta(days=days_offset)
        sample_feedback.append(item)
    
    # Add some recent feedback with performance improvements (simulating a fix)
    recent_base = datetime(2024, 10, 1)
    recent_items = [
        {"feedback": "Performance has improved significantly! Search is much faster now.", "score": 8, "client_type": "medium", "plan_tier": "Professional", "industry": "Corporate", "date": recent_base + timedelta(days=5)},
        {"feedback": "The new UI update looks great and is more intuitive to use.", "score": 9, "client_type": "small", "plan_tier": "Solo", "industry": "Family Law", "date": recent_base + timedelta(days=10)},
        {"feedback": "System stability has improved. No more crashes during presentations.", "score": 8, "client_type": "large", "plan_tier": "Enterprise", "industry": "Corporate", "date": recent_base + timedelta(days=15)}
    ]
    
    sample_feedback.extend(recent_items)
    
    return pd.DataFrame(sample_feedback)

def calculate_what_if_nps(nps_result, detractor_reduction, conversion_type="passive"):
    """Calculate what-if NPS scenarios"""
    current_dist = nps_result['distribution']
    
    if conversion_type == "passive":
        new_detractors = max(0, current_dist['detractors'] - detractor_reduction)
        new_passives = current_dist['passives'] + detractor_reduction
        new_promoters = current_dist['promoters']
    else:  # promoter
        new_detractors = max(0, current_dist['detractors'] - detractor_reduction)
        new_passives = current_dist['passives']
        new_promoters = current_dist['promoters'] + detractor_reduction
    
    new_total = new_promoters + new_passives + new_detractors
    new_nps = ((new_promoters - new_detractors) / new_total * 100) if new_total > 0 else 0
    
    return round(new_nps, 1)

def analyze_milestone_impact(df, feedback_col, score_col, milestone):
    """Analyze before/after impact of improvement milestones"""
    if 'date' not in df.columns:
        return None
    
    milestone_date = pd.to_datetime(milestone['date'])
    before_df = df[df['date'] < milestone_date]
    after_df = df[df['date'] >= milestone_date]
    
    if before_df.empty or after_df.empty:
        return None
    
    before_nps = calculate_nps(before_df, score_col)['score']
    after_nps = calculate_nps(after_df, score_col)['score']
    
    before_sentiment = before_df['sentiment_score'].mean()
    after_sentiment = after_df['sentiment_score'].mean()
    
    # Analyze specific aspect if applicable
    aspect_improvement = None
    if milestone['target_aspect'] in LEGAL_ASPECTS:
        target_aspect = milestone['target_aspect']
        
        before_aspect_issues = sum(before_df['aspects'].apply(
            lambda aspects: any(a['id'] == target_aspect and a['sentiment'] < -0.1 for a in aspects) if aspects else False
        ))
        after_aspect_issues = sum(after_df['aspects'].apply(
            lambda aspects: any(a['id'] == target_aspect and a['sentiment'] < -0.1 for a in aspects) if aspects else False
        ))
        
        if before_aspect_issues > 0:
            improvement_pct = ((before_aspect_issues - after_aspect_issues) / before_aspect_issues) * 100
            aspect_improvement = {
                'before_issues': before_aspect_issues,
                'after_issues': after_aspect_issues,
                'improvement_pct': improvement_pct
            }
    
    return {
        'before_nps': before_nps,
        'after_nps': after_nps,
        'nps_change': after_nps - before_nps,
        'before_sentiment': before_sentiment * 100,
        'after_sentiment': after_sentiment * 100,
        'sentiment_change': (after_sentiment - before_sentiment) * 100,
        'aspect_improvement': aspect_improvement
    }

# Main Streamlit App
def main():
    # Initialize session state
    if 'selected_aspect' not in st.session_state:
        st.session_state.selected_aspect = "All Aspects"
    
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
            help="CSV should contain columns: 'feedback' (text), 'score' (number), and optionally 'date', 'client_type', 'plan_tier'"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success(f"✅ Loaded {len(df)} feedback entries")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {str(e)}")
    else:
        df = create_sample_data()
        st.sidebar.info("📝 Using sample legal feedback data with time series")
    
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
        
        # Enhanced Filters Section
        st.sidebar.subheader("🔍 Filters")
        
        # Client type filter
        client_filter = "All"
        if 'client_type' in df.columns:
            client_types = ['All'] + sorted(list(df['client_type'].unique()))
            client_filter = st.sidebar.selectbox("Client Type:", client_types)
        
        # Plan tier filter
        plan_filter = "All"
        if 'plan_tier' in df.columns:
            plan_tiers = ['All'] + sorted(list(df['plan_tier'].unique()))
            plan_filter = st.sidebar.selectbox("Plan Tier:", plan_tiers)
        
        # Industry filter
        industry_filter = "All"
        if 'industry' in df.columns:
            industries = ['All'] + sorted(list(df['industry'].unique()))
            industry_filter = st.sidebar.selectbox("Industry:", industries)
        
        # Sentiment filter
        sentiment_filter = st.sidebar.selectbox(
            "Sentiment:",
            ["All", "Positive", "Neutral", "Negative"]
        )
        
        # Aspect filter
        aspect_options = ["All Aspects"] + [v['name'] for v in LEGAL_ASPECTS.values()]
        aspect_filter = st.sidebar.selectbox("Focus on Aspect:", aspect_options)
        
        # Date range filter
        date_filter_enabled = False
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            date_filter_enabled = st.sidebar.checkbox("Filter by Date Range")
            
            if date_filter_enabled:
                min_date = df['date'].min().date()
                max_date = df['date'].max().date()
                
                start_date = st.sidebar.date_input("Start Date:", min_date)
                end_date = st.sidebar.date_input("End Date:", max_date)
        
        # Process data
        analyzer = LegalSentimentAnalyzer()
        
        # Apply filters
        filtered_df = df.copy()
        
        if client_filter != "All" and 'client_type' in df.columns:
            filtered_df = filtered_df[filtered_df['client_type'] == client_filter]
        
        if plan_filter != "All" and 'plan_tier' in df.columns:
            filtered_df = filtered_df[filtered_df['plan_tier'] == plan_filter]
        
        if industry_filter != "All" and 'industry' in df.columns:
            filtered_df = filtered_df[filtered_df['industry'] == industry_filter]
        
        if date_filter_enabled and 'date' in df.columns:
            filtered_df = filtered_df[
                (filtered_df['date'].dt.date >= start_date) & 
                (filtered_df['date'].dt.date <= end_date)
            ]
        
        # Analyze sentiment and aspects
        if not filtered_df.empty:
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
            
            # Apply sentiment filter
            if sentiment_filter != "All":
                filtered_df = filtered_df[filtered_df['sentiment_label'] == sentiment_filter.lower()]
            
            # Apply aspect filter
            if aspect_filter != "All Aspects":
                filtered_df = filtered_df[
                    filtered_df['aspects'].apply(
                        lambda aspect_list: any(aspect_filter.lower() in a['name'].lower() for a in aspect_list) if aspect_list else False
                    )
                ]
            
            if filtered_df.empty:
                st.warning("No data matches the selected filters. Please adjust your filters.")
                return
            
            # Calculate metrics
            nps_result = calculate_nps(filtered_df, score_col)
            avg_sentiment = filtered_df['sentiment_score'].mean()
            
            # Analyze trends and alerts
            trend_data, alerts = analyze_trends(filtered_df, feedback_col, score_col)
            
            # Enhanced aspect data with promoter/detractor tracking
            aspect_data = {}
            for aspect_id in LEGAL_ASPECTS.keys():
                aspect_data[aspect_id] = {
                    'count': 0,
                    'total_sentiment': 0,
                    'average_sentiment': 0,
                    'promoter_count': 0,
                    'detractor_count': 0,
                    'passive_count': 0,
                    'feedback_items': []
                }
            
            for _, row in filtered_df.iterrows():
                # Determine NPS category
                nps_category = None
                if not pd.isna(row[score_col]):
                    try:
                        score_val = float(row[score_col])
                        if score_val >= 9:
                            nps_category = 'promoter'
                        elif score_val >= 7:
                            nps_category = 'passive'
                        elif score_val >= 0:
                            nps_category = 'detractor'
                    except (ValueError, TypeError):
                        # Fallback to sentiment
                        if row['sentiment_label'] == 'positive':
                            nps_category = 'promoter'
                        elif row['sentiment_label'] == 'negative':
                            nps_category = 'detractor'
                        else:
                            nps_category = 'passive'
                
                for aspect in row['aspects']:
                    if aspect['id'] in aspect_data:
                        aspect_data[aspect['id']]['count'] += 1
                        aspect_data[aspect['id']]['total_sentiment'] += aspect['sentiment']
                        aspect_data[aspect['id']]['feedback_items'].append(row)
                        
                        if nps_category == 'promoter':
                            aspect_data[aspect['id']]['promoter_count'] += 1
                        elif nps_category == 'detractor':
                            aspect_data[aspect['id']]['detractor_count'] += 1
                        else:
                            aspect_data[aspect['id']]['passive_count'] += 1
            
            # Calculate averages and percentages
            for aspect_id in aspect_data:
                data = aspect_data[aspect_id]
                data['average_sentiment'] = data['total_sentiment'] / data['count'] if data['count'] > 0 else 0
                if data['count'] > 0:
                    data['pct_promoter_mentions'] = (data['promoter_count'] / data['count']) * 100
                    data['pct_detractor_mentions'] = (data['detractor_count'] / data['count']) * 100
                else:
                    data['pct_promoter_mentions'] = 0
                    data['pct_detractor_mentions'] = 0
            
            # Create analysis object
            analysis_data = {
                'nps': nps_result,
                'aspects': aspect_data,
                'processed_feedback': filtered_df
            }
            
            # Display alerts at the top
            if alerts:
                st.subheader("🚨 Intelligence Alerts")
                for alert in alerts:
                    if alert['type'] == 'warning':
                        st.markdown(f"""
                        <div class="alert-card">
                            <h4>{alert['title']}</h4>
                            <p>{alert['message']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif alert['type'] == 'success':
                        st.markdown(f"""
                        <div class="success-card">
                            <h4>{alert['title']}</h4>
                            <p>{alert['message']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info(f"**{alert['title']}:** {alert['message']}")
            
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
            
            # Segment analysis if available
            if 'client_type' in filtered_df.columns:
                st.subheader("📈 Segment Performance")
                
                # NPS by segment
                segment_nps = {}
                segment_counts = {}
                for ctype, group in filtered_df.groupby('client_type'):
                    segment_nps[ctype] = calculate_nps(group, score_col)['score']
                    segment_counts[ctype] = len(group)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_seg_nps = px.bar(
                        x=list(segment_nps.keys()),
                        y=list(segment_nps.values()),
                        labels={'x': 'Client Type', 'y': 'NPS'},
                        title="NPS by Client Segment",
                        color=list(segment_nps.values()),
                        color_continuous_scale=['red', 'yellow', 'green']
                    )
                    st.plotly_chart(fig_seg_nps, use_container_width=True)
                
                with col2:
                    avg_sent_by_type = filtered_df.groupby('client_type')['sentiment_score'].mean() * 100
                    fig_seg_sent = px.bar(
                        x=avg_sent_by_type.index,
                        y=avg_sent_by_type.values,
                        labels={'x': 'Client Type', 'y': 'Avg Sentiment %'},
                        title="Sentiment by Client Segment",
                        color=avg_sent_by_type.values,
                        color_continuous_scale=['red', 'yellow', 'green']
                    )
                    st.plotly_chart(fig_seg_sent, use_container_width=True)
            
            # Tabs for detailed analysis
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🔍 Aspect Analysis", "📈 Sentiment Trends", "🎯 What-If Analysis", "💡 AI Recommendations", "📊 Improvement Impact", "📝 Feedback Explorer"])
            
            with tab1:
                st.subheader("Legal Aspect-Based Sentiment Analysis")
                
                # Enhanced aspect sentiment chart with promoter/detractor breakdown
                aspects_with_data = [(k, v) for k, v in aspect_data.items() if v['count'] > 0]
                
                if aspects_with_data:
                    aspect_names = [LEGAL_ASPECTS[k]['name'] for k, _ in aspects_with_data]
                    aspect_sentiments = [v['average_sentiment'] * 100 for _, v in aspects_with_data]
                    promoter_counts = [v['promoter_count'] for _, v in aspects_with_data]
                    detractor_counts = [v['detractor_count'] for _, v in aspects_with_data]
                    
                    # Stacked bar chart for promoter vs detractor mentions
                    fig_aspects = go.Figure(data=[
                        go.Bar(name='Promoter Mentions', x=aspect_names, y=promoter_counts, marker_color='seagreen'),
                        go.Bar(name='Detractor Mentions', x=aspect_names, y=detractor_counts, marker_color='crimson')
                    ])
                    fig_aspects.update_layout(
                        barmode='group',
                        title="Promoter vs Detractor Mentions by Aspect",
                        height=500,
                        xaxis_title="Legal Aspects",
                        yaxis_title="Mention Count"
                    )
                    st.plotly_chart(fig_aspects, use_container_width=True)
                    
                    # Sentiment score chart
                    fig_sentiment = px.bar(
                        x=aspect_names,
                        y=aspect_sentiments,
                        title="Average Sentiment Score by Aspect",
                        labels={'x': 'Legal Aspects', 'y': 'Sentiment Score (%)'},
                        color=aspect_sentiments,
                        color_continuous_scale=['red', 'yellow', 'green']
                    )
                    st.plotly_chart(fig_sentiment, use_container_width=True)
                    
                    # Aspect details with enhanced metrics
                    st.subheader("📋 Detailed Aspect Analysis")
                    for aspect_id, aspect_data_item in aspects_with_data:
                        aspect = LEGAL_ASPECTS[aspect_id]
                        sentiment_color = "🟢" if aspect_data_item['average_sentiment'] > 0.1 else "🔴" if aspect_data_item['average_sentiment'] < -0.1 else "🟡"
                        
                        with st.expander(f"{sentiment_color} {aspect['icon']} {aspect['name']} - {aspect_data_item['average_sentiment']*100:.1f}% sentiment"):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Mentions", aspect_data_item['count'])
                            with col2:
                                st.metric("Promoter Mentions", f"{aspect_data_item['promoter_count']} ({aspect_data_item['pct_promoter_mentions']:.1f}%)")
                            with col3:
                                st.metric("Detractor Mentions", f"{aspect_data_item['detractor_count']} ({aspect_data_item['pct_detractor_mentions']:.1f}%)")
                            with col4:
                                st.metric("Avg Sentiment", f"{aspect_data_item['average_sentiment']*100:.1f}%")
                            
                            # Show sample feedback for this aspect
                            if aspect_data_item['feedback_items']:
                                st.write("**Sample Feedback:**")
                                sample_feedback = aspect_data_item['feedback_items'][:3]
                                for item in sample_feedback:
                                    sentiment_emoji = "😊" if item['sentiment_score'] > 0.1 else "😞" if item['sentiment_score'] < -0.1 else "😐"
                                    st.write(f"{sentiment_emoji} *{item[feedback_col][:200]}...*")
                                
                                if st.button(f"Show all {aspect['name']} feedback", key=f"show_all_{aspect_id}"):
                                    st.write(f"**All {aspect['name']} Feedback:**")
                                    for item in aspect_data_item['feedback_items']:
                                        sentiment_emoji = "😊" if item['sentiment_score'] > 0.1 else "😞" if item['sentiment_score'] < -0.1 else "😐"
                                        st.write(f"- {sentiment_emoji} {item[feedback_col]} (Score: {item[score_col]}, Sentiment: {item['sentiment_score']:.2f})")
            
            with tab2:
                st.subheader("📈 Sentiment & NPS Trends")
                
                if trend_data is not None and len(trend_data) > 1:
                    # Combined trend chart
                    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Sentiment trend
                    fig_trend.add_trace(
                        go.Scatter(
                            x=trend_data['month'],
                            y=trend_data['avg_sentiment'] * 100,
                            mode='lines+markers',
                            name='Avg Sentiment (%)',
                            line=dict(color='#4B9CD3', width=3),
                            marker=dict(size=8)
                        ),
                        secondary_y=False
                    )
                    
                    # NPS trend
                    fig_trend.add_trace(
                        go.Scatter(
                            x=trend_data['month'],
                            y=trend_data['nps'],
                            mode='lines+markers',
                            name='NPS',
                            line=dict(color='#E74C3C', width=3),
                            marker=dict(size=8)
                        ),
                        secondary_y=False
                    )
                    
                    # Feedback volume
                    fig_trend.add_trace(
                        go.Bar(
                            x=trend_data['month'],
                            y=trend_data['feedback_count'],
                            name='Feedback Volume',
                            marker_color='rgba(176, 190, 197, 0.4)',
                            opacity=0.6
                        ),
                        secondary_y=True
                    )
                    
                    # Add milestone annotations
                    for milestone in IMPROVEMENT_MILESTONES:
                        milestone_date = pd.to_datetime(milestone['date'])
                        if milestone_date >= trend_data['month'].min() and milestone_date <= trend_data['month'].max():
                            fig_trend.add_vline(
                                x=milestone_date,
                                line_dash="dash",
                                line_color="green",
                                annotation_text=milestone['description'],
                                annotation_position="top"
                            )
                    
                    fig_trend.update_xaxes(title_text="Month")
                    fig_trend.update_yaxes(title_text="Sentiment (%) / NPS", secondary_y=False)
                    fig_trend.update_yaxes(title_text="Feedback Count", secondary_y=True)
                    fig_trend.update_layout(title="Sentiment & NPS Trends Over Time", height=500)
                    
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # Trend projections
                    if len(trend_data) >= 3:
                        st.subheader("🔮 Trend Projections")
                        
                        # Simple linear projection
                        recent_sentiment_change = trend_data['avg_sentiment'].iloc[-1] - trend_data['avg_sentiment'].iloc[-2]
                        recent_nps_change = trend_data['nps'].iloc[-1] - trend_data['nps'].iloc[-2]
                        
                        projected_sentiment = (trend_data['avg_sentiment'].iloc[-1] + recent_sentiment_change) * 100
                        projected_nps = trend_data['nps'].iloc[-1] + recent_nps_change
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Projected Next Month Sentiment", f"{projected_sentiment:.1f}%", f"{recent_sentiment_change*100:.1f}%")
                        with col2:
                            st.metric("Projected Next Month NPS", f"{projected_nps:.1f}", f"{recent_nps_change:.1f}")
                
                else:
                    # Static sentiment distribution for non-time series data
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
                st.subheader("🎯 What-If Scenario Analysis")
                
                # Find the worst performing aspect for what-if analysis
                negative_aspects = [(k, v) for k, v in aspect_data.items() if v['average_sentiment'] < -0.1 and v['detractor_count'] > 0]
                negative_aspects.sort(key=lambda x: x[1]['average_sentiment'])
                
                if negative_aspects:
                    worst_aspect_id, worst_data = negative_aspects[0]
                    worst_aspect = LEGAL_ASPECTS[worst_aspect_id]
                    
                    st.write(f"**Scenario: Improving {worst_aspect['name']}**")
                    st.write(f"Current situation: {worst_data['detractor_count']} detractors mentioned {worst_aspect['name']} issues")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        potential_recovery = st.slider(
                            f"How many {worst_aspect['name']} detractors could we convert?",
                            0, worst_data['detractor_count'], worst_data['detractor_count'] // 2,
                            key="detractor_recovery"
                        )
                        
                        conversion_type = st.radio(
                            "Convert detractors to:",
                            ["Passive (7-8)", "Promoter (9-10)"],
                            key="conversion_type"
                        )
                    
                    with col2:
                        if potential_recovery > 0:
                            conv_type = "passive" if "Passive" in conversion_type else "promoter"
                            new_nps = calculate_what_if_nps(nps_result, potential_recovery, conv_type)
                            
                            improvement = new_nps - nps_result['score']
                            
                            st.metric(
                                "Projected NPS",
                                f"{new_nps}",
                                f"+{improvement:.1f}",
                                delta_color="normal"
                            )
                            
                            # Calculate potential business impact
                            if improvement > 0:
                                st.write("**Estimated Business Impact:**")
                                st.write(f"• NPS improvement: +{improvement:.1f} points")
                                st.write(f"• Potential churn reduction: {improvement * 0.5:.1f}%")
                                st.write(f"• Estimated revenue protection: ${improvement * 10000:.0f}")
                
                # Additional scenario analysis
                st.subheader("📊 Additional Scenarios")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Scenario: Overall 10% Sentiment Improvement**")
                    
                    # Calculate how many people would move categories
                    current_negative = len(filtered_df[filtered_df['sentiment_label'] == 'negative'])
                    improved_negative = int(current_negative * 0.9)  # 10% improvement
                    
                    # Simulate moving some negatives to neutral
                    simulated_nps = calculate_what_if_nps(nps_result, current_negative - improved_negative, "passive")
                    
                    st.metric("Projected NPS", f"{simulated_nps}", f"+{simulated_nps - nps_result['score']:.1f}")
                
                with col2:
                    st.write("**Scenario: Convert Top Passives to Promoters**")
                    
                    # Find passives and convert half to promoters
                    passives_to_convert = nps_result['distribution']['passives'] // 2
                    promoter_nps = calculate_what_if_nps(nps_result, -passives_to_convert, "promoter")  # Negative because we're adding promoters
                    
                    st.metric("Projected NPS", f"{promoter_nps}", f"+{promoter_nps - nps_result['score']:.1f}")
            
            with tab4:
                st.subheader("💡 AI-Generated Recommendations")
                
                recommendations = generate_recommendations(analysis_data, trend_data)
                
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
                
                # Download recommendations
                if recommendations:
                    report_lines = [
                        "LEGAL FEEDBACK INTELLIGENCE HUB - RECOMMENDATIONS REPORT",
                        "=" * 60,
                        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        f"Analysis Period: {len(filtered_df)} feedback items",
                        f"Current NPS: {nps_result['score']}",
                        f"Average Sentiment: {avg_sentiment*100:.1f}%",
                        "",
                        "RECOMMENDATIONS:",
                        "-" * 30
                    ]
                    
                    for i, rec in enumerate(recommendations, 1):
                        report_lines.extend([
                            f"{i}. {rec['title']} ({rec['priority']} PRIORITY)",
                            f"   Analysis: {rec['content']}",
                            f"   Impact: {rec['impact']}",
                            "   Actions:",
                            *[f"   • {action}" for action in rec['actions']],
                            ""
                        ])
                    
                    st.download_button(
                        "📄 Download Recommendations Report",
                        "\n".join(report_lines),
                        "recommendations_report.txt",
                        "text/plain"
                    )
            
            with tab5:
                st.subheader("📊 Improvement Impact Analysis")
                
                if 'date' in filtered_df.columns:
                    st.write("**Before/After Analysis of Recent Improvements**")
                    
                    # Analyze each milestone
                    for milestone in IMPROVEMENT_MILESTONES:
                        impact = analyze_milestone_impact(filtered_df, feedback_col, score_col, milestone)
                        
                        if impact:
                            with st.expander(f"📈 {milestone['description']} - {milestone['date']}"):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        "NPS Change",
                                        f"{impact['after_nps']:.1f}",
                                        f"{impact['nps_change']:+.1f}",
                                        delta_color="normal" if impact['nps_change'] >= 0 else "inverse"
                                    )
                                
                                with col2:
                                    st.metric(
                                        "Sentiment Change",
                                        f"{impact['after_sentiment']:.1f}%",
                                        f"{impact['sentiment_change']:+.1f}%",
                                        delta_color="normal" if impact['sentiment_change'] >= 0 else "inverse"
                                    )
                                
                                with col3:
                                    if impact['aspect_improvement']:
                                        asp_imp = impact['aspect_improvement']
                                        st.metric(
                                            f"{LEGAL_ASPECTS[milestone['target_aspect']]['name']} Issues",
                                            f"{asp_imp['after_issues']}",
                                            f"-{asp_imp['before_issues'] - asp_imp['after_issues']} ({asp_imp['improvement_pct']:.1f}% reduction)"
                                        )
                                
                                # Success story generation
                                if impact['nps_change'] > 0 or impact['sentiment_change'] > 0:
                                    st.success(f"✅ **Success Story:** {milestone['description']} resulted in measurable improvement!")
                                    
                                    success_points = []
                                    if impact['nps_change'] > 0:
                                        success_points.append(f"NPS increased by {impact['nps_change']:.1f} points")
                                    if impact['sentiment_change'] > 0:
                                        success_points.append(f"Sentiment improved by {impact['sentiment_change']:.1f}%")
                                    if impact['aspect_improvement'] and impact['aspect_improvement']['improvement_pct'] > 0:
                                        success_points.append(f"{impact['aspect_improvement']['improvement_pct']:.1f}% reduction in related complaints")
                                    
                                    st.write("**Key Results:**")
                                    for point in success_points:
                                        st.write(f"• {point}")
                else:
                    st.info("📅 Upload data with dates to see improvement impact analysis")
                    
                    # Show general improvement tracking
                    st.write("**Continuous Improvement Framework**")
                    st.write("Track the impact of your improvements with these metrics:")
                    
                    improvement_metrics = [
                        "📊 NPS before vs after implementation",
                        "📈 Sentiment score changes by aspect",
                        "📉 Reduction in specific complaint types",
                        "👥 Client satisfaction by segment",
                        "🔄 Feedback volume and velocity changes"
                    ]
                    
                    for metric in improvement_metrics:
                        st.write(f"• {metric}")
            
            with tab6:
                st.subheader("📝 Interactive Feedback Explorer")
                
                # Enhanced filtering options
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    explorer_sentiment = st.selectbox(
                        "Filter by Sentiment:",
                        ["All", "Positive", "Negative", "Neutral"],
                        key="explorer_sentiment"
                    )
                
                with col2:
                    explorer_aspect = st.selectbox(
                        "Filter by Aspect:",
                        ["All"] + [v['name'] for v in LEGAL_ASPECTS.values()],
                        key="explorer_aspect"
                    )
                
                with col3:
                    sort_by = st.selectbox(
                        "Sort by:",
                        ["Sentiment Score", "Confidence", "Date (newest first)", "Score"],
                        key="explorer_sort"
                    )
                
                # Search functionality
                search_term = st.text_input(
                    "🔍 Search feedback text:",
                    placeholder="Enter keywords to search...",
                    key="feedback_search"
                )
                
                # Apply explorer filters
                explorer_df = filtered_df.copy()
                
                if explorer_sentiment != "All":
                    explorer_df = explorer_df[explorer_df['sentiment_label'] == explorer_sentiment.lower()]
                
                if explorer_aspect != "All":
                    explorer_df = explorer_df[
                        explorer_df['aspects'].apply(
                            lambda aspects: any(explorer_aspect.lower() in a['name'].lower() for a in aspects) if aspects else False
                        )
                    ]
                
                if search_term:
                    explorer_df = explorer_df[
                        explorer_df[feedback_col].str.contains(search_term, case=False, na=False)
                    ]
                
                # Sort the data
                if sort_by == "Sentiment Score":
                    explorer_df = explorer_df.sort_values('sentiment_score', ascending=False)
                elif sort_by == "Confidence":
                    explorer_df = explorer_df.sort_values('sentiment_confidence', ascending=False)
                elif sort_by == "Date (newest first)" and 'date' in explorer_df.columns:
                    explorer_df = explorer_df.sort_values('date', ascending=False)
                elif sort_by == "Score":
                    explorer_df = explorer_df.sort_values(score_col, ascending=False)
                
                # Display results
                st.write(f"**Found {len(explorer_df)} matching feedback items**")
                
                # Pagination
                items_per_page = 10
                total_pages = (len(explorer_df) - 1) // items_per_page + 1
                
                if total_pages > 1:
                    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1) - 1
                    start_idx = page * items_per_page
                    end_idx = start_idx + items_per_page
                    page_df = explorer_df.iloc[start_idx:end_idx]
                else:
                    page_df = explorer_df.head(items_per_page)
                
                # Display feedback items
                for idx, (_, row) in enumerate(page_df.iterrows()):
                    sentiment_emoji = "😊" if row['sentiment_score'] > 0.1 else "😞" if row['sentiment_score'] < -0.1 else "😐"
                    confidence_stars = "⭐" * min(5, int(row['sentiment_confidence'] * 5))
                    
                    # Determine urgency
                    urgency = ""
                    if row['sentiment_score'] < -0.5:
                        urgency = "🚨 URGENT"
                    elif row['sentiment_score'] < -0.2:
                        urgency = "⚠️ NEEDS ATTENTION"
                    
                    date_str = ""
                    if 'date' in row and not pd.isna(row['date']):
                        date_str = f" | {pd.to_datetime(row['date']).strftime('%Y-%m-%d')}"
                    
                    with st.expander(f"{sentiment_emoji} Score: {row[score_col]} | Sentiment: {row['sentiment_score']*100:.1f}% | {confidence_stars}{date_str} {urgency}"):
                        st.write(f"**Feedback:** {row[feedback_col]}")
                        
                        # Show additional metadata
                        metadata_cols = []
                        if 'client_type' in row:
                            metadata_cols.append(f"**Client:** {row['client_type']}")
                        if 'plan_tier' in row:
                            metadata_cols.append(f"**Plan:** {row['plan_tier']}")
                        if 'industry' in row:
                            metadata_cols.append(f"**Industry:** {row['industry']}")
                        
                        if metadata_cols:
                            st.write(" | ".join(metadata_cols))
                        
                        # Show detected aspects
                        if row['aspects']:
                            st.write("**Detected Aspects:**")
                            for aspect in row['aspects']:
                                aspect_sentiment_emoji = "😊" if aspect['sentiment'] > 0.1 else "😞" if aspect['sentiment'] < -0.1 else "😐"
                                st.write(f"• {aspect['icon']} {aspect['name']}: {aspect_sentiment_emoji} {aspect['sentiment']*100:.1f}%")
                        
                        # AI Response suggestion for negative feedback
                        if row['sentiment_score'] < -0.2:
                            if st.button(f"💬 Generate Response Suggestion", key=f"response_{idx}"):
                                # Simulated AI response (in real implementation, you'd use OpenAI API)
                                response_templates = [
                                    f"Thank you for your feedback about {row[feedback_col][:50]}... We take all client concerns seriously and are actively working to address this issue. Our team will follow up with you directly to discuss solutions.",
                                    f"We appreciate you bringing this to our attention. Your feedback helps us improve our service. We're implementing changes to address the concerns you've raised and will keep you updated on our progress.",
                                    f"We're sorry to hear about your experience. This doesn't meet our standards, and we're taking immediate action to resolve these issues. A member of our team will contact you within 24 hours to discuss next steps."
                                ]
                                
                                import random
                                suggested_response = random.choice(response_templates)
                                st.write(f"**💬 Suggested Response:**")
                                st.write(f"*{suggested_response}*")
                
                # Summary insights for current view
                if not explorer_df.empty:
                    st.subheader("📋 Current View Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_sentiment_view = explorer_df['sentiment_score'].mean()
                        st.metric("Avg Sentiment", f"{avg_sentiment_view*100:.1f}%")
                    
                    with col2:
                        urgent_count_view = len(explorer_df[explorer_df['sentiment_score'] < -0.5])
                        st.metric("Urgent Items", urgent_count_view)
                    
                    with col3:
                        if score_col in explorer_df.columns:
                            avg_score_view = explorer_df[score_col].mean()
                            st.metric("Avg Score", f"{avg_score_view:.1f}")
                    
                    with col4:
                        total_view = len(explorer_df)
                        st.metric("Total Items", total_view)
        
        else:
            st.warning("No data available after applying filters. Please adjust your filter criteria.")
    
    else:
        st.info("👆 Please upload a CSV file or select 'Use Sample Data' to begin analysis.")
        
        st.markdown("""
        ### 📋 Enhanced CSV Format Requirements
        
        Your CSV file should contain at least these columns:
        - **feedback/comment/text**: The actual feedback text
        - **score/rating/nps**: Numeric score (0-10) or text (promoter/passive/detractor)
        
        Optional columns for enhanced analysis:
        - **date**: Date of feedback (YYYY-MM-DD format)
        - **client_type**: Type of client (solo, small, medium, large)
        - **plan_tier**: Subscription tier (Solo, Professional, Enterprise)
        - **industry**: Practice area (Corporate, Litigation, Family Law, etc.)
        
        ### 🎯 Enhanced Features
        
        This Legal Feedback Intelligence Hub now provides:
        
        **🔍 Advanced Analytics**
        - Aspect-based sentiment with promoter/detractor tracking
        - Multi-dimensional filtering and segmentation
        - Time-series trend analysis with alerts
        - What-if scenario modeling
        
        **💡 AI-Powered Insights**
        - Smart recommendations based on data patterns
        - Automated improvement impact tracking
        - Predictive trend analysis
        - Response suggestion generation
        
        **📊 Executive Dashboards**
        - Real-time alerting system
        - Segment performance comparison
        - Interactive feedback exploration
        - Downloadable reports and recommendations
        
        **🔄 Continuous Improvement**
        - Before/after milestone analysis
        - Success story documentation
        - Trend-based recommendations
        - Business impact quantification
        """)

if __name__ == "__main__":
    main()
