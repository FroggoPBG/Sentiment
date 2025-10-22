import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime, timedelta
import json
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

def simple_sentiment_analysis(text):
    """Enhanced sentiment analysis with more nuanced scoring"""
    if not isinstance(text, str) or len(text.strip()) < 3:
        return 0
    
    text_lower = text.lower()
    
    # Enhanced keyword sets with weights
    positive_words = {
        # High impact positive
        'excellent': 2.0, 'outstanding': 2.0, 'amazing': 2.0, 'fantastic': 2.0, 
        'superb': 2.0, 'brilliant': 2.0, 'perfect': 2.0, 'awesome': 2.0,
        # Medium impact positive
        'great': 1.5, 'wonderful': 1.5, 'good': 1.0, 'helpful': 1.0, 
        'useful': 1.0, 'easy': 1.0, 'simple': 1.0, 'comprehensive': 1.5,
        'reliable': 1.5, 'accurate': 1.5, 'fast': 1.0, 'efficient': 1.5,
        'convenient': 1.0, 'valuable': 1.5, 'effective': 1.5, 'powerful': 1.5,
        'user-friendly': 1.5, 'intuitive': 1.5, 'satisfied': 1.0, 'love': 2.0,
        'recommend': 1.5, 'appreciate': 1.0, 'trustworthy': 1.5, 'authoritative': 1.5
    }
    
    negative_words = {
        # High impact negative
        'terrible': -2.0, 'awful': -2.0, 'horrible': -2.0, 'hate': -2.0,
        'useless': -2.0, 'broken': -2.0, 'disappointing': -1.5,
        # Medium impact negative
        'poor': -1.5, 'bad': -1.5, 'slow': -1.0, 'expensive': -1.0,
        'difficult': -1.0, 'hard': -1.0, 'complicated': -1.0, 'confusing': -1.0,
        'frustrating': -1.5, 'unreliable': -1.5, 'inaccurate': -1.5,
        'outdated': -1.0, 'inadequate': -1.0, 'lacking': -1.0, 'missing': -1.0,
        'faulty': -1.5, 'unhelpful': -1.0, 'avoid': -1.5, 'overpriced': -1.0
    }
    
    # Calculate weighted sentiment
    sentiment_score = 0
    word_count = 0
    
    for word, weight in positive_words.items():
        if word in text_lower:
            sentiment_score += weight
            word_count += 1
    
    for word, weight in negative_words.items():
        if word in text_lower:
            sentiment_score += weight
            word_count += 1
    
    # Normalize by word count
    if word_count > 0:
        sentiment_score = sentiment_score / word_count
    
    return max(-1, min(1, sentiment_score))

def extract_nps_category_from_sentiment(text):
    """Enhanced NPS categorization with confidence scoring"""
    if not isinstance(text, str) or len(text.strip()) < 3:
        return 'Unknown', 0
    
    text_clean = text.replace('"', '').replace('/', '').strip()
    if text_clean.lower() in ['', '.', 'no comment', '/', 'ok']:
        return 'Unknown', 0
    
    sentiment = simple_sentiment_analysis(text_clean)
    
    # Enhanced thresholds with confidence
    if sentiment > 0.3:
        confidence = min(0.95, 0.5 + abs(sentiment))
        return 'Promoter', confidence
    elif sentiment < -0.2:
        confidence = min(0.95, 0.5 + abs(sentiment))
        return 'Detractor', confidence
    elif sentiment > 0:
        confidence = 0.6
        return 'Passive', confidence
    else:
        confidence = 0.7
        return 'Passive', confidence

def calculate_nps_from_categories(categories):
    """Enhanced NPS calculation with confidence intervals"""
    total = len(categories)
    if total == 0:
        return 0, {}, {}
    
    category_counts = Counter([cat[0] if isinstance(cat, tuple) else cat for cat in categories])
    
    promoters = category_counts.get('Promoter', 0)
    passives = category_counts.get('Passive', 0)
    detractors = category_counts.get('Detractor', 0)
    unknown = category_counts.get('Unknown', 0)
    
    known_total = promoters + passives + detractors
    if known_total == 0:
        return 0, category_counts, {}
    
    promoter_pct = (promoters / known_total) * 100
    detractor_pct = (detractors / known_total) * 100
    passive_pct = (passives / known_total) * 100
    
    nps = promoter_pct - detractor_pct
    
    # Calculate confidence interval (simplified)
    if known_total > 10:
        margin_error = 1.96 * np.sqrt((promoter_pct * (100 - promoter_pct) + detractor_pct * (100 - detractor_pct)) / known_total)
    else:
        margin_error = 20  # High uncertainty for small samples
    
    confidence_stats = {
        'nps_lower': nps - margin_error,
        'nps_upper': nps + margin_error,
        'promoter_pct': promoter_pct,
        'passive_pct': passive_pct,
        'detractor_pct': detractor_pct
    }
    
    return nps, category_counts, confidence_stats

def extract_advanced_themes(comments, nps_category=None, min_frequency=2):
    """Advanced theme extraction with sentiment context and business impact"""
    if not comments:
        return {}, {}
    
    valid_comments = []
    for comment in comments:
        if isinstance(comment, str):
            clean_comment = comment.replace('"', '').replace('/', '').strip()
            if len(clean_comment) > 3 and clean_comment.lower() not in ['', '.', 'no comment', '/', 'ok']:
                valid_comments.append(clean_comment)
    
    if not valid_comments:
        return {}, {}
    
    # Enhanced theme patterns with business impact categories
    theme_patterns = {
        # User Experience (High Business Impact)
        'user_experience_positive': {
            'patterns': [
                r'\b(user.?friendly|user.?friend|easy.?to.?use|intuitive|simple.?to.?use)\b',
                r'\bfriendly\b.*\b(interface|platform|system)\b',
                r'\b(interface|platform|system)\b.*\bfriendly\b'
            ],
            'business_impact': 'High',
            'category': 'User Experience',
            'sentiment': 'Positive'
        },
        'navigation_ease': {
            'patterns': [
                r'\b(easy|simple)\b.*\b(navigat|search|find)\b',
                r'\b(navigat|search|find)\b.*\b(easy|simple)\b',
                r'\beasy.?to.?navigat\b'
            ],
            'business_impact': 'High',
            'category': 'User Experience',
            'sentiment': 'Positive'
        },
        'interface_problems': {
            'patterns': [
                r'\bnot.*user.?friendly\b|\binterface.*not.*friendly\b',
                r'\bdifficult.*\b(use|navigat)\b',
                r'\bnot.*easy.*use\b'
            ],
            'business_impact': 'High',
            'category': 'User Experience',
            'sentiment': 'Negative'
        },
        
        # Product Quality (High Business Impact)
        'comprehensive_database': {
            'patterns': [
                r'\b(comprehensive|extensive|wide|broad|large|huge)\b.*\b(database|resources|content|coverage|materials)\b',
                r'\b(database|resources|content|coverage|materials)\b.*\b(comprehensive|extensive|wide|broad|large|huge)\b'
            ],
            'business_impact': 'High',
            'category': 'Product Quality',
            'sentiment': 'Positive'
        },
        'content_accuracy': {
            'patterns': [
                r'\b(reliable|accurate|trustworthy|authoritative|trusted)\b',
                r'\b(accuracy|reliability|trust)\b'
            ],
            'business_impact': 'High',
            'category': 'Product Quality',
            'sentiment': 'Positive'
        },
        'content_currency': {
            'patterns': [
                r'\b(up.?to.?date|current|latest|updated|recent)\b',
                r'\bupdated?\b'
            ],
            'business_impact': 'Medium',
            'category': 'Product Quality',
            'sentiment': 'Positive'
        },
        
        # Search & Technology (Medium Business Impact)
        'search_effectiveness': {
            'patterns': [
                r'\b(good|excellent|great|powerful|effective)\b.*\b(search|engine|searching)\b',
                r'\b(search|engine|searching)\b.*\b(good|excellent|great|powerful|effective)\b'
            ],
            'business_impact': 'Medium',
            'category': 'Technology',
            'sentiment': 'Positive'
        },
        'search_issues': {
            'patterns': [
                r'\bsearch.*\b(not|poor|bad|irrelevant|inaccurate)\b',
                r'\b(not|poor|bad|irrelevant|inaccurate).*search\b',
                r'\bsearch.*result.*not.*relevant\b'
            ],
            'business_impact': 'Medium',
            'category': 'Technology',
            'sentiment': 'Negative'
        },
        'ai_features': {
            'patterns': [
                r'\bai\b|\bartificial.?intelligence\b|\bai.?driven\b|\bai.?tools?\b',
                r'\banalytic|insight\b'
            ],
            'business_impact': 'Medium',
            'category': 'Technology',
            'sentiment': 'Positive'
        },
        'ai_limitations': {
            'patterns': [
                r'\bai.*\b(primitive|basic|limited|poor)\b',
                r'\b(primitive|basic|limited|poor).*ai\b'
            ],
            'business_impact': 'Medium',
            'category': 'Technology',
            'sentiment': 'Negative'
        },
        
        # Performance (Medium Business Impact)
        'performance_positive': {
            'patterns': [
                r'\b(fast|quick|efficient|speedy)\b',
                r'\bload.*\b(quick|fast)\b'
            ],
            'business_impact': 'Medium',
            'category': 'Performance',
            'sentiment': 'Positive'
        },
        'performance_issues': {
            'patterns': [
                r'\b(slow|sluggish|delayed)\b.*\b(search|engine|loading|response|speed)\b',
                r'\b(search|engine|loading|response|speed)\b.*\b(slow|sluggish|delayed)\b',
                r'\btakes.*long\b|\bslow.*load\b'
            ],
            'business_impact': 'Medium',
            'category': 'Performance',
            'sentiment': 'Negative'
        },
        
        # Pricing (High Business Impact)
        'pricing_concerns': {
            'patterns': [
                r'\b(expensive|costly|overpriced|pricey)\b',
                r'\bpric(e|ing)\b.*\b(high|expensive|too.?much)\b',
                r'\btoo.?expensive\b'
            ],
            'business_impact': 'High',
            'category': 'Pricing',
            'sentiment': 'Negative'
        },
        'value_proposition': {
            'patterns': [
                r'\b(value|worth|reasonable)\b.*\b(price|cost|money)\b',
                r'\b(price|cost|money)\b.*\b(value|worth|reasonable)\b'
            ],
            'business_impact': 'High',
            'category': 'Pricing',
            'sentiment': 'Positive'
        },
        
        # Support & Service (Medium Business Impact)
        'customer_service_positive': {
            'patterns': [
                r'\b(good|excellent|great)\b.*\b(customer|support|service)\b',
                r'\b(customer|support|service)\b.*\b(good|excellent|great|responsive)\b',
                r'\bresponsive.*support\b'
            ],
            'business_impact': 'Medium',
            'category': 'Support',
            'sentiment': 'Positive'
        },
        'customer_service_negative': {
            'patterns': [
                r'\b(poor|bad|terrible)\b.*\b(customer|support|service)\b',
                r'\b(customer|support|service)\b.*\b(poor|bad|terrible|unresponsive)\b'
            ],
            'business_impact': 'Medium',
            'category': 'Support',
            'sentiment': 'Negative'
        },
        
        # Specific Features (Medium Business Impact)
        'halsbury_content': {
            'patterns': [
                r'\bhalsbury\b|\bannotated.?ordinanc\b|\baohk\b',
                r'\bhalsbury.?law\b'
            ],
            'business_impact': 'Medium',
            'category': 'Content',
            'sentiment': 'Positive'
        },
        'legal_research_capability': {
            'patterns': [
                r'\blegal.?research\b|\bresearch\b.*\b(legal|law)\b',
                r'\bresearch.*\b(tool|capability|function)\b'
            ],
            'business_impact': 'High',
            'category': 'Core Function',
            'sentiment': 'Positive'
        }
    }
    
    # Count theme occurrences with metadata
    theme_counts = {}
    theme_metadata = {}
    
    for comment in valid_comments:
        comment_lower = comment.lower()
        for theme_name, theme_data in theme_patterns.items():
            for pattern in theme_data['patterns']:
                if re.search(pattern, comment_lower, re.IGNORECASE):
                    theme_counts[theme_name] = theme_counts.get(theme_name, 0) + 1
                    theme_metadata[theme_name] = {
                        'business_impact': theme_data['business_impact'],
                        'category': theme_data['category'],
                        'sentiment': theme_data['sentiment']
                    }
                    break
    
    # Filter by minimum frequency
    filtered_themes = {theme: count for theme, count in theme_counts.items() 
                      if count >= min_frequency}
    filtered_metadata = {theme: meta for theme, meta in theme_metadata.items() 
                        if theme in filtered_themes}
    
    return filtered_themes, filtered_metadata

def generate_business_recommendations(themes_data, nps_score, category_counts):
    """Generate actionable business recommendations based on analysis"""
    recommendations = {
        'immediate_actions': [],
        'strategic_initiatives': [],
        'competitive_advantages': [],
        'risk_mitigation': []
    }
    
    themes, metadata = themes_data
    total_responses = sum(category_counts.values())
    
    # Immediate Actions (High Impact, Negative Sentiment)
    high_impact_negative = [theme for theme, meta in metadata.items() 
                           if meta['business_impact'] == 'High' and meta['sentiment'] == 'Negative']
    
    for theme in high_impact_negative:
        count = themes[theme]
        impact_rate = (count / total_responses) * 100
        if impact_rate > 10:  # Affecting more than 10% of users
            if 'pricing' in theme:
                recommendations['immediate_actions'].append({
                    'priority': 'Critical',
                    'action': 'Review pricing strategy and communicate value proposition',
                    'rationale': f'{impact_rate:.1f}% of users mention pricing concerns',
                    'theme': theme
                })
            elif 'interface' in theme or 'user' in theme:
                recommendations['immediate_actions'].append({
                    'priority': 'High',
                    'action': 'Conduct UX audit and implement quick UI improvements',
                    'rationale': f'{impact_rate:.1f}% of users report usability issues',
                    'theme': theme
                })
            elif 'search' in theme:
                recommendations['immediate_actions'].append({
                    'priority': 'High',
                    'action': 'Optimize search algorithm and relevance ranking',
                    'rationale': f'{impact_rate:.1f}% of users experience search problems',
                    'theme': theme
                })
    
    # Strategic Initiatives (Medium/High Impact areas for improvement)
    if nps_score < 20:
        recommendations['strategic_initiatives'].append({
            'initiative': 'Customer Experience Transformation Program',
            'timeline': '6-12 months',
            'rationale': f'NPS score of {nps_score:.1f} indicates systematic issues requiring comprehensive approach'
        })
    
    # Performance issues
    performance_themes = [theme for theme, meta in metadata.items() 
                         if 'performance' in meta['category'].lower() and meta['sentiment'] == 'Negative']
    if performance_themes:
        recommendations['strategic_initiatives'].append({
            'initiative': 'Platform Performance Optimization',
            'timeline': '3-6 months',
            'rationale': 'Multiple performance-related complaints require infrastructure investment'
        })
    
    # Competitive Advantages (High frequency positive themes)
    high_positive = [(theme, count) for theme, count in themes.items() 
                    if metadata.get(theme, {}).get('sentiment') == 'Positive']
    high_positive.sort(key=lambda x: x[1], reverse=True)
    
    for theme, count in high_positive[:3]:
        impact_rate = (count / total_responses) * 100
        if impact_rate > 15:  # Strong positive theme
            clean_theme = theme.replace('_', ' ').title()
            recommendations['competitive_advantages'].append({
                'strength': clean_theme,
                'leverage_strategy': f'Highlight in marketing and sales materials',
                'impact': f'{impact_rate:.1f}% of users specifically mention this positively'
            })
    
    # Risk Mitigation
    detractor_rate = (category_counts.get('Detractor', 0) / total_responses) * 100
    if detractor_rate > 20:
        recommendations['risk_mitigation'].append({
            'risk': 'High customer churn risk',
            'mitigation': 'Implement proactive customer success program for at-risk accounts',
            'urgency': 'Immediate'
        })
    
    return recommendations

def calculate_theme_correlations(df, comment_column, themes_data):
    """Calculate correlations between themes and NPS categories"""
    themes, metadata = themes_data
    
    # Create binary matrix for theme occurrence
    theme_matrix = pd.DataFrame(0, index=df.index, columns=list(themes.keys()))
    
    for idx, comment in df[comment_column].items():
        if isinstance(comment, str) and len(comment.strip()) > 3:
            comment_lower = comment.lower()
            for theme_name in themes.keys():
                # Check if theme appears in comment (simplified check)
                theme_words = theme_name.replace('_', ' ').split()
                if any(word in comment_lower for word in theme_words):
                    theme_matrix.loc[idx, theme_name] = 1
    
    return theme_matrix

def create_advanced_visualizations(themes_data, nps_data, recommendations):
    """Create advanced visualizations for business insights"""
    themes, metadata = themes_data
    
    # 1. Business Impact vs Frequency Matrix
    fig_matrix = go.Figure()
    
    # Prepare data for matrix
    impact_mapping = {'High': 3, 'Medium': 2, 'Low': 1}
    sentiment_mapping = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
    
    x_vals, y_vals, sizes, colors, texts = [], [], [], [], []
    
    for theme, count in themes.items():
        if theme in metadata:
            meta = metadata[theme]
            x_vals.append(impact_mapping[meta['business_impact']])
            y_vals.append(count)
            sizes.append(count * 3)  # Size based on frequency
            colors.append(sentiment_mapping[meta['sentiment']])
            texts.append(theme.replace('_', ' ').title())
    
    fig_matrix.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='markers+text',
        marker=dict(
            size=sizes,
            color=colors,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Sentiment"),
            line=dict(width=1, color='DarkSlateGrey')
        ),
        text=texts,
        textposition="middle center",
        textfont=dict(size=8),
        name="Themes"
    ))
    
    fig_matrix.update_layout(
        title="Theme Business Impact vs Frequency Matrix",
        xaxis_title="Business Impact",
        yaxis_title="Frequency",
        xaxis=dict(tickmode='array', tickvals=[1, 2, 3], ticktext=['Low', 'Medium', 'High']),
        height=600
    )
    
    return fig_matrix

# Streamlit App with Enhanced Features
st.set_page_config(page_title="Advanced Lexis+ NPS Analytics", page_icon="⚖️", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.recommendation-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #28a745;
    margin: 0.5rem 0;
}
.risk-card {
    background-color: #fff3cd;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ffc107;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Advanced Lexis+ Hong Kong NPS Analytics")
st.markdown("Comprehensive business intelligence platform for customer feedback analysis")

# Sidebar navigation
with st.sidebar:
    st.header("🎯 Navigation")
    page = st.selectbox("Choose Analysis Type", [
        "📊 Executive Dashboard",
        "🔍 Detailed Analysis", 
        "💡 Business Intelligence",
        "📈 Competitive Analysis",
        "🎯 Action Planning"
    ])

# Sample data button
if st.button("📝 Load Sample Data", type="secondary"):
    sample_comments = [
        "it is an important tool for legal research",
        "reliable, easy to navigate, and offers comprehensive legal resources",
        "Good database",
        "It is easy to navigate",
        "comprehensive database.",
        "Many resources on Lexis+ for research and work. Most up to date resources. Reputation of Lexis+",
        "I've always found what i'm looking for on Lexis HK",
        "Easy to navigate",
        "slow search engine speed, and failure to understand searcher's question",
        "Essential access to HKG case law and other resources",
        "Authoritative and comprehensive Hong Kong primary law database, powerfully enhanced by modern AI-driven search and analytical tools",
        "The platform's advanced analytics and AI-driven insights",
        "Good Customer Service",
        "It is useful but quite expensive",
        "very information platform",
        "A helpful and user friendly tool",
        "Trustworthiness of the database.",
        "I need it for my work regularly.",
        "Extensive resources",
        "reliability",
        "Useful research function",
        "It is very user friendly",
        "Abundant resources",
        "It is relatively easy to use, and the database is sufficient for daily usage",
        "easy to use",
        "AI",
        "Case law searches could be more accurate",
        "User friendly.",
        "User friendly",
        "The Annotated Ordinances.",
        "user-friendly",
        "Useful search engine and clear annotated ordinance",
        "AI searching still too primitive",
        "One of the most reputable case law databases in HK",
        "Broad Content Coverage",
        "Very up to date",
        "Good selection of reference materials.",
        "available resources",
        "Comprehensive Legal Content makes it a one-stop platform for legal research",
        "contents that are exclusive to the platform",
        "user-friendly and good customer support",
        "conducting legal research",
        "Reliable source of legal database",
        "convenient",
        "legal research",
        "updated news",
        "Good platform design",
        "Good and easy to use.",
        "Despite inserting the relevant key words to the search engine, the results shown are not directly relevant to the issue I am researching on.",
        "User-friendly interface with seamless platform integration. Comprehensive, accurate legal content tailored for Hong Kong. Enhances productivity and client service quality.",
        "Too expensive",
        "good database, sometimes inconsistent search results",
        "Overall good service and appreciate Halsbury, but pricing may be the reason why the firms are considering alternative services"
    ]
    
    # Create enhanced sample data with additional columns
    sample_df = pd.DataFrame({
        'NPS Verbatim Comments': sample_comments,
        'Response_Date': pd.date_range('2024-01-01', periods=len(sample_comments), freq='D'),
        'Customer_Segment': np.random.choice(['Law Firm', 'Corporate Legal', 'Individual Lawyer', 'Academic'], len(sample_comments)),
        'Subscription_Type': np.random.choice(['Premium', 'Standard', 'Basic'], len(sample_comments)),
        'Years_as_Customer': np.random.randint(1, 10, len(sample_comments))
    })
    
    st.session_state['sample_data'] = sample_df
    st.success("Enhanced sample data loaded! Navigate through different analysis types to explore insights.")

# File upload section
uploaded_file = None
use_sample = False

if 'sample_data' in st.session_state:
    use_sample_data = st.checkbox("Use loaded sample data", value=True)
    if use_sample_data:
        df = st.session_state['sample_data']
        use_sample = True

if not use_sample:
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.stop()

# Main analysis section
if use_sample or uploaded_file is not None:
    
    # Column selection
    st.subheader("🔧 Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        comment_column = st.selectbox("Comment Column:", df.columns.tolist())
    with col2:
        min_frequency = st.slider("Min Theme Frequency:", 1, 10, 2)
    with col3:
        confidence_threshold = st.slider("Confidence Threshold:", 0.5, 0.95, 0.7)
    
    if st.button("🚀 Run Advanced Analysis", type="primary"):
        with st.spinner("Running comprehensive analysis..."):
            
            # Extract and analyze comments
            comments = df[comment_column].dropna().tolist()
            
            # Enhanced categorization with confidence
            nps_results = []
            for comment in comments:
                category, confidence = extract_nps_category_from_sentiment(comment)
                nps_results.append((category, confidence))
            
            # Calculate NPS with confidence intervals
            categories = [result[0] for result in nps_results]
            nps_score, category_counts, confidence_stats = calculate_nps_from_categories(categories)
            
            # Advanced theme extraction
            themes_data = extract_advanced_themes(comments, None, min_frequency)
            themes, metadata = themes_data
            
            # Generate business recommendations
            recommendations = generate_business_recommendations(themes_data, nps_score, category_counts)
            
            # Store results in session state for navigation
            st.session_state['analysis_results'] = {
                'nps_score': nps_score,
                'category_counts': category_counts,
                'confidence_stats': confidence_stats,
                'themes_data': themes_data,
                'recommendations': recommendations,
                'df': df,
                'comment_column': comment_column,
                'nps_results': nps_results
            }

# Display results based on selected page
if 'analysis_results' in st.session_state:
    results = st.session_state['analysis_results']
    
    if page == "📊 Executive Dashboard":
        st.header("📊 Executive Dashboard")
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("NPS Score", f"{results['nps_score']:.1f}", 
                     delta=f"±{(results['confidence_stats']['nps_upper'] - results['nps_score']):.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.metric("Total Responses", len(results['nps_results']))
        
        with col3:
            promoter_pct = results['confidence_stats']['promoter_pct']
            st.metric("Promoters", f"{promoter_pct:.1f}%")
        
        with col4:
            detractor_pct = results['confidence_stats']['detractor_pct']
            st.metric("Detractors", f"{detractor_pct:.1f}%")
        
        # NPS Interpretation and Business Impact
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if results['nps_score'] > 50:
                st.success("🎉 **Excellent Performance**: World-class customer satisfaction")
                business_status = "Market leader with strong competitive advantage"
            elif results['nps_score'] > 0:
                st.info("👍 **Good Performance**: Above industry average")
                business_status = "Competitive position with growth opportunities"
            else:
                st.warning("⚠️ **Needs Improvement**: Below industry benchmark")
                business_status = "Risk of customer churn, immediate action required"
            
            st.write(f"**Business Impact:** {business_status}")
        
        with col2:
            # NPS Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = results['nps_score'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "NPS Score"},
                gauge = {
                    'axis': {'range': [-100, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-100, 0], 'color': "lightcoral"},
                        {'range': [0, 50], 'color': "lightyellow"},
                        {'range': [50, 100], 'color': "lightgreen"}
                    ]
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Top Insights
        st.subheader("🎯 Key Business Insights")
        
        themes, metadata = results['themes_data']
        
        # Critical Issues (High Impact Negative)
        critical_issues = [(theme, count) for theme, count in themes.items() 
                          if metadata.get(theme, {}).get('business_impact') == 'High' 
                          and metadata.get(theme, {}).get('sentiment') == 'Negative']
        
        # Top Strengths (High frequency positive)
        top_strengths = [(theme, count) for theme, count in themes.items() 
                        if metadata.get(theme, {}).get('sentiment') == 'Positive']
        top_strengths.sort(key=lambda x: x[1], reverse=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔴 Critical Issues**")
            if critical_issues:
                for theme, count in critical_issues[:3]:
                    clean_theme = theme.replace('_', ' ').title()
                    st.markdown(f"• **{clean_theme}**: {count} mentions")
            else:
                st.write("No critical issues identified")
        
        with col2:
            st.write("**🟢 Top Strengths**")
            for theme, count in top_strengths[:3]:
                clean_theme = theme.replace('_', ' ').title()
                st.markdown(f"• **{clean_theme}**: {count} mentions")
    
    elif page == "🔍 Detailed Analysis":
        st.header("🔍 Detailed Analysis")
        
        # Enhanced visualizations
        themes, metadata = results['themes_data']
        
        # Business Impact Matrix
        fig_matrix = create_advanced_visualizations(results['themes_data'], results['nps_score'], results['recommendations'])
        st.plotly_chart(fig_matrix, use_container_width=True)
        
        # Category breakdown with themes
        tab1, tab2, tab3 = st.tabs(["🟢 Promoters", "🟡 Passives", "🔴 Detractors"])
        
        df = results['df']
        comment_column = results['comment_column']
        
        # Add NPS categories to dataframe
        df_analysis = df.copy()
        categories = [result[0] for result in results['nps_results']]
        valid_indices = df[comment_column].dropna().index
        category_series = pd.Series(index=df.index, dtype='object')
        category_series.loc[valid_indices] = categories
        df_analysis['NPS_Category'] = category_series
        
        with tab1:
            promoter_data = df_analysis[df_analysis['NPS_Category'] == 'Promoter']
            st.metric("Promoter Count", len(promoter_data))
            
            if len(promoter_data) > 0:
                promoter_comments = promoter_data[comment_column].dropna().tolist()
                promoter_themes, promoter_meta = extract_advanced_themes(promoter_comments, 'Promoter', min_frequency)
                
                if promoter_themes:
                    # Create theme analysis
                    theme_df = pd.DataFrame([
                        {
                            'Theme': theme.replace('_', ' ').title(),
                            'Count': count,
                            'Percentage': f"{(count/len(promoter_comments))*100:.1f}%",
                            'Business Impact': promoter_meta.get(theme, {}).get('business_impact', 'Unknown'),
                            'Category': promoter_meta.get(theme, {}).get('category', 'Unknown')
                        }
                        for theme, count in promoter_themes.items()
                    ]).sort_values('Count', ascending=False)
                    
                    st.dataframe(theme_df, use_container_width=True)
        
        # Similar structure for Passives and Detractors tabs...
    
    elif page == "💡 Business Intelligence":
        st.header("💡 Business Intelligence")
        
        # Recommendations display
        recs = results['recommendations']
        
        st.subheader("🚨 Immediate Actions Required")
        for action in recs['immediate_actions']:
            st.markdown(f"""
            <div class="recommendation-card">
                <h4>🎯 {action['action']}</h4>
                <p><strong>Priority:</strong> {action['priority']}</p>
                <p><strong>Rationale:</strong> {action['rationale']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("📈 Strategic Initiatives")
        for initiative in recs['strategic_initiatives']:
            st.markdown(f"""
            <div class="recommendation-card">
                <h4>🚀 {initiative['initiative']}</h4>
                <p><strong>Timeline:</strong> {initiative['timeline']}</p>
                <p><strong>Rationale:</strong> {initiative['rationale']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("🏆 Competitive Advantages")
        for advantage in recs['competitive_advantages']:
            st.markdown(f"""
            <div class="recommendation-card">
                <h4>✨ {advantage['strength']}</h4>
                <p><strong>Strategy:</strong> {advantage['leverage_strategy']}</p>
                <p><strong>Impact:</strong> {advantage['impact']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("⚠️ Risk Mitigation")
        for risk in recs['risk_mitigation']:
            st.markdown(f"""
            <div class="risk-card">
                <h4>⚠️ {risk['risk']}</h4>
                <p><strong>Mitigation:</strong> {risk['mitigation']}</p>
                <p><strong>Urgency:</strong> {risk['urgency']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif page == "📈 Competitive Analysis":
        st.header("📈 Competitive Analysis")
        
        # Benchmarking section
        st.subheader("🏅 Industry Benchmarking")
        
        # Legal software industry benchmarks (example data)
        industry_benchmarks = {
            'Legal Software Average': 25,
            'SaaS Industry Average': 31,
            'Top Quartile Legal Tech': 45,
            'Lexis+ Hong Kong': results['nps_score']
        }
        
        benchmark_df = pd.DataFrame(list(industry_benchmarks.items()), 
                                   columns=['Category', 'NPS Score'])
        
        fig_benchmark = px.bar(benchmark_df, x='Category', y='NPS Score',
                              title="NPS Benchmarking Analysis",
                              color='NPS Score',
                              color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_benchmark, use_container_width=True)
        
        # Competitive positioning
        st.subheader("🎯 Competitive Positioning")
        
        themes, metadata = results['themes_data']
        
        # Identify differentiators
        differentiators = [(theme, count) for theme, count in themes.items() 
                          if 'halsbury' in theme or 'hong_kong' in theme or 'lexis' in theme]
        
        if differentiators:
            st.write("**🏆 Unique Competitive Advantages:**")
            for theme, count in differentiators:
                clean_theme = theme.replace('_', ' ').title()
                st.write(f"• {clean_theme}: {count} mentions")
        
        # Market opportunity analysis
        st.subheader("📊 Market Opportunity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Growth Opportunities:**")
            if results['nps_score'] > 30:
                st.write("• Market expansion potential")
                st.write("• Premium pricing opportunity")
                st.write("• Reference customer development")
            else:
                st.write("• Customer retention focus")
                st.write("• Product improvement priority")
                st.write("• Competitive defense required")
        
        with col2:
            st.write("**Competitive Threats:**")
            negative_themes = [(theme, count) for theme, count in themes.items() 
                             if metadata.get(theme, {}).get('sentiment') == 'Negative']
            
            if negative_themes:
                for theme, count in sorted(negative_themes, key=lambda x: x[1], reverse=True)[:3]:
                    clean_theme = theme.replace('_', ' ').title()
                    st.write(f"• {clean_theme}: Vulnerability area")
    
    elif page == "🎯 Action Planning":
        st.header("🎯 Action Planning")
        
        # Create action plan template
        st.subheader("📋 90-Day Action Plan")
        
        recs = results['recommendations']
        
        # Priority matrix
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Week 1-2: Quick Wins**")
            for i, action in enumerate(recs['immediate_actions'][:2], 1):
                st.write(f"{i}. {action['action']}")
                st.write(f"   *Owner: [Assign]*")
                st.write(f"   *Success Metric: [Define]*")
        
        with col2:
            st.write("**Week 3-4: Foundation Building**")
            st.write("1. Establish customer feedback loop")
            st.write("   *Owner: [Assign]*")
            st.write("2. Create performance monitoring dashboard")
            st.write("   *Owner: [Assign]*")
        
        # Resource allocation
        st.subheader("💰 Resource Allocation Recommendations")
        
        total_responses = sum(results['category_counts'].values())
        
        if results['nps_score'] < 0:
            st.write("**Critical Situation - Immediate Investment Required:**")
            st.write("• 40% - Customer Experience Improvements")
            st.write("• 30% - Product Quality & Performance")
            st.write("• 20% - Customer Success & Support")
            st.write("• 10% - Marketing & Communication")
        elif results['nps_score'] < 30:
            st.write("**Growth Focus - Balanced Investment:**")
            st.write("• 30% - Product Enhancement")
            st.write("• 25% - Customer Experience")
            st.write("• 25% - Customer Success")
            st.write("• 20% - Market Expansion")
        else:
            st.write("**Market Leadership - Innovation Focus:**")
            st.write("• 35% - Innovation & New Features")
            st.write("• 25% - Market Expansion")
            st.write("• 20% - Customer Experience Excellence")
            st.write("• 20% - Competitive Defense")
        
        # Success metrics
        st.subheader("📈 Success Metrics & KPIs")
        
        metrics_data = {
            'Metric': ['NPS Score', 'Customer Retention Rate', 'Time to Resolution', 'Feature Adoption Rate', 'User Satisfaction Score'],
            'Current': [f"{results['nps_score']:.1f}", 'TBD', 'TBD', 'TBD', 'TBD'],
            'Target (90 days)': [f"{results['nps_score'] + 10:.1f}", '>95%', '<24 hours', '>70%', '>4.0/5.0'],
            'Frequency': ['Monthly', 'Monthly', 'Daily', 'Monthly', 'Quarterly']
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Export action plan
        if st.button("📥 Export Action Plan"):
            action_plan = {
                'analysis_date': datetime.now().isoformat(),
                'nps_score': results['nps_score'],
                'total_responses': total_responses,
                'recommendations': recs,
                'success_metrics': metrics_data
            }
            
            st.download_button(
                label="Download Action Plan (JSON)",
                data=json.dumps(action_plan, indent=2),
                file_name=f"lexis_plus_action_plan_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )

# Footer
st.markdown("---")
st.markdown("*Advanced Lexis+ NPS Analytics - Powered by Business Intelligence*")
