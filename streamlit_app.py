import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import re
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced NPS Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
    .recommendation-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedNPSAnalyzer:
    def __init__(self):
        self.sentiment_weights = {
            'excellent': 2.0, 'amazing': 2.0, 'outstanding': 2.0, 'perfect': 2.0,
            'fantastic': 1.8, 'wonderful': 1.8, 'great': 1.5, 'good': 1.0,
            'love': 1.5, 'like': 1.0, 'satisfied': 1.0, 'happy': 1.2,
            'okay': 0.5, 'average': 0.5, 'fine': 0.5,
            'bad': -1.0, 'poor': -1.2, 'terrible': -2.0, 'awful': -2.0,
            'hate': -1.8, 'dislike': -1.0, 'disappointed': -1.5, 'frustrated': -1.5,
            'slow': -1.0, 'expensive': -1.2, 'cheap': -0.8, 'confusing': -1.0,
            'difficult': -1.0, 'easy': 1.0, 'fast': 1.2, 'helpful': 1.2,
            'friendly': 1.0, 'professional': 1.0, 'reliable': 1.2
        }
        
        self.business_themes = {
            'User Experience': ['easy', 'difficult', 'intuitive', 'confusing', 'design', 'interface', 'navigation'],
            'Product Quality': ['quality', 'reliable', 'features', 'functionality', 'performance', 'works'],
            'Technology': ['fast', 'slow', 'bugs', 'crashes', 'loading', 'technical', 'system'],
            'Performance': ['speed', 'efficiency', 'optimization', 'responsive', 'lag', 'delay'],
            'Pricing': ['price', 'cost', 'expensive', 'cheap', 'value', 'money', 'affordable'],
            'Support': ['support', 'help', 'service', 'staff', 'team', 'assistance', 'response']
        }

    def calculate_weighted_sentiment(self, text):
        """Calculate weighted sentiment score"""
        if pd.isna(text):
            return 0
        
        text_lower = str(text).lower()
        weighted_score = 0
        word_count = 0
        
        # Use TextBlob for base sentiment
        blob = TextBlob(text)
        base_sentiment = blob.sentiment.polarity
        
        # Apply custom weights
        for word, weight in self.sentiment_weights.items():
            if word in text_lower:
                weighted_score += weight
                word_count += 1
        
        # Combine weighted and base sentiment
        if word_count > 0:
            final_score = (weighted_score / word_count * 0.7) + (base_sentiment * 0.3)
        else:
            final_score = base_sentiment
        
        return np.clip(final_score, -2, 2)

    def extract_themes(self, texts):
        """Extract and categorize themes from feedback"""
        theme_data = []
        
        for text in texts:
            if pd.isna(text):
                continue
                
            text_lower = str(text).lower()
            text_sentiment = self.calculate_weighted_sentiment(text)
            
            for theme_category, keywords in self.business_themes.items():
                theme_score = sum(1 for keyword in keywords if keyword in text_lower)
                if theme_score > 0:
                    # Determine business impact
                    if theme_category in ['Product Quality', 'User Experience']:
                        business_impact = 'High'
                    elif theme_category in ['Performance', 'Support']:
                        business_impact = 'Medium'
                    else:
                        business_impact = 'Low'
                    
                    theme_data.append({
                        'theme': theme_category,
                        'sentiment': text_sentiment,
                        'frequency': theme_score,
                        'business_impact': business_impact,
                        'text': text
                    })
        
        return pd.DataFrame(theme_data)

    def calculate_nps_with_confidence(self, scores):
        """Calculate NPS with confidence intervals"""
        scores = pd.Series(scores).dropna()
        
        if len(scores) == 0:
            return 0, 0, 0, 0
        
        promoters = (scores >= 9).sum()
        detractors = (scores <= 6).sum()
        total = len(scores)
        
        nps = ((promoters - detractors) / total) * 100
        
        # Calculate confidence interval (95%)
        p = promoters / total
        d = detractors / total
        
        se_promoters = np.sqrt(p * (1 - p) / total)
        se_detractors = np.sqrt(d * (1 - d) / total)
        se_nps = np.sqrt(se_promoters**2 + se_detractors**2) * 100
        
        margin_error = 1.96 * se_nps
        confidence_lower = nps - margin_error
        confidence_upper = nps + margin_error
        
        return nps, confidence_lower, confidence_upper, margin_error

    def generate_recommendations(self, nps_score, theme_df):
        """Generate actionable recommendations based on analysis"""
        recommendations = {
            'immediate_actions': [],
            'strategic_initiatives': [],
            'competitive_advantages': [],
            'risk_mitigation': []
        }
        
        # Analyze themes for recommendations
        if not theme_df.empty:
            theme_summary = theme_df.groupby(['theme', 'business_impact']).agg({
                'sentiment': 'mean',
                'frequency': 'sum'
            }).reset_index()
            
            # High impact negative themes
            critical_themes = theme_summary[
                (theme_summary['business_impact'] == 'High') & 
                (theme_summary['sentiment'] < -0.5)
            ]
            
            for _, theme in critical_themes.iterrows():
                recommendations['immediate_actions'].append({
                    'action': f"Address {theme['theme'].lower()} issues immediately",
                    'priority': 'High',
                    'timeline': '1-2 weeks',
                    'owner': 'Product Team'
                })
            
            # Positive themes to leverage
            strengths = theme_summary[theme_summary['sentiment'] > 0.8]
            for _, theme in strengths.iterrows():
                recommendations['competitive_advantages'].append({
                    'advantage': f"Strong {theme['theme'].lower()} performance",
                    'action': f"Market your {theme['theme'].lower()} capabilities",
                    'priority': 'Medium'
                })
        
        # NPS-based recommendations
        if nps_score < 0:
            recommendations['risk_mitigation'].extend([
                {
                    'risk': 'High churn potential',
                    'action': 'Implement customer retention program',
                    'timeline': 'Immediate'
                },
                {
                    'risk': 'Negative word-of-mouth',
                    'action': 'Proactive customer outreach and issue resolution',
                    'timeline': '1 week'
                }
            ])
        elif nps_score > 50:
            recommendations['strategic_initiatives'].append({
                'initiative': 'Referral program implementation',
                'rationale': 'High NPS indicates strong advocacy potential',
                'timeline': '1-2 months'
            })
        
        return recommendations

    def create_business_impact_matrix(self, theme_df):
        """Create business impact vs frequency matrix visualization"""
        if theme_df.empty:
            return go.Figure()
        
        theme_summary = theme_df.groupby('theme').agg({
            'sentiment': 'mean',
            'frequency': 'sum',
            'business_impact': lambda x: x.iloc[0]
        }).reset_index()
        
        # Map business impact to numeric values
        impact_map = {'High': 3, 'Medium': 2, 'Low': 1}
        theme_summary['impact_numeric'] = theme_summary['business_impact'].map(impact_map)
        
        fig = px.scatter(
            theme_summary,
            x='frequency',
            y='impact_numeric',
            size='frequency',
            color='sentiment',
            hover_name='theme',
            color_continuous_scale='RdYlGn',
            title='Business Impact vs Theme Frequency Matrix'
        )
        
        fig.update_layout(
            yaxis_title='Business Impact Level',
            xaxis_title='Theme Frequency',
            yaxis=dict(tickmode='array', tickvals=[1, 2, 3], ticktext=['Low', 'Medium', 'High'])
        )
        
        return fig

def main():
    st.markdown('<h1 class="main-header">🚀 Advanced NPS Analysis Platform</h1>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = AdvancedNPSAnalyzer()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose Analysis View:",
        ["📊 Executive Dashboard", "🔍 Detailed Analysis", "💼 Business Intelligence", 
         "🏆 Competitive Analysis", "📋 Action Planning"]
    )
    
    # File upload
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your CSV file",
        type=['csv'],
        help="Upload a CSV file with 'score' and 'feedback' columns"
    )
    
    # Sample data option
    if not uploaded_file:
        if st.sidebar.button("🎯 Use Sample Data"):
            sample_data = {
                'score': [9, 8, 7, 6, 10, 9, 5, 4, 8, 9, 7, 6, 10, 3, 2, 8, 9, 7, 5, 10],
                'feedback': [
                    "Excellent product, love the new features!",
                    "Good overall but could be faster",
                    "Interface is okay, some bugs though",
                    "Too expensive for what it offers",
                    "Perfect! Amazing customer support",
                    "Great quality, very reliable",
                    "Slow performance, needs improvement",
                    "Difficult to use, confusing navigation",
                    "Good value for money",
                    "Outstanding features, highly recommend",
                    "Average experience, nothing special",
                    "Support team was helpful",
                    "Fantastic product quality",
                    "Terrible bugs, crashes frequently",
                    "Awful user experience",
                    "Fast and efficient",
                    "Love the design and functionality",
                    "Decent but room for improvement",
                    "Poor customer service",
                    "Excellent value and performance"
                ]
            }
            st.session_state.df = pd.DataFrame(sample_data)
    
    # Load data
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.sidebar.success(f"✅ Loaded {len(df)} records")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading file: {str(e)}")
            return
    
    if 'df' not in st.session_state:
        st.info("👆 Please upload a CSV file or use sample data to begin analysis")
        return
    
    df = st.session_state.df
    
    # Validate data structure
    required_columns = ['score', 'feedback']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"❌ Missing required columns: {missing_columns}")
        st.info("Your CSV should have 'score' and 'feedback' columns")
        return
    
    # Perform analysis
    nps_score, conf_lower, conf_upper, margin_error = analyzer.calculate_nps_with_confidence(df['score'])
    theme_df = analyzer.extract_themes(df['feedback'])
    recommendations = analyzer.generate_recommendations(nps_score, theme_df)
    
    # Display content based on selected page
    if page == "📊 Executive Dashboard":
        show_executive_dashboard(df, nps_score, conf_lower, conf_upper, theme_df, analyzer)
    elif page == "🔍 Detailed Analysis":
        show_detailed_analysis(df, theme_df, analyzer)
    elif page == "💼 Business Intelligence":
        show_business_intelligence(df, nps_score, theme_df, recommendations)
    elif page == "🏆 Competitive Analysis":
        show_competitive_analysis(nps_score, theme_df)
    elif page == "📋 Action Planning":
        show_action_planning(recommendations, nps_score, theme_df)

def show_executive_dashboard(df, nps_score, conf_lower, conf_upper, theme_df, analyzer):
    """Executive Dashboard View"""
    st.header("📊 Executive Dashboard")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "NPS Score",
            f"{nps_score:.1f}",
            delta=f"±{abs(nps_score - conf_lower):.1f}",
            help="Net Promoter Score with confidence interval"
        )
    
    with col2:
        total_responses = len(df)
        st.metric("Total Responses", total_responses)
    
    with col3:
        avg_score = df['score'].mean()
        st.metric("Average Score", f"{avg_score:.1f}")
    
    with col4:
        # Business status
        if nps_score > 50:
            status = "🟢 Excellent"
        elif nps_score > 0:
            status = "🟡 Good"
        else:
            status = "🔴 Needs Attention"
        st.metric("Business Status", status)
    
    # NPS Gauge Chart
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = nps_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "NPS Score"},
        delta = {'reference': 0},
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
    
    fig_gauge.update_layout(height=400)
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Score Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig_dist = px.histogram(
            df, x='score', 
            title='Score Distribution',
            labels={'score': 'NPS Score', 'count': 'Number of Responses'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_dist.update_layout(bargap=0.1)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Top themes
        if not theme_df.empty:
            theme_summary = theme_df.groupby('theme').agg({
                'sentiment': 'mean',
                'frequency': 'sum'
            }).reset_index().sort_values('frequency', ascending=True)
            
            fig_themes = px.bar(
                theme_summary.tail(6),
                x='frequency',
                y='theme',
                orientation='h',
                title='Top Themes by Frequency',
                color='sentiment',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_themes, use_container_width=True)
    
    # Executive Summary
    st.markdown("---")
    st.subheader("📋 Executive Summary")
    
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**🎯 Key Insights:**")
        st.markdown(f"• NPS Score: **{nps_score:.1f}** (95% CI: {conf_lower:.1f} to {conf_upper:.1f})")
        
        if nps_score > 50:
            st.markdown("• **Market Leader** - Customers are highly satisfied")
        elif nps_score > 0:
            st.markdown("• **Good Performance** - Solid customer satisfaction")
        else:
            st.markdown("• **Risk of Churn** - Immediate attention required")
        
        st.markdown(f"• Based on **{len(df)}** customer responses")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with summary_col2:
        if not theme_df.empty:
            critical_themes = theme_df[
                (theme_df['business_impact'] == 'High') & 
                (theme_df['sentiment'] < 0)
            ]['theme'].unique()
            
            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
            st.markdown("**⚡ Immediate Actions:**")
            if len(critical_themes) > 0:
                for theme in critical_themes[:3]:
                    st.markdown(f"• Address **{theme}** concerns")
            else:
                st.markdown("• Continue current excellent performance")
                st.markdown("• Focus on scaling successful practices")
            st.markdown('</div>', unsafe_allow_html=True)

def show_detailed_analysis(df, theme_df, analyzer):
    """Detailed Analysis View"""
    st.header("🔍 Detailed Analysis")
    
    # Sentiment Analysis
    st.subheader("💭 Sentiment Analysis")
    
    # Calculate sentiment for all feedback
    df['sentiment'] = df['feedback'].apply(analyzer.calculate_weighted_sentiment)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution
        fig_sentiment = px.histogram(
            df, x='sentiment',
            title='Sentiment Score Distribution',
            labels={'sentiment': 'Sentiment Score', 'count': 'Number of Responses'},
            color_discrete_sequence=['#2ecc71']
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
        # Sentiment vs NPS Score correlation
        fig_corr = px.scatter(
            df, x='sentiment', y='score',
            title='Sentiment vs NPS Score Correlation',
            trendline="ols",
            labels={'sentiment': 'Sentiment Score', 'score': 'NPS Score'}
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Theme Analysis
    st.subheader("🎯 Theme Analysis")
    
    if not theme_df.empty:
        # Business Impact Matrix
        fig_matrix = analyzer.create_business_impact_matrix(theme_df)
        st.plotly_chart(fig_matrix, use_container_width=True)
        
        # Theme breakdown table
        theme_summary = theme_df.groupby(['theme', 'business_impact']).agg({
            'sentiment': ['mean', 'count'],
            'frequency': 'sum'
        }).round(2)
        
        theme_summary.columns = ['Avg_Sentiment', 'Mention_Count', 'Total_Frequency']
        theme_summary = theme_summary.reset_index()
        
        st.subheader("📊 Theme Breakdown")
        st.dataframe(theme_summary, use_container_width=True)
        
        # Word Cloud
        if len(df['feedback'].dropna()) > 0:
            st.subheader("☁️ Word Cloud")
            all_text = ' '.join(df['feedback'].dropna().astype(str))
            
            try:
                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    colormap='viridis'
                ).generate(all_text)
                
                fig_wc, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig_wc)
            except Exception as e:
                st.info("Word cloud could not be generated - insufficient text data")
    
    # Detailed Feedback Table
    st.subheader("📝 Detailed Feedback Analysis")
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        score_filter = st.selectbox(
            "Filter by Score Range:",
            ["All", "Promoters (9-10)", "Passives (7-8)", "Detractors (0-6)"]
        )
    
    with col2:
        sentiment_filter = st.selectbox(
            "Filter by Sentiment:",
            ["All", "Positive (>0.5)", "Neutral (-0.5 to 0.5)", "Negative (<-0.5)"]
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    if score_filter == "Promoters (9-10)":
        filtered_df = filtered_df[filtered_df['score'] >= 9]
    elif score_filter == "Passives (7-8)":
        filtered_df = filtered_df[filtered_df['score'].between(7, 8)]
    elif score_filter == "Detractors (0-6)":
        filtered_df = filtered_df[filtered_df['score'] <= 6]
    
    if sentiment_filter == "Positive (>0.5)":
        filtered_df = filtered_df[filtered_df['sentiment'] > 0.5]
    elif sentiment_filter == "Neutral (-0.5 to 0.5)":
        filtered_df = filtered_df[filtered_df['sentiment'].between(-0.5, 0.5)]
    elif sentiment_filter == "Negative (<-0.5)":
        filtered_df = filtered_df[filtered_df['sentiment'] < -0.5]
    
    # Display filtered results
    display_df = filtered_df[['score', 'sentiment', 'feedback']].round(2)
    st.dataframe(display_df, use_container_width=True)
    
    st.info(f"Showing {len(filtered_df)} of {len(df)} total responses")

def show_business_intelligence(df, nps_score, theme_df, recommendations):
    """Business Intelligence View"""
    st.header("💼 Business Intelligence")
    
    # Business Performance Overview
    st.subheader("📈 Business Performance Overview")
    
    # Calculate key business metrics
    promoters = len(df[df['score'] >= 9])
    passives = len(df[df['score'].between(7, 8)])
    detractors = len(df[df['score'] <= 6])
    total = len(df)
    
    # Performance metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Promoters", f"{promoters} ({promoters/total*100:.1f}%)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Passives", f"{passives} ({passives/total*100:.1f}%)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Detractors", f"{detractors} ({detractors/total*100:.1f}%)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        churn_risk = detractors / total * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Churn Risk", f"{churn_risk:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Business Impact Analysis
    st.subheader("💰 Business Impact Analysis")
    
    if not theme_df.empty:
        # Critical issues identification
        critical_issues = theme_df[
            (theme_df['business_impact'] == 'High') & 
            (theme_df['sentiment'] < -0.5)
        ].groupby('theme').size().reset_index(name='frequency')
        
        strengths = theme_df[
            (theme_df['sentiment'] > 0.8)
        ].groupby('theme').size().reset_index(name='frequency')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("**🚨 Critical Issues:**")
            if not critical_issues.empty:
                for _, issue in critical_issues.head(3).iterrows():
                    st.markdown(f"• **{issue['theme']}** ({issue['frequency']} mentions)")
            else:
                st.markdown("• No critical issues identified ✅")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("**🌟 Top Strengths:**")
            if not strengths.empty:
                for _, strength in strengths.head(3).iterrows():
                    st.markdown(f"• **{strength['theme']}** ({strength['frequency']} mentions)")
            else:
                st.markdown("• Continue building on current performance")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Resource Allocation Recommendations
    st.subheader("💡 Resource Allocation Strategy")
    
    if nps_score < 0:
        allocation = {"Crisis Management": 50, "Product Improvement": 30, "Customer Support": 20}
    elif nps_score < 30:
        allocation = {"Product Improvement": 40, "Customer Experience": 35, "Marketing": 25}
    elif nps_score < 60:
        allocation = {"Innovation": 35, "Scaling": 30, "Marketing": 20, "Optimization": 15}
    else:
        allocation = {"Innovation": 40, "Market Expansion": 30, "Brand Building": 20, "Optimization": 10}
    
    # Resource allocation chart
    fig_allocation = px.pie(
        values=list(allocation.values()),
        names=list(allocation.keys()),
        title=f"Recommended Resource Allocation (NPS: {nps_score:.1f})"
    )
    st.plotly_chart(fig_allocation, use_container_width=True)
    
    # Financial Impact Estimation
    st.subheader("💰 Financial Impact Estimation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Revenue Impact Analysis:**")
        
        # Estimate based on industry benchmarks
        if nps_score > 50:
            revenue_impact = "🟢 +15-25% growth potential"
            retention_rate = "🟢 90-95% retention expected"
        elif nps_score > 0:
            revenue_impact = "🟡 +5-15% growth potential"
            retention_rate = "🟡 80-90% retention expected"
        else:
            revenue_impact = "🔴 -10-20% revenue at risk"
            retention_rate = "🔴 60-80% retention expected"
        
        st.markdown(f"• {revenue_impact}")
        st.markdown(f"• {retention_rate}")
    
    with col2:
        st.markdown("**Cost Impact Analysis:**")
        
        if churn_risk > 30:
            acquisition_cost = "🔴 High acquisition costs (+200%)"
            support_cost = "🔴 Increased support burden (+150%)"
        elif churn_risk > 15:
            acquisition_cost = "🟡 Moderate acquisition costs (+50%)"
            support_cost = "🟡 Standard support requirements"
        else:
            acquisition_cost = "🟢 Low acquisition costs (-25%)"
            support_cost = "🟢 Efficient support operations"
        
        st.markdown(f"• {acquisition_cost}")
        st.markdown(f"• {support_cost}")

def show_competitive_analysis(nps_score, theme_df):
    """Competitive Analysis View"""
    st.header("🏆 Competitive Analysis")
    
    # Industry Benchmarks
    st.subheader("📊 Industry Benchmarks")
    
    # Sample industry benchmarks (you can update these with real data)
    industry_benchmarks = {
        "Software/SaaS": 31,
        "E-commerce": 28,
        "Financial Services": 24,
        "Healthcare": 27,
        "Telecommunications": 18,
        "Retail": 25
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Benchmark comparison
        fig_benchmark = go.Figure()
        
        industries = list(industry_benchmarks.keys())
        scores = list(industry_benchmarks.values())
        
        fig_benchmark.add_trace(go.Bar(
            x=industries,
            y=scores,
            name='Industry Average',
            marker_color='lightblue'
        ))
        
        # Add your NPS as a horizontal line
        fig_benchmark.add_hline(
            y=nps_score,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Your NPS: {nps_score:.1f}"
        )
        
        fig_benchmark.update_layout(
            title='NPS vs Industry Benchmarks',
            xaxis_title='Industry',
            yaxis_title='NPS Score',
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_benchmark, use_container_width=True)
    
    with col2:
        # Competitive positioning
        st.markdown("**🎯 Competitive Positioning:**")
        
        if nps_score > 50:
            position = "**Market Leader** 🏆"
            description = "Your NPS puts you in the top tier of most industries"
        elif nps_score > 30:
            position = "**Strong Performer** 💪"
            description = "Above average performance with growth potential"
        elif nps_score > 0:
            position = "**Market Average** ⚖️"
            description = "Meeting market expectations, room for differentiation"
        else:
            position = "**Needs Improvement** 📈"
            description = "Below market standards, urgent action required"
        
        st.markdown(f"• Position: {position}")
        st.markdown(f"• Analysis: {description}")
        
        # Calculate percentile
        all_benchmarks = list(industry_benchmarks.values())
        percentile = (sum(1 for x in all_benchmarks if x < nps_score) / len(all_benchmarks)) * 100
        st.markdown(f"• Percentile: **{percentile:.0f}th** percentile")
    
    # Market Opportunities
    st.subheader("🚀 Market Opportunities")
    
    if not theme_df.empty:
        # Identify unique strengths
        strengths = theme_df[theme_df['sentiment'] > 1.0].groupby('theme').size().reset_index(name='frequency')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("**🌟 Unique Differentiators:**")
            if not strengths.empty:
                for _, strength in strengths.head(3).iterrows():
                    st.markdown(f"• **{strength['theme']}** - High customer satisfaction")
            else:
                st.markdown("• Focus on building distinctive capabilities")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
            st.markdown("**📈 Growth Opportunities:**")
            
            if nps_score > 50:
                st.markdown("• **Referral programs** - Leverage advocacy")
                st.markdown("• **Premium offerings** - Capitalize on satisfaction")
                st.markdown("• **Market expansion** - Enter new segments")
            elif nps_score > 0:
                st.markdown("• **Product innovation** - Differentiate from competitors")
                st.markdown("• **Customer experience** - Move above average")
                st.markdown("• **Targeted improvements** - Address specific pain points")
            else:
                st.markdown("• **Retention focus** - Prevent customer loss")
                st.markdown("• **Quality improvement** - Meet market standards")
                st.markdown("• **Competitive analysis** - Learn from leaders")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Competitive Strategy Matrix
    st.subheader("⚔️ Competitive Strategy Matrix")
    
    # Create a 2x2 matrix based on NPS performance and market position
    fig_matrix = go.Figure()
    
    # Define quadrants
    quadrants = [
        {"name": "Leaders", "x": [30, 100], "y": [30, 100], "color": "rgba(0, 255, 0, 0.2)"},
        {"name": "Challengers", "x": [-100, 30], "y": [30, 100], "color": "rgba(255, 255, 0, 0.2)"},
        {"name": "Followers", "x": [30, 100], "y": [-100, 30], "color": "rgba(255, 165, 0, 0.2)"},
        {"name": "Niche", "x": [-100, 30], "y": [-100, 30], "color": "rgba(255, 0, 0, 0.2)"}
    ]
    
    # Add quadrant backgrounds
    for quad in quadrants:
        fig_matrix.add_shape(
            type="rect",
            x0=quad["x"][0], y0=quad["y"][0],
            x1=quad["x"][1], y1=quad["y"][1],
            fillcolor=quad["color"],
            line=dict(color="rgba(0,0,0,0)")
        )
    
    # Add your position
    market_strength = min(nps_score + 20, 80)  # Simulated market strength
    fig_matrix.add_trace(go.Scatter(
        x=[nps_score],
        y=[market_strength],
        mode='markers+text',
        marker=dict(size=20, color='red'),
        text=['Your Position'],
        textposition="top center",
        name='Your Company'
    ))
    
    # Add competitors (simulated)
    competitor_data = [
        {"name": "Competitor A", "nps": 45, "strength": 70},
        {"name": "Competitor B", "nps": 25, "strength": 60},
        {"name": "Competitor C", "nps": 15, "strength": 40},
    ]
    
    for comp in competitor_data:
        fig_matrix.add_trace(go.Scatter(
            x=[comp["nps"]],
            y=[comp["strength"]],
            mode='markers+text',
            marker=dict(size=15, color='blue'),
            text=[comp["name"]],
            textposition="top center",
            name=comp["name"]
        ))
    
    fig_matrix.update_layout(
        title='Competitive Position Matrix',
        xaxis_title='NPS Score',
        yaxis_title='Market Strength',
        xaxis=dict(range=[-20, 80]),
        yaxis=dict(range=[0, 100]),
        showlegend=False
    )
    
    # Add quadrant labels
    annotations = [
        dict(x=65, y=65, text="Leaders", showarrow=False, font=dict(size=16, color="green")),
        dict(x=5, y=65, text="Challengers", showarrow=False, font=dict(size=16, color="orange")),
        dict(x=65, y=15, text="Followers", showarrow=False, font=dict(size=16, color="blue")),
        dict(x=5, y=15, text="Niche Players", showarrow=False, font=dict(size=16, color="red"))
    ]
    
    fig_matrix.update_layout(annotations=annotations)
    st.plotly_chart(fig_matrix, use_container_width=True)

def show_action_planning(recommendations, nps_score, theme_df):
    """Action Planning View"""
    st.header("📋 Action Planning")
    
    # 90-Day Action Plan
    st.subheader("🎯 90-Day Action Plan")
    
    # Create timeline
    today = datetime.now()
    week_1 = today + timedelta(weeks=1)
    week_4 = today + timedelta(weeks=4)
    week_8 = today + timedelta(weeks=8)
    week_12 = today + timedelta(weeks=12)
    
    # Action plan based on NPS score
    if nps_score < 0:
        action_plan = {
            "Week 1-2 (Crisis Response)": [
                "🚨 Immediate customer outreach to detractors",
                "📞 Emergency support team activation",
                "🔍 Root cause analysis of critical issues",
                "📊 Daily NPS monitoring setup"
            ],
            "Week 3-6 (Stabilization)": [
                "🛠️ Fix identified critical issues",
                "💬 Implement customer feedback loop",
                "📈 Launch retention campaign",
                "🎓 Staff training on customer service"
            ],
            "Week 7-10 (Recovery)": [
                "🔄 Process improvements implementation",
                "📱 Enhanced customer communication",
                "🎁 Customer win-back initiatives",
                "📊 Weekly progress reviews"
            ],
            "Week 11-12 (Monitoring)": [
                "📈 Measure improvement in NPS",
                "🔍 Analyze effectiveness of changes",
                "📋 Plan for sustained improvement",
                "🎯 Set targets for next quarter"
            ]
        }
    elif nps_score < 30:
        action_plan = {
            "Week 1-2 (Assessment)": [
                "🔍 Detailed customer journey analysis",
                "📊 Competitor benchmarking study",
                "💡 Innovation workshop sessions",
                "📝 Customer interview program"
            ],
            "Week 3-6 (Enhancement)": [
                "🚀 Product feature improvements",
                "🎨 User experience optimization",
                "📚 Team skill development",
                "💬 Customer communication enhancement"
            ],
            "Week 7-10 (Implementation)": [
                "🔄 Roll out improvements",
                "📈 A/B testing of changes",
                "🎯 Targeted customer campaigns",
                "📊 Performance monitoring"
            ],
            "Week 11-12 (Optimization)": [
                "📈 Measure NPS improvement",
                "🔧 Fine-tune successful initiatives",
                "📋 Plan scaling strategies",
                "🎯 Set ambitious targets"
            ]
        }
    else:
        action_plan = {
            "Week 1-2 (Leverage)": [
                "🌟 Launch referral program",
                "📣 Amplify positive testimonials",
                "🚀 Identify expansion opportunities",
                "📊 Advanced analytics setup"
            ],
            "Week 3-6 (Innovation)": [
                "💡 Next-generation feature development",
                "🎯 Premium service offerings",
                "🌍 Market expansion planning",
                "🤝 Strategic partnership exploration"
            ],
            "Week 7-10 (Scaling)": [
                "📈 Scale successful practices",
                "🎓 Best practice documentation",
                "🌟 Employee advocacy programs",
                "📊 Advanced NPS analytics"
            ],
            "Week 11-12 (Excellence)": [
                "🏆 Excellence program launch",
                "📈 Set industry-leading targets",
                "🔮 Future trend anticipation",
                "🎯 Next quarter's ambitious goals"
            ]
        }
    
    # Display action plan
    for phase, actions in action_plan.items():
        with st.expander(f"📅 {phase}", expanded=True):
            for action in actions:
                st.markdown(f"• {action}")
    
    # Specific Recommendations
    st.subheader("💡 Specific Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🚨 Immediate Actions (1-2 weeks):**")
        for action in recommendations['immediate_actions']:
            st.markdown(f"• **{action.get('action', 'N/A')}**")
            st.markdown(f"  - Priority: {action.get('priority', 'N/A')}")
            st.markdown(f"  - Owner: {action.get('owner', 'TBD')}")
    
    with col2:
        st.markdown("**📈 Strategic Initiatives (1-3 months):**")
        for initiative in recommendations['strategic_initiatives']:
            st.markdown(f"• **{initiative.get('initiative', 'N/A')}**")
            st.markdown(f"  - Timeline: {initiative.get('timeline', 'TBD')}")
    
    # Resource Allocation
    st.subheader("💰 Resource Allocation Plan")
    
    # Budget allocation based on NPS
    if nps_score < 0:
        budget_allocation = {
            "Customer Support": 40,
            "Product Fixes": 30,
            "Crisis Management": 20,
            "Communication": 10
        }
    elif nps_score < 30:
        budget_allocation = {
            "Product Development": 35,
            "Customer Experience": 25,
            "Marketing": 20,
            "Operations": 20
        }
    else:
        budget_allocation = {
            "Innovation": 40,
            "Marketing & Growth": 30,
            "Excellence Programs": 20,
            "Technology": 10
        }
    
    fig_budget = px.pie(
        values=list(budget_allocation.values()),
        names=list(budget_allocation.keys()),
        title="Recommended Budget Allocation"
    )
    st.plotly_chart(fig_budget, use_container_width=True)
    
    # Success Metrics & KPIs
    st.subheader("📊 Success Metrics & KPIs")
    
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        st.markdown("**🎯 Primary Metrics:**")
        target_nps = max(nps_score + 15, 50) if nps_score < 50 else min(nps_score + 10, 80)
        st.markdown(f"• **NPS Target:** {target_nps:.0f} (Current: {nps_score:.1f})")
        st.markdown(f"• **Response Rate:** >30% (Track monthly)")
        st.markdown(f"• **Customer Retention:** >85%")
        st.markdown(f"• **Time to Resolution:** <24 hours")
    
    with metrics_col2:
        st.markdown("**📈 Secondary Metrics:**")
        st.markdown("• **Feature Adoption Rate:** >60%")
        st.markdown("• **Support Ticket Volume:** <10% of customers")
        st.markdown("• **Customer Effort Score:** >4.0/5.0")
        st.markdown("• **Employee NPS:** >30")
    
    # Export Action Plan
    st.subheader("📤 Export Action Plan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Generate Action Plan Report"):
            action_plan_data = {
                "generated_date": datetime.now().isoformat(),
                "current_nps": nps_score,
                "target_nps": target_nps,
                "action_plan": action_plan,
                "recommendations": recommendations,
                "budget_allocation": budget_allocation,
                "success_metrics": {
                    "primary": [
                        f"NPS Target: {target_nps}",
                        "Response Rate: >30%",
                        "Customer Retention: >85%",
                        "Time to Resolution: <24 hours"
                    ],
                    "secondary": [
                        "Feature Adoption Rate: >60%",
                        "Support Ticket Volume: <10%",
                        "Customer Effort Score: >4.0",
                        "Employee NPS: >30"
                    ]
                }
            }
            
            st.download_button(
                label="📥 Download Action Plan (JSON)",
                data=json.dumps(action_plan_data, indent=2),
                file_name=f"nps_action_plan_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📊 Export Analysis Data"):
            if 'df' in st.session_state:
                csv_data = st.session_state.df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Raw Data (CSV)",
                    data=csv_data,
                    file_name=f"nps_analysis_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
