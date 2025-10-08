import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import re
import io
from collections import Counter
import json

# Page config
st.set_page_config(
    page_title="Strategic Feedback Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .theme-tag {
        background: #007bff;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        font-size: 0.8rem;
        display: inline-block;
    }
    .nps-promoter { background: #28a745; }
    .nps-passive { background: #ffc107; color: #212529; }
    .nps-detractor { background: #dc3545; }
    .correlation-item {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 0.8rem;
        margin: 0.3rem 0;
    }
    .emotion-mapping {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 0.8rem;
        margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StrategicFeedbackAnalyzer:
    def __init__(self):
        self.themes = [
            'user_experience', 'pricing_concerns', 'feature_requests', 
            'technical_issues', 'workflow_friction', 'content_coverage', 
            'integration_needs', 'support_quality', 'performance_issues'
        ]
        self.emotions = ['satisfied', 'frustrated', 'confused', 'excited', 'concerned', 'appreciative', 'disappointed']
        self.legal_terms = ['ordinances', 'regulations', 'compliance', 'documentation', 'annotations', 'legal research', 'statutes', 'provisions']
        self.knowledge_gaps = ["don't know if it can", "haven't tried that feature", "wasn't aware of", "not sure how to", "couldn't find"]
        
        # Regression model for NPS prediction
        self.scaler = StandardScaler()
        self.nps_model = LinearRegression()
        self._train_mock_model()
    
    def _train_mock_model(self):
        # Mock training data for NPS prediction
        X = np.random.rand(100, 3)  # friction, risk, feature scores
        y = 5 + X[:, 0] * -2 + X[:, 1] * -1.5 + X[:, 2] * 2 + np.random.normal(0, 0.5, 100)
        y = np.clip(y, 0, 10)
        
        X_scaled = self.scaler.fit_transform(X)
        self.nps_model.fit(X_scaled, y)
    
    def extract_themes(self, text, nps_score=None):
        """Extract themes with confidence scores and evidence"""
        text_lower = text.lower()
        detected_themes = []
        
        theme_patterns = {
            'user_experience': ['love', 'great', 'amazing', 'excellent', 'wonderful', 'fantastic'],
            'pricing_concerns': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'budget'],
            'feature_requests': ['feature', 'functionality', 'would like', 'wish', 'add', 'include'],
            'technical_issues': ['bug', 'error', 'crash', 'broken', 'not working', 'problem'],
            'workflow_friction': ['manual', 'workaround', 'tedious', 'time-consuming', 'difficult'],
            'content_coverage': ['content', 'coverage', 'complete', 'comprehensive', 'missing'],
            'integration_needs': ['integrate', 'connection', 'sync', 'api', 'export', 'import'],
            'support_quality': ['support', 'help', 'customer service', 'response', 'assistance'],
            'performance_issues': ['slow', 'fast', 'performance', 'speed', 'lag', 'loading']
        }
        
        for theme, keywords in theme_patterns.items():
            matches = [kw for kw in keywords if kw in text_lower]
            if matches:
                confidence = min(0.95, len(matches) * 0.3 + 0.4)
                evidence = ', '.join(matches[:3])
                detected_themes.append({
                    'theme': theme,
                    'confidence': confidence,
                    'evidence': evidence
                })
        
        nps_category = self.categorize_nps(nps_score) if nps_score else 'unknown'
        tags = f"<nps-tag>{','.join([str(i+1) for i in range(len(detected_themes))])}</nps-tag>"
        
        return {
            'themes': detected_themes,
            'nps_category': nps_category,
            'tags': tags
        }
    
    def categorize_nps(self, score):
        """Categorize NPS score"""
        if score >= 9:
            return 'promoter'
        elif score >= 7:
            return 'passive'
        else:
            return 'detractor'
    
    def analyze_emotions(self, text):
        """Detect emotions with intensity"""
        text_lower = text.lower()
        detected_emotions = []
        
        emotion_patterns = {
            'satisfied': ['love', 'great', 'happy', 'pleased', 'satisfied'],
            'frustrated': ['frustrated', 'annoying', 'irritating', 'but', 'however'],
            'confused': ['confused', 'unclear', 'don\'t understand', 'not sure'],
            'excited': ['excited', 'amazing', 'fantastic', 'incredible'],
            'concerned': ['worried', 'concerned', 'afraid', 'nervous'],
            'appreciative': ['thank', 'appreciate', 'grateful', 'thanks'],
            'disappointed': ['disappointed', 'expected', 'hoped', 'wish']
        }
        
        for emotion, keywords in emotion_patterns.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                intensity = min(0.9, matches * 0.3 + 0.4)
                detected_emotions.append({
                    'emotion': emotion,
                    'intensity': intensity
                })
        
        return detected_emotions
    
    def detect_knowledge_gaps(self, text):
        """Detect knowledge gaps in feedback"""
        text_lower = text.lower()
        gaps = []
        
        for gap_phrase in self.knowledge_gaps:
            if gap_phrase in text_lower:
                gaps.append({
                    'gap_type': 'feature_awareness',
                    'phrase': gap_phrase,
                    'implication': 'User education opportunity'
                })
        
        return gaps
    
    def analyze_cross_channel(self, feedback, email):
        """Analyze correlations between feedback and email"""
        correlations = []
        
        if not email:
            return correlations
        
        feedback_lower = feedback.lower()
        email_lower = email.lower()
        
        # Check for common themes
        common_terms = ['export', 'import', 'integration', 'bug', 'error', 'pricing', 'feature']
        
        for term in common_terms:
            if term in feedback_lower and term in email_lower:
                correlations.append({
                    'type': 'theme_match',
                    'term': term,
                    'description': f'"{term}" mentioned in both feedback and email',
                    'risk_level': 'high' if term in ['bug', 'error', 'pricing'] else 'medium'
                })
        
        return correlations
    
    def generate_legal_insights(self, text, industry):
        """Generate legal domain insights"""
        insights = []
        
        if industry != 'legal':
            return insights
        
        text_lower = text.lower()
        
        for term in self.legal_terms:
            if term in text_lower:
                insights.append({
                    'term': term,
                    'context': f'Legal terminology: "{term}" detected',
                    'implication': 'Requires domain expertise in legal workflows'
                })
        
        return insights
    
    def simulate_nps_change(self, current_nps, friction_change, risk_change, feature_change):
        """Simulate NPS changes based on improvements"""
        # Convert changes to model input
        changes = np.array([[friction_change, risk_change, feature_change]])
        changes_scaled = self.scaler.transform(changes)
        
        # Predict impact
        impact = self.nps_model.predict(changes_scaled)[0]
        projected_nps = max(0, min(10, current_nps + impact))
        
        return {
            'current': current_nps,
            'projected': round(projected_nps, 1),
            'change': round(impact, 1),
            'confidence': 0.75
        }
    
    def generate_followup_strategy(self, themes, emotions, nps_category, knowledge_gaps):
        """Generate personalized follow-up strategy"""
        strategy = {
            'what_to_say': [],
            'what_to_do': [],
            'priority': 'medium'
        }
        
        # Map emotions to themes for strategic responses
        frustrated_themes = [t['theme'] for t in themes if any(e['emotion'] == 'frustrated' for e in emotions)]
        
        if 'workflow_friction' in frustrated_themes:
            strategy['what_to_say'].append("We understand the manual processes are frustrating and appreciate you bringing this to our attention.")
            strategy['what_to_do'].append("Schedule workflow optimization session within 1 week")
            strategy['priority'] = 'high'
        
        if knowledge_gaps:
            strategy['what_to_say'].append("We'd love to show you some features that might help with your current challenges.")
            strategy['what_to_do'].append("Send feature tutorial or schedule demo")
        
        # NPS-based strategies
        if nps_category == 'detractor':
            strategy['what_to_say'].append("We value your feedback and want to address your concerns directly.")
            strategy['what_to_do'].append("Immediate follow-up call within 24 hours")
            strategy['priority'] = 'urgent'
        elif nps_category == 'promoter':
            strategy['what_to_say'].append("Thank you for being a valued customer! We'd love to hear about expansion opportunities.")
            strategy['what_to_do'].append("Reach out for upsell conversation")
        
        return strategy
    
    def analyze_feedback(self, feedback, email="", nps_score=8, segment="enterprise", industry="tech"):
        """Complete feedback analysis"""
        # Core analysis
        theme_analysis = self.extract_themes(feedback, nps_score)
        emotions = self.analyze_emotions(feedback)
        knowledge_gaps = self.detect_knowledge_gaps(feedback)
        cross_channel = self.analyze_cross_channel(feedback, email)
        legal_insights = self.generate_legal_insights(feedback, industry)
        followup_strategy = self.generate_followup_strategy(
            theme_analysis['themes'], emotions, theme_analysis['nps_category'], knowledge_gaps
        )
        
        return {
            'theme_analysis': theme_analysis,
            'emotions': emotions,
            'knowledge_gaps': knowledge_gaps,
            'cross_channel': cross_channel,
            'legal_insights': legal_insights,
            'followup_strategy': followup_strategy,
            'nps_score': nps_score,
            'segment': segment,
            'industry': industry
        }

@st.cache_data
def load_analyzer():
    return StrategicFeedbackAnalyzer()

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Strategic Feedback Analyzer</h1>
        <p>Amplify strategic thinking beyond sentiment analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = load_analyzer()
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    analysis_type = st.sidebar.selectbox("Analysis Type", ["Individual Feedback", "Batch Analysis"])
    
    if analysis_type == "Individual Feedback":
        individual_analysis(analyzer)
    else:
        batch_analysis(analyzer)

def individual_analysis(analyzer):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Input Data")
        feedback = st.text_area(
            "Customer Feedback:", 
            value="I love the platform and it usually works great but sometimes I have to manually export files",
            height=120
        )
        email = st.text_area(
            "Related Email (optional):", 
            value="Hi team, the export feature seems to have some issues. I've been doing manual workarounds.",
            height=120
        )
    
    with col2:
        st.subheader("📊 Metadata")
        nps_score = st.slider("NPS Score", 0, 10, 8)
        segment = st.selectbox("Customer Segment", ["enterprise", "smb", "startup"])
        industry = st.selectbox("Industry", ["legal", "tech", "finance", "healthcare"])
    
    if st.button("🔍 Analyze Feedback", type="primary"):
        with st.spinner("Analyzing feedback..."):
            results = analyzer.analyze_feedback(feedback, email, nps_score, segment, industry)
            display_individual_results(analyzer, results)

def display_individual_results(analyzer, results):
    st.subheader("🎯 Analysis Results")
    
    # Theme Analysis
    with st.expander("🏷️ Thematic Analysis & Segmentation", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Detected Themes:**")
            for theme in results['theme_analysis']['themes']:
                confidence = int(theme['confidence'] * 100)
                st.markdown(f"""
                <span class="theme-tag">{theme['theme'].replace('_', ' ')} ({confidence}%)</span>
                """, unsafe_allow_html=True)
            
            # Fixed the problematic line
            evidence_list = [theme['evidence'] for theme in results['theme_analysis']['themes']]
            evidence_text = ', '.join([f'"{evidence}"' for evidence in evidence_list])
            st.write(f"**Evidence:** {evidence_text}")
            st.write(f"**Tags:** {results['theme_analysis']['tags']}")
        
        with col2:
            category = results['theme_analysis']['nps_category']
            st.markdown(f"""
            <span class="theme-tag nps-{category}">{category.upper()} (Score: {results['nps_score']})</span>
            """, unsafe_allow_html=True)
    
    # What-If Simulation
    with st.expander("🔮 What-If Simulation", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            friction_change = st.slider("Friction Reduction", -3, 3, 0, key="friction")
        with col2:
            risk_change = st.slider("Risk Mitigation", -3, 3, 0, key="risk")
        with col3:
            feature_change = st.slider("Feature Enhancement", -3, 3, 0, key="feature")
        
        simulation = analyzer.simulate_nps_change(results['nps_score'], friction_change, risk_change, feature_change)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current NPS", simulation['current'])
        with col2:
            st.metric("Projected NPS", simulation['projected'], simulation['change'])
        with col3:
            st.metric("Confidence", f"{int(simulation['confidence'] * 100)}%")
    
    # Cross-Channel Analysis
    with st.expander("🔗 Cross-Channel & Legal Analysis"):
        if results['cross_channel']:
            st.write("**Cross-Channel Correlations:**")
            for corr in results['cross_channel']:
                risk_color = "red" if corr['risk_level'] == 'high' else "orange"
                st.markdown(f"""
                <div class="correlation-item">
                    <strong>{corr['type']}:</strong> {corr['description']} 
                    <span style="color: {risk_color};">({corr['risk_level']} risk)</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.write("No cross-channel correlations detected")
        
        if results['legal_insights']:
            st.write("**Legal Domain Insights:**")
            for insight in results['legal_insights']:
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{insight['term']}:</strong> {insight['context']}<br>
                    <em>Implication: {insight['implication']}</em>
                </div>
                """, unsafe_allow_html=True)
    
    # Emotion Analysis & Strategy
    with st.expander("😊 Emotion Analysis & Follow-up Strategy"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Detected Emotions:**")
            for emotion in results['emotions']:
                intensity = int(emotion['intensity'] * 100)
                st.markdown(f"""
                <div class="emotion-mapping">
                    <strong>{emotion['emotion']}:</strong> {intensity}% intensity
                </div>
                """, unsafe_allow_html=True)
            
            if results['knowledge_gaps']:
                st.write("**Knowledge Gaps:**")
                for gap in results['knowledge_gaps']:
                    st.write(f"• {gap['phrase']} - {gap['implication']}")
        
        with col2:
            strategy = results['followup_strategy']
            st.write(f"**Priority Level:** {strategy['priority'].upper()}")
            
            st.write("**What to Say:**")
            for item in strategy['what_to_say']:
                st.write(f"• {item}")
            
            st.write("**What to Do:**")
            for item in strategy['what_to_do']:
                st.write(f"• {item}")
    
    # Visualizations
    with st.expander("📊 Visualizations"):
        col1, col2 = st.columns(2)
        
        with col1:
            if results['theme_analysis']['themes']:
                theme_df = pd.DataFrame(results['theme_analysis']['themes'])
                theme_df['confidence_pct'] = theme_df['confidence'] * 100
                theme_df['theme_clean'] = theme_df['theme'].str.replace('_', ' ').str.title()
                
                fig = px.bar(
                    theme_df, 
                    x='theme_clean', 
                    y='confidence_pct',
                    title="Theme Detection Confidence",
                    labels={'confidence_pct': 'Confidence %', 'theme_clean': 'Theme'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # NPS gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = results['nps_score'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "NPS Score"},
                gauge = {
                    'axis': {'range': [None, 10]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 6], 'color': "lightgray"},
                        {'range': [6, 8], 'color': "gray"},
                        {'range': [8, 10], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 9
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    st.subheader("📤 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Download CSV"):
            csv_data = create_csv_export(results)
            st.download_button(
                label="Download Analysis Report",
                data=csv_data,
                file_name="feedback_analysis.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📑 Download Report"):
            report_data = create_text_report(results)
            st.download_button(
                label="Download Text Report",
                data=report_data,
                file_name="feedback_report.txt",
                mime="text/plain"
            )

def batch_analysis(analyzer):
    st.subheader("📊 Batch Analysis")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            # Column mapping
            st.subheader("🔗 Column Mapping")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                feedback_col = st.selectbox("Feedback Text Column", df.columns)
            with col2:
                nps_col = st.selectbox("NPS Score Column", df.columns, index=1 if len(df.columns) > 1 else 0)
            with col3:
                email_col = st.selectbox("Email Column (optional)", ["None"] + list(df.columns))
            
            if st.button("🚀 Process Batch", type="primary"):
                with st.spinner("Processing batch analysis..."):
                    results = process_batch_data(analyzer, df, feedback_col, nps_col, email_col)
                    display_batch_results(results)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def process_batch_data(analyzer, df, feedback_col, nps_col, email_col):
    results = []
    
    for idx, row in df.iterrows():
        feedback = str(row[feedback_col])
        nps = int(row[nps_col]) if pd.notna(row[nps_col]) else 5
        email = str(row[email_col]) if email_col != "None" and pd.notna(row[email_col]) else ""
        
        analysis = analyzer.analyze_feedback(feedback, email, nps)
        results.append(analysis)
    
    return results

def display_batch_results(results):
    st.subheader("📈 Batch Analysis Summary")
    
    # Aggregate statistics
    total_count = len(results)
    avg_nps = np.mean([r['nps_score'] for r in results])
    
    category_counts = Counter([r['theme_analysis']['nps_category'] for r in results])
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Responses", total_count)
    with col2:
        st.metric("Average NPS", f"{avg_nps:.1f}")
    with col3:
        st.metric("Promoters", category_counts.get('promoter', 0))
    with col4:
        st.metric("Detractors", category_counts.get('detractor', 0))
    
    # Theme aggregation
    all_themes = []
    for result in results:
        for theme in result['theme_analysis']['themes']:
            all_themes.append({
                'theme': theme['theme'],
                'category': result['theme_analysis']['nps_category']
            })
    
    theme_df = pd.DataFrame(all_themes)
    
    if not theme_df.empty:
        theme_summary = theme_df.groupby(['theme', 'category']).size().unstack(fill_value=0)
        theme_summary['total'] = theme_summary.sum(axis=1)
        theme_summary = theme_summary.sort_values('total', ascending=False)
        
        st.subheader("🏷️ Theme Distribution")
        
        # Theme frequency chart
        fig = px.bar(
            x=theme_summary.index,
            y=theme_summary['total'],
            title="Most Common Themes",
            labels={'x': 'Theme', 'y': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Stacked chart by NPS category
        fig2 = px.bar(
            theme_summary.reset_index(),
            x='theme',
            y=['promoter', 'passive', 'detractor'],
            title="Theme Distribution by NPS Category",
            labels={'value': 'Count', 'theme': 'Theme'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Export batch results
    if st.button("📊 Export Batch Results"):
        batch_csv = create_batch_csv_export(results)
        st.download_button(
            label="Download Batch Analysis",
            data=batch_csv,
            file_name="batch_analysis.csv",
            mime="text/csv"
        )

def create_csv_export(results):
    data = {
        'NPS_Score': [results['nps_score']],
        'NPS_Category': [results['theme_analysis']['nps_category']],
        'Themes': ['; '.join([t['theme'] for t in results['theme_analysis']['themes']])],
        'Theme_Confidence': ['; '.join([f"{t['confidence']:.2f}" for t in results['theme_analysis']['themes']])],
        'Emotions': ['; '.join([e['emotion'] for e in results['emotions']])],
        'Follow_up_Priority': [results['followup_strategy']['priority']],
        'Knowledge_Gaps': [len(results['knowledge_gaps'])],
        'Cross_Channel_Issues': [len(results['cross_channel'])]
    }
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

def create_text_report(results):
    report = f"""
Strategic Feedback Analysis Report
================================

NPS Score: {results['nps_score']}
Category: {results['theme_analysis']['nps_category'].upper()}
Segment: {results['segment']}
Industry: {results['industry']}

THEMES DETECTED:
"""
    for theme in results['theme_analysis']['themes']:
        report += f"- {theme['theme']}: {theme['confidence']:.0%} confidence\n"
        report += f"  Evidence: {theme['evidence']}\n"

    report += f"\nEMOTIONS:\n"
    for emotion in results['emotions']:
        report += f"- {emotion['emotion']}: {emotion['intensity']:.0%} intensity\n"

    report += f"\nSTRATEGIC RECOMMENDATIONS:\n"
    report += f"Priority: {results['followup_strategy']['priority'].upper()}\n\n"
    
    report += "What to Say:\n"
    for item in results['followup_strategy']['what_to_say']:
        report += f"- {item}\n"
    
    report += "\nWhat to Do:\n"
    for item in results['followup_strategy']['what_to_do']:
        report += f"- {item}\n"

    if results['knowledge_gaps']:
        report += f"\nKNOWLEDGE GAPS DETECTED:\n"
        for gap in results['knowledge_gaps']:
            report += f"- {gap['phrase']}: {gap['implication']}\n"

    return report

def create_batch_csv_export(results):
    data = []
    for i, result in enumerate(results):
        row = {
            'Response_ID': i + 1,
            'NPS_Score': result['nps_score'],
            'NPS_Category': result['theme_analysis']['nps_category'],
            'Themes': '; '.join([t['theme'] for t in result['theme_analysis']['themes']]),
            'Emotions': '; '.join([e['emotion'] for e in result['emotions']]),
            'Priority': result['followup_strategy']['priority'],
            'Knowledge_Gaps': len(result['knowledge_gaps']),
            'Cross_Channel_Issues': len(result['cross_channel'])
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

if __name__ == "__main__":
    main()
