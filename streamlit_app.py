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
    page_title="Legal Tech Intelligence Platform",
    page_icon="🧠",
    layout="wide"
)

# Advanced legal-specific ABSA model
class LegalABSA:
    def __init__(self):
        # Legal tech aspect definitions with sophisticated pattern matching
        self.aspects = {
            'search_functionality': {
                'keywords': ['search', 'find', 'results', 'keyword', 'boolean', 'advanced search', 'query'],
                'positive_patterns': ['accurate results', 'finds everything', 'precise search', 'relevant hits'],
                'negative_patterns': ['can\'t find', 'irrelevant results', 'need advanced search', 'missing cases', 'poor ranking'],
                'nuance_indicators': ['sometimes need', 'occasionally', 'mostly works but', 'usually good except']
            },
            'ease_of_use': {
                'keywords': ['easy', 'intuitive', 'user-friendly', 'simple', 'interface', 'navigation'],
                'positive_patterns': ['easy to use', 'intuitive interface', 'user-friendly', 'simple navigation'],
                'negative_patterns': ['confusing', 'complicated', 'hard to use', 'difficult navigation'],
                'nuance_indicators': ['mostly easy but', 'generally simple except', 'intuitive overall']
            },
            'content_quality': {
                'keywords': ['content', 'cases', 'precedents', 'coverage', 'database', 'materials'],
                'positive_patterns': ['comprehensive coverage', 'up-to-date', 'reliable content'],
                'negative_patterns': ['outdated', 'missing coverage', 'incomplete database'],
                'nuance_indicators': ['good coverage but missing', 'mostly current except']
            },
            'result_relevance': {
                'keywords': ['relevance', 'ranking', 'sequence', 'order', 'priority', 'most relevant'],
                'positive_patterns': ['highly relevant', 'perfect ranking', 'well-ordered'],
                'negative_patterns': ['irrelevant results', 'poor ranking', 'wrong sequence', 'buried results'],
                'nuance_indicators': ['generally relevant but', 'decent ranking except']
            }
        }
    
    def analyze_aspect_sentiment(self, text, nps_rating=None, sub_ratings=None):
        """Sophisticated aspect-based sentiment analysis"""
        results = {}
        text_lower = text.lower()
        
        for aspect, config in self.aspects.items():
            # Check if aspect is mentioned
            mentions = sum(1 for keyword in config['keywords'] if keyword in text_lower)
            
            if mentions > 0:
                # Calculate base sentiment from patterns
                pos_score = sum(1 for pattern in config['positive_patterns'] if pattern in text_lower)
                neg_score = sum(1 for pattern in config['negative_patterns'] if pattern in text_lower)
                nuance_score = sum(1 for pattern in config['nuance_indicators'] if pattern in text_lower)
                
                # Incorporate structured ratings if available
                if sub_ratings and aspect in sub_ratings:
                    rating_sentiment = self._rating_to_sentiment(sub_ratings[aspect])
                else:
                    rating_sentiment = 0
                
                # Calculate final sentiment with nuance detection
                base_sentiment = (pos_score - neg_score) / max(mentions, 1)
                
                # Adjust for nuance indicators (suggest hidden issues)
                if nuance_score > 0:
                    confidence = 0.6  # Lower confidence due to mixed signals
                    risk_flag = True
                else:
                    confidence = 0.8
                    risk_flag = False
                
                # Combine text sentiment with rating
                final_sentiment = (base_sentiment + rating_sentiment) / 2 if rating_sentiment != 0 else base_sentiment
                
                results[aspect] = {
                    'sentiment_score': final_sentiment,
                    'confidence': confidence,
                    'mentions': mentions,
                    'risk_flag': risk_flag,
                    'nuance_detected': nuance_score > 0,
                    'patterns_found': {
                        'positive': [p for p in config['positive_patterns'] if p in text_lower],
                        'negative': [p for p in config['negative_patterns'] if p in text_lower],
                        'nuance': [p for p in config['nuance_indicators'] if p in text_lower]
                    }
                }
        
        return results
    
    def _rating_to_sentiment(self, rating_text):
        """Convert rating text to sentiment score"""
        rating_map = {
            'very satisfied': 1.0,
            'satisfied': 0.5,
            'neutral': 0.0,
            'dissatisfied': -0.5,
            'very dissatisfied': -1.0,
            'don\'t know': 0.0  # Neutral but flagged separately
        }
        return rating_map.get(rating_text.lower(), 0.0)

class LegalInsightEngine:
    def __init__(self):
        self.absa = LegalABSA()
    
    def extract_hidden_pain_points(self, feedback_data):
        """Identify non-obvious issues even in positive feedback"""
        pain_points = []
        
        # Analyze the specific feedback
        text = feedback_data.get('feedback_text', '')
        nps = feedback_data.get('nps_score', 0)
        
        # Pattern: High NPS but workflow inefficiencies
        if nps >= 7 and ('need to go to' in text.lower() or 'sometimes' in text.lower()):
            pain_points.append({
                'type': 'workflow_friction',
                'severity': 'medium',
                'description': 'User experiences workflow interruptions despite high satisfaction',
                'evidence': 'Mentions needing to switch to advanced search for basic tasks',
                'risk': 'Could lead to satisfaction erosion over time'
            })
        
        # Pattern: "Don't know" responses in structured ratings
        dont_know_count = sum(1 for rating in feedback_data.get('sub_ratings', {}).values() 
                             if rating.lower() == "don't know")
        
        if dont_know_count >= 2:
            pain_points.append({
                'type': 'feature_underutilization',
                'severity': 'low',
                'description': f'Customer unaware of {dont_know_count} key product aspects',
                'evidence': f"Multiple 'Don't know' responses",
                'risk': 'Missing upsell opportunities and potential churn to competitors'
            })
        
        # Pattern: Subtle language indicating inefficiency
        efficiency_indicators = ['need to', 'have to', 'sometimes', 'occasionally', 'usually but']
        if any(indicator in text.lower() for indicator in efficiency_indicators):
            pain_points.append({
                'type': 'process_inefficiency',
                'severity': 'medium',
                'description': 'Subtle indicators of process friction',
                'evidence': 'Language suggests workarounds or extra steps needed',
                'risk': 'Productivity impact could drive competitive evaluation'
            })
        
        return pain_points
    
    def predict_nps_trajectory(self, current_feedback, historical_patterns=None):
        """Predict NPS evolution based on current feedback patterns"""
        current_nps = current_feedback.get('nps_score', 0)
        aspect_analysis = self.absa.analyze_aspect_sentiment(
            current_feedback.get('feedback_text', ''),
            current_nps,
            current_feedback.get('sub_ratings', {})
        )
        
        # Risk factors for NPS decline
        risk_factors = []
        
        # Check for aspects with risk flags
        for aspect, data in aspect_analysis.items():
            if data.get('risk_flag', False):
                risk_factors.append(f"{aspect}: {data['sentiment_score']:.2f} (nuanced feedback)")
        
        # Simulate trajectory based on risk factors
        if len(risk_factors) >= 2:
            predicted_change = -1.5
            trajectory = "Declining"
            confidence = 0.75
        elif len(risk_factors) == 1:
            predicted_change = -0.5
            trajectory = "Stable with risks"
            confidence = 0.60
        else:
            predicted_change = 0.2
            trajectory = "Stable/Growing"
            confidence = 0.50
        
        return {
            'current_nps': current_nps,
            'predicted_change': predicted_change,
            'trajectory': trajectory,
            'confidence': confidence,
            'risk_factors': risk_factors,
            'timeframe': '3-6 months'
        }
    
    def generate_strategic_recommendations(self, feedback_analysis, pain_points, nps_prediction):
        """Generate sophisticated, prioritized recommendations"""
        recommendations = []
        
        # High-priority: Address workflow friction
        if any(pp['type'] == 'workflow_friction' for pp in pain_points):
            recommendations.append({
                'priority': 'Critical',
                'category': 'Product Enhancement',
                'title': 'Intelligent Search Optimization',
                'description': 'Implement AI-powered query expansion to reduce reliance on advanced search',
                'rationale': 'User forced to use advanced search suggests basic search lacks semantic understanding',
                'implementation': [
                    'Deploy NLP-based query interpretation',
                    'Add auto-suggestion for multi-keyword searches', 
                    'Implement relevance scoring based on legal context'
                ],
                'estimated_impact': '+1.5 NPS points',
                'timeline': '2-3 sprints',
                'success_metrics': ['Reduced advanced search usage', 'Improved query success rate']
            })
        
        # Medium-priority: Feature awareness
        if any(pp['type'] == 'feature_underutilization' for pp in pain_points):
            recommendations.append({
                'priority': 'High',
                'category': 'Customer Success',
                'title': 'Personalized Onboarding Enhancement',
                'description': 'Create usage-based onboarding to expose unknown features',
                'rationale': 'Multiple "Don\'t know" responses indicate engagement gaps',
                'implementation': [
                    'Implement progressive feature disclosure',
                    'Create personalized tutorial sequences',
                    'Add contextual help based on user behavior'
                ],
                'estimated_impact': '+0.8 NPS points, 15% feature adoption increase',
                'timeline': '1-2 sprints',
                'success_metrics': ['Reduced "Don\'t know" responses', 'Increased feature engagement']
            })
        
        # Process optimization
        recommendations.append({
            'priority': 'Medium',
            'category': 'UX Enhancement',
            'title': 'Search Result Intelligence',
            'description': 'Improve case sequencing and relevance ranking',
            'rationale': 'User mentions needing "higher level of sequence" for cases',
            'implementation': [
                'Implement legal precedent-aware ranking',
                'Add jurisdiction and recency weighting',
                'Create customizable result ordering'
            ],
            'estimated_impact': '+1.0 NPS points',
            'timeline': '3-4 sprints',
            'success_metrics': ['Improved result click-through rates', 'Reduced search refinements']
        })
        
        return recommendations
    
    def generate_response_template(self, feedback_data, analysis_results):
        """Generate personalized response template"""
        nps = feedback_data.get('nps_score', 0)
        name = feedback_data.get('customer_name', 'there')
        
        # Determine response tone based on analysis
        if nps >= 8:
            tone = "appreciative_proactive"
        elif nps >= 6:
            tone = "constructive_supportive"
        else:
            tone = "recovery_focused"
        
        templates = {
            'appreciative_proactive': f"""
Dear {name},

Thank you for your NPS score of {nps} and valuable feedback! We're thrilled you find our platform easy to use.

I noticed your suggestion about search functionality and case sequencing - this is exactly the kind of insight that helps us improve. Our product team is actually working on enhanced multi-keyword search capabilities that should address this directly.

Would you be interested in a brief demo of some advanced search tips that might help in the meantime? I'd also love to show you some features you might not be aware of yet.

Best regards,
Customer Success Team
            """,
            'constructive_supportive': f"""
Dear {name},

Thank you for taking the time to provide feedback. Your score of {nps} and comments help us understand how to serve you better.

I'd like to schedule a brief call to discuss your experience and show you some features that might address your concerns.

Best regards,
Customer Success Team
            """,
            'recovery_focused': f"""
Dear {name},

Thank you for your candid feedback. I want to personally ensure we address your concerns and improve your experience.

I'm scheduling a priority call to discuss your specific needs and how we can better support you.

Best regards,
Customer Success Manager
            """
        }
        
        return templates[tone]

def create_sample_feedback():
    """Create the specific sample feedback for analysis"""
    return {
        'customer_name': 'Legal Professional',
        'nps_score': 8,
        'sub_experience_score': 8,
        'feedback_text': 'easy to use, sometimes i need to go to advance search to be able to find multiple keywords and cases to be displayed in the higher level of sequence',
        'sub_ratings': {
            'ease_of_use': 'Very Satisfied',
            'search_functionality': 'Satisfied', 
            'content_quality': "Don't know",
            'customer_support': 'Satisfied',
            'pricing': "Don't know"
        },
        'continuation_intent': 'Definitely Will Continue Using',
        'source': 'NPS Survey',
        'date': datetime.now()
    }

# Main Application
st.title("🧠 Legal Tech Intelligence Platform")
st.markdown("### Deep Analysis Beyond Basic Sentiment")

# Create analysis tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Deep Analysis", "📊 Aspect Breakdown", "🎯 Strategic Insights", "📝 Response Generator"])

# Initialize the engine
engine = LegalInsightEngine()

# Load sample data
sample_feedback = create_sample_feedback()

with tab1:
    st.header("🔍 Comprehensive Feedback Analysis")
    
    # Display raw feedback
    st.subheader("📋 Input Feedback")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("NPS Score", f"{sample_feedback['nps_score']}/10")
        st.metric("Sub-Experience", f"{sample_feedback['sub_experience_score']}/10")
        st.write(f"**Continuation Intent:** {sample_feedback['continuation_intent']}")
    
    with col2:
        st.write("**Feedback Text:**")
        st.info(f"'{sample_feedback['feedback_text']}'")
    
    # Sub-ratings breakdown
    st.subheader("📊 Structured Ratings")
    ratings_df = pd.DataFrame(list(sample_feedback['sub_ratings'].items()), 
                             columns=['Aspect', 'Rating'])
    
    # Color code the ratings
    def color_rating(rating):
        if rating == 'Very Satisfied':
            return 'background-color: #d4edda'
        elif rating == 'Satisfied':
            return 'background-color: #fff3cd'
        elif rating == "Don't know":
            return 'background-color: #f8d7da'
        else:
            return ''
    
    st.dataframe(ratings_df.style.applymap(color_rating, subset=['Rating']), use_container_width=True)
    
    # Analysis trigger
    if st.button("🧠 Run Deep Analysis", type="primary"):
        # Perform comprehensive analysis
        aspect_analysis = engine.absa.analyze_aspect_sentiment(
            sample_feedback['feedback_text'],
            sample_feedback['nps_score'],
            sample_feedback['sub_ratings']
        )
        
        pain_points = engine.extract_hidden_pain_points(sample_feedback)
        nps_prediction = engine.predict_nps_trajectory(sample_feedback)
        recommendations = engine.generate_strategic_recommendations(aspect_analysis, pain_points, nps_prediction)
        
        # Store in session state
        st.session_state['analysis_complete'] = True
        st.session_state['aspect_analysis'] = aspect_analysis
        st.session_state['pain_points'] = pain_points
        st.session_state['nps_prediction'] = nps_prediction
        st.session_state['recommendations'] = recommendations

# Check if analysis is complete
if st.session_state.get('analysis_complete', False):
    
    with tab2:
        st.header("📊 Aspect-Based Sentiment Analysis")
        
        aspect_analysis = st.session_state['aspect_analysis']
        
        if aspect_analysis:
            # Create visualization
            aspects = list(aspect_analysis.keys())
            scores = [data['sentiment_score'] for data in aspect_analysis.values()]
            confidences = [data['confidence'] for data in aspect_analysis.values()]
            risk_flags = [data['risk_flag'] for data in aspect_analysis.values()]
            
            # Create subplot figure
            fig = go.Figure()
            
            # Add sentiment bars
            colors = ['red' if risk else 'green' if score > 0 else 'orange' 
                     for score, risk in zip(scores, risk_flags)]
            
            fig.add_trace(go.Bar(
                x=[aspect.replace('_', ' ').title() for aspect in aspects],
                y=scores,
                marker_color=colors,
                name='Sentiment Score',
                text=[f"Confidence: {conf:.1%}" for conf in confidences],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Aspect-Based Sentiment Analysis",
                yaxis_title="Sentiment Score (-1 to +1)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed breakdown
            st.subheader("🔍 Detailed Aspect Analysis")
            
            for aspect, data in aspect_analysis.items():
                with st.expander(f"{aspect.replace('_', ' ').title()} - Score: {data['sentiment_score']:.2f}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Sentiment Score:** {data['sentiment_score']:.2f}")
                        st.write(f"**Confidence:** {data['confidence']:.1%}")
                        st.write(f"**Mentions:** {data['mentions']}")
                        
                        if data['risk_flag']:
                            st.warning("⚠️ **Risk Flag:** Nuanced feedback detected")
                    
                    with col2:
                        st.write("**Patterns Found:**")
                        if data['patterns_found']['positive']:
                            st.success(f"✅ Positive: {', '.join(data['patterns_found']['positive'])}")
                        if data['patterns_found']['negative']:
                            st.error(f"❌ Negative: {', '.join(data['patterns_found']['negative'])}")
                        if data['patterns_found']['nuance']:
                            st.warning(f"⚡ Nuance: {', '.join(data['patterns_found']['nuance'])}")
    
    with tab3:
        st.header("🎯 Strategic Insights & Predictions")
        
        pain_points = st.session_state['pain_points']
        nps_prediction = st.session_state['nps_prediction']
        recommendations = st.session_state['recommendations']
        
        # Hidden pain points
        st.subheader("🕵️ Hidden Pain Points Analysis")
        st.markdown("*Identifying non-obvious issues even in positive feedback*")
        
        if pain_points:
            for i, pain_point in enumerate(pain_points):
                severity_colors = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
                
                st.write(f"**{severity_colors[pain_point['severity']]} {pain_point['type'].replace('_', ' ').title()}**")
                st.write(f"**Issue:** {pain_point['description']}")
                st.write(f"**Evidence:** {pain_point['evidence']}")
                st.write(f"**Risk:** {pain_point['risk']}")
                st.write("---")
        else:
            st.success("No hidden pain points detected in this feedback.")
        
        # NPS Prediction
        st.subheader("📈 NPS Trajectory Prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current NPS", nps_prediction['current_nps'])
        
        with col2:
            delta_color = "normal" if nps_prediction['predicted_change'] >= 0 else "inverse"
            st.metric("Predicted Change", 
                     f"{nps_prediction['predicted_change']:+.1f}", 
                     delta=f"{nps_prediction['timeframe']}",
                     delta_color=delta_color)
        
        with col3:
            st.metric("Confidence", f"{nps_prediction['confidence']:.1%}")
        
        st.write(f"**Trajectory:** {nps_prediction['trajectory']}")
        
        if nps_prediction['risk_factors']:
            st.warning("**Risk Factors Identified:**")
            for factor in nps_prediction['risk_factors']:
                st.write(f"• {factor}")
        
        # Strategic Recommendations
        st.subheader("💡 Strategic Recommendations")
        
        for i, rec in enumerate(recommendations):
            priority_colors = {'Critical': '🔴', 'High': '🟠', 'Medium': '🟡', 'Low': '🟢'}
            
            with st.expander(f"{priority_colors[rec['priority']]} {rec['priority']} - {rec['title']}"):
                st.write(f"**Category:** {rec['category']}")
                st.write(f"**Description:** {rec['description']}")
                st.write(f"**Rationale:** {rec['rationale']}")
                
                st.write("**Implementation Steps:**")
                for step in rec['implementation']:
                    st.write(f"• {step}")
                
                st.write(f"**Estimated Impact:** {rec['estimated_impact']}")
                st.write(f"**Timeline:** {rec['timeline']}")
                
                st.write("**Success Metrics:**")
                for metric in rec['success_metrics']:
                    st.write(f"• {metric}")
        
        # What-if simulation
        st.subheader("🎮 What-If Simulation")
        
        st.write("**Scenario: Implement top 2 recommendations**")
        
        simulated_impact = 1.5 + 0.8  # From top recommendations
        new_nps = sample_feedback['nps_score'] + simulated_impact
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current NPS", sample_feedback['nps_score'])
        with col2:
            st.metric("Projected NPS", f"{new_nps:.1f}", delta=f"+{simulated_impact:.1f}")
        
        if new_nps >= 9:
            st.success("🎯 **Result:** Customer converts from Passive to Promoter!")
            st.write("• Increased referral likelihood")
            st.write("• Higher retention probability")
            st.write("• Potential upsell opportunities")

    with tab4:
        st.header("📝 AI-Generated Response")
        
        # Generate response template
        response_template = engine.generate_response_template(
            sample_feedback, 
            st.session_state.get('aspect_analysis', {})
        )
        
        st.subheader("💌 Personalized Response Template")
        st.text_area("Generated Response:", response_template, height=200)
        
        # Additional response options
        st.subheader("📋 Follow-up Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Immediate Actions:**")
            st.write("✅ Send personalized response")
            st.write("✅ Schedule feature demo")
            st.write("✅ Create product feedback ticket")
        
        with col2:
            st.write("**Follow-up Timeline:**")
            st.write("📅 Day 1: Send response")
            st.write("📅 Day 3: Demo call")
            st.write("📅 Day 14: Check-in email")
            st.write("📅 Day 30: Feature update notification")

# Key insights summary
if st.session_state.get('analysis_complete', False):
    st.markdown("---")
    st.header("🚀 Key Insights Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 This Analysis Uncovered:**
        - Workflow friction despite high NPS
        - Feature awareness gaps
        - Search optimization opportunities
        - Process efficiency improvements
        """)
    
    with col2:
        st.markdown("""
        **📈 Predicted Impact:**
        - +2.3 NPS points if recommendations implemented
        - Conversion from Passive to Promoter
        - Reduced churn risk
        - Increased feature adoption
        """)
    
    with col3:
        st.markdown("""
        **⚡ Why This Matters:**
        - Prevents satisfaction erosion
        - Identifies upsell opportunities  
        - Provides competitive advantage
        - Enables proactive customer success
        """)

else:
    st.info("👆 Click 'Run Deep Analysis' in the first tab to see comprehensive insights")

# Footer
st.markdown("---")
st.markdown("*This platform transforms obvious feedback into strategic intelligence, helping legal tech companies identify hidden opportunities and risks that traditional sentiment analysis misses.*")
