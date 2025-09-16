import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re
from collections import defaultdict, Counter

# Page config
st.set_page_config(
    page_title="Deep Feedback Intelligence Analyzer",
    page_icon="🧠",
    layout="wide"
)

class DeepFeedbackAnalyzer:
    def __init__(self):
        # Legal-specific aspect patterns
        self.legal_aspects = {
            'search_functionality': {
                'keywords': ['search', 'find', 'results', 'keyword', 'boolean', 'advanced search', 'query', 'lookup'],
                'positive_signals': ['accurate', 'precise', 'finds everything', 'comprehensive results', 'relevant'],
                'negative_signals': ['can\'t find', 'irrelevant', 'missing', 'poor results', 'incomplete'],
                'friction_signals': ['need to go to', 'have to use', 'sometimes', 'switch to', 'manually']
            },
            'ease_of_use': {
                'keywords': ['easy', 'intuitive', 'user-friendly', 'simple', 'interface', 'navigation', 'usable'],
                'positive_signals': ['easy to use', 'intuitive', 'user-friendly', 'straightforward', 'clear'],
                'negative_signals': ['confusing', 'complicated', 'hard to use', 'difficult', 'unclear'],
                'friction_signals': ['learning curve', 'getting used to', 'figuring out']
            },
            'content_quality': {
                'keywords': ['content', 'cases', 'precedents', 'coverage', 'database', 'materials', 'information'],
                'positive_signals': ['comprehensive', 'up-to-date', 'reliable', 'complete', 'thorough'],
                'negative_signals': ['outdated', 'missing', 'incomplete', 'unreliable', 'limited'],
                'friction_signals': ['some gaps', 'mostly good but', 'generally complete except']
            },
            'performance': {
                'keywords': ['speed', 'fast', 'slow', 'performance', 'loading', 'response', 'lag', 'quick'],
                'positive_signals': ['fast', 'quick', 'responsive', 'instant', 'speedy'],
                'negative_signals': ['slow', 'sluggish', 'laggy', 'takes forever', 'unresponsive'],
                'friction_signals': ['usually fast but', 'mostly quick except', 'sometimes slow']
            },
            'support_quality': {
                'keywords': ['support', 'help', 'service', 'assistance', 'team', 'response', 'customer service'],
                'positive_signals': ['helpful', 'responsive', 'excellent support', 'quick response', 'knowledgeable'],
                'negative_signals': ['unhelpful', 'slow response', 'poor service', 'unresponsive', 'rude'],
                'friction_signals': ['usually helpful but', 'good support except', 'sometimes takes time']
            },
            'pricing_value': {
                'keywords': ['price', 'cost', 'expensive', 'cheap', 'value', 'worth', 'budget', 'fee'],
                'positive_signals': ['good value', 'worth it', 'reasonable', 'affordable', 'cost-effective'],
                'negative_signals': ['expensive', 'overpriced', 'not worth', 'too costly', 'budget strain'],
                'friction_signals': ['mostly worth it but', 'good value except', 'reasonable for most']
            }
        }
        
        # Emotion indicators
        self.emotions = {
            'frustrated': ['frustrated', 'annoying', 'irritated', 'fed up', 'bothered'],
            'disappointed': ['disappointed', 'let down', 'expected more', 'underwhelmed'],
            'confused': ['confused', 'unclear', 'don\'t understand', 'puzzled', 'lost'],
            'satisfied': ['satisfied', 'happy', 'pleased', 'content', 'good'],
            'delighted': ['love', 'amazing', 'fantastic', 'excellent', 'outstanding', 'brilliant'],
            'concerned': ['worried', 'concerned', 'anxious', 'nervous', 'uncertain']
        }
        
        # Hidden risk patterns
        self.risk_patterns = {
            'workflow_friction': ['need to', 'have to', 'sometimes', 'usually but', 'except when'],
            'feature_gaps': ['don\'t know', 'not sure', 'haven\'t tried', 'unaware', 'never used'],
            'competitive_risk': ['compared to', 'vs', 'other tools', 'alternatives', 'competitors'],
            'churn_signals': ['considering', 'looking at', 'might switch', 'evaluating', 'thinking about']
        }

    def analyze_feedback(self, feedback_text, nps_score=None, structured_ratings=None):
        """Comprehensive feedback analysis"""
        
        # 1. Aspect-Based Sentiment Analysis
        aspect_analysis = self._analyze_aspects(feedback_text, structured_ratings)
        
        # 2. Hidden Pain Point Detection
        pain_points = self._detect_hidden_pain_points(feedback_text, nps_score, aspect_analysis)
        
        # 3. Emotion Analysis
        emotions = self._analyze_emotions(feedback_text)
        
        # 4. Risk Assessment
        risks = self._assess_risks(feedback_text, nps_score, aspect_analysis)
        
        # 5. NPS Trajectory Prediction
        trajectory = self._predict_trajectory(feedback_text, nps_score, aspect_analysis, risks)
        
        # 6. Strategic Recommendations
        recommendations = self._generate_recommendations(aspect_analysis, pain_points, risks, emotions)
        
        return {
            'aspect_analysis': aspect_analysis,
            'pain_points': pain_points,
            'emotions': emotions,
            'risks': risks,
            'trajectory': trajectory,
            'recommendations': recommendations
        }
    
    def _analyze_aspects(self, text, structured_ratings=None):
        """Sophisticated aspect-based sentiment analysis"""
        results = {}
        text_lower = text.lower()
        
        for aspect, config in self.legal_aspects.items():
            # Check if aspect is mentioned
            keyword_mentions = sum(1 for keyword in config['keywords'] if keyword in text_lower)
            
            if keyword_mentions > 0:
                # Calculate sentiment signals
                positive_count = sum(1 for signal in config['positive_signals'] if signal in text_lower)
                negative_count = sum(1 for signal in config['negative_signals'] if signal in text_lower)
                friction_count = sum(1 for signal in config['friction_signals'] if signal in text_lower)
                
                # Base sentiment calculation
                if positive_count > negative_count:
                    base_sentiment = 0.7
                elif negative_count > positive_count:
                    base_sentiment = -0.7
                else:
                    base_sentiment = 0.0
                
                # Adjust for friction (hidden issues)
                if friction_count > 0:
                    confidence = 0.6  # Lower confidence due to mixed signals
                    hidden_risk = True
                    if base_sentiment > 0:
                        base_sentiment *= 0.7  # Reduce positive sentiment
                else:
                    confidence = 0.85
                    hidden_risk = False
                
                # Incorporate structured ratings if available
                if structured_ratings and aspect in structured_ratings:
                    rating_sentiment = self._convert_rating_to_sentiment(structured_ratings[aspect])
                    # Weight: 60% text, 40% rating
                    final_sentiment = (base_sentiment * 0.6) + (rating_sentiment * 0.4)
                else:
                    final_sentiment = base_sentiment
                
                results[aspect] = {
                    'sentiment_score': final_sentiment,
                    'confidence': confidence,
                    'mentions': keyword_mentions,
                    'hidden_risk': hidden_risk,
                    'signals': {
                        'positive': positive_count,
                        'negative': negative_count,
                        'friction': friction_count
                    },
                    'evidence': self._extract_evidence(text, config['keywords'])
                }
        
        return results
    
    def _detect_hidden_pain_points(self, text, nps_score, aspect_analysis):
        """Identify non-obvious issues even in positive feedback"""
        pain_points = []
        text_lower = text.lower()
        
        # Pattern 1: High NPS but workflow friction
        if nps_score and nps_score >= 7:
            friction_indicators = ['need to', 'have to', 'sometimes', 'usually but', 'except']
            friction_detected = any(indicator in text_lower for indicator in friction_indicators)
            
            if friction_detected:
                pain_points.append({
                    'type': 'workflow_inefficiency',
                    'severity': 'medium',
                    'description': 'User experiences process friction despite high satisfaction',
                    'evidence': [phrase for phrase in friction_indicators if phrase in text_lower],
                    'impact': 'Could lead to satisfaction erosion and competitive vulnerability',
                    'business_risk': 'Productivity impact may drive users to evaluate alternatives'
                })
        
        # Pattern 2: Positive sentiment with hidden friction
        for aspect, data in aspect_analysis.items():
            if data['sentiment_score'] > 0 and data['hidden_risk']:
                pain_points.append({
                    'type': f'{aspect}_optimization_opportunity',
                    'severity': 'low-medium',
                    'description': f'Positive {aspect.replace("_", " ")} experience with efficiency gaps',
                    'evidence': f'Friction signals detected: {data["signals"]["friction"]} instances',
                    'impact': 'Untapped potential for user satisfaction improvement',
                    'business_risk': 'Competitors with smoother workflows could gain advantage'
                })
        
        # Pattern 3: Knowledge gaps
        knowledge_gaps = ['don\'t know', 'not sure', 'haven\'t tried', 'unaware']
        gap_count = sum(1 for gap in knowledge_gaps if gap in text_lower)
        
        if gap_count > 0:
            pain_points.append({
                'type': 'feature_underutilization',
                'severity': 'low',
                'description': f'Customer unaware of {gap_count} product aspects or features',
                'evidence': [gap for gap in knowledge_gaps if gap in text_lower],
                'impact': 'Missing expansion and upsell opportunities',
                'business_risk': 'Underutilized customers more likely to churn'
            })
        
        return pain_points
    
    def _analyze_emotions(self, text):
        """Detect emotional undertones"""
        text_lower = text.lower()
        detected_emotions = {}
        
        for emotion, indicators in self.emotions.items():
            count = sum(1 for indicator in indicators if indicator in text_lower)
            if count > 0:
                detected_emotions[emotion] = {
                    'intensity': min(count / len(indicators), 1.0),
                    'indicators_found': [ind for ind in indicators if ind in text_lower]
                }
        
        return detected_emotions
    
    def _assess_risks(self, text, nps_score, aspect_analysis):
        """Assess business risks from feedback"""
        risks = []
        text_lower = text.lower()
        
        # Churn risk assessment
        churn_signals = sum(1 for pattern in self.risk_patterns['churn_signals'] if pattern in text_lower)
        competitive_mentions = sum(1 for pattern in self.risk_patterns['competitive_risk'] if pattern in text_lower)
        
        if churn_signals > 0:
            risks.append({
                'type': 'churn_risk',
                'level': 'high' if churn_signals >= 2 else 'medium',
                'description': 'Customer showing consideration of alternatives',
                'indicators': [pattern for pattern in self.risk_patterns['churn_signals'] if pattern in text_lower]
            })
        
        if competitive_mentions > 0:
            risks.append({
                'type': 'competitive_pressure',
                'level': 'medium',
                'description': 'Customer actively comparing with competitors',
                'indicators': [pattern for pattern in self.risk_patterns['competitive_risk'] if pattern in text_lower]
            })
        
        # Satisfaction erosion risk
        hidden_friction_count = sum(1 for data in aspect_analysis.values() if data.get('hidden_risk', False))
        
        if hidden_friction_count >= 2:
            risks.append({
                'type': 'satisfaction_erosion',
                'level': 'medium',
                'description': 'Multiple friction points could compound over time',
                'indicators': f'{hidden_friction_count} aspects with hidden friction detected'
            })
        
        return risks
    
    def _predict_trajectory(self, text, nps_score, aspect_analysis, risks):
        """Predict NPS trajectory"""
        if not nps_score:
            return {'prediction': 'Insufficient data', 'confidence': 0}
        
        # Base prediction from current NPS
        current_segment = 'promoter' if nps_score >= 9 else 'passive' if nps_score >= 7 else 'detractor'
        
        # Risk factor impact
        risk_impact = 0
        for risk in risks:
            if risk['level'] == 'high':
                risk_impact -= 2
            elif risk['level'] == 'medium':
                risk_impact -= 1
        
        # Friction impact
        friction_count = sum(1 for data in aspect_analysis.values() if data.get('hidden_risk', False))
        friction_impact = -0.5 * friction_count
        
        # Positive aspect impact
        positive_aspects = sum(1 for data in aspect_analysis.values() if data['sentiment_score'] > 0.5)
        positive_impact = 0.3 * positive_aspects
        
        # Calculate predicted change
        total_impact = risk_impact + friction_impact + positive_impact
        
        predictions = {
            'current_nps': nps_score,
            'current_segment': current_segment,
            'predicted_change': total_impact,
            'confidence': 0.7 if len(aspect_analysis) >= 2 else 0.5,
            'timeframe': '3-6 months',
            'key_factors': {
                'risk_impact': risk_impact,
                'friction_impact': friction_impact,
                'positive_impact': positive_impact
            }
        }
        
        return predictions
    
    def _generate_recommendations(self, aspect_analysis, pain_points, risks, emotions):
        """Generate strategic recommendations"""
        recommendations = []
        
        # High priority: Address friction in positive aspects
        friction_aspects = [aspect for aspect, data in aspect_analysis.items() 
                          if data.get('hidden_risk', False) and data['sentiment_score'] > 0]
        
        if friction_aspects:
            recommendations.append({
                'priority': 'High',
                'category': 'Optimization',
                'title': f'Optimize {", ".join([a.replace("_", " ").title() for a in friction_aspects[:2]])}',
                'description': 'Address workflow friction in currently positive aspects',
                'rationale': 'Prevent satisfaction erosion and unlock efficiency gains',
                'estimated_impact': f'+{len(friction_aspects) * 0.5:.1f} NPS points',
                'implementation_effort': 'Medium',
                'timeline': '2-3 sprints'
            })
        
        # Medium priority: Address negative aspects
        negative_aspects = [aspect for aspect, data in aspect_analysis.items() 
                           if data['sentiment_score'] < -0.3]
        
        if negative_aspects:
            recommendations.append({
                'priority': 'Critical',
                'category': 'Issue Resolution',
                'title': f'Resolve {negative_aspects[0].replace("_", " ").title()} Issues',
                'description': 'Address primary pain points affecting satisfaction',
                'rationale': 'Direct impact on customer satisfaction and retention',
                'estimated_impact': f'+{len(negative_aspects) * 1.0:.1f} NPS points',
                'implementation_effort': 'High',
                'timeline': '1-2 sprints'
            })
        
        # Feature awareness opportunities
        if any(pp['type'] == 'feature_underutilization' for pp in pain_points):
            recommendations.append({
                'priority': 'Medium',
                'category': 'Customer Success',
                'title': 'Enhanced Feature Onboarding',
                'description': 'Improve awareness and adoption of underutilized features',
                'rationale': 'Increase product value realization and reduce churn risk',
                'estimated_impact': '+0.8 NPS points, 20% feature adoption increase',
                'implementation_effort': 'Low',
                'timeline': '1 sprint'
            })
        
        # Emotional response strategies
        if 'frustrated' in emotions or 'disappointed' in emotions:
            recommendations.append({
                'priority': 'Immediate',
                'category': 'Customer Recovery',
                'title': 'Proactive Customer Outreach',
                'description': 'Personal follow-up to address emotional concerns',
                'rationale': 'Prevent escalation and demonstrate commitment to improvement',
                'estimated_impact': 'Prevent potential -2 to -3 NPS decline',
                'implementation_effort': 'Low',
                'timeline': 'Within 24 hours'
            })
        
        return recommendations
    
    def _convert_rating_to_sentiment(self, rating):
        """Convert rating text to sentiment score"""
        rating_map = {
            'very satisfied': 1.0,
            'satisfied': 0.5,
            'neutral': 0.0,
            'dissatisfied': -0.5,
            'very dissatisfied': -1.0,
            'excellent': 1.0,
            'good': 0.5,
            'fair': 0.0,
            'poor': -0.5,
            'very poor': -1.0,
            'don\'t know': 0.0
        }
        return rating_map.get(rating.lower(), 0.0)
    
    def _extract_evidence(self, text, keywords):
        """Extract relevant sentences as evidence"""
        sentences = text.split('.')
        evidence = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                evidence.append(sentence.strip())
        
        return evidence[:3]  # Return top 3 relevant sentences

# Streamlit App
st.title("🧠 Deep Feedback Intelligence Analyzer")
st.markdown("### Transform any feedback into strategic insights")

# Initialize analyzer
analyzer = DeepFeedbackAnalyzer()

# Input section
st.header("📝 Input Your Feedback")

col1, col2 = st.columns([2, 1])

with col1:
    feedback_text = st.text_area(
        "Feedback Text:",
        height=150,
        placeholder="Paste any client feedback, NPS comment, email, or review here..."
    )

with col2:
    st.write("**Optional Context:**")
    nps_score = st.number_input("NPS Score (0-10)", min_value=0, max_value=10, value=None)
    
    # Quick sample buttons
    if st.button("💼 Corporate Sample"):
        feedback_text = "The contract analysis features are game-changing for our M&A practice, but the integration with our document management system keeps failing. Usually works fine but sometimes we have to manually export files."
        nps_score = 7
    
    if st.button("🔍 Search Issue Sample"):
        feedback_text = "Love the new interface and it's very user-friendly. However, I often need to switch to advanced search to find specific case precedents, and sometimes the results aren't in the right order of relevance."
        nps_score = 8
    
    if st.button("💰 Pricing Concern Sample"):
        feedback_text = "Great tool overall and our team finds it easy to use. The AI features are impressive. But honestly, it's getting expensive for a firm our size and we're looking at alternatives."
        nps_score = 6

# Analysis button
if st.button("🧠 Analyze Feedback", type="primary", use_container_width=True):
    if feedback_text.strip():
        # Run comprehensive analysis
        results = analyzer.analyze_feedback(feedback_text, nps_score)
        
        # Display results
        st.markdown("---")
        st.header("📊 Deep Analysis Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            aspect_count = len(results['aspect_analysis'])
            st.metric("Aspects Analyzed", aspect_count)
        
        with col2:
            pain_point_count = len(results['pain_points'])
            st.metric("Hidden Pain Points", pain_point_count)
        
        with col3:
            risk_count = len(results['risks'])
            st.metric("Business Risks", risk_count)
        
        with col4:
            emotion_count = len(results['emotions'])
            st.metric("Emotions Detected", emotion_count)
        
        # Aspect-based analysis
        if results['aspect_analysis']:
            st.subheader("🎯 Aspect-Based Sentiment Analysis")
            
            # Visualization
            aspects = list(results['aspect_analysis'].keys())
            scores = [data['sentiment_score'] for data in results['aspect_analysis'].values()]
            risks = [data['hidden_risk'] for data in results['aspect_analysis'].values()]
            
            colors = ['orange' if risk else 'green' if score > 0 else 'red' 
                     for score, risk in zip(scores, risks)]
            
            fig = go.Figure(data=go.Bar(
                x=[aspect.replace('_', ' ').title() for aspect in aspects],
                y=scores,
                marker_color=colors,
                text=[f"Risk: {'Yes' if risk else 'No'}" for risk in risks],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Aspect Sentiment Scores (Orange = Hidden Risk Detected)",
                yaxis_title="Sentiment Score",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed breakdown
            for aspect, data in results['aspect_analysis'].items():
                with st.expander(f"🔍 {aspect.replace('_', ' ').title()} - Score: {data['sentiment_score']:.2f}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Sentiment Score:** {data['sentiment_score']:.2f}")
                        st.write(f"**Confidence:** {data['confidence']:.1%}")
                        if data['hidden_risk']:
                            st.warning("⚠️ **Hidden Risk Detected**")
                    
                    with col2:
                        st.write("**Signal Analysis:**")
                        st.write(f"• Positive signals: {data['signals']['positive']}")
                        st.write(f"• Negative signals: {data['signals']['negative']}")
                        st.write(f"• Friction signals: {data['signals']['friction']}")
                    
                    if data['evidence']:
                        st.write("**Evidence from text:**")
                        for evidence in data['evidence']:
                            st.write(f"• *{evidence}*")
        
        # Hidden pain points
        if results['pain_points']:
            st.subheader("🕵️ Hidden Pain Points Analysis")
            
            for pain_point in results['pain_points']:
                severity_colors = {'high': '🔴', 'medium': '🟡', 'low': '🟢', 'low-medium': '🟡'}
                
                st.write(f"**{severity_colors.get(pain_point['severity'], '🔵')} {pain_point['type'].replace('_', ' ').title()}**")
                st.write(f"📄 **Issue:** {pain_point['description']}")
                st.write(f"🔍 **Evidence:** {pain_point['evidence']}")
                st.write(f"📈 **Business Impact:** {pain_point['impact']}")
                st.write(f"⚠️ **Risk:** {pain_point['business_risk']}")
                st.write("---")
        
        # Business risks
        if results['risks']:
            st.subheader("⚠️ Business Risk Assessment")
            
            for risk in results['risks']:
                risk_colors = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
                
                st.write(f"**{risk_colors[risk['level']]} {risk['type'].replace('_', ' ').title()} Risk**")
                st.write(f"📊 **Level:** {risk['level'].title()}")
                st.write(f"📝 **Description:** {risk['description']}")
                st.write(f"🎯 **Indicators:** {risk['indicators']}")
                st.write("---")
        
        # NPS trajectory
        if results['trajectory']['prediction'] != 'Insufficient data':
            st.subheader("📈 NPS Trajectory Prediction")
            
            trajectory = results['trajectory']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if trajectory['current_nps']:
                    st.metric("Current NPS", trajectory['current_nps'])
            
            with col2:
                change = trajectory['predicted_change']
                delta_color = "normal" if change >= 0 else "inverse"
                st.metric("Predicted Change", 
                         f"{change:+.1f}", 
                         delta=f"over {trajectory['timeframe']}",
                         delta_color=delta_color)
            
            with col3:
                st.metric("Confidence", f"{trajectory['confidence']:.1%}")
            
            st.write(f"**Current Segment:** {trajectory['current_segment'].title()}")
            
            # Factor breakdown
            factors = trajectory['key_factors']
            st.write("**Impact Factors:**")
            st.write(f"• Risk factors: {factors['risk_impact']:+.1f}")
            st.write(f"• Friction factors: {factors['friction_impact']:+.1f}")
            st.write(f"• Positive factors: {factors['positive_impact']:+.1f}")
        
        # Strategic recommendations
        if results['recommendations']:
            st.subheader("💡 Strategic Recommendations")
            
            for i, rec in enumerate(results['recommendations']):
                priority_colors = {'Critical': '🔴', 'High': '🟠', 'Medium': '🟡', 'Low': '🟢', 'Immediate': '🚨'}
                
                with st.expander(f"{priority_colors[rec['priority']]} {rec['priority']} - {rec['title']}"):
                    st.write(f"**Category:** {rec['category']}")
                    st.write(f"**Description:** {rec['description']}")
                    st.write(f"**Rationale:** {rec['rationale']}")
                    st.write(f"**Estimated Impact:** {rec['estimated_impact']}")
                    st.write(f"**Implementation Effort:** {rec['implementation_effort']}")
                    st.write(f"**Timeline:** {rec['timeline']}")
        
        # Emotions analysis
        if results['emotions']:
            st.subheader("😊 Emotional Analysis")
            
            emotion_data = []
            for emotion, data in results['emotions'].items():
                emotion_data.append({
                    'Emotion': emotion.title(),
                    'Intensity': f"{data['intensity']:.1%}",
                    'Indicators': ', '.join(data['indicators_found'])
                })
            
            emotion_df = pd.DataFrame(emotion_data)
            st.dataframe(emotion_df, use_container_width=True)
        
        # Summary insights
        st.subheader("🎯 Key Insights Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🔍 What We Found:**")
            if results['pain_points']:
                st.write(f"• {len(results['pain_points'])} hidden pain points")
            if results['risks']:
                st.write(f"• {len(results['risks'])} business risks identified")
            
            friction_aspects = sum(1 for data in results['aspect_analysis'].values() if data.get('hidden_risk'))
            if friction_aspects:
                st.write(f"• {friction_aspects} aspects with hidden friction")
        
        with col2:
            st.markdown("**📈 Business Impact:**")
            if results['trajectory']['prediction'] != 'Insufficient data':
                change = results['trajectory']['predicted_change']
                if change < -1:
                    st.write("• High risk of NPS decline")
                elif change < 0:
                    st.write("• Moderate satisfaction risk")
                else:
                    st.write("• Stable/positive trajectory")
            
            if any('churn' in risk['type'] for risk in results['risks']):
                st.write("• Customer retention at risk")
        
        with col3:
            st.markdown("**💡 Next Steps:**")
            if results['recommendations']:
                urgent_recs = [r for r in results['recommendations'] if r['priority'] in ['Critical', 'Immediate']]
                if urgent_recs:
                    st.write(f"• {len(urgent_recs)} urgent actions needed")
                
                st.write(f"• {len(results['recommendations'])} total recommendations")
            
            st.write("• Follow up with customer")

    else:
        st.warning("⚠️ Please enter some feedback to analyze!")

# Footer
st.markdown("---")
st.markdown("""
### 🚀 **This Analysis Goes Beyond Basic Sentiment By:**
- **🎯 Detecting hidden friction** even in positive feedback
- **🔮 Predicting NPS trajectory** based on subtle patterns  
- **💡 Generating strategic recommendations** with business impact
- **⚠️ Identifying business risks** before they become critical
- **🧠 Understanding emotions** and their implications for retention
""")
