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

# Try to import ML libraries, fallback to rule-based if not available
try:
    from transformers import pipeline
    USE_AI_MODELS = True
    
    @st.cache_resource
    def load_models():
        try:
            # Load lightweight models
            sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            return sentiment_classifier, zero_shot_classifier
        except Exception as e:
            st.warning(f"AI models failed to load: {e}. Using rule-based analysis.")
            return None, None
    
    sentiment_classifier, zero_shot_classifier = load_models()
    if sentiment_classifier is None:
        USE_AI_MODELS = False
        
except ImportError:
    USE_AI_MODELS = False
    st.info("🤖 AI models not available. Using enhanced rule-based analysis.")

# Mock historical data for simple prediction
historical_data = {
    'friction_patterns': {1: 0.5, 2: -1.0, 3: -2.0, 4: -3.0},
    'positive_patterns': {1: 0.3, 2: 0.6, 3: 0.9, 4: 1.2},
    'risk_patterns': {1: -0.5, 2: -1.5, 3: -2.5}
}

class DeepFeedbackAnalyzer:
    def __init__(self):
        # Enhanced legal-specific aspect patterns
        self.legal_aspects = {
            'search_functionality': {
                'keywords': ['search', 'find', 'results', 'keyword', 'boolean', 'advanced search', 'query', 'lookup', 'filter'],
                'positive_signals': ['accurate', 'precise', 'finds everything', 'comprehensive results', 'relevant', 'perfect results'],
                'negative_signals': ['can\'t find', 'irrelevant', 'missing', 'poor results', 'incomplete', 'wrong results'],
                'friction_signals': ['need to go to', 'have to use', 'sometimes', 'switch to', 'manually', 'workaround']
            },
            'ease_of_use': {
                'keywords': ['easy', 'intuitive', 'user-friendly', 'simple', 'interface', 'navigation', 'usable', 'design'],
                'positive_signals': ['easy to use', 'intuitive', 'user-friendly', 'straightforward', 'clear', 'simple'],
                'negative_signals': ['confusing', 'complicated', 'hard to use', 'difficult', 'unclear', 'complex'],
                'friction_signals': ['learning curve', 'getting used to', 'figuring out', 'takes time']
            },
            'content_quality': {
                'keywords': ['content', 'cases', 'precedents', 'coverage', 'database', 'materials', 'information', 'data'],
                'positive_signals': ['comprehensive', 'up-to-date', 'reliable', 'complete', 'thorough', 'accurate'],
                'negative_signals': ['outdated', 'missing', 'incomplete', 'unreliable', 'limited', 'poor quality'],
                'friction_signals': ['some gaps', 'mostly good but', 'generally complete except', 'usually accurate']
            },
            'performance': {
                'keywords': ['speed', 'fast', 'slow', 'performance', 'loading', 'response', 'lag', 'quick', 'time'],
                'positive_signals': ['fast', 'quick', 'responsive', 'instant', 'speedy', 'efficient'],
                'negative_signals': ['slow', 'sluggish', 'laggy', 'takes forever', 'unresponsive', 'crashes'],
                'friction_signals': ['usually fast but', 'mostly quick except', 'sometimes slow', 'occasionally']
            },
            'support_quality': {
                'keywords': ['support', 'help', 'service', 'assistance', 'team', 'response', 'customer service', 'staff'],
                'positive_signals': ['helpful', 'responsive', 'excellent support', 'quick response', 'knowledgeable', 'professional'],
                'negative_signals': ['unhelpful', 'slow response', 'poor service', 'unresponsive', 'rude', 'useless'],
                'friction_signals': ['usually helpful but', 'good support except', 'sometimes takes time', 'mostly responsive']
            },
            'pricing_value': {
                'keywords': ['price', 'cost', 'expensive', 'cheap', 'value', 'worth', 'budget', 'fee', 'money'],
                'positive_signals': ['good value', 'worth it', 'reasonable', 'affordable', 'cost-effective', 'fair price'],
                'negative_signals': ['expensive', 'overpriced', 'not worth', 'too costly', 'budget strain', 'waste of money'],
                'friction_signals': ['mostly worth it but', 'good value except', 'reasonable for most', 'expensive but']
            }
        }
        
        # Enhanced emotion indicators
        self.emotions = {
            'frustrated': ['frustrated', 'annoying', 'irritated', 'fed up', 'bothered', 'aggravated'],
            'disappointed': ['disappointed', 'let down', 'expected more', 'underwhelmed', 'unsatisfied'],
            'confused': ['confused', 'unclear', 'don\'t understand', 'puzzled', 'lost', 'complicated'],
            'satisfied': ['satisfied', 'happy', 'pleased', 'content', 'good', 'fine'],
            'delighted': ['love', 'amazing', 'fantastic', 'excellent', 'outstanding', 'brilliant', 'perfect'],
            'concerned': ['worried', 'concerned', 'anxious', 'nervous', 'uncertain', 'skeptical']
        }
        
        # Risk patterns
        self.risk_patterns = {
            'workflow_friction': ['need to', 'have to', 'sometimes', 'usually but', 'except when', 'workaround'],
            'feature_gaps': ['don\'t know', 'not sure', 'haven\'t tried', 'unaware', 'never used', 'unfamiliar'],
            'competitive_risk': ['compared to', 'vs', 'other tools', 'alternatives', 'competitors', 'similar products'],
            'churn_signals': ['considering', 'looking at', 'might switch', 'evaluating', 'thinking about', 'exploring options']
        }

    def analyze_feedback(self, feedback_text, nps_score=None, structured_ratings=None):
        """Comprehensive feedback analysis with enhanced AI/rule-based hybrid"""
        
        # Preprocess text
        preprocessed_text = self._preprocess_text(feedback_text)
        
        # Choose analysis method based on availability
        if USE_AI_MODELS and sentiment_classifier and zero_shot_classifier:
            aspect_analysis = self._ai_analyze_aspects(preprocessed_text, structured_ratings)
            emotions = self._ai_analyze_emotions(preprocessed_text)
        else:
            aspect_analysis = self._rule_analyze_aspects(preprocessed_text, structured_ratings)
            emotions = self._rule_analyze_emotions(preprocessed_text)
        
        # These use enhanced rule-based methods
        pain_points = self._detect_hidden_pain_points(preprocessed_text, nps_score, aspect_analysis)
        risks = self._assess_risks(preprocessed_text, nps_score, aspect_analysis)
        trajectory = self._predict_trajectory(nps_score, aspect_analysis, risks)
        recommendations = self._generate_recommendations(aspect_analysis, pain_points, risks, emotions)
        
        return {
            'aspect_analysis': aspect_analysis,
            'pain_points': pain_points,
            'emotions': emotions,
            'risks': risks,
            'trajectory': trajectory,
            'recommendations': recommendations,
            'analysis_method': 'AI-Enhanced' if USE_AI_MODELS else 'Enhanced Rule-Based'
        }
    
    def _preprocess_text(self, text):
        """Preprocess text for analysis"""
        text = re.sub(r'\s+', ' ', text.strip().lower())
        return text
    
    def _ai_analyze_aspects(self, text, structured_ratings=None):
        """AI-powered aspect analysis"""
        results = {}
        
        try:
            # Use zero-shot classification
            aspect_labels = list(self.legal_aspects.keys())
            zero_shot_results = zero_shot_classifier(text, candidate_labels=aspect_labels, multi_label=True)
            
            for aspect, score in zip(zero_shot_results['labels'], zero_shot_results['scores']):
                if score > 0.3:  # Relevance threshold
                    # Get sentiment for this aspect
                    sentences = text.split('.')
                    relevant_sentences = [s for s in sentences if any(kw in s for kw in self.legal_aspects[aspect]['keywords'])]
                    snippet = ' '.join(relevant_sentences[:2]) or text[:200]
                    
                    sentiment = sentiment_classifier(snippet)[0]
                    sentiment_score = 1.0 if sentiment['label'] == 'POSITIVE' else -1.0
                    confidence = sentiment['score']
                    
                    # Check for friction signals
                    friction_count = sum(1 for signal in self.legal_aspects[aspect]['friction_signals'] if signal in text)
                    hidden_risk = friction_count > 0 or confidence < 0.7
                    
                    if structured_ratings and aspect in structured_ratings:
                        rating_sentiment = self._convert_rating_to_sentiment(structured_ratings[aspect])
                        sentiment_score = (sentiment_score * 0.6) + (rating_sentiment * 0.4)
                    
                    results[aspect] = {
                        'sentiment_score': sentiment_score,
                        'confidence': confidence,
                        'mentions': len(relevant_sentences),
                        'hidden_risk': hidden_risk,
                        'signals': {
                            'positive': sum(1 for s in self.legal_aspects[aspect]['positive_signals'] if s in text),
                            'negative': sum(1 for s in self.legal_aspects[aspect]['negative_signals'] if s in text),
                            'friction': friction_count
                        },
                        'evidence': relevant_sentences[:3]
                    }
        
        except Exception as e:
            st.warning(f"AI analysis failed: {e}. Falling back to rule-based.")
            return self._rule_analyze_aspects(text, structured_ratings)
        
        return results
    
    def _rule_analyze_aspects(self, text, structured_ratings=None):
        """Enhanced rule-based aspect analysis"""
        results = {}
        
        for aspect, config in self.legal_aspects.items():
            keyword_mentions = sum(1 for keyword in config['keywords'] if keyword in text)
            
            if keyword_mentions > 0:
                positive_count = sum(1 for signal in config['positive_signals'] if signal in text)
                negative_count = sum(1 for signal in config['negative_signals'] if signal in text)
                friction_count = sum(1 for signal in config['friction_signals'] if signal in text)
                
                # Enhanced sentiment calculation
                if positive_count > negative_count:
                    base_sentiment = min(0.8, 0.3 + (positive_count * 0.2))
                elif negative_count > positive_count:
                    base_sentiment = max(-0.8, -0.3 - (negative_count * 0.2))
                else:
                    base_sentiment = 0.0
                
                # Adjust for friction
                if friction_count > 0:
                    confidence = max(0.5, 0.8 - (friction_count * 0.1))
                    hidden_risk = True
                    if base_sentiment > 0:
                        base_sentiment *= (1 - friction_count * 0.1)
                else:
                    confidence = 0.85
                    hidden_risk = False
                
                # Incorporate structured ratings
                if structured_ratings and aspect in structured_ratings:
                    rating_sentiment = self._convert_rating_to_sentiment(structured_ratings[aspect])
                    base_sentiment = (base_sentiment * 0.6) + (rating_sentiment * 0.4)
                
                results[aspect] = {
                    'sentiment_score': base_sentiment,
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
    
    def _ai_analyze_emotions(self, text):
        """AI-powered emotion analysis"""
        try:
            emotion_labels = list(self.emotions.keys())
            zero_shot_results = zero_shot_classifier(text, candidate_labels=emotion_labels, multi_label=True)
            
            detected_emotions = {}
            for emotion, score in zip(zero_shot_results['labels'], zero_shot_results['scores']):
                if score > 0.4:
                    detected_emotions[emotion] = {
                        'intensity': score,
                        'indicators_found': [ind for ind in self.emotions[emotion] if ind in text]
                    }
            
            return detected_emotions
        except:
            return self._rule_analyze_emotions(text)
    
    def _rule_analyze_emotions(self, text):
        """Enhanced rule-based emotion analysis"""
        detected_emotions = {}
        
        for emotion, indicators in self.emotions.items():
            count = sum(1 for indicator in indicators if indicator in text)
            if count > 0:
                intensity = min(count / len(indicators), 1.0)
                detected_emotions[emotion] = {
                    'intensity': intensity,
                    'indicators_found': [ind for ind in indicators if ind in text]
                }
        
        return detected_emotions
    
    def _detect_hidden_pain_points(self, text, nps_score, aspect_analysis):
        """Enhanced hidden pain point detection"""
        pain_points = []
        
        # Pattern 1: High NPS but workflow friction
        if nps_score and nps_score >= 7:
            friction_indicators = ['need to', 'have to', 'sometimes', 'usually but', 'except', 'workaround']
            friction_detected = [indicator for indicator in friction_indicators if indicator in text]
            
            if friction_detected:
                pain_points.append({
                    'type': 'workflow_inefficiency',
                    'severity': 'medium',
                    'description': 'User experiences process friction despite high satisfaction',
                    'evidence': friction_detected,
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
        knowledge_gaps = ['don\'t know', 'not sure', 'haven\'t tried', 'unaware', 'never used']
        gaps_found = [gap for gap in knowledge_gaps if gap in text]
        
        if gaps_found:
            pain_points.append({
                'type': 'feature_underutilization',
                'severity': 'low',
                'description': f'Customer unaware of {len(gaps_found)} product aspects or features',
                'evidence': gaps_found,
                'impact': 'Missing expansion and upsell opportunities',
                'business_risk': 'Underutilized customers more likely to churn'
            })
        
        return pain_points
    
    def _assess_risks(self, text, nps_score, aspect_analysis):
        """Enhanced risk assessment"""
        risks = []
        
        # Churn risk assessment
        churn_signals = [pattern for pattern in self.risk_patterns['churn_signals'] if pattern in text]
        competitive_mentions = [pattern for pattern in self.risk_patterns['competitive_risk'] if pattern in text]
        
        if churn_signals:
            risks.append({
                'type': 'churn_risk',
                'level': 'high' if len(churn_signals) >= 2 else 'medium',
                'description': 'Customer showing consideration of alternatives',
                'indicators': churn_signals
            })
        
        if competitive_mentions:
            risks.append({
                'type': 'competitive_pressure',
                'level': 'medium',
                'description': 'Customer actively comparing with competitors',
                'indicators': competitive_mentions
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
    
    def _predict_trajectory(self, nps_score, aspect_analysis, risks):
        """Enhanced NPS trajectory prediction"""
        if not nps_score:
            return {'prediction': 'Insufficient data', 'confidence': 0}
        
        current_segment = 'promoter' if nps_score >= 9 else 'passive' if nps_score >= 7 else 'detractor'
        
        # Calculate impact factors
        friction_count = sum(1 for data in aspect_analysis.values() if data.get('hidden_risk', False))
        positive_aspects = sum(1 for data in aspect_analysis.values() if data['sentiment_score'] > 0.5)
        risk_count = len(risks)
        
        # Use historical patterns for prediction
        friction_impact = historical_data['friction_patterns'].get(friction_count, 0)
        positive_impact = historical_data['positive_patterns'].get(positive_aspects, 0)
        risk_impact = historical_data['risk_patterns'].get(risk_count, 0)
        
        total_impact = friction_impact + positive_impact + risk_impact
        
        return {
            'current_nps': nps_score,
            'current_segment': current_segment,
            'predicted_change': total_impact,
            'confidence': 0.75 if len(aspect_analysis) >= 2 else 0.55,
            'timeframe': '3-6 months',
            'key_factors': {
                'risk_impact': risk_impact,
                'friction_impact': friction_impact,
                'positive_impact': positive_impact
            }
        }
    
    def _generate_recommendations(self, aspect_analysis, pain_points, risks, emotions):
        """Enhanced recommendation generation"""
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
        
        # Critical priority: Address negative aspects
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
        
        return evidence[:3]

# Streamlit App
st.title("🧠 Deep Feedback Intelligence Analyzer")
st.markdown("### Transform any feedback into strategic insights")

# Show analysis method
if USE_AI_MODELS:
    st.success("🤖 AI-Enhanced Analysis Active")
else:
    st.info("📊 Enhanced Rule-Based Analysis Active")

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
    
    # File upload for batch processing
    uploaded_file = st.file_uploader("Or upload CSV for batch analysis", type="csv")

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
    if feedback_text.strip() or uploaded_file:
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("📊 Processing batch analysis...")
                
                # Process first few rows as demo
                results_list = []
                for i in range(min(3, len(df))):
                    row = df.iloc[i]
                    text = row.get('feedback_text', str(row.iloc[0]))
                    nps = row.get('nps_score', None)
                    
                    result = analyzer.analyze_feedback(text, nps)
                    results_list.append({
                        'row': i+1,
                        'text': text[:100] + "...",
                        'nps': nps,
                        'aspects': len(result['aspect_analysis']),
                        'risks': len(result['risks']),
                        'recommendations': len(result['recommendations'])
                    })
                
                batch_df = pd.DataFrame(results_list)
                st.dataframe(batch_df, use_container_width=True)
                st.info("Batch analysis limited to first 3 rows in demo. Analyzing first row in detail below:")
                
                feedback_text = df.iloc[0].get('feedback_text', str(df.iloc[0, 0]))
                nps_score = df.iloc[0].get('nps_score', None)
                
            except Exception as e:
                st.error(f"Error processing CSV: {e}")
                st.stop()
        
        # Run comprehensive analysis
        with st.spinner("🧠 Running deep analysis..."):
            results = analyzer.analyze_feedback(feedback_text, nps_score)
        
        # Display results
        st.markdown("---")
        st.header("📊 Deep Analysis Results")
        
        # Show analysis method used
        st.info(f"📊 Analysis Method: {results['analysis_method']}")
        
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
                            if evidence.strip():
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
        if results['trajectory'].get('prediction') != 'Insufficient data':
            st.subheader("📈 NPS Trajectory Prediction")
            
            trajectory = results['trajectory']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if trajectory.get('current_nps'):
                    st.metric("Current NPS", trajectory['current_nps'])
            
            with col2:
                change = trajectory.get('predicted_change', 0)
                delta_color = "normal" if change >= 0 else "inverse"
                st.metric("Predicted Change", 
                         f"{change:+.1f}", 
                         delta=f"over {trajectory.get('timeframe', 'N/A')}",
                         delta_color=delta_color)
            
            with col3:
                st.metric("Confidence", f"{trajectory.get('confidence', 0):.1%}")
            
            st.write(f"**Current Segment:** {trajectory.get('current_segment', 'Unknown').title()}")
            
            # Factor breakdown
            if 'key_factors' in trajectory:
                factors = trajectory['key_factors']
                st.write("**Impact Factors:**")
                st.write(f"• Risk factors: {factors.get('risk_impact', 0):+.1f}")
                st.write(f"• Friction factors: {factors.get('friction_impact', 0):+.1f}")
                st.write(f"• Positive factors: {factors.get('positive_impact', 0):+.1f}")
        
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
            if results['trajectory'].get('prediction') != 'Insufficient data':
                change = results['trajectory'].get('predicted_change', 0)
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
- **🤖 AI/Rule-based hybrid** for maximum compatibility
""")
