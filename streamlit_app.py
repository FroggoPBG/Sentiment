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
    page_title="Strategic Feedback Intelligence Analyzer",
    page_icon="🎯",
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

class StrategicFeedbackAnalyzer:
    def __init__(self):
        # **ENHANCEMENT 1: THEMATIC TAGGING AND SEGMENTATION**
        self.themes = [
            'user_experience', 'pricing_concerns', 'feature_requests', 
            'technical_issues', 'workflow_friction', 'content_coverage', 
            'integration_needs', 'support_quality', 'performance_issues'
        ]
        
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
        
        # **ENHANCEMENT 6: EMOTION AND FOLLOW-UP ENHANCEMENTS**
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

        # **ENHANCEMENT 3: LEGAL-SPECIFIC ANALYSIS**
        self.legal_terms = ['ordinances', 'regulations', 'compliance', 'documentation', 'annotations', 'legal research', 'statutes', 'provisions']
        self.knowledge_gaps = ["don't know if it can", "haven't tried that feature", "wasn't aware of", "not sure how to", "couldn't find"]

    # **ENHANCEMENT 1: THEMATIC TAGGING AND SEGMENTATION**
    def _extract_themes(self, text, nps_score=None):
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
        
        nps_category = self._categorize_nps(nps_score) if nps_score else 'unknown'
        tags = f"<nps-tag>{','.join([str(i+1) for i in range(len(detected_themes))])}</nps-tag>"
        
        return {
            'themes': detected_themes,
            'nps_category': nps_category,
            'tags': tags
        }

    def _categorize_nps(self, score):
        """Categorize NPS score"""
        if score >= 9:
            return 'promoter'
        elif score >= 7:
            return 'passive'
        else:
            return 'detractor'

    # **ENHANCEMENT 2: PREDICTIVE WHAT-IF SIMULATIONS**
    def _simulate_nps_change(self, current_nps, friction_change, risk_change, feature_change):
        """Simulate NPS changes based on improvements"""
        # Simple impact calculation
        friction_impact = friction_change * -0.5
        risk_impact = risk_change * -0.3
        feature_impact = feature_change * 0.4
        
        total_impact = friction_impact + risk_impact + feature_impact
        projected_nps = max(0, min(10, current_nps + total_impact))
        
        return {
            'current': current_nps,
            'projected': round(projected_nps, 1),
            'change': round(total_impact, 1),
            'confidence': 0.75
        }

    # **ENHANCEMENT 3: CROSS-CHANNEL AND LEGAL ANALYSIS**
    def _analyze_cross_channel(self, feedback, email):
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

    def _generate_legal_insights(self, text, industry):
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

    def _detect_knowledge_gaps(self, text):
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

    # **ENHANCEMENT 6: EMOTION AND FOLLOW-UP ENHANCEMENTS**
    def _generate_followup_strategy(self, themes, emotions, nps_category, knowledge_gaps):
        """Generate personalized follow-up strategy with emotion-theme mapping"""
        strategy = {
            'what_to_say': [],
            'what_to_do': [],
            'priority': 'medium'
        }
        
        # Map emotions to themes for strategic responses
        frustrated_themes = [t['theme'] for t in themes if any(e in ['frustrated', 'disappointed'] for e in emotions.keys())]
        
        if 'workflow_friction' in frustrated_themes:
            strategy['what_to_say'].append("We understand the manual processes are frustrating and appreciate you bringing this to our attention.")
            strategy['what_to_do'].append("Schedule workflow optimization session within 1 week")
            strategy['priority'] = 'high'
        
        if knowledge_gaps:
            strategy['what_to_say'].append("We'd love to show you some features that might help with your current challenges.")
            strategy['what_to_do'].append("Send feature tutorial or schedule demo")
        
        # Emotion-specific strategies
        if 'frustrated' in emotions:
            strategy['what_to_say'].append("Your frustration is completely valid. Let's address the root cause together.")
            strategy['what_to_do'].append("Immediate call from senior team member")
            strategy['priority'] = 'urgent'
        
        if 'confused' in emotions:
            strategy['what_to_say'].append("Let's clarify these points with a personalized walkthrough.")
            strategy['what_to_do'].append("Schedule 1-on-1 training session")
        
        if 'delighted' in emotions:
            strategy['what_to_say'].append("We're thrilled you're having such a positive experience! We'd love to explore expansion opportunities.")
            strategy['what_to_do'].append("Schedule strategic review for upsell opportunities")
        
        # NPS-based strategies
        if nps_category == 'detractor':
            strategy['what_to_say'].append("We value your feedback and want to address your concerns directly.")
            strategy['what_to_do'].append("Immediate follow-up call within 24 hours")
            strategy['priority'] = 'urgent'
        elif nps_category == 'promoter':
            strategy['what_to_say'].append("Thank you for being a valued customer! We'd love to hear about expansion opportunities.")
            strategy['what_to_do'].append("Reach out for upsell conversation")
        
        return strategy

    def analyze_feedback(self, feedback_text, email="", nps_score=None, segment="enterprise", industry="tech", structured_ratings=None):
        """Enhanced comprehensive feedback analysis"""
        
        # Preprocess text
        preprocessed_text = self._preprocess_text(feedback_text)
        
        # **ENHANCEMENT 1: Thematic analysis**
        theme_analysis = self._extract_themes(preprocessed_text, nps_score)
        
        # Choose analysis method based on availability
        if USE_AI_MODELS and sentiment_classifier and zero_shot_classifier:
            aspect_analysis = self._ai_analyze_aspects(preprocessed_text, structured_ratings)
            emotions = self._ai_analyze_emotions(preprocessed_text)
        else:
            aspect_analysis = self._rule_analyze_aspects(preprocessed_text, structured_ratings)
            emotions = self._rule_analyze_emotions(preprocessed_text)
        
        # **ENHANCEMENT 3: Cross-channel and legal analysis**
        knowledge_gaps = self._detect_knowledge_gaps(preprocessed_text)
        cross_channel = self._analyze_cross_channel(feedback_text, email)
        legal_insights = self._generate_legal_insights(feedback_text, industry)
        
        # Enhanced rule-based methods
        pain_points = self._detect_hidden_pain_points(preprocessed_text, nps_score, aspect_analysis)
        risks = self._assess_risks(preprocessed_text, nps_score, aspect_analysis)
        trajectory = self._predict_trajectory(nps_score, aspect_analysis, risks)
        recommendations = self._generate_recommendations(aspect_analysis, pain_points, risks, emotions)
        
        # **ENHANCEMENT 6: Enhanced follow-up strategy**
        followup_strategy = self._generate_followup_strategy(
            theme_analysis['themes'], emotions, theme_analysis['nps_category'], knowledge_gaps
        )
        
        return {
            'theme_analysis': theme_analysis,
            'aspect_analysis': aspect_analysis,
            'pain_points': pain_points,
            'emotions': emotions,
            'knowledge_gaps': knowledge_gaps,
            'cross_channel': cross_channel,
            'legal_insights': legal_insights,
            'risks': risks,
            'trajectory': trajectory,
            'recommendations': recommendations,
            'followup_strategy': followup_strategy,
            'nps_score': nps_score,
            'segment': segment,
            'industry': industry,
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

# **ENHANCEMENT 4: INTERACTIVE VISUALS AND EXPORTS**
@st.cache_data
def create_csv_export(results):
    """Create CSV export of analysis results"""
    data = {
        'NPS_Score': [results.get('nps_score', 'N/A')],
        'NPS_Category': [results.get('theme_analysis', {}).get('nps_category', 'unknown')],
        'Themes': ['; '.join([t['theme'] for t in results.get('theme_analysis', {}).get('themes', [])])],
        'Theme_Confidence': ['; '.join([f"{t['confidence']:.2f}" for t in results.get('theme_analysis', {}).get('themes', [])])],
        'Emotions': ['; '.join(results.get('emotions', {}).keys())],
        'Follow_up_Priority': [results.get('followup_strategy', {}).get('priority', 'medium')],
        'Knowledge_Gaps': [len(results.get('knowledge_gaps', []))],
        'Cross_Channel_Issues': [len(results.get('cross_channel', []))]
    }
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

def create_text_report(results):
    """Create detailed text report"""
    report = f"""
Strategic Feedback Analysis Report
================================

NPS Score: {results.get('nps_score', 'N/A')}
Category: {results.get('theme_analysis', {}).get('nps_category', 'unknown').upper()}
Segment: {results.get('segment', 'N/A')}
Industry: {results.get('industry', 'N/A')}

THEMES DETECTED:
"""
    for theme in results.get('theme_analysis', {}).get('themes', []):
        report += f"- {theme['theme']}: {theme['confidence']:.0%} confidence\n"
        report += f"  Evidence: {theme['evidence']}\n"

    report += f"\nEMOTIONS:\n"
    for emotion, data in results.get('emotions', {}).items():
        report += f"- {emotion}: {data['intensity']:.0%} intensity\n"

    report += f"\nSTRATEGIC RECOMMENDATIONS:\n"
    report += f"Priority: {results.get('followup_strategy', {}).get('priority', 'medium').upper()}\n\n"
    
    report += "What to Say:\n"
    for item in results.get('followup_strategy', {}).get('what_to_say', []):
        report += f"- {item}\n"
    
    report += "\nWhat to Do:\n"
    for item in results.get('followup_strategy', {}).get('what_to_do', []):
        report += f"- {item}\n"

    if results.get('knowledge_gaps'):
        report += f"\nKNOWLEDGE GAPS DETECTED:\n"
        for gap in results['knowledge_gaps']:
            report += f"- {gap['phrase']}: {gap['implication']}\n"

    return report

# **ENHANCEMENT 5: BATCH PROCESSING WITH AGGREGATION**
def process_batch_data(analyzer, df, feedback_col, nps_col, email_col):
    """Process batch data with aggregation"""
    results = []
    
    for idx, row in df.iterrows():
        feedback = str(row[feedback_col])
        nps = int(row[nps_col]) if pd.notna(row[nps_col]) else 5
        email = str(row[email_col]) if email_col != "None" and pd.notna(row[email_col]) else ""
        
        analysis = analyzer.analyze_feedback(feedback, email, nps)
        results.append(analysis)
    
    return results

def display_batch_aggregation(results):
    """Display aggregated batch results"""
    st.subheader("📈 Batch Analysis Summary")
    
    # Aggregate statistics
    total_count = len(results)
    avg_nps = np.mean([r.get('nps_score', 5) for r in results])
    
    category_counts = Counter([r.get('theme_analysis', {}).get('nps_category', 'unknown') for r in results])
    
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
        for theme in result.get('theme_analysis', {}).get('themes', []):
            all_themes.append({
                'theme': theme['theme'],
                'category': result.get('theme_analysis', {}).get('nps_category', 'unknown')
            })
    
    if all_themes:
        theme_df = pd.DataFrame(all_themes)
        theme_summary = theme_df.groupby(['theme', 'category']).size().unstack(fill_value=0)
        theme_summary['total'] = theme_summary.sum(axis=1)
        theme_summary = theme_summary.sort_values('total', ascending=False)
        
        st.subheader("🏷️ Theme Distribution")
        
        # **ENHANCEMENT 4: Interactive visualizations**
        fig = px.bar(
            x=theme_summary.index,
            y=theme_summary['total'],
            title="Most Common Themes Across Batch",
            labels={'x': 'Theme', 'y': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)

# Streamlit App
st.title("🎯 Strategic Feedback Intelligence Analyzer")
st.markdown("### Advanced thematic analysis with predictive what-if simulations")

# Show analysis method
if USE_AI_MODELS:
    st.success("🤖 AI-Enhanced Analysis Active")
else:
    st.info("📊 Enhanced Rule-Based Analysis Active")

# Initialize analyzer
analyzer = StrategicFeedbackAnalyzer()

# Main input section
st.header("📝 Input Your Feedback")

# **ENHANCEMENT 5: Batch vs Individual Analysis**
analysis_type = st.radio("Analysis Type", ["Individual Feedback", "Batch Analysis"], horizontal=True)

if analysis_type == "Individual Feedback":
    col1, col2 = st.columns([2, 1])

    with col1:
        feedback_text = st.text_area(
            "Feedback Text:",
            height=150,
            placeholder="Paste any client feedback, NPS comment, email, or review here..."
        )
        
        # **ENHANCEMENT 3: Cross-channel analysis**
        email_text = st.text_area(
            "Related Email (optional):",
            height=100,
            placeholder="Paste related email content for cross-channel analysis..."
        )

    with col2:
        st.write("**Optional Context:**")
        nps_score = st.number_input("NPS Score (0-10)", min_value=0, max_value=10, value=None)
        segment = st.selectbox("Customer Segment", ["enterprise", "smb", "startup"])
        industry = st.selectbox("Industry", ["legal", "tech", "finance", "healthcare"])
        
        # Quick sample buttons
        if st.button("💼 Corporate Sample"):
            feedback_text = "The contract analysis features are game-changing for our M&A practice, but the integration with our document management system keeps failing. Usually works fine but sometimes we have to manually export files."
            nps_score = 7
        
        if st.button("🔍 Search Issue Sample"):
            feedback_text = "Love the new interface and it's very user-friendly. However, I often need to switch to advanced search to find specific case precedents, and sometimes the results aren't in the right order of relevance."
            nps_score = 8

    # Analysis button for individual
    if st.button("🎯 Analyze Individual Feedback", type="primary", use_container_width=True):
        if feedback_text.strip():
            with st.spinner("🧠 Running strategic analysis..."):
                results = analyzer.analyze_feedback(
                    feedback_text, 
                    email_text, 
                    nps_score, 
                    segment, 
                    industry
                )
            
            # Display individual results
            st.markdown("---")
            st.header("🎯 Strategic Analysis Results")
            
            # **ENHANCEMENT 1: Thematic Analysis & Segmentation**
            with st.expander("🏷️ Thematic Analysis & Segmentation", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Detected Themes:**")
                    for theme in results['theme_analysis']['themes']:
                        confidence = int(theme['confidence'] * 100)
                        st.markdown(f"🏷️ **{theme['theme'].replace('_', ' ').title()}** ({confidence}% confidence)")
                        st.write(f"   Evidence: {theme['evidence']}")
                    
                    st.write(f"**NPS Tags:** {results['theme_analysis']['tags']}")
                
                with col2:
                    category = results['theme_analysis']['nps_category']
                    category_color = {'promoter': '🟢', 'passive': '🟡', 'detractor': '🔴'}.get(category, '⚪')
                    st.write(f"**NPS Category:** {category_color} {category.upper()}")
                    st.write(f"**Score:** {results.get('nps_score', 'N/A')}")
            
            # **ENHANCEMENT 2: What-If Simulation**
            with st.expander("🔮 What-If NPS Simulation", expanded=True):
                st.write("**Simulate impact of improvements:**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    friction_change = st.slider("Friction Reduction", -3, 3, 0, key="friction")
                with col2:
                    risk_change = st.slider("Risk Mitigation", -3, 3, 0, key="risk")
                with col3:
                    feature_change = st.slider("Feature Enhancement", -3, 3, 0, key="feature")
                
                if results.get('nps_score'):
                    simulation = analyzer._simulate_nps_change(
                        results['nps_score'], 
                        friction_change, 
                        risk_change, 
                        feature_change
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current NPS", simulation['current'])
                    with col2:
                        st.metric("Projected NPS", simulation['projected'], simulation['change'])
                    with col3:
                        st.metric("Confidence", f"{int(simulation['confidence'] * 100)}%")
            
            # **ENHANCEMENT 3: Cross-Channel & Legal Analysis**
            with st.expander("🔗 Cross-Channel & Legal Analysis"):
                if results['cross_channel']:
                    st.write("**Cross-Channel Correlations:**")
                    for corr in results['cross_channel']:
                        risk_color = "🔴" if corr['risk_level'] == 'high' else "🟡"
                        st.write(f"{risk_color} **{corr['type']}:** {corr['description']} ({corr['risk_level']} risk)")
                else:
                    st.write("No cross-channel correlations detected")
                
                if results['legal_insights']:
                    st.write("**Legal Domain Insights:**")
                    for insight in results['legal_insights']:
                        st.write(f"⚖️ **{insight['term']}:** {insight['context']}")
                        st.write(f"   *{insight['implication']}*")
            
            # **ENHANCEMENT 6: Emotion Analysis & Follow-up Strategy**
            with st.expander("😊 Emotion Analysis & Follow-up Strategy"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Detected Emotions:**")
                    for emotion, data in results['emotions'].items():
                        intensity = int(data['intensity'] * 100)
                        st.write(f"😊 **{emotion.title()}:** {intensity}% intensity")
                    
                    if results['knowledge_gaps']:
                        st.write("**Knowledge Gaps:**")
                        for gap in results['knowledge_gaps']:
                            st.write(f"🔍 {gap['phrase']} - {gap['implication']}")
                
                with col2:
                    strategy = results['followup_strategy']
                    st.write(f"**Priority Level:** {strategy['priority'].upper()}")
                    
                    st.write("**What to Say:**")
                    for item in strategy['what_to_say']:
                        st.write(f"💬 {item}")
                    
                    st.write("**What to Do:**")
                    for item in strategy['what_to_do']:
                        st.write(f"📋 {item}")
            
            # **ENHANCEMENT 4: Interactive Visualizations**
            with st.expander("📊 Interactive Visualizations"):
                col1, col2 = st.columns(2)
                
                with col1:
                    if results['theme_analysis']['themes']:
                        theme_df = pd.DataFrame(results['theme_analysis']['themes'])
                        theme_df['confidence_pct'] = theme_df['confidence'] * 100
                        
                        fig = px.bar(
                            theme_df, 
                            x='theme', 
                            y='confidence_pct',
                            title="Theme Detection Confidence",
                            labels={'confidence_pct': 'Confidence %', 'theme': 'Theme'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # NPS gauge chart
                    if results.get('nps_score'):
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
            
            # **ENHANCEMENT 4: Export Options**
            st.subheader("📤 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Download CSV Report"):
                    csv_data = create_csv_export(results)
                    st.download_button(
                        label="Download Analysis CSV",
                        data=csv_data,
                        file_name="feedback_analysis.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📑 Download Text Report"):
                    report_data = create_text_report(results)
                    st.download_button(
                        label="Download Text Report",
                        data=report_data,
                        file_name="feedback_report.txt",
                        mime="text/plain"
                    )

        else:
            st.warning("⚠️ Please enter some feedback to analyze!")

else:  # **ENHANCEMENT 5: Batch Analysis**
    st.subheader("📊 Batch Processing with Aggregation")
    
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
            
            if st.button("🚀 Process Batch Analysis", type="primary"):
                with st.spinner("Processing batch analysis..."):
                    results = process_batch_data(analyzer, df, feedback_col, nps_col, email_col)
                    display_batch_aggregation(results)
                    
                    # Export batch results
                    st.subheader("📤 Export Batch Results")
                    if st.button("📊 Download Batch CSV"):
                        batch_data = []
                        for i, result in enumerate(results):
                            row = {
                                'Response_ID': i + 1,
                                'NPS_Score': result.get('nps_score', 'N/A'),
                                'NPS_Category': result.get('theme_analysis', {}).get('nps_category', 'unknown'),
                                'Themes': '; '.join([t['theme'] for t in result.get('theme_analysis', {}).get('themes', [])]),
                                'Emotions': '; '.join(result.get('emotions', {}).keys()),
                                'Priority': result.get('followup_strategy', {}).get('priority', 'medium'),
                                'Knowledge_Gaps': len(result.get('knowledge_gaps', [])),
                                'Cross_Channel_Issues': len(result.get('cross_channel', []))
                            }
                            batch_data.append(row)
                        
                        batch_df = pd.DataFrame(batch_data)
                        batch_csv = batch_df.to_csv(index=False)
                        st.download_button(
                            label="Download Batch Analysis",
                            data=batch_csv,
                            file_name="batch_analysis.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
### 🚀 **Enhanced Strategic Capabilities:**
- **🏷️ Dynamic thematic tagging** with NPS segmentation and evidence
- **🔮 Interactive what-if simulations** for NPS trajectory prediction  
- **🔗 Cross-channel correlation analysis** between feedback and emails
- **⚖️ Legal domain-specific insights** and terminology detection
- **📊 Interactive visualizations** with Plotly charts and exports
- **📦 Batch processing** with aggregated insights and CSV export
- **😊 Advanced emotion-theme mapping** for strategic follow-up
- **🤖 AI/Rule-based hybrid** analysis for maximum compatibility
""")
