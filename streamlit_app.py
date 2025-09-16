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
    
    def _generate_emotional_strategy(self, emotions, pain_points, risks, nps_score, aspect_analysis):
        """Generate emotional targeting strategy for follow-up communications"""
        
        # Determine current emotional state
        current_emotions = list(emotions.keys()) if emotions else []
        dominant_emotion = max(emotions.items(), key=lambda x: x[1]['intensity'])[0] if emotions else 'neutral'
        
        # Define target emotions based on current state and feedback context
        target_emotions = self._determine_target_emotions(current_emotions, nps_score, pain_points, risks)
        
        # Generate specific email strategies
        email_strategies = self._create_email_strategies(target_emotions, aspect_analysis, pain_points)
        
        # Generate follow-up action plan
        follow_up_actions = self._create_follow_up_actions(target_emotions, pain_points, risks, nps_score)
        
        return {
            'current_emotions': current_emotions,
            'dominant_emotion': dominant_emotion,
            'target_emotions': target_emotions,
            'email_strategies': email_strategies,
            'follow_up_actions': follow_up_actions
        }

    def _determine_target_emotions(self, current_emotions, nps_score, pain_points, risks):
        """Determine what emotions we want to elicit in follow-up"""
        targets = {}
        
        # Base strategy on current emotional state
        if 'frustrated' in current_emotions or 'disappointed' in current_emotions:
            targets['reassurance'] = {
                'priority': 'primary',
                'rationale': 'Counter negative emotions with stability and confidence',
                'intensity_goal': 'high'
            }
            targets['empathy'] = {
                'priority': 'primary', 
                'rationale': 'Acknowledge their frustration to build connection',
                'intensity_goal': 'high'
            }
            targets['hope'] = {
                'priority': 'secondary',
                'rationale': 'Create optimism about resolution and improvement',
                'intensity_goal': 'medium'
            }
        
        elif 'confused' in current_emotions:
            targets['clarity'] = {
                'priority': 'primary',
                'rationale': 'Provide clear understanding and direction',
                'intensity_goal': 'high'
            }
            targets['confidence'] = {
                'priority': 'secondary',
                'rationale': 'Build confidence in their ability to succeed with product',
                'intensity_goal': 'medium'
            }
        
        elif 'satisfied' in current_emotions and nps_score and nps_score >= 7:
            targets['excitement'] = {
                'priority': 'primary',
                'rationale': 'Elevate satisfaction to advocacy and engagement',
                'intensity_goal': 'high'
            }
            targets['curiosity'] = {
                'priority': 'secondary',
                'rationale': 'Drive exploration of additional features and value',
                'intensity_goal': 'medium'
            }
            targets['partnership'] = {
                'priority': 'secondary',
                'rationale': 'Position relationship as collaborative partnership',
                'intensity_goal': 'medium'
            }
        
        elif 'delighted' in current_emotions:
            targets['advocacy'] = {
                'priority': 'primary',
                'rationale': 'Convert delight into active promotion and referrals',
                'intensity_goal': 'high'
            }
            targets['exclusivity'] = {
                'priority': 'secondary',
                'rationale': 'Make them feel like a valued, special customer',
                'intensity_goal': 'medium'
            }
        
        # Risk-based emotional targets
        if any(risk['type'] == 'churn_risk' for risk in risks):
            targets['loyalty'] = {
                'priority': 'critical',
                'rationale': 'Reinforce commitment and long-term value proposition',
                'intensity_goal': 'high'
            }
            targets['investment'] = {
                'priority': 'critical',
                'rationale': 'Highlight their existing investment and future potential',
                'intensity_goal': 'high'
            }
        
        # Default professional emotions
        if not targets:
            targets['appreciation'] = {
                'priority': 'primary',
                'rationale': 'Standard professional acknowledgment and gratitude',
                'intensity_goal': 'medium'
            }
            targets['collaboration'] = {
                'priority': 'secondary',
                'rationale': 'Foster ongoing partnership mindset',
                'intensity_goal': 'medium'
            }
        
        return targets

    def _create_email_strategies(self, target_emotions, aspect_analysis, pain_points):
        """Create specific email content strategies for each target emotion"""
        strategies = {}
        
        for emotion, config in target_emotions.items():
            if emotion == 'reassurance':
                strategies[emotion] = {
                    'email_elements': {
                        'subject_line': 'We hear you - here\'s our action plan',
                        'opening': 'Thank you for your candid feedback. Your experience matters deeply to us.',
                        'body_focus': 'Specific steps we\'re taking to address your concerns',
                        'tone': 'Confident, solution-oriented, empathetic'
                    },
                    'specific_language': [
                        '"We\'ve immediately prioritized..."',
                        '"Our team has already begun working on..."',
                        '"You can expect to see improvements in..."',
                        '"We\'re committed to making this right"'
                    ],
                    'avoid_language': [
                        'Apologetic overuse ("sorry" repeatedly)',
                        'Vague promises ("we\'ll look into it")',
                        'Defensive explanations'
                    ]
                }
            
            elif emotion == 'empathy':
                strategies[emotion] = {
                    'email_elements': {
                        'subject_line': 'I understand your frustration - let\'s fix this together',
                        'opening': 'I can imagine how frustrating this experience has been for you.',
                        'body_focus': 'Validation of their specific pain points with personal acknowledgment',
                        'tone': 'Personal, understanding, validating'
                    },
                    'specific_language': [
                        '"I completely understand why..."',
                        '"That must have been incredibly frustrating when..."',
                        '"You\'re absolutely right that..."',
                        '"I\'d feel the same way if..."'
                    ],
                    'avoid_language': [
                        'Generic empathy ("we understand")',
                        'Minimizing their experience',
                        'Corporate speak'
                    ]
                }
            
            elif emotion == 'excitement':
                strategies[emotion] = {
                    'email_elements': {
                        'subject_line': 'Exciting updates based on your feedback + exclusive preview',
                        'opening': 'Your insights are helping us build something amazing!',
                        'body_focus': 'New features, improvements, and exclusive opportunities',
                        'tone': 'Enthusiastic, forward-looking, collaborative'
                    },
                    'specific_language': [
                        '"Based on your suggestion, we\'ve developed..."',
                        '"You\'ll be among the first to experience..."',
                        '"I\'m excited to show you..."',
                        '"This is going to transform how you..."'
                    ],
                    'avoid_language': [
                        'Overly salesy language',
                        'Generic excitement',
                        'Pressure tactics'
                    ]
                }
            
            elif emotion == 'clarity':
                strategies[emotion] = {
                    'email_elements': {
                        'subject_line': 'Clear answers to your questions + step-by-step guide',
                        'opening': 'Let me provide clear answers to address your questions.',
                        'body_focus': 'Direct explanations, tutorials, and structured guidance',
                        'tone': 'Clear, educational, supportive'
                    },
                    'specific_language': [
                        '"Here\'s exactly how to..."',
                        '"Let me break this down step by step..."',
                        '"The key difference is..."',
                        '"To clarify..."'
                    ],
                    'avoid_language': [
                        'Technical jargon',
                        'Assumptions about knowledge',
                        'Overwhelming detail'
                    ]
                }
            
            elif emotion == 'curiosity':
                strategies[emotion] = {
                    'email_elements': {
                        'subject_line': 'Discovered something that might interest you...',
                        'opening': 'Given how you\'re using [product], I thought you\'d find this intriguing.',
                        'body_focus': 'Relevant features, use cases, and possibilities they haven\'t explored',
                        'tone': 'Intriguing, personalized, consultative'
                    },
                    'specific_language': [
                        '"I noticed you\'re already..."',
                        '"What if you could also..."',
                        '"Many clients in similar situations have discovered..."',
                        '"Have you considered..."'
                    ],
                    'avoid_language': [
                        'Generic feature lists',
                        'Pushy upselling',
                        'Irrelevant suggestions'
                    ]
                }
            
            elif emotion == 'loyalty':
                strategies[emotion] = {
                    'email_elements': {
                        'subject_line': 'Your partnership means everything to us',
                        'opening': 'As one of our valued long-term partners...',
                        'body_focus': 'Shared journey, mutual investment, exclusive benefits',
                        'tone': 'Appreciative, partnership-focused, exclusive'
                    },
                    'specific_language': [
                        '"Over our time working together..."',
                        '"Your continued trust in us..."',
                        '"As a valued partner, you have access to..."',
                        '"We\'re invested in your long-term success"'
                    ],
                    'avoid_language': [
                        'Transactional language',
                        'Generic customer messaging',
                        'Short-term focus'
                    ]
                }
            
            elif emotion == 'advocacy':
                strategies[emotion] = {
                    'email_elements': {
                        'subject_line': 'Help other firms discover what you\'ve experienced',
                        'opening': 'Your success story could help other legal professionals.',
                        'body_focus': 'Referral opportunities, case studies, community involvement',
                        'tone': 'Collaborative, community-focused, empowering'
                    },
                    'specific_language': [
                        '"Would you be willing to share..."',
                        '"Your experience could help..."',
                        '"Join our community of advocates..."',
                        '"We\'d love to feature your success..."'
                    ],
                    'avoid_language': [
                        'Pushy referral requests',
                        'Generic testimonial asks',
                        'One-sided benefit language'
                    ]
                }
        
        return strategies

    def _create_follow_up_actions(self, target_emotions, pain_points, risks, nps_score):
        """Create specific follow-up action plans"""
        actions = []
        
        # Immediate actions (within 24-48 hours)
        immediate_actions = []
        
        if 'reassurance' in target_emotions or 'empathy' in target_emotions:
            immediate_actions.extend([
                {
                    'action': 'Send personalized email from senior team member',
                    'what_to_do': 'Have VP/Director personally acknowledge their feedback',
                    'what_to_include': 'Specific timeline for resolution, direct contact info, escalation commitment',
                    'timeline': 'Within 4 hours',
                    'owner': 'Customer Success Director'
                },
                {
                    'action': 'Schedule emergency response call',
                    'what_to_do': 'Book 30-minute call to discuss concerns directly',
                    'what_to_include': 'Product manager + CS rep, agenda focused on their specific issues',
                    'timeline': 'Within 24 hours',
                    'owner': 'Product Manager + CS Rep'
                }
            ])
        
        if 'clarity' in target_emotions:
            immediate_actions.append({
                'action': 'Create personalized tutorial package',
                'what_to_do': 'Record custom screen-share addressing their specific confusion points',
                'what_to_include': 'Step-by-step video, written guide, practice exercises',
                'timeline': 'Within 24 hours',
                'owner': 'Customer Success Specialist'
            })
        
        # Short-term actions (1-2 weeks)
        short_term_actions = []
        
        if 'excitement' in target_emotions or 'curiosity' in target_emotions:
            short_term_actions.extend([
                {
                    'action': 'Invite to beta program',
                    'what_to_do': 'Offer early access to features that address their interests',
                    'what_to_include': 'Exclusive beta invite, direct feedback channel to product team',
                    'timeline': '1 week',
                    'owner': 'Product Marketing'
                },
                {
                    'action': 'Advanced feature deep-dive session',
                    'what_to_do': 'Host 1-hour session exploring advanced capabilities',
                    'what_to_include': 'Customized demo, advanced tips, Q&A with product expert',
                    'timeline': '1-2 weeks',
                    'owner': 'Solutions Engineer'
                }
            ])
        
        if 'loyalty' in target_emotions:
            short_term_actions.append({
                'action': 'Executive relationship review',
                'what_to_do': 'Schedule quarterly business review with executive sponsor',
                'what_to_include': 'ROI analysis, roadmap preview, strategic planning session',
                'timeline': '2 weeks',
                'owner': 'Account Executive + VP Customer Success'
            })
        
        # Long-term actions (1-3 months)
        long_term_actions = []
        
        if any(pp['type'] == 'workflow_inefficiency' for pp in pain_points):
            long_term_actions.append({
                'action': 'Workflow optimization consultation',
                'what_to_do': 'Comprehensive review of their processes and optimization opportunities',
                'what_to_include': 'Process mapping, efficiency recommendations, implementation support',
                'timeline': '1 month',
                'owner': 'Customer Success Manager + Solutions Consultant'
            })
        
        if 'advocacy' in target_emotions:
            long_term_actions.extend([
                {
                    'action': 'Customer advisory board invitation',
                    'what_to_do': 'Invite to join exclusive customer advisory panel',
                    'what_to_include': 'Quarterly meetings, product roadmap input, networking opportunities',
                    'timeline': '6-8 weeks',
                    'owner': 'Head of Product + Customer Success'
                },
                {
                    'action': 'Case study development',
                    'what_to_do': 'Collaborate on success story documentation',
                    'what_to_include': 'Professional case study, conference speaking opportunity, thought leadership platform',
                    'timeline': '2-3 months',
                    'owner': 'Marketing + Customer Success'
                }
            ])
        
        return {
            'immediate': immediate_actions,
            'short_term': short_term_actions,
            'long_term': long_term_actions,
            'total_actions': len(immediate_actions) + len(short_term_actions) + len(long_term_actions)
        }
    
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
        
        # Emotional Strategy & Follow-up Plan
        st.subheader("💝 Emotional Strategy & Follow-up Plan")

        emotional_strategy = analyzer._generate_emotional_strategy(
            results['emotions'], 
            results['pain_points'], 
            results['risks'], 
            nps_score, 
            results['aspect_analysis']
        )

        # Current vs Target Emotions
        col1, col2 = st.columns(2)

        with col1:
            st.write("**🎭 Current Emotional State:**")
            if emotional_strategy['current_emotions']:
                for emotion in emotional_strategy['current_emotions']:
                    intensity = results['emotions'][emotion]['intensity']
                    st.write(f"• {emotion.title()}: {intensity:.1%} intensity")
            else:
                st.write("• Neutral/Professional tone detected")
            
            if emotional_strategy['dominant_emotion'] != 'neutral':
                st.info(f"**Dominant Emotion:** {emotional_strategy['dominant_emotion'].title()}")

        with col2:
            st.write("**🎯 Target Emotions for Follow-up:**")
            for emotion, config in emotional_strategy['target_emotions'].items():
                priority_icons = {'critical': '🚨', 'primary': '🎯', 'secondary': '📍'}
                st.write(f"{priority_icons[config['priority']]} **{emotion.title()}** ({config['priority']})")
                st.write(f"   *{config['rationale']}*")

        # Email Strategy Details
        st.write("---")
        st.subheader("📧 Specific Email Strategy")

        for emotion, strategy in emotional_strategy['email_strategies'].items():
            with st.expander(f"📝 {emotion.title()} Email Strategy"):
                
                # Email elements
                st.write("**📌 Email Structure:**")
                elements = strategy['email_elements']
                st.write(f"**Subject Line:** {elements['subject_line']}")
                st.write(f"**Opening:** {elements['opening']}")
                st.write(f"**Body Focus:** {elements['body_focus']}")
                st.write(f"**Tone:** {elements['tone']}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**✅ Use This Language:**")
                    for phrase in strategy['specific_language']:
                        st.write(f"• {phrase}")
                
                with col2:
                    st.write("**❌ Avoid This Language:**")
                    for phrase in strategy['avoid_language']:
                        st.write(f"• {phrase}")

        # Follow-up Action Plan
        st.write("---")
        st.subheader("📅 Comprehensive Follow-up Action Plan")

        actions = emotional_strategy['follow_up_actions']

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Actions", actions['total_actions'])
        with col2:
            st.metric("Immediate (24-48h)", len(actions['immediate']))
        with col3:
            st.metric("Short-term (1-2w)", len(actions['short_term']))
        with col4:
            st.metric("Long-term (1-3m)", len(actions['long_term']))

        # Detailed action plans
        action_categories = [
            ('🚨 Immediate Actions (24-48 hours)', actions['immediate'], 'error'),
            ('⚡ Short-term Actions (1-2 weeks)', actions['short_term'], 'warning'), 
            ('📈 Long-term Actions (1-3 months)', actions['long_term'], 'info')
        ]

        for category_name, action_list, alert_type in action_categories:
            if action_list:
                st.write(f"**{category_name}:**")
                
                for i, action in enumerate(action_list, 1):
                    with st.expander(f"{i}. {action['action']} - {action['timeline']}"):
                        st.write(f"**👤 Owner:** {action['owner']}")
                        st.write(f"**🎯 What to Do:** {action['what_to_do']}")
                        st.write(f"**📦 What to Include:** {action['what_to_include']}")
                        st.write(f"**⏰ Timeline:** {action['timeline']}")
                        
                        if alert_type == 'error':
                            st.error("⚠️ **URGENT**: This action requires immediate attention")
                        elif alert_type == 'warning':
                            st.warning("📅 **SCHEDULED**: Add to calendar and assign owner")
                        else:
                            st.info("📋 **PLANNED**: Add to long-term customer success roadmap")

        # Email Template Generator
        st.write("---")
        st.subheader("✉️ Generated Email Template")

        # Select primary target emotion for template
        primary_emotion = next((emotion for emotion, config in emotional_strategy['target_emotions'].items() 
                               if config['priority'] in ['critical', 'primary']), 'appreciation')

        if primary_emotion in emotional_strategy['email_strategies']:
            strategy = emotional_strategy['email_strategies'][primary_emotion]
            
            # Generate sample email
            sample_email = f"""Subject: {strategy['email_elements']['subject_line']}

Dear [Customer Name],

{strategy['email_elements']['opening']}

[SPECIFIC CONTENT BASED ON THEIR FEEDBACK - Focus on: {strategy['email_elements']['body_focus']}]

{strategy['specific_language'][0].strip('"')} [specific details about their situation].

{strategy['specific_language'][1].strip('"')} [concrete next steps].

I'd love to schedule a brief call to discuss this further and ensure we're addressing your needs effectively.

Best regards,
[Your Name]
[Title]
[Direct Contact Information]

P.S. {strategy['specific_language'][-1].strip('"')} - let's make this right together.
"""
            
            st.text_area("📝 Sample Email Template:", sample_email, height=300)
            
            # Copy to clipboard button (simulated)
            if st.button("📋 Copy Template"):
                st.success("Template copied to clipboard! (In real implementation)")

        # Success Metrics to Track
        st.write("---")
        st.subheader("📊 Success Metrics to Track")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**📈 Engagement Metrics:**")
            st.write("• Email open rate")
            st.write("• Response rate and time")
            st.write("• Meeting acceptance rate")
            st.write("• Follow-up email engagement")

        with col2:
            st.write("**🎯 Outcome Metrics:**")
            st.write("• NPS score change (next survey)")
            st.write("• Product usage increase")
            st.write("• Support ticket volume")
            st.write("• Renewal/expansion likelihood")
        
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
- **💝 Creating emotional follow-up strategies** with specific language and actions
- **📧 Generating personalized email templates** and action plans
- **🤖 AI/Rule-based hybrid** for maximum compatibility
""")
