# Add this new method to the DeepFeedbackAnalyzer class

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
