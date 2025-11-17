import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Legal Feedback Intelligence Hub",
    page_icon="⚖️",
    layout="wide"
)

class LegalSentimentAnalyzer:
    """Advanced sentiment analyzer specifically designed for legal feedback analysis."""
    def __init__(self):
        # Legal-specific positive terms
        self.positive_terms = {
            'excellent', 'outstanding', 'exceptional', 'superior', 'impressive',
            'efficient', 'streamlined', 'intuitive', 'user-friendly', 'comprehensive',
            'accurate', 'precise', 'reliable', 'responsive', 'helpful', 'professional',
            'satisfied', 'pleased', 'happy', 'love', 'amazing', 'fantastic',
            'recommend', 'valuable', 'useful', 'effective', 'seamless', 'smooth',
            'fast', 'quick', 'timely', 'organized', 'clear', 'detailed'
        }
        # Legal-specific negative terms
        self.negative_terms = {
            'terrible', 'awful', 'horrible', 'disappointing', 'frustrating',
            'confusing', 'complicated', 'difficult', 'slow', 'buggy', 'broken',
            'unreliable', 'inaccurate', 'outdated', 'missing', 'incomplete',
            'useless', 'waste', 'annoying', 'clunky', 'cumbersome', 'tedious',
            'hate', 'dislike', 'poor', 'bad', 'worst', 'lacking', 'inadequate',
            'unresponsive', 'crashes', 'errors', 'issues', 'problems'
        }
        # Intensifiers that modify sentiment
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'incredibly': 1.8, 'absolutely': 1.7,
            'really': 1.3, 'quite': 1.2, 'somewhat': 0.8, 'slightly': 0.7,
            'barely': 0.5, 'hardly': 0.4, 'completely': 1.9, 'totally': 1.8
        }
        # Legal aspect categories with keywords
        self.legal_aspects = {
            'case_search': {
                'name': 'Case Search & Research',
                'icon': '🔍',
                'keywords': [
                    'case search', 'research', 'find cases', 'legal research',
                    'search results', 'search function', 'database search', 'case law',
                    'precedent', 'citation', 'westlaw', 'lexis'
                ]
            },
            'document_management': {
                'name': 'Document Management',
                'icon': '📄',
                'keywords': [
                    'document', 'filing', 'upload', 'download', 'pdf', 'brief',
                    'pleading', 'contract', 'agreement', 'template', 'document review',
                    'version control'
                ]
            },
            'billing': {
                'name': 'Billing & Time Tracking',
                'icon': '💰',
                'keywords': [
                    'billing', 'invoice', 'time tracking', 'hours', 'rates', 'expense',
                    'cost', 'payment', 'fee', 'pricing', 'timesheet'
                ]
            },
            'client_portal': {
                'name': 'Client Portal',
                'icon': '👥',
                'keywords': [
                    'client portal', 'client access', 'communication', 'client login',
                    'sharing', 'collaboration', 'messages'
                ]
            },
            'calendar': {
                'name': 'Calendar & Scheduling',
                'icon': '📅',
                'keywords': [
                    'calendar', 'schedule', 'appointment', 'deadline', 'court date',
                    'meeting', 'reminder', 'docket'
                ]
            },
            'compliance': {
                'name': 'Compliance & Ethics',
                'icon': '⚖️',
                'keywords': [
                    'compliance', 'ethics', 'rules', 'bar requirements',
                    'professional responsibility', 'conflict check', 'audit trail'
                ]
            },
            'reporting': {
                'name': 'Reporting & Analytics',
                'icon': '📊',
                'keywords': [
                    'report', 'analytics', 'dashboard', 'metrics', 'statistics',
                    'performance', 'insights', 'data', 'trends'
                ]
            },
            'performance': {
                'name': 'System Performance',
                'icon': '⚡',
                'keywords': [
                    'speed', 'performance', 'slow', 'fast', 'loading', 'lag',
                    'responsive', 'crash', 'freeze', 'timeout', 'error', 'bug'
                ]
            },
            'training': {
                'name': 'Training & Support',
                'icon': '🎓',
                'keywords': [
                    'training', 'support', 'help', 'tutorial', 'documentation',
                    'onboarding', 'learning', 'guidance', 'assistance'
                ]
            },
            'integration': {
                'name': 'Integration & Compatibility',
                'icon': '🔗',
                'keywords': [
                    'integration', 'compatibility', 'sync', 'import', 'export',
                    'api', 'third party', 'outlook', 'office', 'accounting software'
                ]
            }
        }

    def analyze_sentiment(self, text):
        """Analyze sentiment of legal feedback text and return score, label, confidence."""
        if not text or not text.strip():
            return {'score': 0.0, 'label': 'neutral', 'confidence': 0.0}
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        positive_score = 0.0
        negative_score = 0.0
        total_words = len(words)

        for i, word in enumerate(words):
            # Check if the previous word is an intensifier
            intensifier = 1.0
            if i > 0 and words[i - 1] in self.intensifiers:
                intensifier = self.intensifiers[words[i - 1]]
            # Increment positive or negative score based on word sentiment
            if word in self.positive_terms:
                positive_score += intensifier
            elif word in self.negative_terms:
                negative_score += intensifier

        # Calculate net sentiment score (normalized by length)
        net_score = (positive_score - negative_score) / total_words if total_words > 0 else 0.0

        # Determine sentiment label and confidence
        if net_score > 0.02:
            label = 'positive'
            confidence = min(abs(net_score) * 10, 1.0)  # up to 1.0
        elif net_score < -0.02:
            label = 'negative'
            confidence = min(abs(net_score) * 10, 1.0)
        else:
            label = 'neutral'
            # Higher confidence for net_score near zero (no strong sentiment words)
            confidence = 1.0 - min(abs(net_score) * 5, 0.5)

        return {'score': net_score, 'label': label, 'confidence': confidence}

    def extract_aspects(self, text):
        """Extract legal aspects mentioned in the text, with context and sentiment for each aspect."""
        text_lower = text.lower()
        found_aspects = []
        for aspect_key, aspect_info in self.legal_aspects.items():
            for keyword in aspect_info['keywords']:
                if keyword in text_lower:
                    # Find the sentence containing this keyword for context
                    sentences = re.split(r'[.!?]+', text)
                    context_sentence = ""
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            context_sentence = sentence.strip()
                            break
                    # Analyze sentiment of that context sentence (to gauge aspect sentiment)
                    sentence_sentiment = self.analyze_sentiment(context_sentence)
                    found_aspects.append({
                        'name': aspect_info['name'],
                        'icon': aspect_info['icon'],
                        'keyword': keyword,
                        'text': context_sentence[:100] + "..." if len(context_sentence) > 100 else context_sentence,
                        'sentiment': sentence_sentiment['score']
                    })
                    break  # Only record the first occurrence per aspect category
        return found_aspects

class CommunicationStyleGenerator:
    """Generate responses based on different communication styles."""
    
    def __init__(self):
        self.styles = {
            'doers': {
                'name': 'Doers',
                'icon': '⚡',
                'description': 'Results-first, action-oriented, brief',
                'characteristics': ['direct', 'concise', 'action-focused', 'time-sensitive']
            },
            'thinkers': {
                'name': 'Thinkers', 
                'icon': '🧠',
                'description': 'Data-driven, structured, detailed',
                'characteristics': ['analytical', 'thorough', 'evidence-based', 'logical']
            },
            'influencers': {
                'name': 'Influencers',
                'icon': '🌟', 
                'description': 'Enthusiastic, people-focused, visionary',
                'characteristics': ['inspiring', 'relationship-focused', 'optimistic', 'big-picture']
            },
            'connectors': {
                'name': 'Connectors',
                'icon': '🤝',
                'description': 'Collaborative, team-beneficial, supportive',
                'characteristics': ['inclusive', 'consensus-building', 'supportive', 'team-oriented']
            }
        }
    
    def generate_response(self, feedback_text, sentiment_result, style='doers'):
        """Generate a response email/message based on communication style."""
        sentiment = sentiment_result['label']
        
        if style == 'doers':
            return self._generate_doers_response(feedback_text, sentiment)
        elif style == 'thinkers':
            return self._generate_thinkers_response(feedback_text, sentiment)
        elif style == 'influencers':
            return self._generate_influencers_response(feedback_text, sentiment)
        elif style == 'connectors':
            return self._generate_connectors_response(feedback_text, sentiment)
        else:
            return self._generate_doers_response(feedback_text, sentiment)
    
    def _generate_doers_response(self, feedback_text, sentiment):
        """Generate action-oriented, brief response."""
        if sentiment == 'positive':
            return {
                'subject': 'Action: Your Feedback - Next Steps',
                'body': """Thank you for your feedback.

KEY ACTIONS:
• Documented your input ✓
• Shared with product team ✓  
• Will update you on improvements

TIMELINE: Updates within 2 weeks.

Questions? Reply directly.

Best regards,
Legal Software Team"""
            }
        elif sentiment == 'negative':
            return {
                'subject': 'URGENT: Resolving Your Concerns',
                'body': """We've received your feedback and are taking immediate action.

IMMEDIATE STEPS:
• Technical team assigned
• Priority escalation initiated
• Direct support contact assigned

NEXT: You'll hear from us within 24 hours with solutions.

Contact: [Direct phone/email]

Regards,
Legal Software Team"""
            }
        else:
            return {
                'subject': 'Your Feedback - Action Plan',
                'body': """Thank you for your input.

ACTIONS TAKEN:
• Feedback logged and categorized
• Relevant teams notified
• Improvement roadmap updated

RESULT: Your suggestions will influence our Q1 updates.

Best,
Legal Software Team"""
            }
    
    def _generate_thinkers_response(self, feedback_text, sentiment):
        """Generate data-driven, structured response."""
        if sentiment == 'positive':
            return {
                'subject': 'Feedback Analysis: Your Input & Our Data-Driven Response',
                'body': """Dear Valued Client,

Thank you for your detailed feedback. Here's our structured analysis and response:

FEEDBACK ANALYSIS:
• Sentiment Score: Positive
• Key Themes Identified: [Performance, Usability, Value]
• Alignment with User Base: 87% of similar feedback trends positive

OUR DATA-DRIVEN APPROACH:
1. Your feedback joins 200+ positive responses this quarter
2. Specific features you mentioned rank in top 3 user satisfaction metrics
3. ROI data shows 23% efficiency improvement for firms using these features

NEXT STEPS:
• Documentation of your use case for product enhancement
• Inclusion in quarterly user satisfaction report
• Invitation to beta test upcoming features

METRICS TO TRACK:
- Feature usage optimization
- Performance benchmarking
- User satisfaction correlation

We appreciate your analytical approach to feedback.

Sincerely,
Legal Software Analytics Team"""
            }
        elif sentiment == 'negative':
            return {
                'subject': 'Technical Analysis: Issue Resolution Framework',
                'body': """Dear Client,

We've conducted a comprehensive analysis of your feedback:

ISSUE CATEGORIZATION:
• Problem Type: [Technical/Process/Interface]
• Severity Level: High Priority
• Impact Assessment: Critical to user workflow

ROOT CAUSE ANALYSIS:
1. Technical diagnostic completed
2. User journey mapping reviewed  
3. System performance metrics analyzed

RESOLUTION METHODOLOGY:
Phase 1: Immediate workaround (24-48 hrs)
Phase 2: Systematic fix implementation (1-2 weeks)
Phase 3: Prevention protocols activated

METRICS & MONITORING:
• Performance benchmarks established
• User experience tracking implemented
• Success criteria defined

DOCUMENTATION PROVIDED:
- Technical specifications
- Timeline with milestones
- Success measurement framework

We'll provide weekly progress reports with quantifiable metrics.

Best regards,
Technical Resolution Team"""
            }
        else:
            return {
                'subject': 'Feedback Assessment: Structured Review & Response',
                'body': """Dear Client,

We've completed a systematic review of your feedback:

ANALYSIS FRAMEWORK:
• Content categorization completed
• User journey impact assessed  
• Feature utilization patterns reviewed

STRUCTURED EVALUATION:
1. Feedback classification: Mixed/Neutral sentiment
2. Priority ranking based on user impact
3. Resource allocation analysis completed

SYSTEMATIC RESPONSE:
Short-term (30 days):
- Immediate improvements where feasible
- User experience optimizations

Medium-term (90 days):  
- Feature enhancement roadmap
- User training resource development

Long-term (6 months):
- Strategic product evolution
- Advanced feature integration

MEASUREMENT CRITERIA:
• User satisfaction metrics
• Feature adoption rates
• Performance benchmarks

We'll provide quarterly progress reports with detailed analytics.

Respectfully,
Product Strategy Team"""
            }
    
    def _generate_influencers_response(self, feedback_text, sentiment):
        """Generate enthusiastic, people-focused response."""
        if sentiment == 'positive':
            return {
                'subject': '🌟 Amazing Feedback! Let\'s Amplify Your Success',
                'body': """Hi there!

WOW! Your feedback absolutely made our day! 🎉

WHAT EXCITES US:
Your success story is exactly why we're passionate about legal technology. When firms like yours thrive, the entire legal community benefits!

AMPLIFYING YOUR SUCCESS:
• Would love to feature your story (with permission)
• Invite you to our user community champions program
• Early access to exciting new features coming soon

VISION ALIGNMENT:
Your feedback reinforces our vision of transforming legal practice through innovative technology. Together, we're reshaping how legal professionals work!

THE BIGGER PICTURE:
Every positive experience like yours inspires us to push boundaries and create even more amazing solutions for the legal community.

LET'S CONNECT:
• Join our monthly user success showcases
• Beta test revolutionary features launching Q1
• Network with other legal innovation leaders

Your enthusiasm is contagious - thank you for being part of our journey!

With excitement,
Legal Innovation Team 🚀"""
            }
        elif sentiment == 'negative':
            return {
                'subject': 'Turning Challenges Into Opportunities - We\'re Here for You!',
                'body': """Hello,

First, thank you for being honest about your experience. Your courage to share challenges helps us grow! 💪

YOUR IMPACT:
Every piece of feedback like yours is a stepping stone toward creating something truly extraordinary for the legal community.

OUR COMMITMENT TO YOU:
• Personal dedication from our leadership team
• Direct line to our innovation experts
• Exclusive access to immediate improvements

TRANSFORMING CHALLENGES:
We see this as an opportunity to:
- Exceed your expectations
- Strengthen our relationship  
- Create solutions that benefit all legal professionals

THE COMMUNITY SPIRIT:
Your feedback joins thousands of legal professionals helping us build the future of legal technology together.

IMMEDIATE SUPPORT:
Our customer success champion will reach out personally within hours to turn this around!

LOOKING FORWARD:
We're confident this experience will become a success story we celebrate together.

Your partner in legal innovation,
Customer Success Leadership Team ✨"""
            }
        else:
            return {
                'subject': '🤝 Your Voice Matters - Building the Future Together',
                'body': """Hello!

Thank you for taking the time to share your perspective with us!

WHY YOUR INPUT INSPIRES US:
Every voice in our legal community shapes the future of legal technology. Your balanced feedback helps us stay grounded while reaching for the stars!

COMMUNITY BUILDING:
• Your insights join our collective wisdom
• Helping us serve the legal profession better
• Contributing to innovative solutions for all

EXCITING OPPORTUNITIES:
• Join our user advisory council
• Influence our product roadmap
• Connect with fellow legal innovators

THE VISION WE SHARE:
Together, we're creating technology that empowers legal professionals to focus on what matters most - serving clients and advancing justice.

NEXT ADVENTURES:
• Exclusive previews of upcoming features
• Invitations to legal innovation events
• Direct input on future developments

Your participation in this journey means everything to us!

With appreciation and excitement,
Legal Community Team 🌟"""
            }
    
    def _generate_connectors_response(self, feedback_text, sentiment):
        """Generate collaborative, supportive response."""
        if sentiment == 'positive':
            return {
                'subject': 'Thank You - Strengthening Our Partnership Together',
                'body': """Dear Valued Partner,

Your positive feedback strengthens our entire legal community, and we're grateful to have you as part of our extended family.

COLLABORATIVE IMPACT:
Your success contributes to the collective advancement of legal technology adoption across the profession.

TEAM BENEFITS:
• Your insights help us better serve all legal professionals
• Your experience guides improvements that benefit everyone
• Your partnership model success for other firms

STRENGTHENING CONNECTIONS:
• Share your success story with peer networks (with your permission)
• Connect you with other successful implementation teams
• Include you in our collaborative user community

SUPPORTING OUR COMMUNITY:
• Mentorship opportunities with firms beginning their journey
• Participation in user support networks
• Contribution to best practices documentation

TOGETHER WE ACHIEVE MORE:
Your feedback represents the collaborative spirit that makes our legal technology community thrive.

CONTINUED PARTNERSHIP:
• Regular check-ins to ensure ongoing success
• Team-based support for any future needs
• Collaborative planning for your firm's growth

Thank you for being such a supportive partner in our shared mission.

In partnership,
Legal Technology Community Team"""
            }
        elif sentiment == 'negative':
            return {
                'subject': 'Working Together - Collaborative Problem Solving',
                'body': """Dear Partner,

Thank you for bringing these challenges to our attention. Strong partnerships are built on open communication and collaborative problem-solving.

OUR TEAM APPROACH:
• Dedicated cross-functional support team assigned
• Collaborative issue resolution process initiated
• Team-based solutions tailored to your needs

SUPPORTING YOUR SUCCESS:
Your challenges become our shared mission. We're committed to working together to find solutions that benefit not just your firm, but our entire user community.

COLLABORATIVE SOLUTIONS:
• Joint problem-solving sessions with your team
• Peer network connections for shared experiences
• Group training and support resources

COMMUNITY STRENGTH:
Your willingness to work with us through challenges strengthens the entire legal technology community and helps us serve everyone better.

PARTNERSHIP COMMITMENT:
• Regular check-ins with your implementation team
• Collaborative timeline development
• Shared success metrics and milestones

BUILDING TOGETHER:
Every challenge we solve together makes our entire community stronger and more resilient.

We're honored to have you as a partner in this journey.

In collaboration,
Partnership Success Team"""
            }
        else:
            return {
                'subject': 'Growing Together - Collaborative Feedback Partnership',
                'body': """Dear Community Partner,

Thank you for sharing your thoughtful perspective. Balanced feedback like yours helps our entire legal technology community grow stronger together.

COLLABORATIVE GROWTH:
• Your insights contribute to our collective learning
• Balanced perspective helps us serve diverse needs
• Community-focused improvements benefit everyone

TEAM-CENTERED APPROACH:
• Include your feedback in collaborative planning sessions
• Share anonymized insights with peer advisory groups
• Integrate suggestions into community-driven roadmap

SUPPORTING COLLECTIVE SUCCESS:
• Connect you with peer networks for shared experiences
• Invitation to collaborative user forums
• Participation in team-based improvement initiatives

PARTNERSHIP DEVELOPMENT:
• Regular collaborative review sessions
• Joint planning for future enhancements
• Shared success measurement frameworks

COMMUNITY BUILDING:
Your balanced perspective helps us maintain focus on solutions that work for the entire legal profession.

STRENGTHENING BONDS:
• Ongoing partnership check-ins
• Team-based support structures
• Collaborative success planning

Together, we create technology solutions that serve our entire legal community.

In partnership and collaboration,
Community Development Team"""
            }

def generate_sample_data():
    """Generate a sample dataframe of legal software feedback with realistic content."""
    np.random.seed(42)
    sample_responses = [
        "The case search functionality is excellent and saves us hours of research time daily.",
        "Document management system is intuitive, but billing integration needs improvement.",
        "Loving the new client portal! Communication with clients has never been easier.",
        "Calendar scheduling works well, though deadline reminders could be more prominent.",
        "System performance is generally good, but search can be slow during peak hours.",
        "Training materials are comprehensive and onboarding was smooth for our team.",
        "Integration with Outlook works perfectly, but accounting software sync has issues.",
        "Compliance tracking is thorough and helps us stay on top of bar requirements.",
        "The research database is comprehensive but the interface feels outdated.",
        "Time tracking is accurate and billing reports are detailed and professional.",
        "Client communication features are great, but document sharing could be more streamlined.",
        "Love the analytics dashboard - gives great insights into our practice metrics.",
        "System occasionally crashes during large document uploads, quite frustrating.",
        "Customer support is responsive and knowledgeable about legal industry needs.",
        "The mobile app works well for basic functions but lacks advanced features.",
        "Conflict checking is thorough and integration with client intake is seamless.",
        "Document templates save significant time, though more customization options are needed.",
        "Reporting functionality is powerful but could benefit from more visual charts.",
        "Security features are robust and give us confidence in client data protection.",
        "User interface is clean and modern, much better than our previous software."
    ]
    
    data = []
    for i in range(200):  # 200 feedback entries
        nps_score = np.random.randint(0, 11)  # 0 to 10 inclusive
        # Choose response tendency based on NPS score
        if nps_score >= 9:
            # likely positive feedback
            candidate_responses = [r for r in sample_responses if any(word in r.lower() for word in ['excellent', 'love', 'great', 'amazing', 'perfect'])]
            response = np.random.choice(candidate_responses) if candidate_responses else np.random.choice(sample_responses)
        elif nps_score <= 6:
            # likely negative feedback
            candidate_responses = [r for r in sample_responses if any(word in r.lower() for word in ['issues', 'slow', 'needs', 'crashes', 'frustrating'])]
            response = np.random.choice(candidate_responses) if candidate_responses else np.random.choice(sample_responses)
        else:
            # neutral or mixed
            response = np.random.choice(sample_responses)
        client_type = np.random.choice(
            ['Small Firm', 'Mid-size Firm', 'Large Firm', 'Solo Practice'],
            p=[0.3, 0.3, 0.2, 0.2]
        )
        practice_area = np.random.choice([
            'Corporate Law', 'Litigation', 'Family Law', 'Criminal Law',
            'Real Estate', 'Personal Injury', 'Employment Law'
        ])
        data.append({
            'NPS_Score': nps_score,
            'Feedback_Text': response,
            'Client_Type': client_type,
            'Practice_Area': practice_area,
            'Respondent_ID': f"Client_{i+1:03d}"
        })
    df_sample = pd.DataFrame(data)
    return df_sample

def calculate_nps_metrics(df):
    """Calculate NPS and related metrics from the dataframe."""
    total = len(df)
    promoters = len(df[df['NPS_Score'] >= 9])
    detractors = len(df[df['NPS_Score'] <= 6])
    passives = len(df[(df['NPS_Score'] >= 7) & (df['NPS_Score'] <= 8)])
    nps_score = ((promoters - detractors) / total * 100) if total > 0 else 0.0
    return {
        'nps_score': nps_score,
        'promoters': promoters,
        'detractors': detractors,
        'passives': passives,
        'total_responses': total,
        'promoter_rate': (promoters / total * 100) if total > 0 else 0.0,
        'detractor_rate': (detractors / total * 100) if total > 0 else 0.0
    }

def create_nps_gauge(nps_score):
    """Create a Plotly gauge chart for NPS score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=nps_score,
        number={'suffix': "%"},
        title={'text': "Net Promoter Score"},
        delta={'reference': 50, 'relative': False, 'position': "top"},
        gauge={
            'axis': {'range': [-100, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-100, 0], 'color': "lightgray"},
                {'range': [0, 50], 'color': "yellow"},
                {'range': [50, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(t=20, b=20, l=40, r=40))
    return fig

def analyze_feedback_aspects(df, analyzer):
    """Analyze each feedback entry for aspect mentions and sentiment."""
    aspect_data = []
    for _, row in df.iterrows():
        aspects = analyzer.extract_aspects(row['Feedback_Text'])
        for asp in aspects:
            sentiment_label = 'Positive' if asp['sentiment'] > 0.1 else 'Negative' if asp['sentiment'] < -0.1 else 'Neutral'
            aspect_data.append({
                'Aspect': asp['name'],
                'Sentiment_Score': asp['sentiment'],
                'Sentiment_Label': sentiment_label,
                'Client_Type': row['Client_Type'],
                'Practice_Area': row['Practice_Area'],
                'NPS_Score': row['NPS_Score'],
                'Context': asp['text']
            })
    return pd.DataFrame(aspect_data)

def main():
    # Title and description
    st.title("⚖️ Legal Feedback Intelligence Hub")
    st.markdown("**Advanced NPS Analysis & Sentiment Intelligence for Legal Software**")

    # Initialize the sentiment analyzer and communication generator
    analyzer = LegalSentimentAnalyzer()
    comm_generator = CommunicationStyleGenerator()

    # Sidebar - Data upload and filtering
    st.sidebar.header("📊 Data Management")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your feedback data (CSV)",
        type=['csv'],
        help="Expecting columns: NPS_Score, Feedback_Text, Client_Type, Practice_Area"
    )
    # Load data
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ Data uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
            df = generate_sample_data()
            st.sidebar.info("Using sample data instead.")
    else:
        df = generate_sample_data()
        st.sidebar.info("📋 Using sample data (randomly generated). Upload a CSV to analyze your own feedback.")
        # Offer sample template download
        sample_csv = df.head(10).to_csv(index=False)
        st.sidebar.download_button(
            label="📥 Download Sample Data Template",
            data=sample_csv,
            file_name="legal_feedback_template.csv",
            mime="text/csv"
        )

    # Use the full dataframe (no date filtering)
    filtered_df = df.copy()
    
    if filtered_df.empty:
        st.warning("No feedback data available.")
        return

    # Calculate NPS metrics for filtered data
    nps_metrics = calculate_nps_metrics(filtered_df)

    # Top-level KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="📊 NPS Score", value=f"{nps_metrics['nps_score']:.1f}", delta="Target: 50+")
    col2.metric(label="🎯 Promoters", value=nps_metrics['promoters'], delta=f"{nps_metrics['promoter_rate']:.1f}%")
    col3.metric(label="⚠️ Detractors", value=nps_metrics['detractors'], delta=f"{nps_metrics['detractor_rate']:.1f}%")
    col4.metric(label="📝 Total Responses", value=nps_metrics['total_responses'])

    # Create tabs for different analysis sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Aspect Analysis", "📈 Sentiment Analysis",
        "💡 AI Recommendations", "📝 Feedback Details",
        "✉️ Email/Transcript Analysis", "📧 Communication Styles"
    ])

    # Tab 1: Aspect Analysis
    with tab1:
        st.subheader("🔍 Legal Aspect Analysis")
        aspect_df = analyze_feedback_aspects(filtered_df, analyzer)
        if not aspect_df.empty:
            # Layout: heatmap on left, top aspects on right
            left_col, right_col = st.columns([2, 1])
            with left_col:
                aspect_summary = aspect_df.groupby(['Aspect', 'Sentiment_Label']).size().unstack(fill_value=0)
                fig = px.imshow(
                    aspect_summary,
                    color_continuous_scale="RdYlGn",
                    labels={'x': 'Sentiment', 'y': 'Aspect', 'color': 'Count'},
                    title="Aspect Sentiment Heatmap"
                )
                fig.update_layout(height=500, margin=dict(t=50, b=50))
                st.plotly_chart(fig, use_container_width=True)
            with right_col:
                st.subheader("📊 Top Aspect Mentions")
                top_aspects = aspect_df['Aspect'].value_counts().head(8)
                for aspect, count in top_aspects.items():
                    avg_sentiment = aspect_df[aspect_df['Aspect'] == aspect]['Sentiment_Score'].mean()
                    sentiment_emoji = "😊" if avg_sentiment > 0.1 else "😞" if avg_sentiment < -0.1 else "😐"
                    st.metric(label=f"{sentiment_emoji} {aspect}", value=f"{count} mentions", delta=f"Avg: {avg_sentiment:.2f}")
        else:
            st.info("No specific legal aspects were mentioned in the selected feedback data.")

    # Tab 2: Sentiment Analysis
    with tab2:
        st.subheader("📈 Sentiment Distribution Analysis")
        
        # Distribution of sentiment labels
        col_left, col_right = st.columns(2)
        with col_left:
            sentiment_labels = [analyzer.analyze_sentiment(text)['label'] for text in filtered_df['Feedback_Text']]
            sentiment_counts = pd.Series(sentiment_labels).value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Overall Sentiment Distribution",
                color_discrete_map={'positive': '#2E8B57', 'neutral': '#DAA520', 'negative': '#DC143C'}
            )
            fig_pie.update_traces(textinfo='label+percent', showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_right:
            nps_sentiment = pd.DataFrame([
                {'NPS_Score': row['NPS_Score'], 'Sentiment': analyzer.analyze_sentiment(row['Feedback_Text'])['label']}
                for _, row in filtered_df.iterrows()
            ])
            fig_box = px.box(
                nps_sentiment, x='Sentiment', y='NPS_Score', color='Sentiment',
                title="NPS Score by Sentiment Category",
                color_discrete_map={'positive': '#2E8B57', 'neutral': '#DAA520', 'negative': '#DC143C'}
            )
            fig_box.update_traces(quartilemethod="inclusive")
            st.plotly_chart(fig_box, use_container_width=True)
            
        # Sentiment by Client Type and Practice Area
        st.subheader("📊 Sentiment by Demographics")
        col_demo1, col_demo2 = st.columns(2)
        
        with col_demo1:
            client_sentiment = pd.DataFrame([
                {
                    'Client_Type': row['Client_Type'], 
                    'Sentiment': analyzer.analyze_sentiment(row['Feedback_Text'])['label']
                }
                for _, row in filtered_df.iterrows()
            ])
            client_sent_counts = client_sentiment.groupby(['Client_Type', 'Sentiment']).size().unstack(fill_value=0)
            fig_client = px.bar(
                client_sent_counts, 
                title="Sentiment by Client Type",
                color_discrete_map={'positive': '#2E8B57', 'neutral': '#DAA520', 'negative': '#DC143C'}
            )
            st.plotly_chart(fig_client, use_container_width=True)
            
        with col_demo2:
            practice_sentiment = pd.DataFrame([
                {
                    'Practice_Area': row['Practice_Area'], 
                    'Sentiment': analyzer.analyze_sentiment(row['Feedback_Text'])['label']
                }
                for _, row in filtered_df.iterrows()
            ])
            practice_sent_counts = practice_sentiment.groupby(['Practice_Area', 'Sentiment']).size().unstack(fill_value=0)
            fig_practice = px.bar(
                practice_sent_counts, 
                title="Sentiment by Practice Area",
                color_discrete_map={'positive': '#2E8B57', 'neutral': '#DAA520', 'negative': '#DC143C'}
            )
            st.plotly_chart(fig_practice, use_container_width=True)

    # Tab 3: AI Recommendations
    with tab3:
        st.subheader("💡 AI-Powered Recommendations")
        aspect_df = analyze_feedback_aspects(filtered_df, analyzer)
        if not aspect_df.empty:
            # Identify top negative aspects to prioritize
            neg_aspects = aspect_df[aspect_df['Sentiment_Label'] == 'Negative']
            priority_issues = neg_aspects['Aspect'].value_counts().head(5)
            st.subheader("🚨 Priority Issues to Address")
            for i, (aspect, count) in enumerate(priority_issues.items(), start=1):
                with st.expander(f"{i}. {aspect} ({count} negative mentions)"):
                    # Show some sample negative feedback for this aspect
                    aspect_feedback = neg_aspects[neg_aspects['Aspect'] == aspect]
                    st.write("**Sample Feedback Snippets:**")
                    for _, row in aspect_feedback.head(3).iterrows():
                        st.write(f"- \"{row['Context']}\"")
                    # Suggest recommendations (if available for this aspect)
                    recommendations = {
                        'Case Search & Research': [
                            "Optimize the search algorithm for faster results.",
                            "Improve search result relevance and filtering options.",
                            "Implement caching to speed up repeated searches."
                        ],
                        'Document Management': [
                            "Streamline the document upload process.",
                            "Introduce bulk document operations (upload/download).",
                            "Enhance version control with clearer change tracking."
                        ],
                        'System Performance': [
                            "Conduct a performance audit and optimize slow processes.",
                            "Upgrade server infrastructure or enable auto-scaling.",
                            "Implement real-time monitoring to catch issues early."
                        ],
                        'Billing & Time Tracking': [
                            "Simplify the time entry interface for users.",
                            "Introduce automatic time tracking suggestions.",
                            "Allow more customization in billing reports."
                        ],
                        'Integration & Compatibility': [
                            "Improve sync stability with accounting software.",
                            "Expand integration documentation for third-party apps.",
                            "Add alerts for any integration failures."
                        ]
                    }
                    if aspect in recommendations:
                        st.write("**Recommended Actions:**")
                        for rec in recommendations[aspect]:
                            st.write(f"✅ {rec}")
            # Highlight top positive aspects
            st.subheader("🌟 Strengths to Leverage")
            pos_aspects = aspect_df[aspect_df['Sentiment_Label'] == 'Positive']
            top_strengths = pos_aspects['Aspect'].value_counts().head(3)
            for aspect, count in top_strengths.items():
                st.success(f"**{aspect}** — {count} positive mentions. Continue excelling here and consider expanding these features.")
        else:
            st.info("No aspect analysis available. Once feedback is provided, recommendations will appear here.")

    # Tab 4: Feedback Details
    with tab4:
        st.subheader("📝 Detailed Feedback Analysis")
        # Filters for detailed view
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            client_filter = st.selectbox("Filter by Client Type", ['All'] + sorted(filtered_df['Client_Type'].unique().tolist()))
        with filter_col2:
            practice_filter = st.selectbox("Filter by Practice Area", ['All'] + sorted(filtered_df['Practice_Area'].unique().tolist()))
        with filter_col3:
            nps_filter = st.selectbox("Filter by NPS Category", ['All', 'Promoters (9-10)', 'Passives (7-8)', 'Detractors (0-6)'])
        # Apply selected filters
        display_df = filtered_df.copy()
        if client_filter != 'All':
            display_df = display_df[display_df['Client_Type'] == client_filter]
        if practice_filter != 'All':
            display_df = display_df[display_df['Practice_Area'] == practice_filter]
        if nps_filter != 'All':
            if nps_filter == 'Promoters (9-10)':
                display_df = display_df[display_df['NPS_Score'] >= 9]
            elif nps_filter == 'Passives (7-8)':
                display_df = display_df[(display_df['NPS_Score'] >= 7) & (display_df['NPS_Score'] <= 8)]
            elif nps_filter == 'Detractors (0-6)':
                display_df = display_df[display_df['NPS_Score'] <= 6]
        st.write(f"Showing **{len(display_df)}** feedback entries after filtering:")
        # Iterate through filtered feedback for detailed display
        for _, row in display_df.iterrows():
            sentiment_result = analyzer.analyze_sentiment(row['Feedback_Text'])
            sentiment_label = sentiment_result['label']
            sentiment_conf = sentiment_result['confidence']
            sentiment_emoji = "😊" if sentiment_label == 'positive' else "😞" if sentiment_label == 'negative' else "😐"
            expander_title = f"{sentiment_emoji} NPS: {row['NPS_Score']} | {row['Client_Type']}"
            with st.expander(expander_title):
                st.write(f"**Feedback:** {row['Feedback_Text']}")
                col_det1, col_det2 = st.columns(2)
                with col_det1:
                    st.write(f"**Practice Area:** {row['Practice_Area']}")
                    st.write(f"**Sentiment:** {sentiment_label.capitalize()} ({sentiment_conf:.1%} confidence)")
                with col_det2:
                    aspects = analyzer.extract_aspects(row['Feedback_Text'])
                    if aspects:
                        st.write("**Mentioned Aspects:**")
                        for asp in aspects:
                            st.write(f"- {asp['icon']} {asp['name']}")
                    else:
                        st.write("**Mentioned Aspects:** None")

    # Tab 5: Email/Transcript Analysis
    with tab5:
        st.subheader("✉️ Email/Transcript Analysis")
        st.write("Paste the text of a client email or conversation transcript below to analyze its sentiment and key aspects:")
        
        # Use session state to persist input text
        if "input_text" not in st.session_state:
            st.session_state.input_text = ""
        
        user_text = st.text_area(
            "Enter email or transcript text:",
            value=st.session_state.input_text,
            placeholder="Paste your email content or call transcript here...",
            height=150
        )
        
        analyze_button = st.button("Analyze Text", type="primary")
        
        if analyze_button:
            if not user_text or not user_text.strip():
                st.warning("Please enter some text to analyze.")
            else:
                # Save current input to session state
                st.session_state.input_text = user_text
                
                with st.spinner("Analyzing text..."):
                    result = analyzer.analyze_sentiment(user_text)
                    aspects_found = analyzer.extract_aspects(user_text)
                
                # Display Overall Sentiment
                st.subheader("📊 Analysis Results")
                sentiment_label = result['label']
                sentiment_score = result['score']
                confidence_pct = result['confidence'] * 100
                emoji = "😊" if sentiment_label == "positive" else "😞" if sentiment_label == "negative" else "😐"
                
                # Colored container for overall sentiment
                bg_color = '#d4edda' if sentiment_label == 'positive' else '#f8d7da' if sentiment_label == 'negative' else '#fff3cd'
                border_color = '#c3e6cb' if sentiment_label == 'positive' else '#f5c6cb' if sentiment_label == 'negative' else '#ffeeba'
                
                container_html = f"""
                <div style="padding: 15px; border-radius: 8px; background-color: {bg_color}; 
                            border: 1px solid {border_color}; margin-bottom: 1rem;">
                    <h4>{emoji} Overall Sentiment: {sentiment_label.capitalize()}</h4>
                    <p><strong>Score:</strong> {sentiment_score:.3f} &nbsp;|&nbsp; 
                       <strong>Confidence:</strong> {confidence_pct:.1f}%</p>
                </div>
                """
                st.markdown(container_html, unsafe_allow_html=True)
                
                # Mini gauge chart for sentiment score
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=sentiment_score,
                    number={'font': {'size': 48}, 'suffix': ""},
                    gauge={
                        'axis': {'range': [-1, 1], 'tickwidth': 1},
                        'bar': {'color': "black"},
                        'steps': [
                            {'range': [-1, -0.1], 'color': "#F8D7DA"},
                            {'range': [-0.1, 0.1], 'color': "#FFF3CD"},
                            {'range': [0.1, 1], 'color': "#D4EDDA"}
                        ],
                    }
                ))
                gauge_fig.update_layout(height=200, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Display detected aspects
                st.subheader("🔍 Detected Legal Aspects")
                
                if aspects_found:
                    st.write("The following aspects were mentioned and their sentiment context:")
                    for asp in aspects_found:
                        asp_name = asp['name']
                        asp_icon = asp['icon']
                        asp_keyword = asp['keyword']
                        asp_text = asp['text']
                        asp_score = asp['sentiment']
                        asp_label = "positive" if asp_score > 0.1 else "negative" if asp_score < -0.1 else "neutral"
                        asp_emoji = "😊" if asp_label == "positive" else "😞" if asp_label == "negative" else "😐"
                        
                        expander_label = f"{asp_icon} {asp_name} — {asp_emoji} {asp_label.capitalize()} ({asp_score*100:.1f}%)"
                        with st.expander(expander_label):
                            st.write(f"**Context Snippet:** \"{asp_text}\"")
                            st.write(f"**Detected Keyword:** *{asp_keyword}*")
                            
                            # Sentiment bar indicator
                            sentiment_color = "#28a745" if asp_score > 0.1 else "#dc3545" if asp_score < -0.1 else "#ffc107"
                            bar_width = min(100, abs(asp_score) * 100)
                            bar_html = f"""
                            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 2px; margin-top: 5px; width: 100%;">
                                <div style="background-color: {sentiment_color}; width: {bar_width}%; height: 8px; border-radius: 5px;"></div>
                            </div>
                            """
                            st.markdown(bar_html, unsafe_allow_html=True)
                else:
                    st.info("No specific legal aspect keywords were detected in the text.")
                    st.write("The content might be general or not related to key product features.")
                
                # Quick insights metrics
                st.subheader("💡 Quick Insights")
                word_count = len(user_text.split())
                reading_time = max(1, word_count // 200)
                
                insight_col1, insight_col2, insight_col3 = st.columns(3)
                insight_col1.metric("Word Count", word_count)
                insight_col2.metric("Estimated Read Time", f"{reading_time} min")
                insight_col3.metric("Aspects Detected", len(aspects_found))
                
                # Recommended action based on overall sentiment
                if sentiment_label == "negative":
                    st.warning("**Recommended Action:** The sentiment is negative. Prioritize a prompt response and address the concerns mentioned.")
                elif sentiment_label == "positive":
                    st.success("**Recommended Action:** The sentiment is positive. Consider following up for more details or a potential testimonial.")
                else:
                    st.info("**Recommended Action:** The sentiment is neutral. Monitor for any specific questions or requests to address.")
        
        # Example texts for quick testing
        st.subheader("📝 Example Texts to Try")
        
        examples = {
            "Positive Client Email": "Hi team, I wanted to reach out and thank you for the excellent case search functionality. Our research time has been cut in half since implementing your system. The document management is also very intuitive and our whole firm adapted quickly. Great work!",
            "Negative Support Email": "I'm having serious issues with the billing module. It has been crashing repeatedly when I try to generate invoices, and the time tracking interface is confusing. This is affecting our ability to bill clients properly. Please help urgently.",
            "Mixed Feedback": "The legal research database is comprehensive and I love the integration with our case management. However, the system performance has been quite slow lately, especially during peak hours. The client portal works well though."
        }
        
        for example_name, example_text in examples.items():
            if st.button(f"Load: {example_name}"):
                st.session_state.input_text = example_text

    # Tab 6: Communication Style-Based Messaging
    with tab6:
        st.subheader("📧 Communication Style-Based Response Generator")
        st.write("Generate personalized responses based on different communication preferences and styles.")
        
        # Input section
        col_input1, col_input2 = st.columns([2, 1])
        
        with col_input1:
            input_feedback = st.text_area(
                "Client Feedback to Respond To:",
                placeholder="Paste the client feedback you want to respond to...",
                height=120,
                key="comm_feedback_input"
            )
        
        with col_input2:
            st.write("**Select Communication Style:**")
            selected_style = st.radio(
                "Choose Style:",
                options=['doers', 'thinkers', 'influencers', 'connectors'],
                format_func=lambda x: f"{comm_generator.styles[x]['icon']} {comm_generator.styles[x]['name']} - {comm_generator.styles[x]['description']}"
            )
        
        # Style descriptions
        st.subheader("🎯 Communication Style Guide")
        style_cols = st.columns(4)
        
        for i, (style_key, style_info) in enumerate(comm_generator.styles.items()):
            with style_cols[i]:
                is_selected = style_key == selected_style
                border_style = "border: 3px solid #28a745;" if is_selected else "border: 1px solid #ddd;"
                style_html = f"""
                <div style="{border_style} padding: 10px; border-radius: 8px; margin-bottom: 10px;">
                    <h4>{style_info['icon']} {style_info['name']}</h4>
                    <p><em>{style_info['description']}</em></p>
                    <strong>Characteristics:</strong>
                    <ul>
                """
                for char in style_info['characteristics']:
                    style_html += f"<li>{char}</li>"
                style_html += "</ul></div>"
                st.markdown(style_html, unsafe_allow_html=True)
        
        # Generate response button
        if st.button("Generate Response", type="primary", key="generate_comm_response"):
            if not input_feedback or not input_feedback.strip():
                st.warning("Please enter some feedback to respond to.")
            else:
                with st.spinner("Generating personalized response..."):
                    # Analyze sentiment first
                    sentiment_result = analyzer.analyze_sentiment(input_feedback)
                    
                    # Generate response based on style
                    response = comm_generator.generate_response(
                        input_feedback, sentiment_result, selected_style
                    )
                    
                    # Display the generated response
                    st.subheader(f"📧 Generated Response - {comm_generator.styles[selected_style]['icon']} {comm_generator.styles[selected_style]['name']} Style")
                    
                    # Response preview container
                    with st.container():
                        st.write("**Subject Line:**")
                        st.code(response['subject'], language=None)
                        
                        st.write("**Email Body:**")
                        st.text_area(
                            "Response (editable):",
                            value=response['body'],
                            height=400,
                            key="generated_response_editable"
                        )
                    
                    # Action buttons
                    response_col1, response_col2, response_col3 = st.columns(3)
                    
                    with response_col1:
                        if st.button("📋 Copy to Clipboard", key="copy_response"):
                            st.success("Response ready to copy! Use Ctrl+A, Ctrl+C on the text above.")
                    
                    with response_col2:
                        # Download as text file
                        response_text = f"Subject: {response['subject']}\n\n{response['body']}"
                        st.download_button(
                            label="📥 Download Response",
                            data=response_text,
                            file_name=f"response_{selected_style}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
                            mime="text/plain",
                            key="download_response"
                        )
                    
                    with response_col3:
                        if st.button("🔄 Generate Alternative", key="regenerate_response"):
                            st.rerun()
        
        # Sample feedback examples for each style
        st.subheader("📝 Sample Feedback Examples")
        st.write("Try these examples to see how different communication styles handle various types of feedback:")
        
        example_feedbacks = {
            "Positive Technical": "The case search feature has significantly improved our research efficiency. The integration with our existing workflow is seamless and the team adapted quickly. Very satisfied with the performance.",
            
            "Negative Performance": "The system has been running extremely slowly this week. Documents take forever to load and the billing module crashed twice yesterday. This is impacting our client work and needs immediate attention.",
            
            "Mixed Feature Request": "Overall the software works well for our basic needs. However, we really need better reporting capabilities and the mobile app could use more features. The training was helpful but we need ongoing support.",
            
            "Complex Integration Issue": "We're having trouble with the Outlook integration. Emails sync sometimes but not always, and the calendar appointments don't appear consistently. Our IT team has tried troubleshooting but needs technical support."
        }
        
        example_cols = st.columns(2)
        
        for i, (example_name, example_text) in enumerate(example_feedbacks.items()):
            with example_cols[i % 2]:
                with st.expander(f"📋 {example_name}"):
                    st.write(f"**Sample Feedback:** {example_text}")
                    if st.button(f"Use This Example", key=f"comm_example_{i}"):
                        st.session_state['comm_example_feedback'] = example_text
                        st.success(f"Example loaded! Scroll up to generate responses.")
        
        # Load example if selected
        if 'comm_example_feedback' in st.session_state:
            st.session_state.pop('comm_example_feedback')
            st.rerun()

if __name__ == "__main__":
    main()
