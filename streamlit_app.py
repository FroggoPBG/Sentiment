import streamlit as st
import pandas as pd
import re
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def extract_nps_category_from_sentiment(text):
    """
    Extract NPS category from text sentiment since your data doesn't have explicit categories
    """
    if not isinstance(text, str) or len(text.strip()) < 3:
        return 'Unknown'
    
    # Clean text for analysis
    text_clean = text.replace('"', '').replace('/', '').strip()
    if text_clean.lower() in ['', '.', 'no comment', '/']:
        return 'Unknown'
    
    # Use TextBlob for sentiment analysis
    blob = TextBlob(text_clean)
    polarity = blob.sentiment.polarity
    
    # Define thresholds for legal software feedback
    if polarity > 0.2:
        return 'Promoter'
    elif polarity < -0.1:
        return 'Detractor'
    else:
        return 'Passive'

def calculate_nps_from_categories(categories):
    """Calculate NPS from category counts"""
    total = len(categories)
    if total == 0:
        return 0, {}
    
    category_counts = Counter(categories)
    
    promoters = category_counts.get('Promoter', 0)
    passives = category_counts.get('Passive', 0)
    detractors = category_counts.get('Detractor', 0)
    unknown = category_counts.get('Unknown', 0)
    
    # Calculate percentages (excluding unknown)
    known_total = promoters + passives + detractors
    if known_total == 0:
        return 0, category_counts
    
    promoter_pct = (promoters / known_total) * 100
    detractor_pct = (detractors / known_total) * 100
    
    nps = promoter_pct - detractor_pct
    
    return nps, category_counts

def extract_legal_software_themes(comments, nps_category=None, min_frequency=2):
    """
    Extract themes specifically relevant to legal software/database feedback
    """
    if not comments:
        return {}
    
    # Filter valid comments
    valid_comments = []
    for comment in comments:
        if isinstance(comment, str):
            clean_comment = comment.replace('"', '').replace('/', '').strip()
            if len(clean_comment) > 3 and clean_comment.lower() not in ['', '.', 'no comment', '/', 'ok']:
                valid_comments.append(clean_comment)
    
    if not valid_comments:
        return {}
    
    st.write(f"Analyzing {len(valid_comments)} valid comments...")
    
    # Legal software specific themes
    theme_patterns = {
        # Positive themes
        'user_friendly': [
            r'\b(user.?friendly|user.?friend|easy.?to.?use|intuitive|simple.?to.?use)\b',
            r'\bfriendly\b.*\b(interface|platform|system)\b',
            r'\b(interface|platform|system)\b.*\bfriendly\b'
        ],
        'comprehensive_database': [
            r'\b(comprehensive|extensive|wide|broad|large|huge)\b.*\b(database|resources|content|coverage|materials)\b',
            r'\b(database|resources|content|coverage|materials)\b.*\b(comprehensive|extensive|wide|broad|large|huge)\b'
        ],
        'easy_navigation': [
            r'\b(easy|simple)\b.*\b(navigat|search|find)\b',
            r'\b(navigat|search|find)\b.*\b(easy|simple)\b',
            r'\beasy.?to.?navigat\b'
        ],
        'good_search_engine': [
            r'\b(good|excellent|great|powerful|effective)\b.*\b(search|engine|searching)\b',
            r'\b(search|engine|searching)\b.*\b(good|excellent|great|powerful|effective)\b'
        ],
        'reliable_accurate': [
            r'\b(reliable|accurate|trustworthy|authoritative|trusted)\b',
            r'\b(accuracy|reliability|trust)\b'
        ],
        'up_to_date': [
            r'\b(up.?to.?date|current|latest|updated|recent)\b',
            r'\bupdated?\b'
        ],
        'ai_features': [
            r'\bai\b|\bartificial.?intelligence\b|\bai.?driven\b|\bai.?tools?\b',
            r'\banalytic|insight\b'
        ],
        'halsbury_content': [
            r'\bhalsbury\b|\bannotated.?ordinanc\b|\baohk\b',
            r'\bhalsbury.?law\b'
        ],
        'good_customer_service': [
            r'\b(good|excellent|great)\b.*\b(customer|support|service)\b',
            r'\b(customer|support|service)\b.*\b(good|excellent|great|responsive)\b',
            r'\bresponsive.*support\b'
        ],
        'useful_helpful': [
            r'\b(useful|helpful|valuable|beneficial)\b',
            r'\bhelpful.*research\b'
        ],
        
        # Negative themes
        'expensive_pricing': [
            r'\b(expensive|costly|overpriced|pricey)\b',
            r'\bpric(e|ing)\b.*\b(high|expensive|too.?much)\b',
            r'\btoo.?expensive\b'
        ],
        'slow_performance': [
            r'\b(slow|sluggish|delayed)\b.*\b(search|engine|loading|response|speed)\b',
            r'\b(search|engine|loading|response|speed)\b.*\b(slow|sluggish|delayed)\b',
            r'\btakes.*long\b|\bslow.*load\b'
        ],
        'search_issues': [
            r'\bsearch.*\b(not|poor|bad|irrelevant|inaccurate)\b',
            r'\b(not|poor|bad|irrelevant|inaccurate).*search\b',
            r'\bsearch.*result.*not.*relevant\b'
        ],
        'interface_problems': [
            r'\bnot.*user.?friendly\b|\binterface.*not.*friendly\b',
            r'\bdifficult.*\b(use|navigat)\b',
            r'\bnot.*easy.*use\b'
        ],
        'missing_features': [
            r'\bmissing\b|\bnot.*available\b|\bcould.*not.*find\b',
            r'\broom.*improvement\b|\bcould.*improve\b'
        ],
        
        # Neutral/mixed themes
        'moderate_satisfaction': [
            r'\b(moderate|average|okay|acceptable|satisfactory|fine)\b',
            r'\bnot.*often.*use\b|\bseldom.*use\b|\brarely.*use\b'
        ],
        'specific_use_case': [
            r'\bfor.*\b(research|work|practice|legal)\b',
            r'\buse.*for\b|\bneed.*for\b'
        ]
    }
    
    # Count theme occurrences
    theme_counts = {}
    
    for comment in valid_comments:
        comment_lower = comment.lower()
        for theme, patterns in theme_patterns.items():
            for pattern in patterns:
                if re.search(pattern, comment_lower, re.IGNORECASE):
                    theme_counts[theme] = theme_counts.get(theme, 0) + 1
                    break  # Count each theme only once per comment
    
    # Add word frequency analysis for legal terms
    legal_terms = {
        'case_law': [r'\bcase.?law\b|\bcases?\b|\bjudgment\b|\bprecedent\b'],
        'legal_research': [r'\blegal.?research\b|\bresearch\b'],
        'database_mention': [r'\bdatabase\b|\bdb\b'],
        'hong_kong_specific': [r'\bhong.?kong\b|\bhk\b'],
        'lexis_brand': [r'\blexis\b|\blexisnexis\b'],
        'ordinances': [r'\bordinanc\b|\bstatut\b|\blegislation\b'],
        'practical_guidance': [r'\bpractical\b|\bguidance\b|\btemplate\b'],
        'workflow': [r'\bworkflow\b|\bproductivity\b|\befficient\b']
    }
    
    for comment in valid_comments:
        comment_lower = comment.lower()
        for term, patterns in legal_terms.items():
            for pattern in patterns:
                if re.search(pattern, comment_lower, re.IGNORECASE):
                    theme_counts[f"{term}_mentions"] = theme_counts.get(f"{term}_mentions", 0) + 1
                    break
    
    # Filter by minimum frequency
    filtered_themes = {theme: count for theme, count in theme_counts.items() 
                      if count >= min_frequency}
    
    return filtered_themes

# Streamlit App
st.set_page_config(page_title="Lexis+ NPS Analysis", page_icon="⚖️", layout="wide")

st.title("⚖️ Lexis+ Hong Kong NPS Analysis")
st.markdown("Comprehensive analysis of Lexis+ Hong Kong customer feedback and NPS scoring")

# Sample data button
if st.button("📝 Load Sample Data (Your 330 Comments)", type="secondary"):
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
    
    # Create DataFrame
    sample_df = pd.DataFrame({'NPS Verbatim Comments': sample_comments})
    st.session_state['sample_data'] = sample_df
    st.success("Sample data loaded! You can now analyze it using the batch analysis below.")

# Analysis Type Selection
analysis_type = st.radio(
    "Analysis Type",
    ("Individual Feedback", "Batch Analysis"),
    horizontal=True
)

if analysis_type == "Individual Feedback":
    st.subheader("📝 Individual Feedback Analysis")
    
    comment = st.text_area(
        "Customer Comment:",
        placeholder="Enter a Lexis+ Hong Kong customer comment here...",
        height=100
    )
    
    if st.button("Analyze Individual Feedback", type="primary"):
        if comment:
            # Determine NPS category from sentiment
            nps_category = extract_nps_category_from_sentiment(comment)
            
            # Theme extraction
            themes = extract_legal_software_themes([comment], nps_category, min_frequency=1)
            
            st.subheader("📊 Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted NPS Category", nps_category)
                
                # Color coding for categories
                if nps_category == "Promoter":
                    st.success("✅ Positive feedback")
                elif nps_category == "Passive":
                    st.warning("⚠️ Neutral feedback")
                elif nps_category == "Detractor":
                    st.error("❌ Negative feedback")
                else:
                    st.info("ℹ️ Unknown category")
            
            with col2:
                if themes:
                    st.write("**Identified Themes:**")
                    for theme, count in sorted(themes.items(), key=lambda x: x[1], reverse=True):
                        clean_theme = theme.replace('_', ' ').replace(' mentions', '').title()
                        st.write(f"• {clean_theme}")
                else:
                    st.write("No specific themes identified in this comment.")
                    
            # Show sentiment analysis
            blob = TextBlob(comment)
            st.write(f"**Sentiment Score:** {blob.sentiment.polarity:.2f} (Range: -1 to 1)")
        else:
            st.warning("Please enter a customer comment to analyze.")

else:  # Batch Analysis
    st.subheader("📊 Batch Processing with Aggregation")
    
    # Check if sample data is loaded
    uploaded_file = None
    use_sample = False
    
    if 'sample_data' in st.session_state:
        use_sample_data = st.checkbox("Use loaded sample data", value=True)
        if use_sample_data:
            df = st.session_state['sample_data']
            use_sample = True
            st.write("Using sample data:")
            st.dataframe(df.head())
    
    if not use_sample:
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                # Read file based on extension
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error("Unsupported file format. Please upload CSV or Excel files only.")
                    st.stop()
                    
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.stop()
    
    # Proceed with analysis if we have data
    if use_sample or uploaded_file is not None:
        # Column selection
        st.subheader("🔧 Column Configuration")
        
        comment_column = st.selectbox(
            "Select Comment Column:",
            options=df.columns.tolist(),
            help="Column containing customer feedback text"
        )
        
        # Add minimum frequency setting
        min_frequency = st.slider(
            "Minimum frequency for themes:",
            min_value=1,
            max_value=10,
            value=3,
            help="Themes appearing less than this number will be filtered out"
        )
        
        if st.button("🚀 Analyze All Comments", type="primary"):
            with st.spinner("Processing your Lexis+ feedback data... This may take a moment."):
                
                # Extract comments and categorize by sentiment
                comments = df[comment_column].dropna().tolist()
                
                # Predict NPS categories based on sentiment
                nps_categories = []
                for comment in comments:
                    category = extract_nps_category_from_sentiment(comment)
                    nps_categories.append(category)
                
                # Calculate NPS
                nps_score, category_counts = calculate_nps_from_categories(nps_categories)
                
                # Display results
                st.subheader("📈 Lexis+ Hong Kong NPS Analysis Results")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("NPS Score", f"{nps_score:.1f}")
                
                with col2:
                    st.metric("Total Responses", len(nps_categories))
                
                with col3:
                    promoter_pct = (category_counts.get('Promoter', 0) / len(nps_categories)) * 100
                    st.metric("Promoters", f"{promoter_pct:.1f}%")
                
                with col4:
                    detractor_pct = (category_counts.get('Detractor', 0) / len(nps_categories)) * 100
                    st.metric("Detractors", f"{detractor_pct:.1f}%")
                
                # Interpretation
                if nps_score > 50:
                    st.success("🎉 Excellent NPS Score! Customers are very satisfied with Lexis+")
                elif nps_score > 0:
                    st.info("👍 Good NPS Score. Room for improvement in customer satisfaction.")
                else:
                    st.warning("⚠️ NPS Score needs attention. Focus on addressing customer concerns.")
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Category distribution pie chart
                    fig_pie = px.pie(
                        values=list(category_counts.values()),
                        names=list(category_counts.keys()),
                        title="NPS Category Distribution",
                        color_discrete_map={
                            'Promoter': '#2E8B57',
                            'Passive': '#FFD700', 
                            'Detractor': '#DC143C',
                            'Unknown': '#808080'
                        }
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # NPS gauge chart
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
                                {'range': [-100, 0], 'color': "lightcoral"},
                                {'range': [0, 50], 'color': "lightyellow"},
                                {'range': [50, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Overall themes analysis
                st.subheader("🎯 Key Themes in Customer Feedback")
                
                overall_themes = extract_legal_software_themes(comments, None, min_frequency)
                
                if overall_themes:
                    # Create themes dataframe
                    themes_data = []
                    for theme, count in overall_themes.items():
                        clean_theme = theme.replace('_', ' ').replace(' mentions', '').title()
                        percentage = (count / len(comments)) * 100
                        themes_data.append({
                            'Theme': clean_theme, 
                            'Count': count, 
                            'Percentage': f"{percentage:.1f}%"
                        })
                    
                    themes_df = pd.DataFrame(themes_data).sort_values('Count', ascending=False)
                    
                    # Display themes table
                    st.dataframe(themes_df.head(15), use_container_width=True)
                    
                    # Themes visualization
                    top_themes = themes_df.head(12)
                    fig_themes = px.bar(
                        top_themes, 
                        x='Count', 
                        y='Theme', 
                        orientation='h',
                        title="Most Common Themes in Lexis+ Feedback",
                        color='Count',
                        color_continuous_scale='viridis'
                    )
                    fig_themes.update_layout(height=500)
                    st.plotly_chart(fig_themes, use_container_width=True)
                
                # Category-specific analysis
                st.subheader("📝 Detailed Analysis by NPS Category")
                
                # Add categories to dataframe
                df_with_nps = df.copy()
                df_with_nps['NPS_Category'] = nps_categories
                
                tab1, tab2, tab3 = st.tabs(["🟢 Promoters", "🟡 Passives", "🔴 Detractors"])
                
                with tab1:
                    promoter_comments = df_with_nps[df_with_nps['NPS_Category'] == 'Promoter'][comment_column].dropna().tolist()
                    if promoter_comments:
                        promoter_themes = extract_legal_software_themes(promoter_comments, 'Promoter', min_frequency)
                        
                        st.write(f"**{len(promoter_comments)} Promoter comments analyzed**")
                        
                        if promoter_themes:
                            promoter_data = []
                            for theme, count in promoter_themes.items():
                                clean_theme = theme.replace('_', ' ').replace(' mentions', '').title()
                                percentage = (count / len(promoter_comments)) * 100
                                promoter_data.append({
                                    'Theme': clean_theme,
                                    'Count': count,
                                    'Percentage': f"{percentage:.1f}%"
                                })
                            
                            promoter_df = pd.DataFrame(promoter_data).sort_values('Count', ascending=False)
                            st.dataframe(promoter_df)
                            
                            # Top positive themes
                            st.write("**What Promoters Love About Lexis+:**")
                            for _, row in promoter_df.head(5).iterrows():
                                st.write(f"• {row['Theme']}: {row['Count']} mentions ({row['Percentage']})")
                        else:
                            st.write("No themes found above the minimum frequency threshold.")
                    else:
                        st.write("No Promoter comments found.")
                
                with tab2:
                    passive_comments = df_with_nps[df_with_nps['NPS_Category'] == 'Passive'][comment_column].dropna().tolist()
                    if passive_comments:
                        passive_themes = extract_legal_software_themes(passive_comments, 'Passive', min_frequency)
                        
                        st.write(f"**{len(passive_comments)} Passive comments analyzed**")
                        
                        if passive_themes:
                            passive_data = []
                            for theme, count in passive_themes.items():
                                clean_theme = theme.replace('_', ' ').replace(' mentions', '').title()
                                percentage = (count / len(passive_comments)) * 100
                                passive_data.append({
                                    'Theme': clean_theme,
                                    'Count': count,
                                    'Percentage': f"{percentage:.1f}%"
                                })
                            
                            passive_df = pd.DataFrame(passive_data).sort_values('Count', ascending=False)
                            st.dataframe(passive_df)
                            
                            st.write("**Key Insights from Passive Users:**")
                            for _, row in passive_df.head(5).iterrows():
                                st.write(f"• {row['Theme']}: {row['Count']} mentions ({row['Percentage']})")
                        else:
                            st.write("No themes found above the minimum frequency threshold.")
                    else:
                        st.write("No Passive comments found.")
                
                with tab3:
                    detractor_comments = df_with_nps[df_with_nps['NPS_Category'] == 'Detractor'][comment_column].dropna().tolist()
                    if detractor_comments:
                        detractor_themes = extract_legal_software_themes(detractor_comments, 'Detractor', min_frequency)
                        
                        st.write(f"**{len(detractor_comments)} Detractor comments analyzed**")
                        
                        if detractor_themes:
                            detractor_data = []
                            for theme, count in detractor_themes.items():
                                clean_theme = theme.replace('_', ' ').replace(' mentions', '').title()
                                percentage = (count / len(detractor_comments)) * 100
                                detractor_data.append({
                                    'Theme': clean_theme,
                                    'Count': count,
                                    'Percentage': f"{percentage:.1f}%"
                                })
                            
                            detractor_df = pd.DataFrame(detractor_data).sort_values('Count', ascending=False)
                            st.dataframe(detractor_df)
                            
                            st.write("**Areas for Improvement (Detractor Concerns):**")
                            for _, row in detractor_df.head(5).iterrows():
                                st.write(f"• {row['Theme']}: {row['Count']} mentions ({row['Percentage']})")
                        else:
                            st.write("No themes found above the minimum frequency threshold.")
                    else:
                        st.write("No Detractor comments found.")
                
                # Actionable insights
                st.subheader("💡 Key Insights & Recommendations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Strengths to Leverage:**")
                    if overall_themes:
                        positive_themes = ['user_friendly', 'comprehensive_database', 'easy_navigation', 
                                         'good_search_engine', 'reliable_accurate', 'up_to_date']
                        for theme in positive_themes:
                            if theme in overall_themes:
                                count = overall_themes[theme]
                                clean_theme = theme.replace('_', ' ').title()
                                st.write(f"✅ {clean_theme}: {count} mentions")
                
                with col2:
                    st.write("**Areas for Improvement:**")
                    if overall_themes:
                        negative_themes = ['expensive_pricing', 'slow_performance', 'search_issues', 
                                         'interface_problems', 'missing_features']
                        for theme in negative_themes:
                            if theme in overall_themes:
                                count = overall_themes[theme]
                                clean_theme = theme.replace('_', ' ').title()
                                st.write(f"⚠️ {clean_theme}: {count} mentions")
                
                # Download results
                st.subheader("💾 Download Analysis Results")
                
                results_df = df_with_nps.copy()
                results_df['NPS_Score'] = nps_score
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Complete Analysis as CSV",
                    data=csv,
                    file_name="lexis_plus_nps_analysis_results.csv",
                    mime="text/csv"
                )

# Sidebar
with st.sidebar:
    st.header("⚖️ About Lexis+ Analysis")
    st.write("""
    This tool analyzes Lexis+ Hong Kong customer feedback using:
    
    **🤖 AI-Powered Analysis:**
    - Sentiment-based NPS categorization
    - Legal software specific theme extraction
    - Pattern recognition for legal terminology
    
    **📊 Key Metrics:**
    - NPS Score calculation
    - Theme frequency analysis
    - Category-specific insights
    
    **🎯 Focus Areas:**
    - User experience
    - Database quality
    - Search functionality
    - Pricing concerns
    - Technical performance
    """)
    
    st.header("📋 Expected Data Format")
    st.write("CSV with 'NPS Verbatim Comments' column containing customer feedback text.")
    
    st.header("🔍 Sample Themes Detected")
    st.write("""
    - User Friendly Interface
    - Comprehensive Database
    - Search Engine Quality
    - Pricing Concerns
    - AI Features
    - Halsbury Content
    - Customer Service
    - Performance Issues
    """)
