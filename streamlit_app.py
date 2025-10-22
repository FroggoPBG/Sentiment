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
import string

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

def extract_nps_category(text):
    """
    Extract NPS category from text using keyword matching
    Returns: 'Promoter', 'Passive', 'Detractor', or 'Unknown'
    """
    if not isinstance(text, str):
        return 'Unknown'
    
    text_lower = text.lower()
    
    # Define keywords for each category
    promoter_keywords = ['promoter', 'promote', 'promoting']
    passive_keywords = ['passive', 'neutral']
    detractor_keywords = ['detractor', 'detract', 'detracting']
    
    # Check for exact matches first
    if any(keyword in text_lower for keyword in promoter_keywords):
        return 'Promoter'
    elif any(keyword in text_lower for keyword in passive_keywords):
        return 'Passive'
    elif any(keyword in text_lower for keyword in detractor_keywords):
        return 'Detractor'
    else:
        return 'Unknown'

def calculate_nps_from_categories(categories):
    """
    Calculate NPS from category counts
    NPS = (% Promoters - % Detractors)
    """
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

def preprocess_text(text):
    """Clean and preprocess text for analysis"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def extract_themes_advanced(comments, nps_category=None, min_frequency=2):
    """
    Advanced theme extraction using multiple approaches
    """
    if not comments:
        return {}
    
    # Filter out empty comments
    valid_comments = [str(comment) for comment in comments if isinstance(comment, str) and len(str(comment).strip()) > 5]
    
    if not valid_comments:
        return {}
    
    st.write(f"Analyzing {len(valid_comments)} valid comments for {nps_category or 'all categories'}...")
    
    # Approach 1: Advanced keyword patterns based on NPS category
    theme_patterns = {}
    
    if nps_category == 'Promoter' or nps_category is None:
        theme_patterns.update({
            'excellent_service': [
                r'\b(excellent|outstanding|exceptional|amazing|fantastic|great|wonderful|superb|brilliant|perfect|awesome)\b.*\b(service|support|experience|team|staff)\b',
                r'\b(service|support|experience|team|staff)\b.*\b(excellent|outstanding|exceptional|amazing|fantastic|great|wonderful|superb|brilliant|perfect|awesome)\b'
            ],
            'fast_delivery': [
                r'\b(fast|quick|speedy|rapid|prompt|immediate|instant)\b.*\b(delivery|shipping|dispatch|arrival)\b',
                r'\b(delivery|shipping|dispatch|arrival)\b.*\b(fast|quick|speedy|rapid|prompt|immediate|instant)\b'
            ],
            'high_quality': [
                r'\b(high|excellent|great|superior|premium|top|best)\b.*\b(quality|standard|grade)\b',
                r'\b(quality|standard|grade)\b.*\b(high|excellent|great|superior|premium|top|best)\b'
            ],
            'easy_use': [
                r'\b(easy|simple|user.friendly|intuitive|straightforward|effortless)\b.*\b(use|navigate|understand|setup)\b',
                r'\b(use|navigate|understand|setup)\b.*\b(easy|simple|user.friendly|intuitive|straightforward|effortless)\b'
            ],
            'helpful_staff': [
                r'\b(helpful|friendly|kind|courteous|professional|knowledgeable|supportive)\b.*\b(staff|team|support|employee|representative)\b',
                r'\b(staff|team|support|employee|representative)\b.*\b(helpful|friendly|kind|courteous|professional|knowledgeable|supportive)\b'
            ],
            'value_money': [
                r'\b(value|worth|good.*deal|affordable|reasonable|fair.*price)\b',
                r'\b(money|price|cost|pricing)\b.*\b(good|great|excellent|fair|reasonable)\b'
            ]
        })
    
    if nps_category == 'Detractor' or nps_category is None:
        theme_patterns.update({
            'poor_service': [
                r'\b(poor|bad|terrible|awful|horrible|disappointing|unprofessional|rude)\b.*\b(service|support|experience|team|staff)\b',
                r'\b(service|support|experience|team|staff)\b.*\b(poor|bad|terrible|awful|horrible|disappointing|unprofessional|rude)\b'
            ],
            'slow_delivery': [
                r'\b(slow|delayed|late|overdue|never.*arrived|took.*long)\b.*\b(delivery|shipping|dispatch|arrival)\b',
                r'\b(delivery|shipping|dispatch|arrival)\b.*\b(slow|delayed|late|overdue|never.*arrived|took.*long)\b'
            ],
            'poor_quality': [
                r'\b(poor|low|bad|terrible|cheap|inferior|defective|faulty)\b.*\b(quality|standard|grade|product)\b',
                r'\b(quality|standard|grade|product)\b.*\b(poor|low|bad|terrible|cheap|inferior|defective|faulty)\b'
            ],
            'technical_issues': [
                r'\b(technical|system|website|app|platform|software)\b.*\b(issue|problem|error|bug|glitch|crash|down|broken)\b',
                r'\b(issue|problem|error|bug|glitch|crash|down|broken)\b.*\b(technical|system|website|app|platform|software)\b'
            ],
            'expensive': [
                r'\b(expensive|overpriced|costly|too.*much|high.*price|rip.*off)\b',
                r'\b(price|cost|pricing|money)\b.*\b(high|expensive|too.*much|unreasonable)\b'
            ],
            'difficult_use': [
                r'\b(difficult|hard|complicated|confusing|frustrating)\b.*\b(use|navigate|understand|setup|figure.*out)\b',
                r'\b(use|navigate|understand|setup|figure.*out)\b.*\b(difficult|hard|complicated|confusing|frustrating)\b'
            ]
        })
    
    if nps_category == 'Passive' or nps_category is None:
        theme_patterns.update({
            'average_experience': [
                r'\b(average|okay|acceptable|decent|standard|normal|fine|alright)\b.*\b(experience|service|product|quality)\b',
                r'\b(experience|service|product|quality)\b.*\b(average|okay|acceptable|decent|standard|normal|fine|alright)\b'
            ],
            'mixed_feelings': [
                r'\b(mixed|unsure|uncertain|neutral|so.*so|could.*better)\b',
                r'\b(good.*bad|positive.*negative|like.*dislike)\b'
            ],
            'room_improvement': [
                r'\b(could.*improve|room.*improvement|needs.*work|better|enhance)\b',
                r'\b(improvement|improve|better|enhance)\b'
            ]
        })
    
    # Count theme occurrences
    theme_counts = {}
    
    for comment in valid_comments:
        comment_lower = comment.lower()
        for theme, patterns in theme_patterns.items():
            for pattern in patterns:
                if re.search(pattern, comment_lower, re.IGNORECASE):
                    theme_counts[theme] = theme_counts.get(theme, 0) + 1
                    break  # Count each theme only once per comment
    
    # Approach 2: Sentiment-based word frequency analysis
    preprocessed_comments = [preprocess_text(comment) for comment in valid_comments]
    all_words = ' '.join(preprocessed_comments).split()
    
    # Get most common meaningful words
    word_freq = Counter(all_words)
    common_words = dict(word_freq.most_common(20))
    
    # Filter out very common words that don't add meaning
    meaningless_words = {'would', 'could', 'should', 'really', 'much', 'well', 'also', 'get', 'got', 'one', 'two', 'first', 'last', 'time', 'way', 'make', 'made', 'take', 'come', 'go', 'see', 'know', 'think', 'want', 'need', 'use', 'said', 'say', 'tell', 'give', 'find', 'work', 'call', 'try', 'ask', 'look', 'feel', 'seem', 'become', 'leave', 'put', 'mean', 'keep', 'let', 'begin', 'help', 'talk', 'turn', 'start', 'show', 'hear', 'play', 'run', 'move', 'live', 'believe', 'bring', 'happen', 'write', 'provide', 'sit', 'stand', 'lose', 'pay', 'meet', 'include', 'continue', 'set', 'learn', 'change', 'lead', 'understand', 'watch', 'follow', 'stop', 'create', 'speak', 'read', 'allow', 'add', 'spend', 'grow', 'open', 'walk', 'win', 'offer', 'remember', 'love', 'consider', 'appear', 'buy', 'wait', 'serve', 'die', 'send', 'expect', 'build', 'stay', 'fall', 'cut', 'reach', 'kill', 'remain'}
    
    filtered_words = {word: count for word, count in common_words.items() 
                     if word not in meaningless_words and count >= min_frequency}
    
    # Add word frequency themes
    for word, count in filtered_words.items():
        theme_name = f"mentions_{word}"
        theme_counts[theme_name] = count
    
    # Approach 3: Sentiment analysis
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    for comment in valid_comments:
        blob = TextBlob(comment)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            positive_count += 1
        elif polarity < -0.1:
            negative_count += 1
        else:
            neutral_count += 1
    
    # Add sentiment themes
    if positive_count >= min_frequency:
        theme_counts['positive_sentiment'] = positive_count
    if negative_count >= min_frequency:
        theme_counts['negative_sentiment'] = negative_count
    if neutral_count >= min_frequency:
        theme_counts['neutral_sentiment'] = neutral_count
    
    # Filter themes by minimum frequency
    filtered_themes = {theme: count for theme, count in theme_counts.items() 
                      if count >= min_frequency}
    
    return filtered_themes

# Streamlit App
st.set_page_config(page_title="NPS Category Analysis Tool", page_icon="📊", layout="wide")

st.title("🎯 NPS Category Analysis Tool")
st.markdown("Upload your data with NPS categories (Promoter, Passive, Detractor) and customer comments for comprehensive analysis.")

# Analysis Type Selection
analysis_type = st.radio(
    "Analysis Type",
    ("Individual Feedback", "Batch Analysis"),
    horizontal=True
)

if analysis_type == "Individual Feedback":
    st.subheader("📝 Individual Feedback Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        nps_category = st.selectbox(
            "Select NPS Category:",
            ["Promoter", "Passive", "Detractor"]
        )
    
    with col2:
        st.write("")  # Spacing
    
    comment = st.text_area(
        "Customer Comment:",
        placeholder="Enter the customer's feedback here...",
        height=100
    )
    
    if st.button("Analyze Individual Feedback", type="primary"):
        if comment:
            # Theme extraction based on the selected category
            themes = extract_themes_advanced([comment], nps_category)
            
            st.subheader("📊 Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("NPS Category", nps_category)
                
                # Color coding for categories
                if nps_category == "Promoter":
                    st.success("✅ Positive feedback")
                elif nps_category == "Passive":
                    st.warning("⚠️ Neutral feedback")
                else:
                    st.error("❌ Negative feedback")
            
            with col2:
                if themes:
                    st.write("**Identified Themes:**")
                    for theme, count in sorted(themes.items(), key=lambda x: x[1], reverse=True):
                        clean_theme = theme.replace('_', ' ').title()
                        st.write(f"• {clean_theme}")
                else:
                    st.write("No specific themes identified in this comment.")
        else:
            st.warning("Please enter a customer comment to analyze.")

else:  # Batch Analysis
    st.subheader("📊 Batch Processing with Aggregation")
    
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
            
            # Column selection
            st.subheader("🔧 Column Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                nps_column = st.selectbox(
                    "Select NPS Category Column:",
                    options=df.columns.tolist(),
                    help="Column containing Promoter, Passive, or Detractor categories"
                )
            
            with col2:
                comment_column = st.selectbox(
                    "Select Comment Column:",
                    options=df.columns.tolist(),
                    help="Column containing customer feedback text"
                )
            
            # Add minimum frequency setting
            min_frequency = st.slider(
                "Minimum frequency for themes (how many times a theme should appear to be included):",
                min_value=1,
                max_value=10,
                value=2,
                help="Themes appearing less than this number will be filtered out"
            )
            
            if st.button("Analyze Batch Data", type="primary"):
                with st.spinner("Processing your data... This may take a moment for large datasets."):
                    # Extract NPS categories
                    nps_categories = []
                    for value in df[nps_column]:
                        if pd.isna(value):
                            nps_categories.append('Unknown')
                        else:
                            category = extract_nps_category(str(value))
                            nps_categories.append(category)
                    
                    # Calculate NPS
                    nps_score, category_counts = calculate_nps_from_categories(nps_categories)
                    
                    # Display results
                    st.subheader("📈 NPS Analysis Results")
                    
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
                    
                    # Theme analysis
                    st.subheader("🎯 Comprehensive Theme Analysis")
                    
                    # Add NPS categories to dataframe for filtering
                    df_with_nps = df.copy()
                    df_with_nps['NPS_Category'] = nps_categories
                    
                    # Overall themes
                    st.write("### Overall Themes Across All Categories")
                    comments = df[comment_column].dropna().tolist()
                    overall_themes = extract_themes_advanced(comments, None, min_frequency)
                    
                    if overall_themes:
                        # Create a more readable theme dataframe
                        themes_data = []
                        for theme, count in overall_themes.items():
                            clean_theme = theme.replace('_', ' ').replace('mentions ', '').title()
                            themes_data.append({'Theme': clean_theme, 'Count': count, 'Percentage': f"{(count/len(comments)*100):.1f}%"})
                        
                        themes_df = pd.DataFrame(themes_data).sort_values('Count', ascending=False)
                        
                        # Display top themes
                        st.dataframe(themes_df.head(15))
                        
                        # Visualization
                        top_themes = themes_df.head(10)
                        fig_themes = px.bar(
                            top_themes, 
                            x='Count', 
                            y='Theme', 
                            orientation='h',
                            title="Top 10 Most Common Themes"
                        )
                        fig_themes.update_layout(height=400)
                        st.plotly_chart(fig_themes, use_container_width=True)
                    
                    # Category-specific themes
                    st.subheader("📝 Detailed Themes by NPS Category")
                    
                    tab1, tab2, tab3 = st.tabs(["🟢 Promoters", "🟡 Passives", "🔴 Detractors"])
                    
                    with tab1:
                        promoter_comments = df_with_nps[df_with_nps['NPS_Category'] == 'Promoter'][comment_column].dropna().tolist()
                        if promoter_comments:
                            promoter_themes = extract_themes_advanced(promoter_comments, 'Promoter', min_frequency)
                            
                            if promoter_themes:
                                st.write(f"**Analysis of {len(promoter_comments)} Promoter comments:**")
                                
                                # Create dataframe for better display
                                promoter_data = []
                                for theme, count in promoter_themes.items():
                                    clean_theme = theme.replace('_', ' ').replace('mentions ', '').title()
                                    promoter_data.append({
                                        'Theme': clean_theme, 
                                        'Count': count, 
                                        'Percentage': f"{(count/len(promoter_comments)*100):.1f}%"
                                    })
                                
                                promoter_df = pd.DataFrame(promoter_data).sort_values('Count', ascending=False)
                                st.dataframe(promoter_df)
                                
                                # Visualization for promoters
                                if len(promoter_df) > 0:
                                    fig_promoter = px.bar(
                                        promoter_df.head(8), 
                                        x='Count', 
                                        y='Theme', 
                                        orientation='h',
                                        title="Promoter Themes",
                                        color_discrete_sequence=['#2E8B57']
                                    )
                                    st.plotly_chart(fig_promoter, use_container_width=True)
                            else:
                                st.write("No themes found above the minimum frequency threshold.")
                        else:
                            st.write("No Promoter comments found in the data.")
                    
                    with tab2:
                        passive_comments = df_with_nps[df_with_nps['NPS_Category'] == 'Passive'][comment_column].dropna().tolist()
                        if passive_comments:
                            passive_themes = extract_themes_advanced(passive_comments, 'Passive', min_frequency)
                            
                            if passive_themes:
                                st.write(f"**Analysis of {len(passive_comments)} Passive comments:**")
                                
                                # Create dataframe for better display
                                passive_data = []
                                for theme, count in passive_themes.items():
                                    clean_theme = theme.replace('_', ' ').replace('mentions ', '').title()
                                    passive_data.append({
                                        'Theme': clean_theme, 
                                        'Count': count, 
                                        'Percentage': f"{(count/len(passive_comments)*100):.1f}%"
                                    })
                                
                                passive_df = pd.DataFrame(passive_data).sort_values('Count', ascending=False)
                                st.dataframe(passive_df)
                                
                                # Visualization for passives
                                if len(passive_df) > 0:
                                    fig_passive = px.bar(
                                        passive_df.head(8), 
                                        x='Count', 
                                        y='Theme', 
                                        orientation='h',
                                        title="Passive Themes",
                                        color_discrete_sequence=['#FFD700']
                                    )
                                    st.plotly_chart(fig_passive, use_container_width=True)
                            else:
                                st.write("No themes found above the minimum frequency threshold.")
                        else:
                            st.write("No Passive comments found in the data.")
                    
                    with tab3:
                        detractor_comments = df_with_nps[df_with_nps['NPS_Category'] == 'Detractor'][comment_column].dropna().tolist()
                        if detractor_comments:
                            detractor_themes = extract_themes_advanced(detractor_comments, 'Detractor', min_frequency)
                            
                            if detractor_themes:
                                st.write(f"**Analysis of {len(detractor_comments)} Detractor comments:**")
                                
                                # Create dataframe for better display
                                detractor_data = []
                                for theme, count in detractor_themes.items():
                                    clean_theme = theme.replace('_', ' ').replace('mentions ', '').title()
                                    detractor_data.append({
                                        'Theme': clean_theme, 
                                        'Count': count, 
                                        'Percentage': f"{(count/len(detractor_comments)*100):.1f}%"
                                    })
                                
                                detractor_df = pd.DataFrame(detractor_data).sort_values('Count', ascending=False)
                                st.dataframe(detractor_df)
                                
                                # Visualization for detractors
                                if len(detractor_df) > 0:
                                    fig_detractor = px.bar(
                                        detractor_df.head(8), 
                                        x='Count', 
                                        y='Theme', 
                                        orientation='h',
                                        title="Detractor Themes",
                                        color_discrete_sequence=['#DC143C']
                                    )
                                    st.plotly_chart(fig_detractor, use_container_width=True)
                            else:
                                st.write("No themes found above the minimum frequency threshold.")
                        else:
                            st.write("No Detractor comments found in the data.")
                    
                    # Download results
                    st.subheader("💾 Download Results")
                    
                    # Prepare comprehensive download data
                    results_df = df_with_nps.copy()
                    results_df['NPS_Score'] = nps_score
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Analysis Results as CSV",
                        data=csv,
                        file_name="nps_comprehensive_analysis_results.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please ensure your file contains the correct NPS categories (Promoter, Passive, Detractor) and comment columns.")
            st.write("Also make sure you have internet connection for the first run to download required language models.")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This tool provides comprehensive NPS analysis using advanced text processing.
    
    **Features:**
    - Pattern-based theme detection
    - Word frequency analysis
    - Sentiment analysis
    - Category-specific insights
    
    **Expected Data Format:**
    - NPS Category Column: 'Promoter', 'Passive', 'Detractor'
    - Comment Column: Customer feedback text
    
    **NPS Calculation:**
    NPS = % Promoters - % Detractors
    
    **Score Interpretation:**
    - Above 50: Excellent
    - 0 to 50: Good
    - Below 0: Needs Improvement
    """)
    
    st.header("📊 Sample Data Format")
    sample_data = pd.DataFrame({
        'NPS_Category': ['Promoter', 'Detractor', 'Passive', 'Promoter'],
        'Customer_Comment': [
            'Excellent service and fast delivery!',
            'Poor quality and slow response.',
            'Average experience, nothing special.',
            'Amazing product, highly recommend!'
        ]
    })
    st.dataframe(sample_data)
