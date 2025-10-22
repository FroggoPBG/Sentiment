import streamlit as st
import pandas as pd
import re
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go

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

def extract_themes(comments, nps_category=None):
    """Extract common themes from comments, optionally filtered by NPS category"""
    
    # Common positive themes
    positive_themes = {
        'excellent service': r'\b(excellent|outstanding|exceptional|amazing|fantastic|great|wonderful)\s+(service|support|experience|product|quality)\b',
        'fast delivery': r'\b(fast|quick|speedy|rapid|prompt)\s+(delivery|shipping|dispatch)\b',
        'helpful staff': r'\b(helpful|friendly|kind|courteous|professional)\s+(staff|team|support|service)\b',
        'high quality': r'\b(high|great|excellent|superior|premium)\s+(quality|standard)\b',
        'easy to use': r'\b(easy|simple|user.friendly|intuitive|straightforward)\s+(to\s+use|interface|platform|system)\b',
        'value for money': r'\b(value|worth|good\s+deal|affordable|reasonable)\s+(for\s+money|price|pricing)\b'
    }
    
    # Common negative themes
    negative_themes = {
        'poor service': r'\b(poor|bad|terrible|awful|horrible|disappointing)\s+(service|support|experience|quality)\b',
        'slow delivery': r'\b(slow|delayed|late|overdue)\s+(delivery|shipping|dispatch)\b',
        'unhelpful staff': r'\b(unhelpful|rude|unprofessional|dismissive)\s+(staff|team|support|service)\b',
        'technical issues': r'\b(technical|system|website|app|platform)\s+(issues|problems|errors|bugs|glitches)\b',
        'poor quality': r'\b(poor|low|bad|terrible)\s+(quality|standard)\b',
        'expensive': r'\b(expensive|overpriced|costly|too\s+much|high\s+price)\b'
    }
    
    # Neutral themes
    neutral_themes = {
        'average experience': r'\b(average|okay|acceptable|decent|standard)\s+(experience|service|product|quality)\b',
        'mixed feelings': r'\b(mixed|unsure|uncertain|neutral|average)\s+(feelings|opinion|experience)\b'
    }
    
    # Combine all themes based on NPS category
    if nps_category == 'Promoter':
        all_themes = positive_themes
    elif nps_category == 'Detractor':
        all_themes = negative_themes
    elif nps_category == 'Passive':
        all_themes = neutral_themes
    else:
        all_themes = {**positive_themes, **negative_themes, **neutral_themes}
    
    theme_counts = {}
    
    for comment in comments:
        if isinstance(comment, str):
            comment_lower = comment.lower()
            for theme, pattern in all_themes.items():
                if re.search(pattern, comment_lower, re.IGNORECASE):
                    theme_counts[theme] = theme_counts.get(theme, 0) + 1
    
    return theme_counts

# Streamlit App
st.set_page_config(page_title="NPS Category Analysis Tool", page_icon="📊", layout="wide")

st.title("🎯 NPS Category Analysis Tool")
st.markdown("Upload your data with NPS categories (Promoter, Passive, Detractor) and customer comments for analysis.")

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
            themes = extract_themes([comment], nps_category)
            
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
                        st.write(f"• {theme.title()}")
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
            
            if st.button("Analyze Batch Data", type="primary"):
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
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [-100, 0], 'color': "lightgray"},
                                {'range': [0, 50], 'color': "yellow"},
                                {'range': [50, 100], 'color': "green"}
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
                st.subheader("🎯 Theme Analysis")
                
                # Overall themes
                comments = df[comment_column].dropna().tolist()
                overall_themes = extract_themes(comments)
                
                if overall_themes:
                    themes_df = pd.DataFrame(
                        list(overall_themes.items()), 
                        columns=['Theme', 'Count']
                    ).sort_values('Count', ascending=False)
                    
                    fig_themes = px.bar(
                        themes_df, 
                        x='Count', 
                        y='Theme', 
                        orientation='h',
                        title="Most Common Themes"
                    )
                    fig_themes.update_layout(height=400)
                    st.plotly_chart(fig_themes, use_container_width=True)
                
                # Category-specific themes
                st.subheader("📝 Themes by NPS Category")
                
                tab1, tab2, tab3 = st.tabs(["Promoters", "Passives", "Detractors"])
                
                # Add NPS categories to dataframe for filtering
                df_with_nps = df.copy()
                df_with_nps['NPS_Category'] = nps_categories
                
                with tab1:
                    promoter_comments = df_with_nps[df_with_nps['NPS_Category'] == 'Promoter'][comment_column].dropna().tolist()
                    promoter_themes = extract_themes(promoter_comments, 'Promoter')
                    
                    if promoter_themes:
                        st.write(f"**{len(promoter_comments)} Promoter comments analyzed**")
                        for theme, count in sorted(promoter_themes.items(), key=lambda x: x[1], reverse=True):
                            st.write(f"• {theme.title()}: {count} mentions")
                    else:
                        st.write("No specific themes identified for Promoters.")
                
                with tab2:
                    passive_comments = df_with_nps[df_with_nps['NPS_Category'] == 'Passive'][comment_column].dropna().tolist()
                    passive_themes = extract_themes(passive_comments, 'Passive')
                    
                    if passive_themes:
                        st.write(f"**{len(passive_comments)} Passive comments analyzed**")
                        for theme, count in sorted(passive_themes.items(), key=lambda x: x[1], reverse=True):
                            st.write(f"• {theme.title()}: {count} mentions")
                    else:
                        st.write("No specific themes identified for Passives.")
                
                with tab3:
                    detractor_comments = df_with_nps[df_with_nps['NPS_Category'] == 'Detractor'][comment_column].dropna().tolist()
                    detractor_themes = extract_themes(detractor_comments, 'Detractor')
                    
                    if detractor_themes:
                        st.write(f"**{len(detractor_comments)} Detractor comments analyzed**")
                        for theme, count in sorted(detractor_themes.items(), key=lambda x: x[1], reverse=True):
                            st.write(f"• {theme.title()}: {count} mentions")
                    else:
                        st.write("No specific themes identified for Detractors.")
                
                # Download results
                st.subheader("💾 Download Results")
                
                # Prepare download data
                results_df = df_with_nps[[nps_column, comment_column, 'NPS_Category']].copy()
                results_df['NPS_Score'] = nps_score
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Analysis Results as CSV",
                    data=csv,
                    file_name="nps_analysis_results.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please ensure your file contains the correct NPS categories (Promoter, Passive, Detractor) and comment columns.")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This tool analyzes NPS (Net Promoter Score) data based on customer categories and feedback.
    
    **Expected Data Format:**
    - NPS Category Column: Should contain 'Promoter', 'Passive', or 'Detractor'
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
