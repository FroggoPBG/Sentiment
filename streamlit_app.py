import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="🎭",
    layout="wide"
)

# Initialize analyzer
@st.cache_resource
def load_analyzer():
    return SentimentIntensityAnalyzer()

analyzer = load_analyzer()

# Main app
st.title("🎭 Sentiment Analysis Tool")
st.markdown("### Analyze the emotional tone of any text using VADER sentiment analysis")

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.write("This tool uses VADER (Valence Aware Dictionary and sEntiment Reasoner) to analyze sentiment in text.")
    st.write("**Compound Score:**")
    st.write("• Positive: ≥ 0.05")
    st.write("• Neutral: -0.05 to 0.05")
    st.write("• Negative: ≤ -0.05")

# Main input area
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area(
        "Enter your text here:",
        height=150,
        placeholder="Type or paste your text here to analyze its sentiment..."
    )

with col2:
    st.write("**Quick Examples:**")
    if st.button("😊 Positive Example"):
        text_input = "I absolutely love this! It's amazing and wonderful!"
    if st.button("😢 Negative Example"):
        text_input = "This is terrible and I hate it so much."
    if st.button("😐 Neutral Example"):
        text_input = "The weather today is cloudy with a chance of rain."

# Analysis button
if st.button("🔍 Analyze Sentiment", type="primary"):
    if text_input.strip():
        # Get sentiment scores
        scores = analyzer.polarity_scores(text_input)
        
        # Create columns for results
        col1, col2, col3 = st.columns([1, 1, 1])
        
        # Overall sentiment
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = "POSITIVE"
            emoji = "😊"
            color = "green"
        elif compound <= -0.05:
            sentiment = "NEGATIVE"
            emoji = "😢"
            color = "red"
        else:
            sentiment = "NEUTRAL"
            emoji = "😐"
            color = "gray"
        
        # Display overall result
        st.markdown("---")
        st.markdown(f"## {emoji} Overall Sentiment: **{sentiment}**")
        st.markdown(f"**Compound Score: {compound:.3f}**")
        
        # Detailed scores
        st.markdown("### Detailed Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=['Positive', 'Negative', 'Neutral'],
                    y=[scores['pos'], scores['neg'], scores['neu']],
                    marker_color=['#2E8B57', '#DC143C', '#708090'],
                    text=[f"{scores['pos']:.3f}", f"{scores['neg']:.3f}", f"{scores['neu']:.3f}"],
                    textposition='auto'
                )
            ])
            fig_bar.update_layout(
                title="Sentiment Component Scores",
                yaxis_title="Score",
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Positive', 'Negative', 'Neutral'],
                values=[scores['pos'], scores['neg'], scores['neu']],
                marker_colors=['#2E8B57', '#DC143C', '#708090']
            )])
            fig_pie.update_layout(
                title="Sentiment Distribution",
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Metrics
        st.markdown("### Score Details")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Positive", f"{scores['pos']:.3f}")
        with col2:
            st.metric("Negative", f"{scores['neg']:.3f}")
        with col3:
            st.metric("Neutral", f"{scores['neu']:.3f}")
        with col4:
            st.metric("Compound", f"{scores['compound']:.3f}")
        
        # Text analysis
        st.markdown("### Text Analysis")
        word_count = len(text_input.split())
        char_count = len(text_input)
        
        st.write(f"**Word Count:** {word_count}")
        st.write(f"**Character Count:** {char_count}")
        
    else:
        st.warning("⚠️ Please enter some text to analyze!")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit and VADER Sentiment Analysis*")
