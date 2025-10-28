import streamlit as st
import pandas as pd
import numpy as np

# Test if basic Streamlit works
st.title("🚀 NPS Analysis Platform - Test Version")

st.write("If you can see this, Streamlit is working!")

# Simple file uploader test
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("✅ File loaded successfully!")
        st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        st.write("Column names:", list(df.columns))
        st.dataframe(df.head())
        
        if 'score' in df.columns and 'feedback' in df.columns:
            st.write("✅ Required columns found!")
            
            # Simple NPS calculation
            score_col = df['score'].str.lower().str.strip()
            promoters = score_col.str.contains('promoter', na=False).sum()
            passives = score_col.str.contains('passive', na=False).sum()
            detractors = score_col.str.contains('detractor', na=False).sum()
            
            total = len(df)
            if total > 0:
                nps = ((promoters - detractors) / total) * 100
                st.metric("NPS Score", f"{nps:.1f}")
                st.write(f"Promoters: {promoters}")
                st.write(f"Passives: {passives}")
                st.write(f"Detractors: {detractors}")
        else:
            st.error("Missing 'score' or 'feedback' columns")
            
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload a CSV file")

st.write("🎯 Basic test complete!")
