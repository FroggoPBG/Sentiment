import streamlit as st
import anthropic

# Page config
st.set_page_config(page_title="Client Email Assistant", page_icon="✉️", layout="wide")

# Title
st.title("✉️ Client Communication Assistant")
st.markdown("Generate professional emails with the right tone, style, and language")

# Sidebar for API key
with st.sidebar:
    st.header("🔑 Setup")
    api_key = st.text_input("Enter your Anthropic API Key", type="password", help="Get your key from console.anthropic.com")
    
    if api_key:
        st.success("API Key set! ✓")
    else:
        st.warning("Please enter your API key to continue")
    
    st.divider()
    
    st.header("⚙️ Email Settings")
    
    # Communication style
    communication_style = st.selectbox(
        "Communication Style",
        [
            "Professional & Formal",
            "Friendly & Casual", 
            "Empathetic & Supportive",
            "Direct & Concise",
            "Apologetic & Reassuring",
            "Enthusiastic & Positive"
        ]
    )
    
    # Situation type
    situation_type = st.selectbox(
        "Situation Type",
        [
            "General Inquiry Response",
            "Issue/Complaint Resolution",
            "Follow-up Email",
            "Apology",
            "Good News/Update",
            "Request for Information",
            "Thank You Note",
            "Introduction/Cold Outreach",
            "Meeting Request",
            "Escalation"
        ]
    )
    
    # Language
    target_language = st.selectbox(
        "Language",
        [
            "English",
            "Spanish",
            "French",
            "German",
            "Italian",
            "Portuguese",
            "Chinese (Simplified)",
            "Japanese",
            "Korean",
            "Arabic",
            "Hindi",
            "Russian"
        ]
    )
    
    st.divider()
    
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Be specific about the situation
    - Include relevant context
    - Mention key points to address
    - Review before sending!
    """)

# Main content
if not api_key:
    st.info("👈 Please enter your Anthropic API key in the sidebar to get started")
    st.markdown("""
    ### How to get your API key:
    1. Go to [console.anthropic.com](https://console.anthropic.com/)
    2. Sign up or log in
    3. Go to API Keys
    4. Create a new key
    5. Copy and paste it in the sidebar
    """)
else:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input Details")
        
        # Client information
        client_name = st.text_input(
            "Client Name (optional)",
            placeholder="e.g., Sarah Johnson"
        )
        
        company_name = st.text_input(
            "Company Name (optional)",
            placeholder="e.g., Acme Corp"
        )
        
        # Situation description
        situation = st.text_area(
            "Describe the Situation *",
            placeholder="e.g., Client's shipment was delayed by 5 days due to weather. They need it urgently for an event this weekend.",
            height=150,
            help="Be as specific as possible about what happened"
        )
        
        # Key points
        key_points = st.text_area(
            "Key Points to Address",
            placeholder="e.g., Apologize for delay, explain weather issue, offer express shipping refund, provide new delivery date",
            height=100
        )
        
        # Additional context
        additional_context = st.text_input(
            "Additional Context",
            placeholder="e.g., VIP customer, second incident this month, contract renewal coming up"
        )
        
        # Tone adjustments
        with st.expander("🎚️ Advanced Options"):
            urgency = st.select_slider(
                "Urgency Level",
                options=["Low", "Medium", "High", "Critical"],
                value="Medium"
            )
            
            formality = st.select_slider(
                "Formality Level",
                options=["Very Casual", "Casual", "Neutral", "Formal", "Very Formal"],
                value="Neutral"
            )
            
            include_cta = st.checkbox("Include Call-to-Action", value=True)
        
        # Generate button
        generate_button = st.button(
            "✨ Generate Email",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        st.subheader("📧 Generated Email")
        
        if generate_button:
            if not situation.strip():
                st.error("⚠️ Please describe the situation first!")
            else:
                with st.spinner("✍️ Crafting your email..."):
                    # Build prompt
                    prompt = f"""You are a professional email writing assistant. Generate a client-facing email based on the following details:

**Situation Type:** {situation_type}
**Communication Style:** {communication_style}
**Situation:** {situation}
"""
                    
                    if client_name:
                        prompt += f"\n**Client Name:** {client_name}"
                    
                    if company_name:
                        prompt += f"\n**Company:** {company_name}"
                    
                    if key_points:
                        prompt += f"\n**Key Points to Address:** {key_points}"
                    
                    if additional_context:
                        prompt += f"\n**Additional Context:** {additional_context}"
                    
                    prompt += f"\n**Urgency Level:** {urgency}"
                    prompt += f"\n**Formality Level:** {formality}"
                    
                    prompt += f"""

Please generate an email that:
1. Matches the {communication_style} style
2. Is appropriate for a {situation_type} scenario
3. Addresses all key points mentioned
4. Is client-focused and professional
5. Has a clear, compelling subject line
6. Uses the appropriate level of urgency and formality
"""
                    
                    if include_cta:
                        prompt += "\n7. Includes a clear call-to-action"
                    
                    prompt += """

Format your response EXACTLY like this:

**SUBJECT:** [Write a clear subject line]

**EMAIL:**

[Write the complete email body here]

**TONE CHECK:** [Brief note on the tone used]
"""
                    
                    if target_language != "English":
                        prompt += f"\n\nAfter generating the email in English, provide a translation in {target_language}. Show both versions clearly labeled."
                    
                    try:
                        # Create Anthropic client
                        client = anthropic.Anthropic(api_key=api_key)
                        
                        # Call API
                        message = client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=2500,
                            messages=[
                                {"role": "user", "content": prompt}
                            ]
                        )
                        
                        # Get response
                        email_content = message.content[0].text
                        
                        # Display result
                        st.markdown(email_content)
                        
                        # Action buttons
                        st.divider()
                        
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.download_button(
                                label="📥 Download",
                                data=email_content,
                                file_name="client_email.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        with col_b:
                            if st.button("🔄 Regenerate", use_container_width=True):
                                st.rerun()
                        
                        with col_c:
                            st.button("📋 Copy", use_container_width=True, help="Click to copy to clipboard")
                        
                        # Feedback
                        st.divider()
                        st.markdown("**Was this email helpful?**")
                        col_f1, col_f2 = st.columns(2)
                        with col_f1:
                            st.button("👍 Yes", use_container_width=True)
                        with col_f2:
                            st.button("👎 Needs work", use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.info("Please check your API key and try again.")
        
        else:
            st.info("👈 Fill in the details on the left and click 'Generate Email'")
            
            # Show example
            with st.expander("📖 See Example"):
                st.markdown("""
                **Example Input:**
                - **Situation:** Customer's order arrived damaged
                - **Style:** Empathetic & Supportive
                - **Key Points:** Apologize, offer replacement, provide discount
                
                **Example Output:**
                
                **SUBJECT:** We're Sorry - Immediate Replacement for Your Order #12345
                
                **EMAIL:**
                
                Dear Sarah,
                
                I'm truly sorry to hear that your recent order arrived damaged. I understand how disappointing this must be, especially when you were looking forward to receiving your items in perfect condition...
                """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Powered by Claude AI • Always review emails before sending • Keep your API key secure</p>
</div>
""", unsafe_allow_html=True)
