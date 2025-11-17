import streamlit as st
import anthropic
import os

# Initialize the Anthropic client
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

st.set_page_config(page_title="Client Communication Assistant", page_icon="✉️", layout="wide")

st.title("✉️ Client Communication Assistant")
st.markdown("Generate professional emails with the right tone and style for any client situation")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    # Communication style selection
    communication_style = st.selectbox(
        "Communication Style",
        ["Professional & Formal", "Friendly & Casual", "Empathetic & Supportive", 
         "Direct & Concise", "Apologetic & Reassuring"]
    )
    
    # Sentiment/Situation type
    situation_type = st.selectbox(
        "Situation Type",
        ["General Inquiry Response", "Issue/Complaint Resolution", "Follow-up", 
         "Apology", "Good News/Update", "Request for Information", 
         "Thank You", "Introduction/Cold Outreach"]
    )
    
    # Translation
    target_language = st.selectbox(
        "Target Language",
        ["English (Original)", "Spanish", "French", "German", "Italian", 
         "Portuguese", "Chinese (Simplified)", "Japanese", "Korean", "Arabic"]
    )
    
    st.divider()
    st.markdown("**Tips:**")
    st.markdown("- Be specific about the client situation")
    st.markdown("- Include relevant details or context")
    st.markdown("- Specify any key points to address")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Details")
    
    # Client context
    client_name = st.text_input("Client Name (optional)", placeholder="e.g., John Smith")
    
    # Situation description
    situation = st.text_area(
        "Describe the Situation",
        placeholder="e.g., Client reported a bug in the software that caused data loss. They're frustrated and need immediate resolution.",
        height=150
    )
    
    # Key points to address
    key_points = st.text_area(
        "Key Points to Address (optional)",
        placeholder="e.g., Acknowledge the issue, explain what happened, outline resolution steps, offer compensation",
        height=100
    )
    
    # Additional context
    additional_context = st.text_input(
        "Additional Context (optional)",
        placeholder="e.g., Premium customer, previous incident last month"
    )
    
    generate_button = st.button("✨ Generate Email", type="primary", use_container_width=True)

with col2:
    st.subheader("Generated Email")
    
    if generate_button:
        if not situation.strip():
            st.error("Please describe the situation first!")
        else:
            with st.spinner("Crafting your email..."):
                # Build the prompt for Claude
                prompt = f"""Generate a professional email for the following client communication scenario:

**Situation Type:** {situation_type}
**Communication Style:** {communication_style}
**Situation Description:** {situation}
"""
                
                if client_name:
                    prompt += f"\n**Client Name:** {client_name}"
                
                if key_points:
                    prompt += f"\n**Key Points to Address:** {key_points}"
                
                if additional_context:
                    prompt += f"\n**Additional Context:** {additional_context}"
                
                prompt += f"""

Please generate an appropriate email that:
1. Matches the specified communication style
2. Addresses the situation appropriately
3. Includes all key points mentioned
4. Is professional and client-focused
5. Has a clear subject line

Format the response as:
**Subject:** [subject line]

**Email Body:**
[email content]
"""
                
                if target_language != "English (Original)":
                    prompt += f"\n\nAfter generating the email in English, translate it to {target_language}. Provide both versions."
                
                try:
                    # Call Claude API
                    message = client.messages.create(
                        model="claude-4.5-sonnet-20250101",
                        max_tokens=2000,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    
                    # Display the generated email
                    email_content = message.content[0].text
                    st.markdown(email_content)
                    
                    # Copy to clipboard button
                    st.divider()
                    st.download_button(
                        label="📋 Download Email",
                        data=email_content,
                        file_name="generated_email.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating email: {str(e)}")
    else:
        st.info("👈 Fill in the details and click 'Generate Email' to create your client communication")

# Footer with examples
st.divider()
with st.expander("📚 Example Scenarios"):
    st.markdown("""
    **Issue Resolution Example:**
    - Situation: "Customer's order was delayed by 3 days due to shipping error"
    - Style: Apologetic & Reassuring
    - Key Points: Apologize, explain cause, offer discount on next order
    
    **Follow-up Example:**
    - Situation: "Following up on product demo from last week, client seemed interested"
    - Style: Friendly & Casual
    - Key Points: Reference demo, offer trial period, schedule next call
    
    **Good News Example:**
    - Situation: "Client's requested feature has been implemented ahead of schedule"
    - Style: Professional & Formal
    - Key Points: Announce feature, explain benefits, offer training session
    """)

st.markdown("---")
st.markdown("*Powered by Claude 4.5 Sonnet - Always review and customize emails before sending*")
