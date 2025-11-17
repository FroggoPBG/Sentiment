import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Client Email Builder", page_icon="✉️", layout="wide")

st.title("✉️ Client Email Builder Tool")
st.markdown("Build professional client emails with templates and best practices")

# Sidebar
with st.sidebar:
    st.header("⚙️ Email Configuration")
    
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
    
    situation_type = st.selectbox(
        "Email Type",
        [
            "General Inquiry Response",
            "Issue/Complaint Resolution",
            "Follow-up Email",
            "Apology",
            "Good News/Update",
            "Request for Information",
            "Thank You Note",
            "Meeting Request",
            "Price Quote",
            "Order Confirmation"
        ]
    )
    
    urgency = st.select_slider(
        "Urgency Level",
        options=["Low", "Medium", "High", "Critical"],
        value="Medium"
    )
    
    target_language = st.selectbox(
        "Language",
        ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Chinese"]
    )

# Email templates
TEMPLATES = {
    "Professional & Formal": {
        "greeting": "Dear {name},",
        "opening": "I hope this message finds you well.",
        "closing": "Best regards,\n{sender_name}\n{sender_title}",
        "tone_tips": "Use complete sentences, avoid contractions, maintain distance"
    },
    "Friendly & Casual": {
        "greeting": "Hi {name}!",
        "opening": "Hope you're doing great!",
        "closing": "Cheers,\n{sender_name}",
        "tone_tips": "Use contractions, be warm, show personality"
    },
    "Empathetic & Supportive": {
        "greeting": "Dear {name},",
        "opening": "Thank you for reaching out to us.",
        "closing": "We're here to help,\n{sender_name}",
        "tone_tips": "Acknowledge feelings, show understanding, offer support"
    },
    "Direct & Concise": {
        "greeting": "Hello {name},",
        "opening": "",
        "closing": "Thanks,\n{sender_name}",
        "tone_tips": "Get to the point quickly, use bullet points, be clear"
    },
    "Apologetic & Reassuring": {
        "greeting": "Dear {name},",
        "opening": "I sincerely apologize for the inconvenience.",
        "closing": "With our sincere apologies,\n{sender_name}",
        "tone_tips": "Take responsibility, explain action steps, restore confidence"
    },
    "Enthusiastic & Positive": {
        "greeting": "Hi {name}!",
        "opening": "Great to hear from you!",
        "closing": "Looking forward to connecting,\n{sender_name}",
        "tone_tips": "Use exclamation points (sparingly), be upbeat, show excitement"
    }
}

SITUATION_TEMPLATES = {
    "Issue/Complaint Resolution": {
        "subject": "Re: {issue} - We're Here to Help",
        "structure": [
            "1. Acknowledge the issue",
            "2. Apologize for inconvenience",
            "3. Explain what happened (briefly)",
            "4. Outline solution/next steps",
            "5. Provide timeline",
            "6. Offer direct contact"
        ],
        "key_phrases": [
            "I understand your frustration",
            "We take full responsibility",
            "Here's what we're doing to fix this",
            "You can expect [specific action] by [date]",
            "Please don't hesitate to contact me directly"
        ]
    },
    "Apology": {
        "subject": "Our Sincere Apologies - {issue}",
        "structure": [
            "1. Start with clear apology",
            "2. Acknowledge specific issue",
            "3. Take responsibility (no excuses)",
            "4. Explain resolution",
            "5. Prevent future occurrence",
            "6. Make it right (compensation if applicable)"
        ],
        "key_phrases": [
            "I sincerely apologize",
            "This is not the experience we want you to have",
            "We take full responsibility",
            "Here's what we're doing to make this right",
            "We've implemented [change] to prevent this"
        ]
    },
    "Good News/Update": {
        "subject": "Great News About {topic}!",
        "structure": [
            "1. Lead with the good news",
            "2. Provide details",
            "3. Explain benefits",
            "4. Next steps (if any)",
            "5. Thank them"
        ],
        "key_phrases": [
            "I'm pleased to inform you",
            "Good news!",
            "This means that",
            "We're excited to",
            "Thank you for your patience"
        ]
    },
    "General Inquiry Response": {
        "subject": "Re: Your Inquiry About {topic}",
        "structure": [
            "1. Thank them for inquiry",
            "2. Answer their question(s)",
            "3. Provide additional helpful info",
            "4. Invite follow-up questions",
            "5. Close warmly"
        ],
        "key_phrases": [
            "Thank you for your interest",
            "To answer your question",
            "Additionally, you might find it helpful",
            "Please let me know if you need anything else",
            "Happy to help further"
        ]
    },
    "Follow-up Email": {
        "subject": "Following Up: {topic}",
        "structure": [
            "1. Reference previous conversation",
            "2. Purpose of follow-up",
            "3. New information/question",
            "4. Clear call-to-action",
            "5. Timeline"
        ],
        "key_phrases": [
            "I wanted to follow up on",
            "As discussed",
            "Just checking in",
            "Could you please",
            "By [date] would be ideal"
        ]
    },
    "Thank You Note": {
        "subject": "Thank You!",
        "structure": [
            "1. Express gratitude",
            "2. Be specific about what you're thanking for",
            "3. Mention impact/value",
            "4. Look forward",
            "5. Warm close"
        ],
        "key_phrases": [
            "Thank you so much for",
            "I really appreciate",
            "This means a lot because",
            "Looking forward to",
            "Grateful for your"
        ]
    },
    "Meeting Request": {
        "subject": "Meeting Request: {topic}",
        "structure": [
            "1. State purpose",
            "2. Propose time/duration",
            "3. Suggest agenda",
            "4. Provide options",
            "5. Request confirmation"
        ],
        "key_phrases": [
            "I'd like to schedule a meeting to discuss",
            "Would you be available for",
            "The agenda would include",
            "I'm flexible on timing",
            "Please let me know what works best"
        ]
    },
    "Request for Information": {
        "subject": "Information Request: {topic}",
        "structure": [
            "1. Context/reason for request",
            "2. Specific information needed",
            "3. Why you need it",
            "4. Deadline",
            "5. Thank them in advance"
        ],
        "key_phrases": [
            "I'm reaching out to request",
            "Specifically, I need",
            "This will help us to",
            "If possible, by [date]",
            "Thank you in advance"
        ]
    },
    "Price Quote": {
        "subject": "Quote for {product/service}",
        "structure": [
            "1. Thank for interest",
            "2. Present quote clearly",
            "3. Explain what's included",
            "4. Highlight value",
            "5. Next steps",
            "6. Validity period"
        ],
        "key_phrases": [
            "Thank you for your interest",
            "I'm pleased to provide a quote",
            "This includes",
            "The total investment is",
            "This quote is valid until"
        ]
    },
    "Order Confirmation": {
        "subject": "Order Confirmation #{order_number}",
        "structure": [
            "1. Confirm order received",
            "2. Order details/summary",
            "3. Next steps/timeline",
            "4. Tracking info (if applicable)",
            "5. Support contact",
            "6. Thank them"
        ],
        "key_phrases": [
            "Thank you for your order",
            "Order details",
            "Expected delivery",
            "You can track your order",
            "If you have questions, contact"
        ]
    }
}

TRANSLATIONS = {
    "Spanish": {
        "Dear": "Estimado/a",
        "Hi": "Hola",
        "Hello": "Hola",
        "Thank you": "Gracias",
        "Best regards": "Saludos cordiales",
        "Sincerely": "Atentamente"
    },
    "French": {
        "Dear": "Cher/Chère",
        "Hi": "Salut",
        "Hello": "Bonjour",
        "Thank you": "Merci",
        "Best regards": "Cordialement",
        "Sincerely": "Sincèrement"
    },
    "German": {
        "Dear": "Liebe/Lieber",
        "Hi": "Hallo",
        "Hello": "Guten Tag",
        "Thank you": "Danke",
        "Best regards": "Mit freundlichen Grüßen",
        "Sincerely": "Hochachtungsvoll"
    },
    "Italian": {
        "Dear": "Caro/Cara",
        "Hi": "Ciao",
        "Hello": "Salve",
        "Thank you": "Grazie",
        "Best regards": "Cordiali saluti",
        "Sincerely": "Cordialmente"
    },
    "Portuguese": {
        "Dear": "Prezado/a",
        "Hi": "Oi",
        "Hello": "Olá",
        "Thank you": "Obrigado/a",
        "Best regards": "Atenciosamente",
        "Sincerely": "Cordialmente"
    },
    "Chinese": {
        "Dear": "尊敬的",
        "Hi": "你好",
        "Hello": "您好",
        "Thank you": "谢谢",
        "Best regards": "此致敬礼",
        "Sincerely": "真诚地"
    }
}

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Email Details")
    
    # Sender info
    sender_name = st.text_input("Your Name *", placeholder="e.g., John Smith")
    sender_title = st.text_input("Your Title", placeholder="e.g., Customer Success Manager")
    sender_email = st.text_input("Your Email", placeholder="e.g., john.smith@company.com")
    
    st.divider()
    
    # Recipient info
    client_name = st.text_input("Client Name *", placeholder="e.g., Sarah Johnson")
    company_name = st.text_input("Company Name", placeholder="e.g., Acme Corp")
    
    st.divider()
    
    # Email content
    subject_topic = st.text_input("Main Topic/Issue", placeholder="e.g., Delayed Shipment Order #12345")
    
    situation = st.text_area(
        "Situation Description *",
        placeholder="Describe the situation you're addressing...",
        height=100
    )
    
    key_points = st.text_area(
        "Key Points to Include",
        placeholder="List the main points you want to cover (one per line)",
        height=80
    )
    
    additional_notes = st.text_input(
        "Additional Notes",
        placeholder="e.g., VIP customer, urgent, follow-up from phone call"
    )

with col2:
    st.subheader("📧 Email Builder")
    
    if sender_name and client_name and situation:
        # Get templates
        style_template = TEMPLATES.get(communication_style, TEMPLATES["Professional & Formal"])
        situation_template = SITUATION_TEMPLATES.get(situation_type, {})
        
        # Build email
        st.markdown("### Suggested Subject Line:")
        suggested_subject = situation_template.get("subject", "Re: {topic}").format(
            issue=subject_topic or "Your Inquiry",
            topic=subject_topic or "Your Request",
            order_number="[ORDER#]"
        )
        
        subject_line = st.text_input(
            "Subject:",
            value=suggested_subject,
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Email body
        st.markdown("### Email Body:")
        
        # Greeting
        greeting = style_template["greeting"].format(name=client_name)
        if target_language != "English" and target_language in TRANSLATIONS:
            for eng, trans in TRANSLATIONS[target_language].items():
                greeting = greeting.replace(eng, trans)
        
        # Opening
        opening = style_template["opening"]
        
        # Body structure
        st.info(f"**Recommended Structure for {situation_type}:**")
        if "structure" in situation_template:
            for step in situation_template["structure"]:
                st.markdown(f"- {step}")
        
        st.divider()
        
        # Text areas for each section
        email_greeting = st.text_area("Greeting:", value=greeting, height=50)
        
        email_opening = st.text_area("Opening:", value=opening, height=50)
        
        email_body = st.text_area(
            "Main Content:",
            placeholder="Write your main message here...\n\nUse the structure and key phrases below as a guide.",
            height=200
        )
        
        # Closing
        closing = style_template["closing"].format(
            sender_name=sender_name,
            sender_title=sender_title or ""
        )
        if target_language != "English" and target_language in TRANSLATIONS:
            for eng, trans in TRANSLATIONS[target_language].items():
                closing = closing.replace(eng, trans)
        
        email_closing = st.text_area("Closing:", value=closing, height=80)
        
        # Signature
        if sender_email:
            email_signature = st.text_area(
                "Signature:",
                value=f"{sender_name}\n{sender_title}\n{sender_email}",
                height=80
            )
        else:
            email_signature = ""

# Helper panel
st.divider()

col_help1, col_help2 = st.columns(2)

with col_help1:
    st.markdown("### 💡 Writing Tips")
    if communication_style in TEMPLATES:
        st.info(f"**{communication_style}:** {TEMPLATES[communication_style]['tone_tips']}")
    
    if urgency == "High" or urgency == "Critical":
        st.warning("⚡ **High Urgency Tips:**\n- State urgency in subject\n- Be clear about timeframe\n- Provide direct contact info\n- Keep it concise")
    
    if situation_type in SITUATION_TEMPLATES and "key_phrases" in SITUATION_TEMPLATES[situation_type]:
        st.markdown("**Suggested Phrases:**")
        for phrase in SITUATION_TEMPLATES[situation_type]["key_phrases"][:3]:
            st.markdown(f"- _{phrase}_")

with col_help2:
    st.markdown("### ✅ Email Checklist")
    
    checklist = st.container()
    with checklist:
        st.checkbox("Clear subject line", value=bool(subject_topic))
        st.checkbox("Personalized greeting", value=bool(client_name))
        st.checkbox("States purpose clearly")
        st.checkbox("Addresses all key points")
        st.checkbox("Appropriate tone")
        st.checkbox("Clear next steps/call-to-action")
        st.checkbox("Professional closing")
        st.checkbox("Contact information included")
        st.checkbox("Proofread for errors")

# Final output
st.divider()
st.markdown("## 📬 Final Email")

if sender_name and client_name:
    final_email = f"""**SUBJECT:** {subject_line if 'subject_line' in locals() else '[Add subject]'}

{email_greeting if 'email_greeting' in locals() else ''}

{email_opening if 'email_opening' in locals() else ''}

{email_body if 'email_body' in locals() else '[Write your main message here]'}

{email_closing if 'email_closing' in locals() else ''}

{email_signature if 'email_signature' in locals() else ''}
"""
    
    st.text_area("", value=final_email, height=400, label_visibility="collapsed")
    
    # Action buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        st.download_button(
            label="📥 Download Email",
            data=final_email,
            file_name=f"email_{client_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col_btn2:
        if st.button("📋 Copy to Clipboard", use_container_width=True):
            st.success("✓ Email copied!")
    
    with col_btn3:
        if st.button("🔄 Start Over", use_container_width=True):
            st.rerun()

else:
    st.info("👈 Fill in the required fields (Your Name, Client Name, Situation) to build your email")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>💼 Professional Email Builder • Always review before sending • Customize for your specific needs</p>
</div>
""", unsafe_allow_html=True)
