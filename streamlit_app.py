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

# Questionnaire templates based on situation type
QUESTIONNAIRES = {
    "Issue/Complaint Resolution": [
        {
            "question": "What is the main issue the client is experiencing?",
            "type": "text",
            "placeholder": "e.g., Product defect, delayed delivery, billing error"
        },
        {
            "question": "How long has this issue been ongoing?",
            "type": "radio",
            "options": ["Just happened", "1-3 days", "1 week", "More than a week"]
        },
        {
            "question": "Has the client contacted you before about this?",
            "type": "radio",
            "options": ["First time", "Second time", "Multiple times"]
        },
        {
            "question": "What is the impact on the client?",
            "type": "multiselect",
            "options": ["Financial loss", "Time wasted", "Inconvenience", "Frustration", "Work disruption", "Other"]
        },
        {
            "question": "Do you have a solution ready?",
            "type": "radio",
            "options": ["Yes, immediate fix", "Yes, but needs time", "Investigating", "Need more info from client"]
        },
        {
            "question": "What action will you take?",
            "type": "text",
            "placeholder": "e.g., Send replacement, issue refund, escalate to management"
        },
        {
            "question": "When can the issue be resolved?",
            "type": "text",
            "placeholder": "e.g., Within 24 hours, by Friday, within 3-5 business days"
        },
        {
            "question": "Is compensation appropriate?",
            "type": "radio",
            "options": ["Yes - discount/refund", "Yes - free product/service", "No", "Maybe - needs approval"]
        }
    ],
    "Apology": [
        {
            "question": "What specifically went wrong?",
            "type": "text",
            "placeholder": "e.g., Missed deadline, incorrect information provided, poor service"
        },
        {
            "question": "Whose fault was it?",
            "type": "radio",
            "options": ["Our error", "System/technical issue", "Third-party vendor", "Miscommunication"]
        },
        {
            "question": "Did this affect multiple clients?",
            "type": "radio",
            "options": ["Only this client", "Small group", "Many clients", "All clients"]
        },
        {
            "question": "How severe was the impact?",
            "type": "radio",
            "options": ["Minor inconvenience", "Moderate issue", "Major problem", "Critical/urgent"]
        },
        {
            "question": "What are you doing to fix it?",
            "type": "text",
            "placeholder": "e.g., Correcting the error, implementing new process, training team"
        },
        {
            "question": "How will you prevent this in the future?",
            "type": "text",
            "placeholder": "e.g., Added quality checks, updated system, new policy"
        },
        {
            "question": "What will you offer to make it right?",
            "type": "text",
            "placeholder": "e.g., Full refund, priority service, discount on next purchase"
        }
    ],
    "Good News/Update": [
        {
            "question": "What's the good news?",
            "type": "text",
            "placeholder": "e.g., Issue resolved, new feature launched, order shipped early"
        },
        {
            "question": "Were they expecting this update?",
            "type": "radio",
            "options": ["Yes, they were waiting", "Partial - they knew something was coming", "No, pleasant surprise"]
        },
        {
            "question": "How does this benefit them?",
            "type": "multiselect",
            "options": ["Saves time", "Saves money", "Better quality", "More features", "Faster service", "Peace of mind"]
        },
        {
            "question": "Is any action required from them?",
            "type": "radio",
            "options": ["No action needed", "Optional action", "Required action", "Choice to make"]
        },
        {
            "question": "When does this take effect?",
            "type": "text",
            "placeholder": "e.g., Immediately, starting Monday, next billing cycle"
        }
    ],
    "General Inquiry Response": [
        {
            "question": "What is the client asking about?",
            "type": "text",
            "placeholder": "e.g., Product features, pricing, availability, process"
        },
        {
            "question": "How detailed should your answer be?",
            "type": "radio",
            "options": ["Quick yes/no", "Brief explanation", "Detailed information", "Comprehensive guide"]
        },
        {
            "question": "Do you have all the information they need?",
            "type": "radio",
            "options": ["Yes, complete answer", "Mostly, minor gaps", "Partial info only", "Need to research"]
        },
        {
            "question": "Should you provide additional resources?",
            "type": "multiselect",
            "options": ["Link to documentation", "Video tutorial", "FAQ page", "Product demo", "Case study", "Contact for specialist"]
        },
        {
            "question": "Is this a sales opportunity?",
            "type": "radio",
            "options": ["No, just information", "Maybe - show interest", "Yes - soft sell", "Yes - include pricing/CTA"]
        }
    ],
    "Follow-up Email": [
        {
            "question": "What are you following up on?",
            "type": "text",
            "placeholder": "e.g., Previous email, phone call, meeting, proposal"
        },
        {
            "question": "How long has it been since last contact?",
            "type": "radio",
            "options": ["1-2 days", "3-5 days", "1 week", "2+ weeks"]
        },
        {
            "question": "Did they say they'd respond by a certain time?",
            "type": "radio",
            "options": ["Yes, and deadline passed", "Yes, deadline approaching", "No specific timeline", "They said they'd contact me"]
        },
        {
            "question": "What do you need from them?",
            "type": "text",
            "placeholder": "e.g., Decision, information, approval, meeting confirmation"
        },
        {
            "question": "Is there urgency?",
            "type": "radio",
            "options": ["No rush", "Somewhat time-sensitive", "Urgent - impacts timeline", "Critical - deadline approaching"]
        },
        {
            "question": "What's your relationship with this client?",
            "type": "radio",
            "options": ["New prospect", "Existing client", "Long-term partner", "VIP/high-value"]
        }
    ],
    "Thank You Note": [
        {
            "question": "What are you thanking them for?",
            "type": "text",
            "placeholder": "e.g., Purchase, referral, meeting, feedback, patience"
        },
        {
            "question": "How significant was their action?",
            "type": "radio",
            "options": ["Small gesture", "Standard business", "Above and beyond", "Exceptional/game-changing"]
        },
        {
            "question": "What impact did it have?",
            "type": "text",
            "placeholder": "e.g., Helped us improve, won new business, met deadline"
        },
        {
            "question": "Do you want to mention future collaboration?",
            "type": "radio",
            "options": ["No - just thanks", "Yes - general mention", "Yes - specific opportunity", "Yes - ask for continued partnership"]
        }
    ],
    "Meeting Request": [
        {
            "question": "What's the meeting purpose?",
            "type": "text",
            "placeholder": "e.g., Discuss project update, resolve issue, explore partnership"
        },
        {
            "question": "How long do you need?",
            "type": "radio",
            "options": ["15 minutes", "30 minutes", "1 hour", "More than 1 hour"]
        },
        {
            "question": "Is this meeting time-sensitive?",
            "type": "radio",
            "options": ["Flexible timing", "Prefer within a week", "Within 2-3 days", "Urgent - ASAP"]
        },
        {
            "question": "Who else should attend?",
            "type": "text",
            "placeholder": "e.g., Just you two, include technical team, management"
        },
        {
            "question": "Meeting format preference?",
            "type": "radio",
            "options": ["Phone call", "Video call", "In-person", "Their choice"]
        },
        {
            "question": "Should they prepare anything?",
            "type": "text",
            "placeholder": "e.g., Review proposal, bring questions, have data ready"
        }
    ],
    "Request for Information": [
        {
            "question": "What information do you need?",
            "type": "text",
            "placeholder": "e.g., Account details, technical specs, feedback, documents"
        },
        {
            "question": "Why do you need this information?",
            "type": "text",
            "placeholder": "e.g., Complete order, troubleshoot issue, prepare proposal"
        },
        {
            "question": "Is there a deadline?",
            "type": "radio",
            "options": ["No deadline", "Soft deadline - would be helpful", "Hard deadline - required", "Urgent - today/tomorrow"]
        },
        {
            "question": "How should they provide the information?",
            "type": "radio",
            "options": ["Email reply", "Fill out form", "Upload document", "Phone call", "Any method works"]
        }
    ],
    "Price Quote": [
        {
            "question": "What are you quoting?",
            "type": "text",
            "placeholder": "e.g., Product name, service package, custom solution"
        },
        {
            "question": "Did they request this quote?",
            "type": "radio",
            "options": ["Yes, they asked", "Follow-up to inquiry", "Proactive offer", "Renewal/upsell"]
        },
        {
            "question": "Are there multiple pricing options?",
            "type": "radio",
            "options": ["Single price", "2-3 tiers", "Custom packages", "Volume discounts available"]
        },
        {
            "question": "What's included in the price?",
            "type": "multiselect",
            "options": ["Product/service", "Support", "Training", "Implementation", "Warranty", "Ongoing maintenance"]
        },
        {
            "question": "Are there any special offers?",
            "type": "text",
            "placeholder": "e.g., Early bird discount, bundle deal, limited-time offer"
        },
        {
            "question": "How long is the quote valid?",
            "type": "text",
            "placeholder": "e.g., 30 days, end of month, until [date]"
        }
    ],
    "Order Confirmation": [
        {
            "question": "Order number/ID:",
            "type": "text",
            "placeholder": "e.g., #12345"
        },
        {
            "question": "What did they order?",
            "type": "text",
            "placeholder": "e.g., Product name, quantity, specifications"
        },
        {
            "question": "When will it be delivered/completed?",
            "type": "text",
            "placeholder": "e.g., 3-5 business days, by [date], immediate access"
        },
        {
            "question": "Is tracking available?",
            "type": "radio",
            "options": ["Yes - include tracking", "Yes - will send separately", "No tracking", "Not applicable"]
        },
        {
            "question": "Any special instructions or next steps?",
            "type": "text",
            "placeholder": "e.g., Setup required, check email for access, contact for installation"
        }
    ]
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

# Initialize session state
if 'questionnaire_completed' not in st.session_state:
    st.session_state.questionnaire_completed = False
if 'questionnaire_answers' not in st.session_state:
    st.session_state.questionnaire_answers = {}

# Main content
st.divider()

# Step 1: Basic Info
st.markdown("## 👤 Step 1: Basic Information")

col_basic1, col_basic2 = st.columns(2)

with col_basic1:
    sender_name = st.text_input("Your Name *", placeholder="e.g., John Smith")
    sender_title = st.text_input("Your Title", placeholder="e.g., Customer Success Manager")
    sender_email = st.text_input("Your Email", placeholder="e.g., john.smith@company.com")

with col_basic2:
    client_name = st.text_input("Client Name *", placeholder="e.g., Sarah Johnson")
    company_name = st.text_input("Company Name", placeholder="e.g., Acme Corp")
    subject_topic = st.text_input("Main Topic/Issue", placeholder="e.g., Delayed Shipment Order #12345")

st.divider()

# Step 2: Guided Questionnaire
st.markdown("## 🎯 Step 2: Let's Understand the Situation")
st.markdown("*Answer these questions to help craft the perfect email*")

if situation_type in QUESTIONNAIRES:
    questionnaire = QUESTIONNAIRES[situation_type]
    
    # Create a container for questionnaire
    with st.container():
        st.markdown(f"### 📋 Questions for: **{situation_type}**")
        
        answers = {}
        
        for idx, q in enumerate(questionnaire):
            st.markdown(f"**Q{idx + 1}: {q['question']}**")
            
            if q['type'] == 'text':
                answers[q['question']] = st.text_input(
                    f"Answer {idx + 1}",
                    placeholder=q.get('placeholder', ''),
                    key=f"q_{idx}",
                    label_visibility="collapsed"
                )
            
            elif q['type'] == 'radio':
                answers[q['question']] = st.radio(
                    f"Select {idx + 1}",
                    options=q['options'],
                    key=f"q_{idx}",
                    label_visibility="collapsed"
                )
            
            elif q['type'] == 'multiselect':
                answers[q['question']] = st.multiselect(
                    f"Select all that apply {idx + 1}",
                    options=q['options'],
                    key=f"q_{idx}",
                    label_visibility="collapsed"
                )
            
            st.markdown("")  # spacing
        
        # Generate key points button
        if st.button("✨ Generate Key Points from Answers", type="primary", use_container_width=True):
            st.session_state.questionnaire_answers = answers
            st.session_state.questionnaire_completed = True
            st.success("✓ Key points generated! Scroll down to see them.")
else:
    st.info("Select an email type from the sidebar to see guided questions.")

st.divider()

# Step 3: Review Generated Key Points
st.markdown("## 📝 Step 3: Key Points & Situation Description")

if st.session_state.questionnaire_completed and st.session_state.questionnaire_answers:
    st.success("✓ Generated from your answers above")
    
    # Generate key points from answers
    generated_points = []
    generated_situation = []
    
    for question, answer in st.session_state.questionnaire_answers.items():
        if answer:  # Only include non-empty answers
            if isinstance(answer, list):  # multiselect
                if answer:
                    generated_points.append(f"• {question}: {', '.join(answer)}")
            else:
                # For situation description, combine key answers
                if "issue" in question.lower() or "wrong" in question.lower() or "news" in question.lower():
                    generated_situation.append(str(answer))
                else:
                    generated_points.append(f"• {answer}")
    
    default_situation = " ".join(generated_situation) if generated_situation else ""
    default_points = "\n".join(generated_points) if generated_points else ""
else:
    default_situation = ""
    default_points = ""
    st.info("👆 Complete the questionnaire above to auto-generate this section")

situation = st.text_area(
    "Situation Description *",
    value=default_situation,
    placeholder="Describe the situation you're addressing...",
    height=100
)

key_points = st.text_area(
    "Key Points to Include",
    value=default_points,
    placeholder="• Point 1\n• Point 2\n• Point 3",
    height=150,
    help="Edit or add more points as needed"
)

additional_notes = st.text_input(
    "Additional Notes/Context",
    placeholder="e.g., VIP customer, urgent, follow-up from phone call"
)

if st.button("🔄 Reset Questionnaire", help="Start over with new answers"):
    st.session_state.questionnaire_completed = False
    st.session_state.questionnaire_answers = {}
    st.rerun()

st.divider()

# Step 4: Email Builder
st.markdown("## 📧 Step 4: Build Your Email")

if sender_name and client_name and situation:
    # Get templates
    style_template = TEMPLATES.get(communication_style, TEMPLATES["Professional & Formal"])
    situation_template = SITUATION_TEMPLATES.get(situation_type, {})
    
    col_email1, col_email2 = st.columns([1, 1])
    
    with col_email1:
        # Build email
        st.markdown("### Subject Line")
        suggested_subject = situation_template.get("subject", "Re: {topic}").format(
            issue=subject_topic or "Your Inquiry",
            topic=subject_topic or "Your Request",
            order_number="[ORDER#]",
            product_service="[Product/Service]"
        )
        
        subject_line = st.text_input(
            "Subject:",
            value=suggested_subject,
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Email body sections
        st.markdown("### Email Content")
        
        # Greeting
        greeting = style_template["greeting"].format(name=client_name)
        if target_language != "English" and target_language in TRANSLATIONS:
            for eng, trans in TRANSLATIONS[target_language].items():
                greeting = greeting.replace(eng, trans)
        
        email_greeting = st.text_area("Greeting:", value=greeting, height=50)
        
        # Opening
        opening = style_template["opening"]
        email_opening = st.text_area("Opening:", value=opening, height=50)
        
        # Main body
        email_body = st.text_area(
            "Main Content:",
            placeholder="Write your main message here...\n\nUse the key points and structure as a guide.",
            height=250
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
    
    with col_email2:
        st.markdown("### 💡 Writing Guide")
        
        # Tone tips
        st.info(f"**{communication_style}:** {style_template['tone_tips']}")
        
        # Structure
        if "structure" in situation_template:
            with st.expander("📌 Recommended Structure", expanded=True):
                for step in situation_template["structure"]:
                    st.markdown(f"- {step}")
        
        # Key phrases
        if "key_phrases" in situation_template:
            with st.expander("💬 Suggested Phrases", expanded=True):
                for phrase in situation_template["key_phrases"]:
                    st.markdown(f"- _{phrase}_")
        
        # Urgency indicator
        if urgency == "High" or urgency == "Critical":
            st.warning(f"⚡ **{urgency} Urgency**\n- State urgency in subject\n- Be clear about timeframe\n- Provide direct contact\n- Keep it concise")
        
        # Your key points reference
        if key_points:
            with st.expander("📋 Your Key Points (Reference)", expanded=True):
                st.markdown(key_points)
        
        # Email checklist
        st.markdown("### ✅ Quality Checklist")
        with st.container():
            st.checkbox("Clear subject line", value=bool(subject_topic))
            st.checkbox("Personalized greeting", value=bool(client_name))
            st.checkbox("States purpose clearly")
            st.checkbox("Addresses all key points")
            st.checkbox("Appropriate tone")
            st.checkbox("Clear next steps/call-to-action")
            st.checkbox("Professional closing")
            st.checkbox("Contact information included")
            st.checkbox("Proofread for errors")

else:
    st.info("👈 Complete Step 1 (Basic Information) and Step 3 (Situation Description) to build your email")

# Final output
st.divider()
st.markdown("## 📬 Final Email Preview")

if sender_name and client_name:
    final_email = f"""SUBJECT: {subject_line if 'subject_line' in locals() else '[Add subject]'}

{email_greeting if 'email_greeting' in locals() else ''}

{email_opening if 'email_opening' in locals() else ''}

{email_body if 'email_body' in locals() else '[Write your main message here]'}

{email_closing if 'email_closing' in locals() else ''}

{email_signature if 'email_signature' in locals() else ''}
"""
    
    st.text_area("", value=final_email, height=400, label_visibility="collapsed")
    
    # Action buttons
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    
    with col_btn1:
        st.download_button(
            label="📥 Download",
            data=final_email,
            file_name=f"email_{client_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col_btn2:
        if st.button("📋 Copy", use_container_width=True):
            st.success("✓ Email copied!")
    
    with col_btn3:
        if st.button("🔄 Start Over", use_container_width=True):
            st.session_state.questionnaire_completed = False
            st.session_state.questionnaire_answers = {}
            st.rerun()
    
    with col_btn4:
        # Save as draft (could expand this to actually save)
        if st.button("💾 Save Draft", use_container_width=True):
            st.success("✓ Draft saved!")

else:
    st.info("👈 Fill in Step 1 to see your email preview")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>💼 Professional Email Builder • 4-Step Guided Process • Always review before sending</p>
</div>
""", unsafe_allow_html=True)
