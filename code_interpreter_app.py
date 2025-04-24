import streamlit as st
import pandas as pd
from openai import OpenAI
import plotly.express as px
import io
import numpy as np
import traceback
import time
from PIL import Image
import hashlib

# --- Page Configuration ---
st.set_page_config(
    page_title="DataInsight Pro",
    layout="wide",  # Set wide layout for the app
    initial_sidebar_state="collapsed",
)

# Initialize session state for login status and app settings
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
#if 'openai_key' not in st.session_state:
    # Default key (will be overridden by user input)
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = "gpt-4o-mini"  # Default model
# Default analysis prompt - store in session state so users can modify it
if 'analysis_prompt' not in st.session_state:
    st.session_state.analysis_prompt = """Please analyze this dataset comprehensively, focusing on business insights. Include:

1. Data Overview: Summary of the dataset structure and key metrics
2. Data Quality Assessment: Missing values, outliers, and data integrity issues
3. Key Insights: Identify important patterns, trends, and relationships
4. Visualization: Create clear, professional visualizations to illustrate key findings
5. Business Recommendations: Provide actionable recommendations based on the data

Present your analysis in a well-structured format with clear sections and professional language."""

# Predefined credentials (in production, use a more secure approach)
CORRECT_USERNAME = "llm_analysis"
CORRECT_PASSWORD = "test123456"  # In production, store hashed passwords

# --- Authentication function ---
def authenticate(username, password):
    return username == CORRECT_USERNAME and password == CORRECT_PASSWORD

# --- Login Page Styling ---
login_style = """
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Page Background */
    .stApp {
        background: linear-gradient(145deg, #ff7a59 0%, #ff9980 50%, #ffffff 100%);
    }

    /* Center the login card within the WIDE layout */
    /* Target the main block container when NOT authenticated */
    section[data-testid="stSidebar"] + section div[data-testid="stVerticalBlock"] {
        display: flex;
        flex-direction: column;
        align-items: center; /* Center children horizontally */
        padding-top: 10vh;   /* Add padding from top */
    }

    /* Style the login card itself (adjust selector if needed) */
    /* This targets the container holding the login elements */
    div[data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"]:not(:has(div.app-header)) /* Exclude main app containers */ {
        background-color: white;
        padding: 2.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        max-width: 450px; /* Limit width */
        width: 100%;      /* Take available width up to max-width */
        /* margin: 0 auto; is implicitly handled by parent flex */
    }

    /* Login Header (Logo, Title, Subtitle) */
    .login-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .login-header svg { /* Adjust logo size/margin if needed */
        margin-bottom: 1rem;
    }
    .login-title {
        color: #33475b;
        font-weight: 600;
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    .login-subtitle {
        color: #516f90;
        font-size: 1rem;
    }
    .orange-highlight { color: #ff7a59; }

    /* Error Message */
    .error-message {
        color: #e34c4c;
        background-color: #ffeae5;
        border-left: 3px solid #e34c4c;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        font-size: 0.9rem;
    }

    /* Streamlit Input Styling */
    .stTextInput label {
        display: block;
        margin-bottom: 0.5rem;
        font-weight: 500;
        color: #33475b;
    }
    .stTextInput input {
        border: 1px solid #cbd6e2;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: border-color 0.2s ease;
    }
    .stTextInput input:focus {
        border-color: #ff7a59;
        outline: none;
        box-shadow: 0 0 0 2px rgba(255, 122, 89, 0.2);
    }

    /* Streamlit Checkbox Styling */
    .stCheckbox {
        margin-bottom: 1.5rem;
    }
    .stCheckbox label {
        color: #516f90;
        font-size: 0.9rem;
    }
    
    /* Streamlit Button Styling */
    .stButton button {
        width: 100%;
        background-color: #ff7a59;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.85rem;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        background-color: #ff8f73;
    }

    /* Footer Styling */
    .login-footer {
        text-align: center;
        margin-top: 2rem;
        color: #516f90;
        font-size: 0.85rem;
        width: 100%; /* Ensure footer is centered below card */
    }
    
    /* Logo Animation */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .logo-pulse {
        animation: pulse 2s infinite ease-in-out;
        display: inline-block; /* Needed for animation */
    }

</style>
"""

# --- Main Application Logic ---
if not st.session_state.authenticated:
    # Apply the CSS styling
    st.markdown(login_style, unsafe_allow_html=True)

    # Login error container - Place it before the card for visibility
    login_error = st.empty()

    # Use a container to group login elements, styled by CSS above
    with st.container():
        # Header (Logo, Title, Subtitle) using Markdown
        st.markdown("""
            <div class="login-header">
                <div class="logo-pulse">
                    <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                        <rect width="10" height="16" fill="#ff7a59" rx="2"/>
                        <rect x="11" width="10" height="26" fill="#00a4bd" rx="2"/>
                        <rect x="22" width="10" height="22" fill="#7c98b6" rx="2"/>
                    </svg>
                </div>
                <h1 class="login-title">DataInsight <span class="orange-highlight">Pro</span></h1>
                <p class="login-subtitle">Sign in to access your analytics dashboard</p>
            </div>
        """, unsafe_allow_html=True)

        # Streamlit functional input fields
        username = st.text_input(
            "Username",
            placeholder="Enter your username",
            key="username_input" # Keep key if needed for state
        )
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            key="password_input" # Keep key if needed for state
        )
        remember_me = st.checkbox("Remember me for 30 days")

        # Streamlit login button
        login_pressed = st.button("Sign In", type="primary", use_container_width=True)

        # Footer inside the container to be centered with the card
        st.markdown('<div class="login-footer">© 2025 DataInsight Pro • All rights reserved</div>', unsafe_allow_html=True)

    # Authentication check
    if login_pressed:
        if authenticate(username, password):
            st.session_state.authenticated = True
            # REMOVED: st.set_page_config(layout="wide") 
            st.rerun()
        else:
            # Display error message using the placeholder
            login_error.markdown("""
            <div style="display: flex; justify-content: center;">
                <div class="error-message" style="max-width: 450px; width: 100%; margin: 0 auto 1.5rem auto;">
                    Invalid username or password. Please try again.
                </div>
            </div>
            """, unsafe_allow_html=True)
            # Add a small delay and clear the error after a few seconds
            time.sleep(3)
            login_error.empty()

else:
    # --- Add sidebar with settings ---
    with st.sidebar:
        st.markdown("<h3 style='margin-top:0'>Settings</h3>", unsafe_allow_html=True)
        
        # API Key input with toggle to show/hide
        st.markdown("### OpenAI API Key")
        show_api_key = st.checkbox("Show API Key", value=False)
        api_key_input = st.text_input(
            "Enter your OpenAI API Key",
            type="default" if show_api_key else "password",
            value=st.session_state.openai_key,
            placeholder="sk-..."
        )
        
        # Update the API key in session state
        if api_key_input != st.session_state.openai_key:
            st.session_state.openai_key = api_key_input
        
        # Model selection
        st.markdown("### Model Selection")
        model_options = [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4.1",
            "gpt-4.1-mini",
            #"o3-mini",
            #"gpt-o3"

            
        ]
        selected_model = st.selectbox(
            "Choose LLM Model",
            options=model_options,
            index=model_options.index(st.session_state.model_choice) if st.session_state.model_choice in model_options else 0,
            help="Select the OpenAI model to use for analysis. More powerful models provide better insights but may cost more."
        )
        
        # Update the model choice in session state
        if selected_model != st.session_state.model_choice:
            st.session_state.model_choice = selected_model
        
        # Add some information about models
        st.markdown("""
        <div style="font-size:0.85rem; color:#516f90; margin-top:0.75rem;">
        <strong>Model comparison:</strong>
        <ul style="padding-left:1rem; margin-top:0.5rem;">
        <li><strong>gpt-4o-mini</strong>: Fast and cost-effective with solid performance</li>
        <li><strong>gpt-4o</strong>: High-quality insights and advanced reasoning</li>
        <li><strong>gpt-4.1</strong>: Enhanced coding and long-context understanding</li>
        <li><strong>gpt-4.1-mini</strong>: Balanced performance with reduced latency and cost</li>
        #<li><strong>gpt-o3-mini</strong>: Efficient reasoning with multimodal capabilities</li>
        #<li><strong>gpt-o3</strong>: Advanced reasoning with strong visual and tool integration</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        
        # Add prompt customization
        st.markdown("### Analysis Prompt")
        st.markdown("""
        <div style="font-size:0.85rem; color:#516f90; margin-bottom:0.5rem;">
        Customize the instructions for the AI analysis
        </div>
        """, unsafe_allow_html=True)
        
        # Text area for editing the prompt
        custom_prompt = st.text_area(
            "Prompt",
            value=st.session_state.analysis_prompt,
            height=300,
            label_visibility="collapsed",
            help="Edit the prompt to customize the analysis instructions"
        )
        
        # Update prompt in session state if changed
        if custom_prompt != st.session_state.analysis_prompt:
            st.session_state.analysis_prompt = custom_prompt
            
        # Reset prompt button
        if st.button("Reset to Default Prompt", use_container_width=True):
            st.session_state.analysis_prompt = """Please analyze this dataset comprehensively, focusing on business insights. Include:

1. Data Overview: Summary of the dataset structure and key metrics
2. Data Quality Assessment: Missing values, outliers, and data integrity issues
3. Key Insights: Identify important patterns, trends, and relationships
4. Visualization: Create clear, professional visualizations to illustrate key findings
5. Business Recommendations: Provide actionable recommendations based on the data

Present your analysis in a well-structured format with clear sections and professional language."""
            st.rerun()
        
        # Divider before logout
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Logout button
        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()
    
    # Initialize the OpenAI client with the key from session state
    client = OpenAI(api_key=st.session_state.openai_key)

    # --- HubSpot/SaaS-Inspired CSS Styling ---
    st.markdown(
        """
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif;
            color: #33475b; /* HubSpot dark blue text */
        }
        
        /* App Background and Layout */
        .stApp {
            background-color: #f5f8fa; /* Light gray background similar to HubSpot */
        }
        
        /* Main Content Container */
        .main .block-container {
            padding: 2rem 3rem !important; /* Ensure padding for wide layout */
            max-width: 1200px;
            margin: 0 auto;
        }
        
        /* Header Styling */
        h1 {
            color: #33475b; /* Dark blue */
            font-weight: 700;
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        
        /* App Logo & Title Container */
        .app-header {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            padding: 1.5rem;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        /* Subheaders */
        h3 {
            color: #33475b;
            font-weight: 600;
            font-size: 1.5rem;
            margin-top: 1rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
        }
        h3 svg {
            margin-right: 0.5rem;
        }
        h6 {
            font-weight: 600;
            color: #516f90;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        
        /* Cards */
        .card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        /* Buttons - HubSpot orange primary color */
        .stButton > button {
            background-color: #ff7a59; /* HubSpot orange */
            border: none;
            border-radius: 4px;
            padding: 10px 20px;
            font-weight: 600;
            color: white;
            transition: all 0.2s ease;
            box-shadow: 0 2px 4px rgba(255, 122, 89, 0.2);
        }
        .stButton > button:hover {
            background-color: #ff8f73; /* Lighter orange */
            box-shadow: 0 4px 8px rgba(255, 122, 89, 0.3);
            transform: translateY(-1px);
        }
        .stButton > button:active {
            background-color: #e05e39; /* Darker orange */
            transform: translateY(0);
        }
        
        /* Secondary Buttons */
        .secondary-button button {
            background-color: #fff;
            color: #33475b;
            border: 1px solid #cbd6e2;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        .secondary-button button:hover {
            background-color: #f5f8fa;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        /* Radio Buttons - More modern, flat style */
        .stRadio [role="radiogroup"] {
            display: flex;
            gap: 1rem;
            padding: 0.5rem 0;
        }
        .stRadio [role="radio"] {
            border: 1px solid #cbd6e2;
            border-radius: 4px;
            padding: 12px 16px;
            transition: all 0.2s ease;
        }
        .stRadio [aria-checked="true"] {
            background-color: #eaf0f6; /* Light blue when selected */
            border-color: #ff7a59; /* Orange accent */
            box-shadow: 0 0 0 1px #ff7a59;
        }
        
        /* File Uploader */
        [data-testid="stFileUploader"] {
            background-color: white;
            border: 2px dashed #cbd6e2;
            border-radius: 6px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        [data-testid="stFileUploadDropzone"] {
            min-height: 140px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        [data-testid="stFileUploadDropzone"] button {
            background-color: #ff7a59; /* Changed to orange */
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: 600;
            margin-top: 1rem;
        }
        [data-testid="stFileUploadDropzone"] button:hover {
            background-color: #ff8f73;
        }
        
        /* DataFrames */
        .stDataFrame {
            border: 1px solid #eaf0f6;
            border-radius: 6px;
            overflow: hidden;
        }
        .stDataFrame [data-testid="stTable"] {
            border-collapse: separate;
            border-spacing: 0;
        }
        .stDataFrame th {
            background-color: #f5f8fa;
            color: #33475b;
            font-weight: 600;
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eaf0f6;
        }
        .stDataFrame td {
            padding: 12px;
            border-bottom: 1px solid #f5f8fa;
            font-size: 0.9rem;
        }
        
        /* Alert/Info Boxes */
        .stAlert {
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: flex-start;
        }
        .info-box {
            background-color: #eaf5ff;
            border-left: 4px solid #0091ae;
            color: #33475b;
        }
        .success-box {
            background-color: #edf8f0;
            border-left: 4px solid #00bf6f;
            color: #33475b;
        }
        .warning-box {
            background-color: #fef8e3;
            border-left: 4px solid #ffc21d;
            color: #33475b;
        }
        .error-box {
            background-color: #ffeae5;
            border-left: 4px solid #ff7a59;
            color: #33475b;
        }
        
        /* Progress Indicators */
        .progress-container {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        .progress-step {
            display: flex;
            align-items: center;
            margin-bottom: 0.75rem;
        }
        .progress-icon {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background-color: #eaf0f6;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 0.75rem;
            font-size: 12px;
            font-weight: 600;
        }
        .progress-icon.completed {
            background-color: #00bf6f;
            color: white;
        }
        .progress-icon.active {
            background-color: #ff7a59; /* Changed to orange */
            color: white;
        }
        .progress-text {
            font-size: 0.9rem;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 1.5rem;
            margin-top: 3rem;
            color: #7c98b6;
            font-size: 0.9rem;
            border-top: 1px solid #eaf0f6;
        }
        .footer a {
            color: #ff7a59; /* Changed to orange */
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        
        /* Markdown Content */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
        .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #33475b;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        .stMarkdown p {
            color: #5a6e84;
            line-height: 1.6;
            margin-bottom: 1rem;
        }
        .stMarkdown a {
            color: #ff7a59; /* Changed to orange */
            text-decoration: none;
        }
        .stMarkdown a:hover {
            text-decoration: underline;
        }
        .stMarkdown ul, .stMarkdown ol {
            margin-bottom: 1rem;
            color: #5a6e84;
        }
        
        /* Custom Classes for App Components */
        .section-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #33475b;
            margin-bottom: 1rem;
            margin-top: 2rem;
        }
        .header-icon {
            margin-right: 8px;
            vertical-align: middle;
        }
        .step-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background-color: #ff7a59; /* Changed to orange */
            color: white;
            font-weight: 600;
            margin-right: 10px;
        }
        
        /* User Account Section */
        .user-account {
            position: absolute;
            top: 1.5rem;
            right: 1.5rem;
            display: flex;
            align-items: center;
            background-color: white;
            border-radius: 50px;
            padding: 0.5rem 1rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .user-avatar {
            width: 30px;
            height: 30px;
            background-color: #ff7a59;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 0.9rem;
            margin-right: 0.75rem;
        }
        .user-info {
            display: flex;
            flex-direction: column;
        }
        .user-name {
            font-weight: 600;
            font-size: 0.9rem;
            color: #33475b;
        }
        .user-role {
            font-size: 0.75rem;
            color: #516f90;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # User account display in top right corner
    st.markdown(
        f"""
        <div class="user-account">
            <div class="user-avatar">{CORRECT_USERNAME[0].upper()}</div>
            <div class="user-info">
                <div class="user-name">{CORRECT_USERNAME}</div>
                <div class="user-role">Data Analyst</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Backend Functions ---
    def use_openai_assistant(df):
        """
        Use OpenAI Assistant with Code Interpreter to analyze the data.
        """
        try:
            # Create a progress tracker
            progress_container = st.empty()

            with progress_container.container():
                st.markdown('<div class="progress-container card">', unsafe_allow_html=True) # Add card class

                # Step 1: Uploading data
                st.markdown(
                    '<div class="progress-step">'
                    '<div class="progress-icon active">1</div>'
                    '<div class="progress-text">Preparing and uploading data...</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

                # Convert DataFrame to CSV for upload
                csv_data = df.to_csv(index=False)

                # Upload the file using BytesIO for binary support
                file_response = client.files.create(
                    file=io.BytesIO(csv_data.encode('utf-8')),
                    purpose="assistants"
                )
                file_id = file_response.id

                # Update progress for step 1 as completed
                st.markdown(
                    '<div class="progress-step">'
                    '<div class="progress-icon completed">✓</div>'
                    '<div class="progress-text">Data uploaded successfully</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

                # Step 2: Creating assistant
                st.markdown(
                    '<div class="progress-step">'
                    '<div class="progress-icon active">2</div>'
                    '<div class="progress-text">Initializing AI assistant...</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

                # Create an assistant with the code interpreter tool and selected model
                assistant = client.beta.assistants.create(
                    name="Data Analysis Assistant",
                    instructions="You are a helpful data analysis assistant. Analyze the provided dataset in a high level. Present your findings in a clear, business-oriented manner. Structure your analysis with clear headers, insights, and recommendations. Where appropriate, create professional-looking visualizations. Focus on actionable business insights.",
                    model=st.session_state.model_choice,  # Use selected model
                    tools=[{"type": "code_interpreter"}]
                )

                # Update progress for step 2
                st.markdown(
                    '<div class="progress-step">'
                    '<div class="progress-icon completed">✓</div>'
                    '<div class="progress-text">AI assistant initialized</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

                # Step 3: Creating thread and sending message
                st.markdown(
                    '<div class="progress-step">'
                    '<div class="progress-icon active">3</div>'
                    '<div class="progress-text">Setting up analysis parameters...</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

                # Create thread and send message
                thread = client.beta.threads.create()
                message = client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=[{
                        "type": "text",
                        "text": st.session_state.analysis_prompt  # Use the custom prompt
                    }],
                    attachments=[{"file_id": file_id, "tools": [{"type": "code_interpreter"}]}]
                )

                # Update progress for step 3
                st.markdown(
                    '<div class="progress-step">'
                    '<div class="progress-icon completed">✓</div>'
                    '<div class="progress-text">Analysis parameters configured</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

                # Step 4: Running the analysis
                st.markdown(
                    '<div class="progress-step">'
                    '<div class="progress-icon active">4</div>'
                    '<div class="progress-text">Executing in-depth data analysis...</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

                # Run the assistant
                run = client.beta.threads.runs.create(
                    thread_id=thread.id,
                    assistant_id=assistant.id
                )

                # Separate progress tracking for analysis
                analysis_progress = st.progress(0)
                # Status message
                status_message = st.empty()

                # Poll for completion
                stages = ["Initializing", "Analyzing data structure", "Performing statistical analysis",
                         "Identifying trends", "Creating visualizations", "Formulating recommendations",
                         "Finalizing report"]
                current_stage = 0

                while run.status in ["queued", "in_progress", "requires_action"]:
                    # Update progress animation
                    current_stage = (current_stage + 1) % len(stages)
                    progress_value = min(0.05 + (current_stage / len(stages)) * 0.7, 0.95)  # Cap at 95%
                    analysis_progress.progress(progress_value)
                    status_message.markdown(f"<div style='text-align: center; color: #516f90;'>{stages[current_stage]}...</div>", unsafe_allow_html=True)

                    if run.status == "requires_action":
                        # Handle tool calls if needed
                        try:
                            run = client.beta.threads.runs.submit_tool_outputs(
                                thread_id=thread.id,
                                run_id=run.id,
                                tool_outputs=[]
                            )
                        except Exception as tool_err:
                            st.error(f"Error during analysis: {tool_err}")
                            break
                    else:
                        # Check status again
                        run = client.beta.threads.runs.retrieve(
                            thread_id=thread.id,
                            run_id=run.id
                        )
                    time.sleep(2)

                # Complete the progress bar
                analysis_progress.progress(1.0)
                status_message.markdown("<div style='text-align: center; color: #00bf6f;'>Analysis complete!</div>", unsafe_allow_html=True)

                # Update final progress step
                st.markdown(
                    '<div class="progress-step">'
                    '<div class="progress-icon completed">✓</div>'
                    '<div class="progress-text">Analysis completed successfully</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

            st.markdown('</div>', unsafe_allow_html=True) # Close progress-container

            # Clear the progress container
            progress_container.empty()

            # Retrieve and display results
            messages = client.beta.threads.messages.list(
                thread_id=thread.id,
                order='asc'
            )

            # Create result container with HubSpot-style card
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📊 Analysis Results")

            assistant_responded = False
            for msg in messages.data:
                if msg.role == "assistant":
                    assistant_responded = True
                    for content_part in msg.content:
                        if content_part.type == "text":
                            # Process markdown to enhance with HubSpot styling
                            text = content_part.text.value
                            # Enhance headers with HubSpot styling
                            text = text.replace("# ", "## ")  # Make H1 into H2 for better sizing
                            st.markdown(text, unsafe_allow_html=True) # Allow HTML in results

                        elif content_part.type == "image_file":
                            try:
                                image_file_id = content_part.image_file.file_id
                                image_response = client.files.content(image_file_id)
                                image_bytes = image_response.read()
                                image_obj = Image.open(io.BytesIO(image_bytes))

                                # Display image with caption styling
                                st.image(image_obj, use_column_width=True)
                                st.markdown("<div style='text-align:center; color:#516f90; font-size:0.9rem; margin-bottom:1.5rem;'>Figure: Data visualization generated by AI analysis</div>", unsafe_allow_html=True)
                            except Exception as img_err:
                                st.warning(f"Unable to display a visualization: {img_err}")

            if not assistant_responded:
                st.warning("No analysis results were returned. Please try again.")

            # Close the card div
            st.markdown("</div>", unsafe_allow_html=True)

            # Clean up resources
            try:
                client.beta.assistants.delete(assistant.id)
                client.files.delete(file_id)
            except Exception as cleanup_err:
                pass  # Silently handle cleanup errors

            return True

        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.code(traceback.format_exc()) # Add traceback for debugging
            return False

    def create_sample_dataframe():
        """
        Create a sample dataset for demonstration purposes.
        """
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'sales': np.random.randint(100, 500, size=30),
            'customers': np.random.randint(10, 50, size=30),
            'advertising_spend': np.random.randint(50, 200, size=30),
            'region': np.random.choice(['North', 'South', 'East', 'West'], size=30),
            'promotion': np.random.choice([True, False], size=30)
        })
        df['sales_per_customer'] = df['sales'] / df['customers']
        # Add some potential quality issues for demo
        df.loc[np.random.choice(df.index, 3, replace=False), 'sales'] = np.nan
        df.loc[np.random.choice(df.index, 2, replace=False), 'customers'] = 0 # Avoid division by zero
        df['customers'] = df['customers'].replace(0, 1) # Replace 0 with 1 after check
        df['sales_per_customer'] = df['sales'] / df['customers']
        return df

    # App Header with logo
    st.markdown("""
    <div class="app-header">
        <h1>📊 DataInsight <span style="color:#ff7a59;">Pro</span></h1>
        <p style="color:#516f90; font-size:1.1rem;">Powered by AI-driven data analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Main container for instructions
    st.markdown("""
    <div class="card">
        <p style="font-size:1.05rem; line-height:1.6;">
            DataInsight Pro uses advanced AI to analyze your business data and uncover actionable insights.
            Simply upload your dataset or use our sample data to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Step 1: Data Source Selection
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="step-number">1</span>Select Your Data</div>', unsafe_allow_html=True)

    data_option = st.radio(
        "Choose data source:", # Add a label
        ["Use demo dataset", "Upload your own data"],
        horizontal=True,
        key="data_source_radio"
    )

    df = None

    # Handle data options
    if data_option == "Use demo dataset":
        st.markdown("<h6>Business Sales Demo Dataset</h6>", unsafe_allow_html=True)
        if 'demo_df' not in st.session_state:
            st.session_state.demo_df = create_sample_dataframe()
        df = st.session_state.demo_df

        # Layout with columns
        col1, col2 = st.columns([3,1])
        with col1:
            st.dataframe(df.head(), use_container_width=True)
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.markdown('<div class="secondary-button">', unsafe_allow_html=True)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="business_sales_data.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("""
            <div style="font-size:0.85rem; color:#516f90; margin-top:1rem;">
            <strong>About this dataset:</strong><br>
            30 days of sales data with customer counts, advertising spend, regional breakdown, and promotion status. Includes sample missing values.
            </div>
            """, unsafe_allow_html=True)

    else: # Upload your own data
        st.markdown("<h6>Upload Your Business Data</h6>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag and drop your file here or click to browse",
            type=["csv", "xlsx", "json"],
            label_visibility="collapsed",
            key="file_uploader"
        )

        if uploaded_file is not None:
            # Use session state to store uploaded df
            if 'uploaded_df' not in st.session_state or st.session_state.get('uploaded_file_name') != uploaded_file.name:
                 try:
                    # Read file based on type
                    if uploaded_file.name.endswith('.csv'):
                        st.session_state.uploaded_df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.xlsx'):
                        st.session_state.uploaded_df = pd.read_excel(uploaded_file)
                    elif uploaded_file.name.endswith('.json'):
                        st.session_state.uploaded_df = pd.read_json(uploaded_file)
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.success(f"✅ Successfully loaded: {uploaded_file.name}")
                 except Exception as e:
                    st.error(f"Error reading file: {e}")
                    if 'uploaded_df' in st.session_state: del st.session_state.uploaded_df
                    if 'uploaded_file_name' in st.session_state: del st.session_state.uploaded_file_name
            
            # Assign df if it exists in session state
            if 'uploaded_df' in st.session_state:
                df = st.session_state.uploaded_df
                st.dataframe(df.head(), use_container_width=True)
                
                # Display quick stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", f"{df.shape[0]:,}")
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    # Safely count unique data types
                    try:
                        num_dtypes = len(df.dtypes.unique())
                    except:
                        num_dtypes = "N/A" 
                    st.metric("Data Types", num_dtypes)

        else: # No file uploaded yet
             if 'uploaded_df' in st.session_state: del st.session_state.uploaded_df # Clear previous upload
             if 'uploaded_file_name' in st.session_state: del st.session_state.uploaded_file_name
             st.markdown("""
                <div class="info-box" style="padding:1rem; border-radius:6px;">
                    <div style="display:flex; align-items:start;">
                        <div style="margin-right:10px; font-size:1.5rem;">ℹ️</div>
                        <div>
                            <p style="margin:0; font-size:0.9rem;">
                                <strong>Supported file types:</strong> CSV, Excel (xlsx), and JSON<br>
                                <strong>Recommended size:</strong> Less than 200MB for optimal performance
                            </p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


    st.markdown('</div>', unsafe_allow_html=True)  # Close the data selection card

    # Step 2: Analysis Section & Results Display Area
    analysis_results_area = st.container() # Container for analysis button and results

    with analysis_results_area:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><span class="step-number">2</span>Generate Analysis</div>', unsafe_allow_html=True)

        if df is not None:
            st.markdown("""
            <p style="margin-bottom:1.5rem;">
            Click below to start your AI-powered data analysis. The system will examine your data and generate
            insights, visualizations, and recommendations tailored to your dataset.
            </p>
            """, unsafe_allow_html=True)
            
            # Display the current model choice
            st.markdown(f"""
            <div style="display: flex; align-items: center; padding: 0.5rem 0.75rem; background-color: #f5f8fa; border-radius: 4px; margin-bottom: 1rem; font-size: 0.9rem;">
                <span style="margin-right: 0.5rem; color: #516f90;">🤖</span>
                <span>Analysis will be performed using <strong>{st.session_state.model_choice}</strong> with a custom prompt (editable in sidebar)</span>
            </div>
            """, unsafe_allow_html=True)

            analyze_button = st.button("🚀 Run Business Analysis", use_container_width=True, key="analyze_button")

            if analyze_button:
                # Clear previous results if any
                if 'analysis_done' in st.session_state:
                    del st.session_state['analysis_done'] 
                    
                with st.spinner("AI analysis in progress... Please wait."):
                    analysis_successful = use_openai_assistant(df.copy()) # Pass a copy
                    if analysis_successful:
                        st.session_state['analysis_done'] = True
                        st.success("Analysis complete!")
                    else:
                        st.error("Analysis failed. Please check the error message above.")
                        if 'analysis_done' in st.session_state:
                             del st.session_state['analysis_done'] 

        else: # No dataframe available
            st.markdown("""
            <div class="info-box" style="padding:1rem; border-radius:6px;">
                <div style="display:flex; align-items:start;">
                    <div style="margin-right:10px; font-size:1.5rem;">👆</div>
                    <div>
                        <p style="margin:0;">Please select or upload data in Step 1 to enable analysis.</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # Close the analysis trigger card

    # Footer (ensure it's at the bottom)
    st.markdown("<hr style='border-top: 1px solid #eaf0f6; margin-top: 3rem;'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="footer">
        <p>© 2025 DataInsight Pro • Powered by <a href="https://openai.com/" target="_blank">OpenAI</a> & <a href="https://streamlit.io/" target="_blank">Streamlit</a></p>
        <p style="font-size:0.8rem; margin-top:0.5rem;">
            Need custom analytics for your business? <a href="#">Contact us</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
