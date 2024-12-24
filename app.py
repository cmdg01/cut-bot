import streamlit as st
from anthropic import Anthropic
from datetime import datetime
import json
import plotly.express as px
import pandas as pd
import anthropic

# Initialize Anthropic client with API key from Streamlit secrets
client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# Page configuration
st.set_page_config(
    page_title="CUT Virtual Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Improved CSS with dark mode support
st.markdown("""
<style>
    /* Base styles with CSS variables for theming */
    :root {
        --background-primary: #ffffff;
        --background-secondary: #f8f9fa;
        --text-primary: #1a1a1a;
        --text-secondary: #495057;
        --accent-primary: #1f4068;
        --accent-secondary: #162b47;
        --user-message-bg: #e3eeff;
        --bot-message-bg: #f8f9fa;
        --border-color: #e0e0e0;
        --shadow-color: rgba(0,0,0,0.1);
    }

    /* Dark mode styles */
    @media (prefers-color-scheme: dark) {
        :root {
            --background-primary: #1a1a1a;
            --background-secondary: #2d2d2d;
            --text-primary: #ffffff;
            --text-secondary: #cccccc;
            --accent-primary: #4a89dc;
            --accent-secondary: #357abd;
            --user-message-bg: #2c3e50;
            --bot-message-bg: #34495e;
            --border-color: #404040;
            --shadow-color: rgba(0,0,0,0.2);
        }
    }

    /* Main container styles */
    .main {
        background-color: var(--background-primary);
        color: var(--text-primary);
    }

    /* Button styles */
    .stButton>button {
        background-color: var(--accent-primary);
        color: #ffffff;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        border: none;
        transition: all 0.3s ease;
        font-weight: 500;
        box-shadow: 0 2px 4px var(--shadow-color);
    }

    .stButton>button:hover {
        background-color: var(--accent-secondary);
        transform: translateY(-1px);
        box-shadow: 0 4px 6px var(--shadow-color);
    }

    /* Chat message styles */
    .chat-message {
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 2px 8px var(--shadow-color);
        border: 1px solid var(--border-color);
        transition: transform 0.2s ease;
    }

    .chat-message:hover {
        transform: translateY(-2px);
    }

    .user-message {
        background-color: var(--user-message-bg);
        border-left: 4px solid var(--accent-primary);
    }

    .bot-message {
        background-color: var(--bot-message-bg);
        border-left: 4px solid var(--text-secondary);
    }

    /* Usage stats panel */
    .usage-stats {
        font-size: 0.9rem;
        color: var(--text-secondary);
        padding: 1rem;
        background-color: var(--background-secondary);
        border-radius: 8px;
        border: 1px solid var(--border-color);
        margin-top: 1rem;
    }

    /* Typography */
    .stTitle {
        color: var(--accent-primary);
        font-weight: 700;
        margin-bottom: 1.5rem;
    }

    .stSubheader {
        color: var(--text-secondary);
        font-weight: 500;
        margin-bottom: 1rem;
    }

    /* Sidebar improvements */
    .sidebar .sidebar-content {
        background-color: var(--background-secondary);
        padding: 1rem;
        border-radius: 8px;
    }

    /* Quick links */
    .quick-link {
        display: block;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        background-color: var(--background-primary);
        border-radius: 6px;
        color: var(--text-primary);
        text-decoration: none;
        transition: all 0.2s ease;
        border: 1px solid var(--border-color);
    }

    .quick-link:hover {
        background-color: var(--accent-primary);
        color: #ffffff;
        transform: translateX(5px);
    }

    /* Feedback buttons */
    .feedback-button {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: 1px solid var(--border-color);
        background-color: var(--background-secondary);
        color: var(--text-primary);
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .feedback-button:hover {
        background-color: var(--accent-primary);
        color: #ffffff;
    }

    /* Chat input */
    .stTextInput>div>div>input {
        background-color: var(--background-secondary);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }

    /* Selectbox */
    .stSelectbox>div>div {
        background-color: var(--background-secondary);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--background-secondary);
        color: var(--text-primary);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }

    /* Plotly chart background */
    .js-plotly-plot .plotly .main-svg {
        background-color: var(--background-secondary) !important;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--background-secondary);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--accent-primary);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-secondary);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat_counter' not in st.session_state:
    st.session_state.chat_counter = 0
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []

def load_conversation_prompts():
    """Load predefined conversation prompts"""
    return {
        "Academic Programs": [
            "What programs are offered in the School of Engineering?",
            "Tell me about the MBA program",
            "What are the entry requirements for undergraduate programs?"
        ],
        "Fees & Payments": [
            "What are the tuition fees for engineering programs?",
            "How can I pay my fees?",
            "Are there any payment plans available?"
        ],
        "Events & Schedule": [
            "When is the next orientation week?",
            "What events are happening this month?",
            "Show me the examination schedule"
        ],
        "Student Services": [
            "How do I access the library resources?",
            "What sports facilities are available?",
            "Tell me about student accommodation options"
        ]
    }

def get_claude_response(prompt):
    """Get response from Claude API with improved error handling"""
    try:
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            temperature=0.7,
            system="You are a helpful virtual assistant for Chinhoyi University of Technology. Provide accurate and relevant information about the university's programs, policies, and services. Be concise but informative.",
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        return message.content[0].text
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return "I apologize, but I'm having trouble processing your request at the moment. Please try again in a few moments."

def display_chat_history():
    """Display chat history with enhanced styling and animations"""
    for idx, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            st.markdown(f"""
                <div class="chat-message user-message">
                    <div style="font-weight: 500; margin-bottom: 0.5rem;">You:</div>
                    <div>{message["content"]}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message bot-message">
                    <div style="font-weight: 500; margin-bottom: 0.5rem;">Assistant:</div>
                    <div>{message["content"]}</div>
                </div>
            """, unsafe_allow_html=True)
            get_feedback(message["content"], idx)

def get_feedback(response, idx):
    """Collect user feedback with improved UI"""
    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        if st.button("👍", key=f"positive_{idx}", help="This response was helpful"):
            st.session_state.feedback_data.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "response": response,
                "feedback": "positive"
            })
            st.success("Thank you for your positive feedback!")
    with col2:
        if st.button("👎", key=f"negative_{idx}", help="This response needs improvement"):
            st.session_state.feedback_data.append({
                "timestamp": datetime.now().strftime("%Y-%m--%d %H:%M:%S"),
                "response": response,
                "feedback": "negative"
            })
            st.error("Thank you for your feedback. We'll work on improving.")

def show_usage_stats():
    """Display usage statistics with enhanced visualizations"""
    if st.session_state.chat_counter > 0:
        st.sidebar.markdown("### Chat Statistics")
        st.sidebar.markdown(f"""
            <div class="usage-stats">
                <div style="margin-bottom: 0.5rem;">
                    <strong>Messages sent:</strong> {st.session_state.chat_counter}
                </div>
                <div>
                    <strong>Session duration:</strong> {datetime.now().strftime('%H:%M:%S')}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.feedback_data:
            df = pd.DataFrame(st.session_state.feedback_data)
            feedback_counts = df['feedback'].value_counts()
            
            # Create a more aesthetically pleasing pie chart
            fig = px.pie(
                values=feedback_counts.values,
                names=feedback_counts.index,
                title='Response Feedback Distribution',
                color_discrete_map={'positive': '#28a745', 'negative': '#dc3545'},
                hole=0.3
            )
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=30, l=0, r=0, b=0)
            )
            st.sidebar.plotly_chart(fig, use_container_width=True)

def main():
    # Main layout with improved structure
    st.title("🎓 Chinhoyi University of Technology")
    st.subheader("Virtual Assistant")
    
    # Sidebar with enhanced layout
    with st.sidebar:
        st.image("https://via.placeholder.com/150", caption="CUT Logo")
        
        # Quick navigation with styled links
        st.markdown("### Quick Links")
        st.markdown("""
            <a href="http://student.cut.ac.zw" target="_blank" class="quick-link">
                📚 Student Portal
            </a>
            <a href="http://elearning.cut.ac.zw" target="_blank" class="quick-link">
                💻 E-Learning Platform
            </a>
            <a href="http://library.cut.ac.zw" target="_blank" class="quick-link">
                📖 Library Resources
            </a>
        """, unsafe_allow_html=True)
        
        # Suggested prompts with improved UI
        st.markdown("### Suggested Questions")
        prompts = load_conversation_prompts()
        selected_category = st.selectbox("Select a category:", list(prompts.keys()))
        
        if selected_category:
            selected_prompt = st.selectbox("Choose a question:", prompts[selected_category])
            if st.button("Ask Question", key="suggested_prompt"):
                st.session_state.chat_history.append({"role": "user", "content": selected_prompt})
                with st.spinner("Thinking..."):
                    response = get_claude_response(selected_prompt)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.session_state.chat_counter += 1
                st.rerun()
        
        show_usage_stats()
        
        # Enhanced help section
        with st.expander("Help & Tips"):
            st.markdown("""
                ### How to use the chatbot:
                1. Type your question in the chat input below
                2. Browse suggested questions by category
                3. Use quick links for direct access to resources
                4. Provide feedback to help improve responses
                
                ### Tips for better results:
                - Be specific in your questions
                - Use complete sentences
                - Check suggested questions for examples
                - Give feedback to help improve the system
                
                ### Need more help?
                Contact support at support@cut.ac.zw
            """)
    
    # Main chat interface
    chat_container = st.container()
    with chat_container:
        display_chat_history()
    
    # User input with improved styling
    user_query = st.chat_input("Ask me anything about CUT...", key="chat_input")
    
    if user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Get and display assistant response
        with st.spinner("Thinking..."):
            response = get_claude_response(user_query)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Update chat counter
        st.session_state.chat_counter += 1
        st.rerun()

if __name__ == "__main__":
    main()
        
