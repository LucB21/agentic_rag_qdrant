import streamlit as st
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from agentic_rag.main import RagFlow, GuideOutline

# Configure Streamlit page
st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'question_submitted' not in st.session_state:
    st.session_state.question_submitted = False
if 'response' not in st.session_state:
    st.session_state.response = None
if 'question' not in st.session_state:
    st.session_state.question = ""
if 'selected_topic' not in st.session_state:
    st.session_state.selected_topic = ""
if 'show_confirmation' not in st.session_state:
    st.session_state.show_confirmation = False
if 'force_continue' not in st.session_state:
    st.session_state.force_continue = False

def reset_form():
    """Reset the form and go back to the main page"""
    st.session_state.question_submitted = False
    st.session_state.response = None
    st.session_state.question = ""
    st.session_state.show_confirmation = False
    st.session_state.force_continue = False
    st.rerun()

def confirmation_page():
    """Display the confirmation page when question is not in topic"""
    st.title("🤖 Agentic RAG System - Topic Confirmation")
    st.markdown("---")
    
    # Display question and topic
    st.markdown("### Your Question")
    st.info(f"**Topic:** {st.session_state.selected_topic}")
    st.write(f"**Question:** {st.session_state.question}")
    
    st.markdown("---")
    
    # Warning message
    st.warning("⚠️ Your question doesn't seem to be related to the selected topic.")
    st.markdown("### What would you like to do?")
    
    # Add disclaimer on confirmation page too
    st.info("💡 **Note**: Continuing anyway will generate an AI response that may not be accurate for off-topic questions.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("**Option 1:** Continue with this question anyway")
        if st.button("🚀 Continue Anyway", type="primary", use_container_width=True):
            st.session_state.force_continue = True
            st.session_state.show_confirmation = False
            st.session_state.response = None  # Reset response to reprocess
            st.rerun()
        
        st.markdown("**Option 2:** Ask a different question")
        if st.button("🔙 Ask Different Question", type="secondary", use_container_width=True):
            reset_form()

def process_question(question: str, topic: str, force_continue: bool = False):
    """Process the user question using the RagFlow"""
    try:
        # Initialize the flow
        flow = RagFlow()
        
        # Set the question and sector, and skip relevance check if force_continue is True
        flow.set_question_and_sector(question, [topic], skip_relevance_check=force_continue)
        
        # Start the flow
        flow.get_user_question()
        
        # Evaluate relevance
        relevance_result = flow.evaluate_relevance()
        
        if relevance_result == "failed" and not force_continue:
            # Only show confirmation if this is the first attempt (not force_continue)
            return {
                "error": False,
                "relevance_failed": True,
                "message": f"Your question doesn't seem to be related to the topic '{topic}'."
            }
        
        # Execute RAG search
        flow.rag_search()
        
        # Execute web search
        flow.web_search()
        
        # Synthesize information
        result = flow.synthesize_information()
        
        return {
            "error": False,
            "relevance_failed": False,
            "result": result
        }
        
    except Exception as e:
        return {
            "error": True,
            "relevance_failed": False,
            "message": f"An error occurred while processing your question: {str(e)}"
        }

def main_page():
    """Display the main page with topic selection and question input"""
    st.title("🤖 Agentic RAG System")
    st.markdown("---")
    
    # Center column for better layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Ask a Question")
        
        # Disclaimer before question input
        st.info("⚠️ **Disclaimer**: This system uses AI to generate answers. The responses may contain inaccuracies and should be verified from authoritative sources.")
        
        # Topic selection dropdown
        topics = ["Financial Services", "Energy"]  # Based on your main.py sectors
        selected_topic = st.selectbox(
            "Select a topic:",
            topics,
            index=0,
            help="Choose the topic area for your question"
        )
        
        st.session_state.selected_topic = selected_topic
        
        # Question input
        question = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="Type your question here...",
            help="Ask anything related to the selected topic"
        )
        
        st.session_state.question = question
        
        # Submit button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            if st.button("🚀 Submit Question", type="primary", use_container_width=True):
                if question.strip():
                    st.session_state.question_submitted = True
                    st.session_state.force_continue = False  # Reset for new questions
                    st.session_state.response = None  # Reset response cache
                    st.rerun()
                else:
                    st.error("Please enter a question before submitting.")

def results_page():
    """Display the results page"""
    st.title("🤖 Agentic RAG System - Results")
    st.markdown("---")
    
    # Back button
    if st.button("← Back to Main Page", type="secondary"):
        reset_form()
        return
    
    # Display question and topic
    st.markdown("### Your Question")
    st.info(f"**Topic:** {st.session_state.selected_topic}")
    st.write(f"**Question:** {st.session_state.question}")
    
    st.markdown("---")
    
    # Process the question if not already processed
    if st.session_state.response is None:
        with st.spinner("🔍 Processing your question... This may take a moment."):
            response = process_question(
                st.session_state.question, 
                st.session_state.selected_topic, 
                st.session_state.force_continue
            )
            st.session_state.response = response
    
    # Display results
    response = st.session_state.response
    
    if response["error"]:
        st.error(response["message"])
        if st.button("Try Again", type="primary"):
            reset_form()
    elif response.get("relevance_failed", False):
        # Show confirmation dialog
        st.session_state.show_confirmation = True
        st.rerun()
    else:
        st.markdown("### 🎯 Answer")
        
        result = response["result"]
        
        # Display the final output
        if "output" in result and result["output"]:
            # Create tabs for different information
            tab1, tab2, tab3 = st.tabs(["📝 Final Answer", "📚 RAG Context", "🔍 Search Summary"])
            
            with tab1:
                st.markdown("#### Final Synthesized Answer")
                try:
                    # Handle different types of CrewAI output
                    if hasattr(result["output"]["answer"], 'raw'):
                        st.markdown(result["output"]["answer"].raw)
                    elif hasattr(result["output"]["answer"], 'content'):
                        st.markdown(result["output"]["answer"].content)
                    elif isinstance(result["output"]["answer"], dict):
                        if 'final_output' in result["output"]["answer"]:
                            st.markdown(result["output"]["answer"]['final_output'])
                        else:
                            st.json(result["output"]["answer"])
                    else:
                        st.markdown(str(result["output"]["answer"]))
                except Exception as e:
                    st.error(f"Error displaying output: {str(e)}")
                    st.json(result["output"]["answer"])
                
                # Disclaimer after AI-generated answer
                st.warning("⚠️ **AI-Generated Content Disclaimer**: This answer was generated by artificial intelligence and may contain inaccuracies, outdated information, or biases. Please verify the information from reliable, authoritative sources before making any decisions based on this content.")
            
            with tab2:
                st.markdown("#### Retrieved Context from RAG")
                if result["chunk"]:
                    if isinstance(result["chunk"], dict):
                        # Display in a more readable format
                        for key, value in result["chunk"].items():
                            st.markdown(f"**{key}:**")
                            st.text(str(value))
                            st.markdown("---")
                    else:
                        st.text(str(result["chunk"]))
                else:
                    st.info("No RAG context available.")
            
            with tab3:
                st.markdown("#### Web Search Summary")
                if result["search_summary"]:
                    if isinstance(result["search_summary"], dict):
                        st.json(result["search_summary"])
                    else:
                        st.markdown(str(result["search_summary"]))
                else:
                    st.info("No web search summary available.")
        else:
            st.warning("No output generated from the synthesis process.")
        
        # Option to ask another question
        st.markdown("---")
        if st.button("🔄 Ask Another Question", type="primary"):
            reset_form()

def main():
    """Main Streamlit application"""
    # Apply custom CSS for better styling
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton > button {
        width: 100%;
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
    .stTextArea > div > div {
        background-color: #f0f2f6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
    }
    .stAlert {
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Route to appropriate page based on session state
    if st.session_state.show_confirmation:
        confirmation_page()
    elif st.session_state.question_submitted:
        results_page()
    else:
        main_page()

if __name__ == "__main__":
    main()