import streamlit as st
import httpx
import json
from datetime import datetime
from pydantic import BaseModel
import time
from httpx import TimeoutException, ConnectError
import requests

st.set_page_config(page_title="Chatbot Platform", layout="wide")

# Initialize session state variables
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "token" not in st.session_state:
    st.session_state.token = None
if "current_chatbot" not in st.session_state:
    st.session_state.current_chatbot = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False

# Add these constants at the top
API_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 30.0  # 30 seconds
UPLOAD_TIMEOUT = 120.0  # 2 minutes

class ChatbotUpdate(BaseModel):
    name: str
    description: str
    temperature: float
    system_prompt: str
    prompt_template: str

# Create a helper function for API calls
def make_api_request(method, endpoint, **kwargs):
    url = f"{API_URL}{endpoint}"
    timeout = kwargs.pop('timeout', DEFAULT_TIMEOUT)
    
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.request(method, url, **kwargs)
            return response
    except TimeoutException:
        st.error("The server is taking too long to respond. Please try again.")
        return None
    except ConnectError:
        st.error("Could not connect to the server. Please check if the server is running.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def get_auth_header():
    """Get the authentication header using the token from session state."""
    if not st.session_state.token:
        raise Exception("No authentication token found")
    return {
        "Authorization": f"Bearer {st.session_state.token}",
        "Content-Type": "application/json"
    }

def signup():
    st.title("Sign Up")
    with st.form("signup_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        full_name = st.text_input("Full Name")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Sign Up")
        
        if submit:
            if password != confirm_password:
                st.error("Passwords do not match!")
                return
                
            # Call the register API
            try:
                response = httpx.post(
                    "http://localhost:8000/api/auth/register",
                    json={
                        "username": username,
                        "email": email,
                        "full_name": full_name,
                        "password": password
                    }
                )
                
                if response.status_code == 200:
                    st.success("Account created successfully! Please login.")
                    st.session_state.show_signup = False
                    st.experimental_rerun()
                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    st.error(f"Failed to create account: {error_detail}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

def login():
    st.title("Welcome to Chatbot Platform")
    
    # Add tabs for Login and Sign Up
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    # Login Tab
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                response = make_api_request(
                    'POST',
                    '/api/auth/login',
                    data={"username": username, "password": password}
                )
                
                if response and response.status_code == 200:
                    st.session_state.token = response.json()["access_token"]
                    st.session_state.authenticated = True
                    st.session_state.current_page = "home"
                    st.experimental_rerun()
                elif response:  # If we got a response but status code wasn't 200
                    st.error("Invalid credentials")
    
    # Sign Up Tab
    with tab2:
        with st.form("signup_form"):
            new_username = st.text_input("Username")
            email = st.text_input("Email")
            full_name = st.text_input("Full Name")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            signup_submit = st.form_submit_button("Sign Up")
            
            if signup_submit:
                if new_password != confirm_password:
                    st.error("Passwords do not match!")
                    return
                
                response = make_api_request(
                    'POST',
                    '/api/auth/register',
                    json={
                        "username": new_username,
                        "email": email,
                        "full_name": full_name,
                        "password": new_password
                    }
                )
                
                if response and response.status_code == 200:
                    st.success("Account created successfully! Please login.")
                    time.sleep(2)
                    st.experimental_rerun()
                elif response:
                    error_detail = response.json().get("detail", "Unknown error")
                    st.error(f"Failed to create account: {error_detail}")

def create_chatbot():
    st.subheader("Create New Chatbot")
    
    with st.form("create_chatbot_form"):
        name = st.text_input("Chatbot Name")
        description = st.text_area("Description", "Default description")
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7)
        
        if st.form_submit_button("Create Chatbot"):
            if not name:
                st.error("Please enter a name for your chatbot")
                return
                
            try:
                # Get the current user's username from session state
                user_id = st.session_state.get('username', 'default_user')
                
                response = make_api_request(
                    'POST',
                    '/api/chatbots',
                    headers=get_auth_header(),
                    json={
                        "name": name,
                        "description": description,
                        "temperature": temperature,
                        "user_id": user_id
                    }
                )
                
                if response and response.status_code == 201:
                    st.success("Chatbot created successfully!")
                    # Store the new chatbot's ID and name in session state
                    chatbot_data = response.json()
                    st.session_state.current_chatbot = chatbot_data["id"]
                    st.session_state.current_chatbot_name = name
                    # Redirect to upload documents page instead of home
                    st.session_state.current_page = "upload_documents"
                    st.experimental_rerun()
                elif response:
                    st.error(f"Failed to create chatbot: {response.text}")
                else:
                    st.error("Failed to get response from server")
            except Exception as e:
                st.error(f"Error creating chatbot: {str(e)}")

def configure_bot_page():
    st.title("Configure Your Chatbot")
    
    with st.form("configure_bot_form"):
        # Basic Configuration
        name = st.text_input("Name", st.session_state.current_chatbot_name)
        description = st.text_area("Description", "Enter your chatbot's description here")
        
        # Advanced Configuration
        st.subheader("Chatbot Behavior")
        
        # Predefined templates
        template_options = {
            "Custom": "Write your own custom prompt",
            "Professional Assistant": """You are a professional AI assistant. Format responses as:
📌 **Key Points:**
- Clear, concise bullet points
- Professional language
- Factual information

📌 **Analysis:**
[Detailed professional analysis]""",
            "Teacher": """You are a patient teacher. Format responses as:
📌 **Simple Explanation:**
[Easy to understand explanation]

📌 **Examples:**
1. [First example]
2. [Second example]

📌 **Practice Question:**
[Related question for learning]""",
            "Technical Expert": """You are a technical expert. Format responses with:
📌 **Technical Summary:**
[Technical explanation]

📌 **Code Example:** (if applicable)"""
        }

        selected_template = st.selectbox(
            "Choose a Prompt Template",
            options=list(template_options.keys())
        )

        if selected_template == "Custom":
            system_prompt = st.text_area(
                "Custom System Prompt",
                value="""You are a helpful AI assistant. Format your responses as:

📌 **Answer:**
[Your answer here]

📌 **Explanation:**
- [Point 1]
- [Point 2]
- [Point 3]""",
                height=300
            )
        else:
            system_prompt = template_options[selected_template]
            st.text_area("Selected Template (Read Only)", value=system_prompt, height=300, disabled=True)

        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.0,
            value=0.7
        )

        # Preview Section
        if st.checkbox("Show Preview"):
            st.markdown("#### Response Preview")
            preview_context = "The sky appears blue due to Rayleigh scattering."
            preview_query = "Why is the sky blue?"
            preview = f"""System: {system_prompt}

Context: {preview_context}

Question: {preview_query}"""
            st.code(preview)

        configure_button = st.form_submit_button("Save Configuration")
        
        if configure_button:
            try:
                headers = {
                    "Authorization": f"Bearer {st.session_state.token}",
                    "Content-Type": "application/json"
                }
                
                update_data = {
                    "name": name,
                    "description": description,
                    "temperature": temperature,
                    "system_prompt": system_prompt,
                    "prompt_template": """System: {system_prompt}

Context: {context}

Question: {query}"""
                }
                
                response = make_api_request(
                    'PUT',
                    f'/api/chatbots/{st.session_state.current_chatbot}',
                    headers=headers,
                    json=update_data
                )
                
                if response and response.status_code == 200:
                    st.success("Configuration saved successfully!")
                    st.session_state.current_page = "upload_documents"
                    st.experimental_rerun()
                else:
                    st.error(f"Failed to update configuration: {response.text if response else 'No response from server'}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

def upload_documents_page():
    st.title("Upload Documents")
    
    # File Upload Section
    st.header("Upload Files")
    st.info("Supported formats: PDF, TXT, DOCX (Max size: 10MB)")
    
    uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'docx'])
    
    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "FileType": uploaded_file.type,
            "Size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.write("File Details:", file_details)
        
        if st.button("Upload and Process File"):
            try:
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Processing file..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    
                    # Start upload
                    response = make_api_request(
                        'POST',
                        f'/api/chatbots/{st.session_state.current_chatbot}/documents',
                        headers={"Authorization": f"Bearer {st.session_state.token}"},
                        files=files,
                        timeout=300.0
                    )
                    
                    if response and response.status_code == 200:
                        result = response.json()
                        
                        # Poll for progress
                        while True:
                            status_response = make_api_request(
                                'GET',
                                f'/api/chatbots/{st.session_state.current_chatbot}',
                                headers={"Authorization": f"Bearer {st.session_state.token}"}
                            )
                            
                            if status_response and status_response.status_code == 200:
                                status_data = status_response.json()
                                progress = status_data.get('progress', 0)
                                status = status_data.get('processing_status', '')
                                
                                # Update progress bar and status
                                progress_bar.progress(int(progress))
                                status_text.text(f"Status: {status}")
                                
                                if status == 'completed':
                                    st.success(f"Successfully processed {uploaded_file.name}")
                                    st.write(f"Chunks processed: {result.get('chunks_processed', 0)}")
                                    st.write(f"Vectors uploaded: {result.get('vectors_uploaded', 0)}")
                                    break
                                elif status == 'failed':
                                    st.error(f"Processing failed: {status_data.get('error_message', 'Unknown error')}")
                                    break
                                
                                time.sleep(1)  # Poll every second
                    else:
                        error_msg = response.json() if response else "No response from server"
                        st.error(f"Error processing file: {error_msg}")
                        
            except Exception as e:
                st.error(f"Error uploading file: {str(e)}")
    
    # URL Processing Section
    st.header("Process URL")
    st.info("Enter a URL to extract and process its content")
    url = st.text_input("Enter URL:")
    
    if url:
        if st.button("Process URL"):
            with st.spinner("Processing URL..."):
                try:
                    headers = {
                        "Authorization": f"Bearer {st.session_state.token}",
                        "Content-Type": "application/json"
                    }
                    
                    response = make_api_request(
                        'POST',
                        f'/api/chatbots/{st.session_state.current_chatbot}/urls',
                        headers=headers,
                        json={"url": url},
                        timeout=UPLOAD_TIMEOUT
                    )
                    
                    if response and response.status_code == 200:
                        st.success("URL processed successfully!")
                    else:
                        error_msg = response.json() if response else "No response from server"
                        st.error(f"Error processing URL: {error_msg}")
                
                except Exception as e:
                    st.error(f"Error processing URL: {str(e)}")
    
    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back to Configuration"):
            st.session_state.current_page = "configure_bot"
            st.experimental_rerun()
    with col2:
        if st.button("🏠 Go to Home"):
            st.session_state.current_page = "home"
            st.experimental_rerun()
    with col3:
        if st.button("Start Chatting →"):
            st.session_state.current_page = "chat"
            st.experimental_rerun()

    # Display uploaded documents
    st.markdown("---")
    st.header("Uploaded Documents")
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        response = make_api_request(
            'GET',
            f'/api/chatbots/{st.session_state.current_chatbot}',
            headers=headers
        )
        
        if response and response.status_code == 200:
            chatbot = response.json()
            if chatbot.get('documents'):
                for doc in chatbot['documents']:
                    st.text(f"📄 {doc}")
            else:
                st.info("No documents uploaded yet")
        else:
            st.warning("Could not fetch uploaded documents")
    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")

def chat_page():
    st.title("Chat with Your Bot")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response with timeout handling
        try:
            with st.spinner("Thinking..."):
                response = httpx.post(
                    f"http://localhost:8000/api/chatbots/{st.session_state.current_chatbot}/chat",
                    headers=headers,
                    json={"text": prompt},
                    timeout=45.0
                )
            
            if response.status_code == 200:
                response_data = response.json()
                bot_response = response_data["response"]
                
                # Add context indicator if no context was found
                if not response_data.get("context_used", True):
                    bot_response = "⚠️ *No relevant context found for this query.*\n\n" + bot_response
                
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                with st.chat_message("assistant"):
                    st.markdown(bot_response)
            else:
                error_message = "I apologize, but I encountered an error. Please try again."
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                with st.chat_message("assistant"):
                    st.error(error_message)
        
        except httpx.TimeoutException:
            timeout_message = "I apologize, but I'm taking too long to respond. Please try asking your question again."
            st.session_state.messages.append({"role": "assistant", "content": timeout_message})
            with st.chat_message("assistant"):
                st.warning(timeout_message)
        
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant"):
                st.error(error_message)

def home_page():
    st.title("Your Chatbots")
    
    if st.button("Create New Chatbot"):
        st.session_state.current_page = "create_bot"
        st.experimental_rerun()
    
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    response = make_api_request('GET', '/api/chatbots', headers=headers)
    
    if response and response.status_code == 200:
        chatbots = response.json()
        if not chatbots:
            st.info("You don't have any chatbots yet. Create one to get started!")
        for chatbot in chatbots:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"### {chatbot['name']}")
                st.write(chatbot['description'])
            with col2:
                if st.button("Chat", key=f"chat_{chatbot['id']}"):
                    st.session_state.current_chatbot = chatbot['id']
                    st.session_state.current_page = "chat"
                    st.experimental_rerun()

def main():
    if not st.session_state.authenticated:
        login()
    else:
        # Sidebar for navigation
        with st.sidebar:
            st.title("Navigation")
            if st.button("Home"):
                st.session_state.current_page = "home"
                st.experimental_rerun()
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.token = None
                st.session_state.current_chatbot = None
                st.session_state.current_page = "home"
                st.experimental_rerun()
        
        # Main content based on current page
        if st.session_state.current_page == "home":
            home_page()
        elif st.session_state.current_page == "create_bot":
            create_chatbot()
        elif st.session_state.current_page == "configure_bot":
            configure_bot_page()
        elif st.session_state.current_page == "upload_documents":
            upload_documents_page()
        elif st.session_state.current_page == "chat":
            chat_page()

if __name__ == "__main__":
    main() 