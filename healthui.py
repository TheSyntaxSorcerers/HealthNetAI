import os
import logging
import requests
from dotenv import load_dotenv
from typing import Optional
import streamlit as st

# ----------------------------
# Environment Setup
# ----------------------------
load_dotenv()  # Load environment variables from .env
LAMBDALABS_API_KEY = os.getenv("LAMBDALABS_API_KEY")

if not LAMBDALABS_API_KEY:
    raise ValueError("LAMBDALABS_API_KEY is not set. Check your .env file.")

# ----------------------------
# Environment and Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------------
# Custom Exceptions
# ----------------------------
class APIConnectionError(Exception):
    """Custom exception for API connection and authentication issues."""
    pass

# ----------------------------
# HealthNet AI (API Integration)
# ----------------------------
class HealthNetAI:
    def __init__(self, 
                 api_base_url: str = "https://api.lambdalabs.com/v1",
                 api_key: Optional[str] = None,
                 model: str = "llama3.1-70b-instruct-berkeley"):
        self.api_key = api_key or os.getenv("LAMBDALABS_API_KEY")
        self.api_base_url = api_base_url
        self.model = model
        if not self.api_key:
            raise APIConnectionError("API key is missing. Set it via environment variable LAMBDALABS_API_KEY.")
        
    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> tuple:
        """Generate a response from the HealthNetAI model."""
        try:
            logger.info("Sending request to Lambda Labs API.")
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            response = requests.post(f"{self.api_base_url}/completions", headers=headers, json=data)
            
            if response.status_code == 200:
                logger.info("Received response from Lambda Labs API.")
                ai_text = response.json().get("choices", [{}])[0].get("text", "No response received.")
                treatment_suggestion = self.extract_treatment(ai_text)
                return ai_text, treatment_suggestion
            else:
                logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                raise APIConnectionError(f"API request failed with status code {response.status_code}.")
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise APIConnectionError(f"API request failed: {e}")
    
    def extract_treatment(self, ai_text: str) -> str:
        """Simple method to extract treatment-related suggestions from AI's response."""
        if "consult" in ai_text.lower():
            return "Consider consulting a healthcare professional."
        elif "take" in ai_text.lower():
            return "Consider taking over-the-counter medications or prescribed drugs."
        else:
            return "Further analysis or consultation is recommended."

# ----------------------------
# Follow-up Question Function
# ----------------------------
def get_follow_up_question(response: str) -> Optional[str]:
    """Generate a follow-up question based on the AI's response."""
    if "fever" in response.lower():
        return "Do you also have a cough or sore throat?"
    elif "pain" in response.lower():
        return "Where exactly is the pain located?"
    elif "nausea" in response.lower():
        return "Are you experiencing vomiting along with the nausea?"
    else:
        return "Can you provide more details about your symptoms?"

# ----------------------------
# Streamlit Application
# ----------------------------
def main():
    st.set_page_config(page_title="HealthNet AI Chatbot", layout="wide")
    st.title("🤖 HealthNet AI Chatbot")
    st.write("Generate insights and responses using HealthNet AI.")

    # Sidebar: Model and Parameters
    st.sidebar.header("🔧 Configuration")
    available_models = [
        "llama3.1-70b-instruct-berkeley", 
        "llama3.2-13b-instruct", 
        "llama2-7b-chat"
    ]
    selected_model = st.sidebar.selectbox("Select Model", available_models)

    max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=1024, value=512, step=50)
    temperature = st.sidebar.slider("Temperature (Creativity)", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

    # Initialize conversation state
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "follow_up_question" not in st.session_state:
        st.session_state.follow_up_question = None

    # Input area
    st.write("### 💬 Chat with the AI")
    user_input = st.text_area("Enter your symptoms:", "")

    # Handle follow-up responses
    if st.session_state.follow_up_question:
        follow_up_response = st.text_area(st.session_state.follow_up_question)
        if st.button("Submit Follow-Up Response"):
            if not follow_up_response:
                st.warning("Please enter a response to the follow-up question.")
            else:
                st.session_state.conversation.append({"user": follow_up_response, "ai": "Processing..."})
                st.session_state.follow_up_question = None
                process_input(follow_up_response, selected_model, max_tokens, temperature)
    else:
        if st.button("Generate Response"):
            if not user_input:
                st.warning("Please enter a symptom before generating a response.")
            else:
                process_input(user_input, selected_model, max_tokens, temperature)

    # Display conversation history
    st.write("### 🗂️ Conversation History")
    for idx, turn in enumerate(st.session_state.conversation):
        st.write(f"**You:** {turn['user']}")
        st.write(f"**AI:** {turn['ai']}")
        st.write("---")

    # Clear conversation history
    if st.button("Clear Conversation"):
        st.session_state.conversation = []
        st.session_state.follow_up_question = None
        st.success("Conversation history cleared.")

def process_input(user_input, model, max_tokens, temperature):
    """Process user input and update the conversation."""
    try:
        healthnet_ai = HealthNetAI(model=model)
        conversation_context = "\n".join(
            [f"User: {c['user']}\nAI: {c['ai']}" for c in st.session_state.conversation]
        )
        full_prompt = f"{conversation_context}\nUser: {user_input}\nAI:"
        response, treatment = healthnet_ai.generate_response(full_prompt, max_tokens, temperature)

        # Update conversation
        st.session_state.conversation.append({"user": user_input, "ai": response})
        st.success("Response Generated:")
        st.write(response)
        st.write("### Suggested Treatment:")
        st.write(treatment)

        # Follow-up question
        follow_up_question = get_follow_up_question(response)
        if follow_up_question:
            st.session_state.follow_up_question = follow_up_question
            st.write(f"### AI: {follow_up_question}")

    except APIConnectionError as e:
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
