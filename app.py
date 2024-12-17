import os
import logging
import requests
from dotenv import load_dotenv
from typing import List, Optional
import streamlit as st

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom Exceptions
class APIConnectionError(Exception):
    """Custom exception for API connection and authentication issues."""
    pass

class HealthNetAI:
    def __init__(self, 
                 api_base_url: str = "https://api.lambdalabs.com/v1",
                 api_key: Optional[str] = None,
                 model: str = "llama3.1-70b-instruct-berkeley"):
        """
        Initialize HealthNet AI with Lambda Labs API.

        Args:
            api_base_url (str): Base URL for the API endpoint
            api_key (Optional[str]): API authentication key
            model (str): Model to use for API calls
        
        Raises:
            APIConnectionError: If API initialization fails
        """
        self.api_key = api_key or os.getenv('LAMBDALABS_API_KEY')
        if not self.api_key:
            raise APIConnectionError(
                "No API key found. Set LAMBDALABS_API_KEY in .env or pass directly."
            )
        
        self.base_url = api_base_url
        self.model = model

    def _verify_api_connection(self):
        """
        Verify initial connection to the Lambda Labs API.
        
        Raises:
            APIConnectionError: If connection or authentication fails
        """
        try:
            # Endpoint to check API connection (example)
            response = requests.get(
                f"{self.base_url}/models/{self.model}",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()  # Raise an error for invalid responses
            logger.info("Successfully connected to Lambda Labs API")
        except requests.exceptions.RequestException as e:
            error_msg = f"API Connection Error: {e}"
            logger.error(error_msg)
            raise APIConnectionError(error_msg)

    def get_diagnosis(self, symptoms: str) -> List[str]:
        """
        Generate medical diagnoses by calling Lambda Labs API.

        Args:
            symptoms (str): Comma-separated patient symptoms

        Returns:
            List[str]: Potential medical diagnoses
        """
        if not symptoms or not isinstance(symptoms, str):
            raise ValueError("Invalid symptoms input")

        prompt = (
            f"HealthNet AI Diagnostic Analysis\n"
            f"Patient Symptoms: {symptoms}\n\n"
            "Provide a detailed medical diagnosis. For each potential condition, include:\n"
            "1. Diagnosis name\n"
            "2. Likelihood based on symptoms\n"
            "3. Brief clinical reasoning\n\n"
            "Response format:\n"
            "Diagnosis: [Condition]\n"
            "Likelihood: [Low/Medium/High]\n"
            "Reasoning: [Explanation]\n\n"
        )

        try:
            response = requests.post(
                f"{self.base_url}/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": 300,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_return_sequences": 3
                }
            )
            response.raise_for_status()  # Raise an error for invalid responses
            results = response.json().get("choices", [])
            diagnoses = [result.get('text').strip() for result in results if result.get('text') and len(result.get('text')) > 50]
            return diagnoses
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return []

class HealthAdvisor:
    def __init__(self):
        """
        Initialize medical advisory system with predefined guidelines.
        """
        self._test_recommendations = {
            "Hypothyroidism": [
                "Thyroid Stimulating Hormone (TSH)",
                "Free T4 Hormone",
                "Comprehensive Metabolic Panel"
            ],
            "default": [
                "Complete Blood Count",
                "Basic Metabolic Panel",
                "Comprehensive Health Screening"
            ]
        }
        self._treatment_guidelines = {
            "Hypothyroidism": (
                "Comprehensive Treatment Plan:\n"
                "- Synthetic thyroid hormone replacement\n"
                "- Regular thyroid function monitoring\n"
                "- Dietary and lifestyle modifications"
            ),
            "default": (
                "General Medical Advisory:\n"
                "- Consult healthcare professional\n"
                "- Comprehensive diagnostic evaluation\n"
                "- Personalized treatment planning"
            )
        }

    def get_tests(self, diagnosis: str) -> List[str]:
        """
        Recommend diagnostic tests based on potential condition.
        
        Args:
            diagnosis (str): Identified medical condition
        
        Returns:
            List[str]: Recommended diagnostic tests
        """
        return self._test_recommendations.get(diagnosis, self._test_recommendations["default"])

    def get_treatment(self, diagnosis: str) -> str:
        """
        Provide treatment recommendations.
        
        Args:
            diagnosis (str): Identified medical condition
        
        Returns:
            str: Detailed treatment advisory
        """
        return self._treatment_guidelines.get(diagnosis, self._treatment_guidelines["default"])

def main_cli():
    try:
        healthnet_ai = HealthNetAI()
        health_advisor = HealthAdvisor()

        # Prompt user for symptom input
        print("Enter symptoms (comma-separated, e.g., 'fatigue, weight gain, dry skin'):")
        symptoms = input("Symptom Set: ")
        
        if not symptoms.strip():
            print("No symptoms entered. Exiting...")
            return
        
        print(f"\n🩺 HealthNet AI Diagnostic Analysis: {symptoms}")
        print("-" * 50)
        try:
            diagnoses = healthnet_ai.get_diagnosis(symptoms)
            if not diagnoses:
                print("No diagnoses could be generated.")
                return
            for idx, diagnosis in enumerate(diagnoses, 1):
                print(f"\nDiagnosis {idx}:\n{diagnosis}")
                condition = diagnosis.split('\n')[0].split(': ')[-1].strip()
                tests = health_advisor.get_tests(condition)
                treatment = health_advisor.get_treatment(condition)
                print("\nRecommended Tests:")
                for test in tests:
                    print(f"  ✔️ {test}")
                print("\nTreatment Advisory:")
                print(treatment)
        except ValueError as ve:
            logger.error(f"Invalid input: {ve}")
    except APIConnectionError as ace:
        logger.critical(f"API Connection Failed: {ace}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

def main_streamlit():
    st.title("HealthNet AI Diagnostic Assistant")
    
    st.write(
        "This tool uses Lambda Labs' HealthNet AI to generate diagnoses based on your symptoms. "
        "Simply enter your symptoms, and it will provide potential diagnoses, recommended tests, and treatment advice."
    )
    
    # Input box for symptoms
    symptoms = st.text_input("Enter symptoms (comma-separated, e.g., 'fatigue, weight gain, dry skin')")

    if symptoms:
        try:
            healthnet_ai = HealthNetAI()
            health_advisor = HealthAdvisor()

            st.write(f"🩺 **HealthNet AI Diagnostic Analysis**: {symptoms}")
            st.markdown("-" * 50)
            diagnoses = healthnet_ai.get_diagnosis(symptoms)
            
            if not diagnoses:
                st.write("No diagnoses could be generated.")
            else:
                for idx, diagnosis in enumerate(diagnoses, 1):
                    st.subheader(f"Diagnosis {idx}:")
                    st.write(diagnosis)
                    
                    # Extract condition from the diagnosis
                    condition = diagnosis.split('\n')[0].split(': ')[-1].strip()
                    
                    # Get recommended tests and treatments
                    tests = health_advisor.get_tests(condition)
                    treatment = health_advisor.get_treatment(condition)
                    
                    st.write("### Recommended Tests:")
                    for test in tests:
                        st.write(f"✔️ {test}")
                    
                    st.write("### Treatment Advisory:")
                    st.write(treatment)
        except APIConnectionError as ace:
            st.error(f"API Connection Failed: {ace}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    # Uncomment one of the following lines depending on how you want to run the app
    # For CLI-based interaction
    main_cli()

    # For Streamlit-based interaction
    # main_streamlit()
