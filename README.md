# HealthNet AI Chatbot

🤖 **HealthNet AI** is a Streamlit-powered chatbot application that integrates with the **Lambda Labs API** to generate AI-driven responses and treatment suggestions for user-provided symptoms. The app provides interactive conversations and suggests follow-up questions for enhanced user engagement.

---

## Features

- **Symptom Analysis**: Input symptoms to receive AI-generated insights and treatment suggestions.
- **Follow-up Questions**: The chatbot intelligently generates follow-up questions to better understand symptoms.
- **Customizable AI Model**: Choose from multiple AI models (e.g., `llama3.1-70b-instruct-berkeley`, `llama3.2-13b-instruct`, and more).
- **Conversation History**: View and maintain a history of interactions with the chatbot.
- **Clear History**: Reset the chat interface with a single click.

---

## Prerequisites

Before running the application, ensure the following requirements are met:

1. **Python**: Python 3.8 or above
2. **Environment Variables**:
   - Set up a `.env` file in the root directory containing your Lambda Labs API key:
     ```env
     LAMBDALABS_API_KEY=your_api_key_here
     ```
3. **Dependencies**: Install required libraries using `pip`.

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   **Dependencies Include:**
   - `streamlit`
   - `requests`
   - `python-dotenv`

3. Add your API key in `.env` as shown above.

---

## Usage

Run the Streamlit application using the following command:
```bash
streamlit run <filename>.py
```

### Application Walkthrough:

1. **Select AI Model**: Use the sidebar to choose from available models.
2. **Configure Parameters**: Adjust `max_tokens` (output length) and `temperature` (creativity) via sliders.
3. **Chat with AI**:
   - Enter your symptoms in the text area.
   - Click **"Generate Response"** to receive AI-generated suggestions.
4. **Follow-up Questions**:
   - If the AI detects a need for more information, it will prompt relevant follow-up questions.
5. **View History**: Review previous user-AI interactions in the conversation history section.
6. **Clear Chat**: Reset the session by clicking **"Clear Conversation"**.

---

## Configuration

### Environment Variables
The application uses the Lambda Labs API, requiring an API key. Store this key securely in a `.env` file.

```env
LAMBDALABS_API_KEY=your_api_key_here
```

### Available AI Models
The app supports the following models:
- `llama3.1-70b-instruct-berkeley`
- `llama3.2-13b-instruct`
- `llama2-7b-chat`

You can expand or modify the model list in the sidebar configuration section.

---

## Code Structure

- **Environment Setup**: Loads API keys from `.env` using `dotenv`.
- **HealthNetAI Class**:
  - Handles API requests to Lambda Labs for text generation.
  - Extracts treatment-related suggestions from AI-generated responses.
- **Follow-up Question Logic**: Analyzes responses to suggest follow-up questions.
- **Streamlit UI**:
  - Input: Accepts user symptoms and follow-up responses.
  - Output: Displays AI responses, treatment suggestions, and history.

---

## Error Handling

- **APIConnectionError**: Custom exception raised for API connection or authentication issues.
- **Validation**: Ensures proper API key setup and valid user input before generating responses.

---

## Example Output

1. **User Input**: "I have a fever and headache."
2. **AI Response**:
   > "It seems like you might have flu-like symptoms. Consider consulting a healthcare professional for further analysis."
3. **Suggested Treatment**:
   > "Consider consulting a healthcare professional."
4. **Follow-up Question**:
   > "Do you also have a cough or sore throat?"

---

## Troubleshooting

- **API Key Error**: Ensure your `LAMBDALABS_API_KEY` is correctly stored in `.env`.
- **Missing Dependencies**: Run `pip install -r requirements.txt` to install all dependencies.
- **Streamlit Issues**: Ensure Streamlit is installed and up to date:
  ```bash
  pip install --upgrade streamlit
  ```

---

## Contributing

Feel free to open issues or submit pull requests for improvements. Contributions are welcome!

---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

## Author

[Your Name]  
Email: your.email@example.com

---

## Acknowledgments

- **Lambda Labs** for providing the API.
- **Streamlit** for creating an interactive UI framework.

---

Enjoy using HealthNet AI! 🚀
