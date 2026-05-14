import os
import sys
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    print("The 'openai' library is not installed. Please install it using: pip install openai")
    sys.exit(1)

# Load environment variables from a .env file if it exists
load_dotenv()

SYSTEM_PROMPT = """
You are a helpful, friendly, and empathetic medical assistant. 
Your goal is to answer general health-related queries in a clear, easy-to-understand manner.

CRITICAL SAFETY GUIDELINES (YOU MUST FOLLOW THESE):
1. You are an AI, NOT a licensed medical professional.
2. DO NOT provide specific medical diagnoses for the user's specific symptoms.
3. DO NOT prescribe medications, recommend specific dosages, or tell the user to take a specific medication for their condition.
4. ALWAYS include a clear disclaimer in your response stating that your advice is for informational purposes only and that the user should consult a qualified healthcare provider for medical advice, diagnoses, or treatment.
5. If the user's query indicates a potential medical emergency (e.g., severe chest pain, difficulty breathing, suicidal thoughts), immediately advise them to seek emergency medical attention (like calling 911 or visiting the nearest ER).

Always respond warmly and professionally.
"""

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not found.")
        print("Please set it in your environment or in a .env file.")
        api_key = input("Alternatively, enter your OpenAI API key now: ").strip()
        if not api_key:
            print("API key is required to run this chatbot. Exiting.")
            sys.exit(1)
            
    # Set the key in the environment for the OpenAI client to pick up
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI()

def check_local_safety_filters(user_input):
    """A simple keyword-based safety filter before sending to the LLM."""
    danger_words = ['suicide', 'kill myself', 'heart attack', 'stroke', 'emergency', 'chest pain', "can't breathe"]
    user_input_lower = user_input.lower()
    
    for word in danger_words:
        if word in user_input_lower:
            return True
    return False

def run_chatbot():
    print("Initializing General Health Query Chatbot...\n")
    client = get_openai_client()
    
    print("-" * 50)
    print("Welcome to the General Health Query Chatbot!")
    print("I can answer general health questions.")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("-" * 50)
    print("Example queries:")
    print("  - 'What causes a sore throat?'")
    print("  - 'Is paracetamol safe for children?'")
    print("-" * 50)
    
    # Initialize conversation history with the system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("\nChatbot: Stay healthy! Goodbye.")
                break
                
            if not user_input:
                continue

            # 1. Local Safety Filter
            if check_local_safety_filters(user_input):
                print("\nChatbot: [SAFETY ALERT] It sounds like you might be experiencing a medical emergency. Please call emergency services (e.g., 911) or go to the nearest emergency room immediately.")
                continue

            # Add user message to history
            messages.append({"role": "user", "content": user_input})
            
            # 2. Call OpenAI API
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )
                
                bot_response = response.choices[0].message.content
                print(f"\nChatbot: {bot_response}")
                
                # Add bot response to history to maintain context
                messages.append({"role": "assistant", "content": bot_response})
                
            except Exception as api_err:
                print(f"\nChatbot Error: Failed to get a response from the API. ({api_err})")
                # Remove the failed user message from history
                messages.pop()
                
        except KeyboardInterrupt:
            print("\n\nChatbot: Stay healthy! Goodbye.")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            break

if __name__ == "__main__":
    run_chatbot()
