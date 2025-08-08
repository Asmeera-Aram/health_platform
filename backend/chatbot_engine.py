# chatbot_engine.py

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma"  # or "gemma:2b" or any other local model you loaded

def get_health_advice(disease):
    """
    Sends the disease name to Ollama and returns the recommended advice.
    """
    prompt = (
        f"I have been diagnosed with {disease}. "
        "Please provide a detailed explanation of the condition, common symptoms, "
        "preventive measures, and treatment recommendations in simple language."
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data['response'].strip()

    except requests.exceptions.RequestException as e:
        return f"❌ Error connecting to Ollama: {e}"
    except Exception as e:
        return f"❌ Unexpected error: {e}"

# Test it directly
if __name__ == "__main__":
    disease_input = input("Enter diagnosed disease: ")
    print("\n💡 Health Advice from Gemma:\n")
    print(get_health_advice(disease_input))
