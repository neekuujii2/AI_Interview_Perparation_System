import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
print("Loaded key:", os.environ.get("GOOGLE_API_KEY"))

# Load API key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in environment variables.")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Default model (adjust as needed)
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.0-flash")


def get_response_from_llm(prompt: str) -> str:
    """
    Calls Google Gemini and returns raw string response.
    """
    try:
        model = genai.GenerativeModel(LLM_MODEL)

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=1000,
            )
        )

        return response.text

    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {e}")


def parse_json_response(response: str):
    """
    Safely parse JSON from Gemini response.
    Automatically strips code fences and ignores extra text.
    """
    try:
        clean = (
            response.replace("```json", "")
            .replace("```", "")
            .strip()
        )

        # Find first JSON object if model added extra text
        if "{" in clean and "}" in clean:
            json_str = clean[clean.index("{"): clean.rindex("}") + 1]
            return json.loads(json_str)

        return None

    except Exception:
        return None
