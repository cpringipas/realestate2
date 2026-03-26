import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import typing_extensions as typing

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# Initialize the model with system instructions
system_instruction = (
    "You are a ruthless real estate investor based in Cyprus. You evaluate properties in Euros (€) and size in square meters (sqm). "
    "You heavily scrutinize cash flow, ROI, and local market dynamics (e.g., Limassol is expensive but has high rental demand, Nicosia is stable, Paphos is tourist-driven). "
    "You must penalize listings heavily if they do not clarify the status of the Title Deeds or VAT. "
    "Grade the deal from 1 to 100 based purely on financial value in the Cyprus market."
)
model = genai.GenerativeModel('gemini-3.0-flash', system_instruction=system_instruction)

# Define the expected JSON schema using TypedDict
class ValuationResult(typing.TypedDict):
    score: int
    justification: str

def evaluate_valuation(session_memory, retries=2):
    """
    Evaluator with Guardrails: Uses Gemini to analyze session memory and provide a 
    valuation score in a strict JSON format. Retries if the schema is broken.
    """
    if not api_key:
        return {"error": "GOOGLE_API_KEY not found. Please set it to use the Evaluator."}
    
    history_str = "\n".join([f"{h['role']}: {h['message']}" for h in session_memory.history])
    
    prompt = f"""
    Analyze the following property data and session history:
    - Initial Description: {session_memory.initial_description}
    - Follow-up Clarifications: {session_memory.follow_ups}
    - Market Data (Neighborhood Avg): {session_memory.market_data}
    
    Full Session History:
    {history_str}
    
    TASK:
    1. Analyze the listing price against the market average.
    2. Factor in the user's clarifications.
    3. Calculate a valuation score from 1 to 100 based on the investment value.
    
    The output must strictly be a JSON object with two keys: 'score' (an integer) and 'justification' (a string explaining the brutal truth in under 50 words).
    """

    for attempt in range(retries + 1):
        try:
            # Enforce JSON output and schema strictly at the API generation level
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=ValuationResult,
                )
            )
            data = json.loads(response.text)
            
            # Ensure score is within bounds
            if "score" in data and "justification" in data:
                data["score"] = max(1, min(100, int(data["score"])))
                return data
            else:
                raise ValueError("Missing required JSON keys.")
                
        except (json.JSONDecodeError, ValueError, Exception) as e:
            if attempt == retries:
                return {
                    "score": 0, 
                    "justification": f"Error: Failed to generate a valid valuation after retries. ({str(e)})"
                }
            continue # Retry
