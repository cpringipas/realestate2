import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import typing_extensions as typing
from rag_system import retrieve_context

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# Initialize the model
model = genai.GenerativeModel('gemini-2.5-flash')

# Define the expected JSON schema using TypedDict
class ValuationResult(typing.TypedDict):
    score: int
    justification: str
    estimated_monthly_rent: typing.Optional[int]
    gross_rental_yield_percentage: typing.Optional[float]
    best_investment_strategy: str
    red_flags: typing.List[str]
    legal_disclaimer: str

def evaluate_valuation(session_memory, status_placeholder=None, retries=2, property_category="Residential"):
    """
    Evaluator with Router Pattern: Uses Gemini to analyze session memory based on property category.
    """
    if not api_key:
        return {"error": "GOOGLE_API_KEY not found. Please set it to use the Evaluator."}
    
    # Select System Instruction based on Category
    if property_category == "Land":
        system_instruction = (
            "You are a Cyprus Land Development Expert. Evaluate this plot of land. "
            "You MUST focus heavily on Building Density (Συντελεστής Δόμησης), Coverage Ratio, zoning laws, and VAT on land sales. "
            "Ignore rental yield and monthly rent. Grade the plot based on development potential from 1 to 100. "
            "Analyze the feasibility of construction and potential profit upon development completion."
        )
        response_schema = ValuationResult
    else: # Default to Residential or Commercial
        system_instruction = (
            "You are a ruthless real estate investor based in Cyprus. You evaluate properties in Euros (€) and size in square meters (sqm). "
            "You heavily scrutinize cash flow, ROI, and local market dynamics. "
            "You must penalize listings heavily if they do not clarify the status of the Title Deeds or VAT. "
            "You must treat the Agent Insider Knowledge as the absolute truth. "
            "Grade the deal from 1 to 100 based purely on financial value in the Cyprus market. "
            "Realistically estimate the monthly rent and calculate the gross rental yield percentage."
        )
        response_schema = ValuationResult

    # Retrieve legal facts from RAG
    legal_facts = retrieve_context(session_memory.initial_description)
    
    history_str = "\n".join([f"{h['role']}: {h['message']}" for h in session_memory.history])
    
    prompt = f"""
    SYSTEM: {system_instruction}
    
    Analyze the following property data and session history:
    - Initial Description: {session_memory.initial_description}
    - Follow-up Clarifications: {session_memory.follow_ups}
    - Market Data (Neighborhood Avg): {session_memory.market_data}
    - Agent Condition Rating: {getattr(session_memory, 'condition_rating', 5)}/10
    - Agent Location Rating: {getattr(session_memory, 'location_rating', 5)}/10
    - Agent Insider Knowledge: {getattr(session_memory, 'insider_knowledge', 'None provided')}
    
    Verified Cyprus Legal & Market Context:
    {legal_facts}
    
    Full Session History:
    {history_str}
    
    TASK:
    1. Analyze the listing price against market potential.
    2. Factor in the user's clarifications and agent's ratings.
    3. Calculate a valuation score from 1 to 100.
    4. For Land: Focus on Building Density and Coverage. For Residential: Estimate rent and yield.
    5. Recommend the best investment strategy.
    6. List any legal/financial red flags (VAT, Title Deeds, Zoning, etc.).
    7. Generate a legal_disclaimer aggressively warning about Cyprus-specific risks.
    
    IMPORTANT: You MUST base your legal warnings and VAT math strictly on the Verified Context provided.
    
    The output must strictly be a JSON object matching the requested schema. The justification should explain the brutal truth in under 50 words.
    """

    # Step 1: Draft Generation
    draft_data = None
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
                draft_data = data
                break
            else:
                raise ValueError("Missing required JSON keys.")
                
        except (json.JSONDecodeError, ValueError, Exception) as e:
            if attempt == retries:
                return {
                    "score": 0, 
                    "justification": f"Error: Failed to generate a valid draft valuation after retries. ({str(e)})",
                    "estimated_monthly_rent": 0,
                    "gross_rental_yield_percentage": 0.0,
                    "best_investment_strategy": "Unknown",
                    "red_flags": [f"Evaluation error: {str(e)}"],
                    "legal_disclaimer": "Unable to provide legal disclaimer due to evaluation error."
                }
            continue # Retry

    # UI Update for Reflection
    if status_placeholder is not None:
        status_placeholder.empty()
        status_placeholder.info('Draft complete. Senior AI Partner is reflecting and verifying the math...')

    # Step 2: Reflection and Verification
    reflection_prompt = f"""
    You are a Senior Real Estate Partner reviewing a junior analyst's draft. Review the provided draft valuation. 
    1. Did it strictly obey the Agent Insider Knowledge? 
    2. Is the Gross Rental Yield mathematically sound based on the estimated rent and purchase price? 
    3. Are the Red Flags and Legal warnings penalized harshly enough in the score? 
    
    Fix any mistakes, adjust the score if necessary, and return the final, polished valuation.
    
    Original Input:
    - Initial Description: {session_memory.initial_description}
    - Follow-up Clarifications: {session_memory.follow_ups}
    - Market Data (Neighborhood Avg): {session_memory.market_data}
    - Agent Condition Rating: {getattr(session_memory, 'condition_rating', 5)}/10
    - Agent Location Rating: {getattr(session_memory, 'location_rating', 5)}/10
    - Agent Insider Knowledge: {getattr(session_memory, 'insider_knowledge', 'None provided')}
    
    Draft Valuation from Junior Analyst:
    {json.dumps(draft_data, indent=2)}
    """

    for attempt in range(retries + 1):
        try:
            response = model.generate_content(
                reflection_prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=ValuationResult,
                )
            )
            final_data = json.loads(response.text)
            
            if "score" in final_data and "justification" in final_data:
                final_data["score"] = max(1, min(100, int(final_data["score"])))
                return final_data
            else:
                raise ValueError("Missing required JSON keys.")
                
        except (json.JSONDecodeError, ValueError, Exception) as e:
            if attempt == retries:
                # If reflection fails, return the draft but note the failure
                draft_data["red_flags"].append(f"Senior Partner review failed: {str(e)}")
                return draft_data
            continue
