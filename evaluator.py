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
    estimated_yearly_expenses: typing.Optional[int]
    gross_rental_yield_percentage: typing.Optional[float]
    listing_price_per_sqm: typing.Optional[int]
    market_avg_price_per_sqm: typing.Optional[int]
    pr_eligibility_status: str
    best_investment_strategy: str
    red_flags: typing.List[str]
    legal_disclaimer: str
    location_intelligence_summary: typing.Optional[str]

def evaluate_valuation(session_memory, status_placeholder=None, retries=2, property_category="Residential", image=None, drive_times=None, title_deeds_status=None, vat_status=None, structural_dampness=None, roof_waterproofing=None, mep_status=None, energy_efficiency=None, unauthorized_extensions=None, capex_estimate=0, developer_track_record=None, construction_stage=None, mep_climate_specs=None, solar_pv_system=None, planning_deviations=None, legal_doc_text=None, inspection_image=None, nearby_cafes=0, nearby_restaurants=0, nearby_parks=0, target_language="English"):
    """
    Evaluator with Router Pattern: Uses Gemini to analyze session memory and optional image based on property category.
    """
    if not api_key:
        return {"error": "GOOGLE_API_KEY not found. Please set it to use the Evaluator."}
    
    pr_law = (
        "Cyprus PR Immigration Law: You must determine if this property qualifies for the Cyprus Fast-Track Permanent Residency. "
        "The strict rules are: The purchase price must be €300,000 or higher, AND the property MUST be a brand new build (or off-plan/under construction). "
        "Resales strictly DO NOT qualify. If it clearly meets both rules, output 'Eligible for Fast-Track PR 🇨🇾'. "
        "If it is a resale or under €300k, output 'Not Eligible for PR'. "
        "If the price is >= €300k but it is unclear if it is new, output 'Price qualifies, but must verify New Build status'."
    )
    
    location_intel_instruction = (
        "Location Intelligence (Drive Times): You will be provided with real-time drive times to key destinations. "
        "If the drive times are missing or empty, DO NOT penalize the score. Only penalize the score if the drive times are successfully provided AND they are over 20 minutes. "
        "If penalized, explicitly mention these exact drive times in your Red Flags and Final Justification."
    )
    
    neighborhood_vibe_instruction = (
        "Neighborhood Vibe & Walkability: You must analyze the amenity counts provided (cafes, restaurants, parks within 1km). "
        "If there are many cafes and restaurants, explicitly boost the score and highlight the area as a vibrant, walkable neighborhood with high tenant demand for young professionals or short-term rentals. "
        "If the counts are near zero, classify it as a car-dependent, quiet residential area, which may lower short-term rental yields but appeal to families. "
        "Always mention the specific amenity counts in your final justification."
    )

    agent_override_instruction = (
        "Agent Overrides: If the agent marks Title Deeds as 'Issued & Clean' or 'Pending (Trusted Developer)', YOU MUST NOT penalize the score for Title Deed risks, even if it is a new build. "
        "If the agent specifies a VAT status (5% or 19%), you MUST use that exact percentage for your final investment cost calculations and remove any Red Flags about VAT uncertainty."
    )

    technical_assessment_instruction = (
        "Technical & Structural Assessment: If the agent provides a Hard CapEx Estimate > 0, you MUST add that exact amount to the purchase price before calculating ROI. "
        "You MUST evaluate the structural, MEP, and legal extension inputs. Heavily penalize the score for Concrete Spalling, Failing Roofs, or Major Unauthorized Extensions, "
        "and explicitly mention these technical risks in the justification."
    )

    new_build_assessment_instruction = (
        "New Build Assessment: If the property is a new build, heavily weight the Developer Track Record. "
        "Penalize the score for Tier 3 or High Risk developers. If MEP Specs are 'Basic Provisions Only' or Solar is 'PV Provisions Only', "
        "you MUST explicitly warn the user about hidden completion costs in the Red Flags. If Major Planning Deviations exist, flag an immediate Title Deed delay risk."
    )

    multimodal_instruction = (
        "Multimodal Analysis: If an inspection photo is provided, you MUST perform a visual structural and technical analysis. "
        "Override the user's condition rating if the photo shows severe damage (like spalling, dampness) or poor developer finishes. "
        "If legal document text is provided, you MUST act as a Cyprus real estate lawyer. Scan the text for encumbrances, "
        "VAT clauses, or Title Deed issues, and override any user assumptions with the hard legal facts found in the document."
    )

    translation_instruction = (
        f"CRITICAL RULE: You MUST output all string values in your final JSON response (including Justification, Red Flags, Strategy, and Disclaimer) "
        f"entirely in the requested {target_language}. The JSON keys must remain in English so the code parses correctly, "
        f"but the actual text the user reads must be translated fluently into {target_language} using professional real estate terminology."
    )

    # Select System Instruction based on Category
    if property_category == "Land":
        system_instruction = (
            "You are a Cyprus Land Development Expert. Evaluate this plot of land. "
            "You MUST focus heavily on Building Density (Συντελεστής Δόμησης), Coverage Ratio, zoning laws, and VAT on land sales. "
            "Ignore rental yield and monthly rent. Grade the plot based on development potential from 1 to 100. "
            "Analyze the feasibility of construction and potential profit upon development completion. "
            "Aggressively hunt for 'Area market analysis' or 'Average price per m2' for land in the scraped text to fill market_avg_price_per_sqm. "
            "If missing, estimate it based on the city and zoning potential. "
            "If an image is provided, analyze the terrain, access roads, and neighboring structures to verify the description.\n"
            + pr_law + "\n" + location_intel_instruction + "\n" + neighborhood_vibe_instruction + "\n" + agent_override_instruction + "\n" + new_build_assessment_instruction + "\n" + multimodal_instruction + "\n" + translation_instruction
        )
    else: # Default to Residential or Commercial
        system_instruction = (
            "You are a ruthless real estate investor based in Cyprus. You evaluate properties in Euros (€) and size in square meters (sqm). "
            "You heavily scrutinize cash flow, ROI, and local market dynamics. "
            "You must penalize listings heavily if they do not clarify the status of the Title Deeds or VAT. "
            "You must treat the Agent Insider Knowledge as the absolute truth. "
            "Grade the deal from 1 to 100 based purely on financial value in the Cyprus market. "
            "Realistically estimate the monthly rent and calculate the gross rental yield percentage. "
            "Estimate realistic Cyprus yearly expenses (communal fees typically €50-150/month for apartments, refuse tax ~€150-200/year, "
            "and maintenance calculated as 0.5% to 2% of the price based on the condition rating). "
            "Aggressively hunt for 'Area market analysis' or 'Average price per m2' in the scraped text to fill market_avg_price_per_sqm. "
            "If missing, estimate it based on the city and property type (e.g. Limassol apartments avg ~4500/sqm). "
            "If an image is provided, analyze the visual condition of the property. Brutally penalize the score if the image shows mold, "
            "deterioration, or poor maintenance that the marketing text tries to hide or downplay.\n"
            + pr_law + "\n" + location_intel_instruction + "\n" + neighborhood_vibe_instruction + "\n" + agent_override_instruction + "\n" + technical_assessment_instruction + "\n" + new_build_assessment_instruction + "\n" + multimodal_instruction + "\n" + translation_instruction
        )

    # Retrieve legal facts from RAG
    legal_facts = retrieve_context(session_memory.initial_description)
    
    history_str = "\n".join([f"{h['role']}: {h['message']}" for h in session_memory.history])
    
    prompt_content = [f"""
    SYSTEM: {system_instruction}
    
    Analyze the following property data and session history:
    - Initial Description: {session_memory.initial_description}
    - Follow-up Clarifications: {session_memory.follow_ups}
    - Market Data (Neighborhood Avg): {session_memory.market_data}
    - Agent Condition Rating: {getattr(session_memory, 'condition_rating', 5)}/10
    - Agent Location Rating: {getattr(session_memory, 'location_rating', 5)}/10
    - Agent Insider Knowledge: {getattr(session_memory, 'insider_knowledge', 'None provided')}
    - Title Deeds Status: {title_deeds_status if title_deeds_status else 'Unknown'}
    - VAT Status: {vat_status if vat_status else 'Unknown'}
    - Real-Time Drive Times: {drive_times if drive_times else 'No location verified for drive times.'}
    - Nearby Amenities (Within 1km): {nearby_cafes} cafes, {nearby_restaurants} restaurants, {nearby_parks} parks
    
    Technical & Structural Assessment Details:
    - Structural & Dampness: {structural_dampness}
    - Roof Waterproofing: {roof_waterproofing}
    - MEP Status: {mep_status}
    - Energy Efficiency: {energy_efficiency}
    - Unauthorized Extensions: {unauthorized_extensions}
    - Agent Estimated CapEx / Repair Cost: €{capex_estimate}
    
    Technical Assessment (New Builds / Off-Plan) Details:
    - Developer Track Record: {developer_track_record}
    - Construction Stage: {construction_stage}
    - MEP & Climate Specs: {mep_climate_specs}
    - Solar / PV System: {solar_pv_system}
    - Planning Deviations: {planning_deviations}
    
    Extracted Legal Document Text (if any):
    {legal_doc_text if legal_doc_text else 'None provided'}

    Verified Cyprus Legal & Market Context:
    {legal_facts}
    
    Full Session History:
    {history_str}
    
    TASK:
    1. Analyze the listing price against market potential.
    2. Factor in the user's clarifications and agent's ratings.
    3. Calculate a valuation score from 1 to 100.
    4. For Land: Focus on Building Density and Coverage. For Residential: Estimate rent and yield.
    5. Estimate yearly operating expenses (communal, taxes, maintenance).
    6. Calculate listing_price_per_sqm (Total Price / Total Size).
    7. Extract or estimate market_avg_price_per_sqm for this specific area/type.
    8. Recommend the best investment strategy.
    9. List any legal/financial red flags (VAT, Title Deeds, Zoning, etc. and Technical Risks).
    10. Generate a legal_disclaimer aggressively warning about Cyprus-specific risks.
    11. Determine PR eligibility status based on the Cyprus PR Immigration Law.
    12. Include a summary of drive times in 'location_intelligence_summary' and ensure they are used in the score and justification as instructed.
    13. Analyze amenity counts and reflect their impact on the score and justification as per the Neighborhood Vibe & Walkability instructions.
    
    IMPORTANT: You MUST base your legal warnings and VAT math strictly on the Verified Context provided.
    
    The output must strictly be a JSON object matching the requested schema. The justification should explain the brutal truth in under 50 words.
    """]
    
    if image:
        prompt_content.append(image)
    if inspection_image:
        prompt_content.append(inspection_image)

    # Step 1: Draft Generation
    draft_data = None
    for attempt in range(retries + 1):
        try:
            # Enforce JSON output and schema strictly at the API generation level
            response = model.generate_content(
                prompt_content,
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
