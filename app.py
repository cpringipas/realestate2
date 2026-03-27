import streamlit as st
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv
import typing_extensions as typing
from evaluator import evaluate_valuation
import requests

# Load environment variables
load_dotenv()

# Configure the Gemini API
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# Set page title
st.set_page_config(page_title="🏠 Cyprus Real Estate Valuation Agent", layout="centered")

class SessionState:
    """
    Memory Management: Stores the history of the current valuation session.
    """
    def __init__(self):
        self.initial_description = ""
        self.follow_ups = {}
        self.market_data = None
        self.history = []
        self.condition_rating = 5
        self.location_rating = 5
        self.insider_knowledge = ""

    def add_history(self, role, message):
        self.history.append({"role": role, "message": message})

# Mock database for neighborhood average prices
NEIGHBORHOOD_MARKET_DATA = {
    "limassol": 4000,
    "nicosia": 2500,
    "paphos": 2800,
    "larnaca": 2600
}

class ExtractionResult(typing.TypedDict):
    property_type: typing.Optional[str]
    bedrooms: typing.Optional[int]
    bathrooms: typing.Optional[int]
    price: typing.Optional[int]
    size_in_sqm: typing.Optional[int]
    city: typing.Optional[str]

def scrape_listing(url):
    """
    Helper function to fetch a webpage and extract readable text from the body.
    """
    try:
        jina_url = f'https://r.jina.ai/{url}'
        response = requests.get(jina_url, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.error(f"Error scraping listing: {e}")
        return ""

def fetch_market_data(location):
    """
    Tool Use: Fetches average price for a given location from a mock database.
    """
    if not location:
        return "No location provided."
    location_key = location.lower().strip()
    return NEIGHBORHOOD_MARKET_DATA.get(location_key, "No market data available for this location.")

def planner(description):
    """
    Planner function that uses Gemini API to intelligently extract property details.
    Returns a dictionary of extracted data and a list of keys that are missing.
    """
    if not api_key:
        return {
            "property_type": None,
            "bedrooms": None,
            "bathrooms": None,
            "price": None,
            "size_in_sqm": None,
            "city": None
        }, ["API Key missing"]
    
    # Initialize the model with the same version as evaluator.py
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    You are an expert real estate data extractor. Extract the following details from the Cyprus property listing provided below:
    1. property_type (The type of property, e.g., Apartment, House, Villa, Studio, Plot)
    2. bedrooms (The number of bedrooms. Extract only the integer.)
    3. bathrooms (The number of bathrooms. Extract only the integer.)
    4. price (The total price in Euros. Extract only the integer.)
    5. size_in_sqm (The total area in square meters. Extract only the integer.)
    6. city (The city or town in Cyprus, e.g., Limassol, Nicosia, Paphos, Larnaca)

    Listing: "{description}"

    If a field is not present or cannot be determined precisely from the text, return null for that field.
    Be smart: understand conversational phrases like "total square meters of apartment is 130".
    """
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=ExtractionResult,
            )
        )
        extracted_data = json.loads(response.text)
    except Exception:
        # Fallback to empty values if AI extraction fails
        extracted_data = {
            "property_type": None,
            "bedrooms": None,
            "bathrooms": None,
            "price": None,
            "size_in_sqm": None,
            "city": None
        }
    
    missing = []
    # Only trigger warnings for these specific fields if they are null/None
    if not extracted_data.get('property_type'):
        missing.append("property_type")
    if extracted_data.get('bedrooms') is None:
        missing.append("bedrooms")
    if extracted_data.get('bathrooms') is None:
        missing.append("bathrooms")
    if extracted_data.get('price') is None:
        missing.append("price")
    if extracted_data.get('size_in_sqm') is None:
        missing.append("size_in_sqm")
    if not extracted_data.get('city'):
        missing.append("city")
        
    return extracted_data, missing

def route_property_type(text):
    """
    Router Pattern: Classifies the listing text into one of three categories.
    """
    if not api_key:
        return "Residential"
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"""
    Classify the following Cyprus property listing into strictly one of these three categories:
    - Residential (e.g., Apartments, Houses, Villas, Studios)
    - Land (e.g., Plots, Fields, Residential Land)
    - Commercial (e.g., Offices, Shops, Warehouses, Hotels)

    Listing Text: "{text}"

    Output only the category name.
    """
    try:
        response = model.generate_content(prompt)
        category = response.text.strip()
        if category in ["Residential", "Land", "Commercial"]:
            return category
        return "Residential" # Default
    except Exception:
        return "Residential"

def main():
    st.title("🏠 Cyprus Real Estate Valuation Agent")
    
    listing_url = st.text_input("Listing URL (optional, e.g., Bazaraki link):", placeholder="https://www.bazaraki.com/adv/...")
    listing = st.text_area("Or paste your property listing here:", height=200, placeholder="e.g., 2 bedroom apartment in Limassol, 100 sqm, 2 bathrooms...")
    
    with st.expander("Agent On-the-Ground Assessment (Optional)"):
        condition_rating = st.slider("Condition & Finish Quality (1-10)", min_value=1, max_value=10, value=5)
        location_rating = st.slider("Street Vibe & Location Reality (1-10)", min_value=1, max_value=10, value=5)
        insider_knowledge = st.text_area("Agent Insider Knowledge (e.g., Seller motivation, hidden repair costs, realistic closing price)")

    if "evaluate_clicked" not in st.session_state:
        st.session_state.evaluate_clicked = False

    if st.button("Evaluate Deal"):
        st.session_state.evaluate_clicked = True
        # Clear previous extraction results when button is clicked
        if "extracted_data" in st.session_state:
            del st.session_state["extracted_data"]

    if st.session_state.evaluate_clicked:
        # Determine source of text
        if "listing_to_process" not in st.session_state:
            if listing_url:
                with st.spinner("Scraping listing..."):
                    st.session_state.listing_to_process = scrape_listing(listing_url)
            else:
                st.session_state.listing_to_process = listing

        listing_to_process = st.session_state.listing_to_process

        with st.expander('View Raw Scraped Text', expanded=False):
            st.text(listing_to_process)

        if not listing_to_process:
            st.error("Please provide a listing URL or paste the listing text.")
            st.session_state.evaluate_clicked = False
            if "listing_to_process" in st.session_state:
                del st.session_state["listing_to_process"]
            return

        # Router Phase
        if "property_category" not in st.session_state:
            with st.spinner("Routing to specialist..."):
                st.session_state.property_category = route_property_type(listing_to_process)
        
        property_category = st.session_state.property_category
        st.success(f"Routing to: {property_category} Specialist")

        # Intelligent Extraction Phase
        if "extracted_data" not in st.session_state:
            with st.spinner("Analyzing listing details..."):
                st.session_state.extracted_data, st.session_state.missing_info = planner(listing_to_process)
        
        extracted_data = st.session_state.extracted_data
        missing_info = st.session_state.missing_info
        
        # Human in the Loop: Follow-up inputs for missing fields
        follow_ups = {}
        all_provided = True
        
        if missing_info:
            st.warning("Additional information is required to provide an accurate valuation.")
            for info in missing_info:
                label = {
                    "property_type": "Property Type (e.g., Apartment, House)",
                    "bedrooms": "Number of Bedrooms",
                    "bathrooms": "Number of Bathrooms",
                    "price": "Price in Euros",
                    "size_in_sqm": "Size in square meters",
                    "city": "City in Cyprus"
                }.get(info, info)
                
                user_val = st.text_input(f"Please provide the {label}:", key=f"input_{info}")
                if not user_val:
                    all_provided = False
                else:
                    follow_ups[info] = user_val
        
        if all_provided:
            # Combine AI extracted data with manual follow-up inputs
            final_data = extracted_data.copy()
            for k, v in follow_ups.items():
                final_data[k] = v

            # Display extracted/provided details beautifully
            st.subheader("Property Details")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Type", str(final_data.get("property_type", "N/A")))
            with col2:
                st.metric("Bedrooms", str(final_data.get("bedrooms", "N/A")))
            with col3:
                st.metric("Size", f"{final_data.get('size_in_sqm', 'N/A')} sqm")
            with col4:
                price_val = final_data.get("price")
                display_price = f"€{int(price_val):,}" if price_val and str(price_val).isdigit() else f"€{price_val}"
                st.metric("Price", display_price)
                
            st.divider()

            # Prepare session state for evaluator
            state = SessionState()
            state.initial_description = listing_to_process
            state.condition_rating = condition_rating
            state.location_rating = location_rating
            state.insider_knowledge = insider_knowledge
            
            for k, v in follow_ups.items():
                state.add_history("agent", f"Asked for missing {k}")
                state.add_history("user", str(v))

            state.follow_ups = follow_ups
            state.add_history("user", listing_to_process)

            # Determine location for market data
            current_location = final_data.get("city")
            
            # Fetch market data
            market_avg = fetch_market_data(current_location)
            state.market_data = market_avg

            status_placeholder = st.empty()
            with st.spinner(f"{property_category} AI Specialist drafting valuation..."):
                valuation_result = evaluate_valuation(state, status_placeholder, property_category=property_category)
            
            # Clear all status messages
            status_placeholder.empty()

            if "error" in valuation_result:
                st.error(valuation_result["error"])
            else:
                st.divider()
                st.subheader("Final Valuation Result")
                
                # Final Score using st.metric
                st.metric("Final Valuation Score", f"{valuation_result.get('score', 0)}/100")
                
                # Justification, Rent, Yield, and Red Flags using st.write as requested
                st.write(f"**Justification:** {valuation_result.get('justification', 'N/A')}")
                
                rent = valuation_result.get('estimated_monthly_rent')
                if rent:
                    st.write(f"**Estimated Monthly Rent:** €{int(rent):,}")
                else:
                    st.write("**Estimated Monthly Rent:** N/A")
                
                yield_val = valuation_result.get('gross_rental_yield_percentage')
                if yield_val:
                    st.write(f"**Gross Rental Yield:** {yield_val}%")
                else:
                    st.write("**Gross Rental Yield:** N/A")
                
                red_flags = valuation_result.get('red_flags', [])
                if red_flags:
                    st.write("**Red Flags Detected:**")
                    for flag in red_flags:
                        st.write(f"- {flag}")
                else:
                    st.write("**Red Flags Detected:** None identified.")
                
                # Best Investment Strategy and Legal Disclaimer
                st.write(f"**Best Investment Strategy:** {valuation_result.get('best_investment_strategy', 'N/A')}")
                st.write(f"**Legal Disclaimer:** {valuation_result.get('legal_disclaimer', 'N/A')}")
                
                st.divider()
                # Option to reset
                if st.button("Start New Evaluation"):
                    st.session_state.evaluate_clicked = False
                    if "extracted_data" in st.session_state:
                        del st.session_state["extracted_data"]
                    if "listing_to_process" in st.session_state:
                        del st.session_state["listing_to_process"]
                    st.rerun()

if __name__ == "__main__":
    main()
