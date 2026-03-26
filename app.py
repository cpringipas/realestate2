from evaluator import evaluate_valuation

class SessionState:
    """
    Memory Management: Stores the history of the current valuation session.
    """
    def __init__(self):
        self.initial_description = ""
        self.follow_ups = {}
        self.market_data = None
        self.history = []

    def add_history(self, role, message):
        self.history.append({"role": role, "message": message})

    def __str__(self):
        return (f"Initial: {self.initial_description}\n"
                f"Follow-ups: {self.follow_ups}\n"
                f"Market Data: {self.market_data}")

# Mock database for neighborhood average prices
NEIGHBORHOOD_MARKET_DATA = {
    "limassol": 4000,
    "nicosia": 2500,
    "paphos": 2800,
    "larnaca": 2600
}

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
    Planner function that checks for 'City in Cyprus' and 'Size in square meters'.
    Returns a list of missing required information.
    """
    missing = []
    if "city" not in description.lower():
        missing.append("City in Cyprus")
    if "size" not in description.lower():
        missing.append("Size in square meters")
    return missing

def main():
    print("--- Real Estate Valuation Agent ---")
    
    while True:
        state = SessionState()
        user_input = input("\nEnter a real estate listing description (or 'quit' to exit): ")
        
        if user_input.lower() == 'quit':
            break

        state.initial_description = user_input
        state.add_history("user", user_input)

        # 1. Planner & Human in the Loop: Check for missing information
        missing_info = planner(user_input)
        current_location = None
        
        # Simple extraction if already in description
        for loc in NEIGHBORHOOD_MARKET_DATA.keys():
            if loc in user_input.lower():
                current_location = loc
                break

        while missing_info:
            info_needed = missing_info.pop(0)
            answer = input(f"Missing {info_needed}. Please provide the {info_needed}: ")
            
            # Memory Management: Store follow-up data
            state.follow_ups[info_needed] = answer
            state.add_history("agent", f"Asked for {info_needed}")
            state.add_history("user", answer)
            
            if info_needed == "City in Cyprus":
                current_location = answer

        # 2. Tool Use: Fetch market data
        print(f"\n[Tool Use] Fetching market data for: {current_location}...")
        market_avg = fetch_market_data(current_location)
        state.market_data = market_avg

        # 3. Evaluator: Pass Memory to Gemini
        print("\n[Evaluator] Running final valuation...")
        valuation_result = evaluate_valuation(state)

        # 4. Output Final JSON result
        print("\n--- Final Valuation Result ---")
        print(valuation_result)

if __name__ == "__main__":
    main()
