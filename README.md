# AI Real Estate Valuation Agent

This project is an AI-powered Real Estate Valuation Agent designed to act as a **ruthless real estate investor**. It prioritizes ROI, financial value, and market data over aesthetics, providing a brutal, analytical evaluation of property listings.

## Agentic Design Patterns

The system implements several core Agentic Design Patterns to ensure a robust and intelligent valuation process:

### 1. Tool Use
The agent has the ability to fetch real-world (mocked) market data. If a location is identified (e.g., Downtown, Suburbs, Uptown, Riverside), it automatically retrieves the neighborhood's average price to compare against the listing.

### 2. Memory Management
A dedicated `SessionState` class manages the lifecycle of a valuation. It stores:
- The initial property description.
- Follow-up information gathered from the user.
- Market data retrieved via tools.
- A full history of the interaction for context-aware final evaluation.

### 3. Human-in-the-Loop
If critical data points like **price** or **location** are missing from the initial description, the system pauses and prompts the user for clarification. This ensures the AI always has the necessary facts before making a financial judgment.

### 4. Evaluator with Guardrails
The final evaluation is performed by a specialized LLM prompt that enforces a strict JSON schema. The agent must provide:
- A **score from 1 to 100** (where 100 is a perfect deal).
- A concise, "ruthless" justification for the score.

## Setup Instructions

To get the AI Real Estate Valuation Agent running locally, follow these steps:

1.  **Install Requirements**
    Ensure you have Python installed, then install the necessary dependencies:
    ```bash
    pip install google-generativeai python-dotenv typing-extensions
    ```

2.  **Configure Environment Variables**
    Create a `.env` file in the root directory of the project and add your Google API Key:
    ```env
    GOOGLE_API_KEY=your_actual_api_key_here
    ```

3.  **Run the Application**
    Start the interactive valuation agent by running:
    ```bash
    python app.py
    ```

## Core Mandates
- The system must always output a score from 1 to 100.
- The system must prioritize gathering missing data from the user before finalizing a valuation.
