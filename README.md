# Support Ticket Categorization Model Comparison

This Python project turns the supplied learning notebook into a Streamlit demo. Run OpenAI only, Jev only, or compare both on the same five ticket fields:

- Category: Technical Issue, Hardware Issue, Data Recovery, Needs Review / Ambiguous, or Out of Scope / Unrelated
- Device
- Problem type
- User impact: Major, Moderate, Minor, or Not Applicable (for unrelated requests)
- Priority: Low, Medium, High, or Not Applicable (for unrelated requests)

The app reports end-to-end elapsed time and lets you download each provider's results as CSV. It includes the original 21-row `support_ticket_data.csv`.

## Important comparison detail

Both providers make one request per ticket for the same five classification fields. The OpenAI output uses a JSON schema, while Jev uses typed Choice questions and returns confidence for each field. The default human-feedback threshold is 0.70 and can be changed in the sidebar. A ticket is marked **Human Review Required** if any Jev field confidence is below the selected threshold. Failed Jev requests are also routed for human review.

The timing is wall-clock time for sequential API requests, including network and service overhead. It is not a measure of model-only inference time. For a presentation, run the comparison more than once and report the average.

## Setup (Windows PowerShell)

1. Install Python 3.10 or newer.
2. Open PowerShell in this project folder.
3. Create and activate a virtual environment:

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and add your API keys:

   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

   Set `OPENAI_API_KEY` and `TYPESAFE_API_KEY`. The default model IDs are `gpt-4o-mini` and `jev-latest`. Change the model names in `.env` if needed. For a custom OpenAI-compatible endpoint or proxy, set `OPENAI_BASE_URL`; leave it blank to use OpenAI's default endpoint. To route Jev through a custom endpoint or proxy, set `TYPESAFE_BASE_URL` to its API root; leave it blank to use the SDK's default endpoint (`https://api.typesafe.ai`).

5. Start the app:

   ```powershell
   streamlit run app.py
   ```

6. In the browser, select **Compare OpenAI and Jev**, **OpenAI only**, or **Jev only**. Choose how many tickets to process, set the Jev human-feedback threshold if applicable, then click **Run selected model(s)**. Try 3–5 tickets first to keep the initial demo short.

## CSV format

The input CSV must include these headers:

```csv
support_tick_id,support_ticket_text
```

Use the included data file or upload another CSV in the app.

For edge-case practice, upload `data/sample_tickets_edge_cases.csv`. It contains 30 synthetic tickets with only ticket IDs and text. Compare the results with `data/sample_ticket_answer_key.csv`; keep the answer-key file separate from the model input. The set covers all five category outcomes, Major/Moderate/Minor impact, Low/Medium/High priority, Not Applicable for unrelated requests, vague or conflicting descriptions, and out-of-scope requests.

## Project structure

```text
app.py                         Streamlit dashboard
src/categorizer.py             Provider calls, shared labels, and timing
data/support_ticket_data.csv   Original learning-project data
data/sample_tickets_edge_cases.csv  30 synthetic ambiguous and unrelated test tickets
data/sample_ticket_answer_key.csv   Expected labels and review notes for edge cases
.env.example                   API key/model settings template
requirements.txt               Python dependencies
```

API keys stay in your local `.env` file, which is ignored by Git. Do not put real keys in source files or share them in screenshots.
