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

## Prerequisites and provider accounts

All run modes require Python 3.10 or newer, an internet connection, and the packages in `requirements.txt`.

| Run mode | Account and credential required |
| --- | --- |
| OpenAI only | An OpenAI API Platform account and an API key stored as `OPENAI_API_KEY`. Create/manage the key in the [OpenAI API dashboard](https://platform.openai.com/api-keys). |
| Jev only | A TypeSafe account and API key stored as `TYPESAFE_API_KEY`. Create the key from the [TypeSafe console](https://console.typesafe.ai/). |
| Compare both | Both accounts and both API keys. |

A ChatGPT subscription by itself does not include API usage; OpenAI API billing is managed separately. API calls may incur provider charges, so check your plan, usage limits, and current pricing before running a large batch. See [OpenAI API billing](https://help.openai.com/en/articles/9039756-managing-billing-settings-on-the-chatgpt-web-and-api-platform) and the [TypeSafe Python SDK quickstart](https://docs.typesafe.ai/sdk/python).

The app only requires the key for the selected provider mode. The custom endpoint settings (`OPENAI_BASE_URL` and `TYPESAFE_BASE_URL`) are optional; leave them blank to use each provider's default endpoint.


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

## How to test

1. Start the app and select **Compare OpenAI and Jev**.
2. Keep the default edge-case dataset, or upload `data/sample_tickets_edge_cases.csv`.
3. For a quick check, set **Tickets to process** to `3`. For the full run, set it to `30`.
4. Set the Jev human-feedback threshold (default `0.70`) and click **Run selected model(s)**.
5. Review the timing table, both result tables, and any red **Human Review Required** cells. Download the result CSVs and compare ticket labels with `data/sample_ticket_answer_key.csv`.

Run the same dataset several times before presenting a timing comparison. The calls run sequentially and include network and provider overhead, so a single run is only a snapshot.


### Comparison results

#### Example run

This example is taken from the screenshot supplied with the project feedback. It used 5 tickets, `gpt-4o-mini`, and `jev-latest`; both providers completed without errors.

| Provider | Model | Tickets | Total time (s) | Average per ticket (s) | Errors |
| --- | --- | ---: | ---: | ---: | ---: |
| OpenAI | `gpt-4o-mini` | 5 | 9.320 | 1.864 | 0 |
| TypeSafe Jev | `jev-latest` | 5 | 1.934 | 0.387 | 0 |

In this run, TypeSafe Jev completed about **4.82× faster** than OpenAI. This is one observed run, not a guaranteed performance result; timing varies with network conditions, provider load, and account configuration.

<img width="1896" height="836" alt="image" src="https://github.com/user-attachments/assets/58746a7f-6ed3-4aa6-8105-345522c30c58" />


#### Template for your own comparison

Use the same dataset size and model settings for each provider. Record the values displayed by the app; repeat the run three times and calculate the average if you want a more stable comparison.

| Run | Tickets | OpenAI total time (s) | OpenAI avg / ticket (s) | Jev total time (s) | Jev avg / ticket (s) | Jev human reviews | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| 2 | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| 3 | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| Average | ___ | ___ | ___ | ___ | ___ | ___ | ___ |


Use the app's **Elapsed-time comparison** table for total and average time. The average is total time divided by the number of tickets. The human-review count is the number of Jev rows whose status is **Human Review Required**. For an OpenAI-only run, Jev columns do not apply; for a Jev-only run, OpenAI columns do not apply.

## Project structure

```text
app.py                         Streamlit dashboard
src/categorizer.py             Provider calls, shared labels, and timing
data/sample_tickets_edge_cases.csv  30 synthetic ambiguous and unrelated test tickets
.env.example                   API key/model settings template
requirements.txt               Python dependencies
```

API keys stay in your local `.env` file, which is ignored by Git. Do not put real keys in source files or share them in screenshots.
