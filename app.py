"""Streamlit dashboard: run OpenAI, TypeSafe Jev, or both on support tickets and compare timing.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.categorizer import ProviderError, compare_summaries, run_provider

load_dotenv()  # Read API keys from .env (never commit this file).

# Bundled sample data. The first file that exists is used when no CSV is uploaded.
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_DATASETS = ["sample_tickets_edge_cases.csv", "support_ticket_data.csv"]
REQUIRED_COLUMNS = {"support_tick_id", "support_ticket_text"}
DEFAULT_CONFIDENCE_THRESHOLD = 0.70

st.set_page_config(page_title="Support Ticket Model Comparison", page_icon="🎫", layout="wide")
st.title("Support Ticket Categorization: OpenAI vs TypeSafe Jev")
st.write(
    "Classify the same support-ticket fields using OpenAI, TypeSafe Jev, or both, then compare elapsed time. "
    "Use the bundled 30 synthetic edge-case tickets or upload your own CSV."
)
st.info(
    "Jev returns typed decisions and confidence scores. A ticket is marked for human feedback when any Jev "
    "field confidence is missing or below the selected threshold, or its category is Needs Review / Ambiguous."
)


def _review_status_style(value: str) -> str:
    """CSS for the Jev review-status cell: red = needs a human, green = auto-accepted."""
    if value == "Human Review Required":
        return "background-color: #FEE2E2; color: #991B1B; font-weight: bold"
    if value == "No human feedback needed":
        return "background-color: #DCFCE7; color: #166534; font-weight: bold"
    return ""


def _styled_results(results: pd.DataFrame):
    """Colour the review_status column when present (Jev results only)."""
    if "review_status" in results.columns:
        return results.style.map(_review_status_style, subset=["review_status"])
    return results


# --- Sidebar: run mode, threshold, data upload ---------------------------------
with st.sidebar:
    st.header("Run settings")
    run_mode = st.radio(
        "Choose model run",
        ["Compare OpenAI and Jev", "OpenAI only", "Jev only"],
        index=0,
    )
    confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
    if run_mode != "OpenAI only":
        confidence_threshold = st.slider(
            "Jev human-feedback threshold",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_CONFIDENCE_THRESHOLD,
            step=0.05,
            help="A ticket needs review if any Jev field confidence is missing or below this value, or its category is Needs Review / Ambiguous.",
        )
    uploaded = st.file_uploader("Upload a CSV (optional)", type=["csv"])
    max_tickets = st.number_input("Tickets to process", min_value=1, max_value=500, value=30, step=1)
    st.caption("For a quick demo, start with 3–5 tickets. Each ticket uses one API request per selected provider.")

# --- Load ticket data ------------------------------------------------------------
if uploaded is not None:
    try:
        tickets = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
        st.stop()
else:
    default_csv = next(
        (os.path.join(DATA_DIR, name) for name in DEFAULT_DATASETS if os.path.exists(os.path.join(DATA_DIR, name))),
        None,
    )
    if default_csv is None:
        st.error("No bundled dataset found in the `data/` folder. Upload a CSV from the sidebar.")
        st.stop()
    tickets = pd.read_csv(default_csv)

missing = REQUIRED_COLUMNS - set(tickets.columns)
if missing:
    st.error("CSV is missing required columns: " + ", ".join(sorted(missing)))
    st.stop()

# Skip empty ticket text and IDs before capping the batch size.
valid_rows = (
    tickets["support_tick_id"].notna()
    & tickets["support_ticket_text"].notna()
    & tickets["support_tick_id"].astype(str).str.strip().ne("")
    & tickets["support_ticket_text"].astype(str).str.strip().ne("")
)
skipped_rows = int((~valid_rows).sum())
tickets = tickets.loc[valid_rows].head(int(max_tickets)).reset_index(drop=True)
if skipped_rows:
    st.warning(f"Skipped {skipped_rows} row(s) with an empty ticket ID or text.")
if tickets.empty:
    st.error("No tickets with both an ID and text were found in the CSV.")
    st.stop()
st.subheader("Ticket data")
st.caption(
    f"Loaded {len(tickets)} ticket(s). Only ticket text is sent to the selected model APIs; IDs stay in local results."
)
st.dataframe(tickets[["support_tick_id", "support_ticket_text"]], width="stretch", hide_index=True)

# --- Credential check (keys are read from the environment, never displayed) ----
configured_openai = bool(os.getenv("OPENAI_API_KEY"))
configured_jev = bool(os.getenv("TYPESAFE_API_KEY"))
left, right = st.columns(2)
left.metric("OpenAI key", "Configured" if configured_openai else "Missing")
right.metric("TypeSafe key", "Configured" if configured_jev else "Missing")

provider_selection = {
    "Compare OpenAI and Jev": ["OpenAI", "Jev"],
    "OpenAI only": ["OpenAI"],
    "Jev only": ["Jev"],
}
selected_providers = provider_selection[run_mode]
missing_credentials = [
    provider
    for provider in selected_providers
    if (provider == "OpenAI" and not configured_openai)
    or (provider == "Jev" and not configured_jev)
]
if missing_credentials:
    st.caption(
        "Add the missing key(s) to `.env` before running: " + ", ".join(missing_credentials)
    )

# --- Run the selected providers sequentially and show results --------------------
if st.button(
    "Run selected model(s)",
    type="primary",
    disabled=bool(missing_credentials),
):
    summaries = []
    for provider in selected_providers:
        st.subheader(f"Running {provider}")
        progress = st.progress(0.0, text="Starting")
        try:
            summary = run_provider(
                provider,
                tickets,
                progress=progress.progress,
                jev_confidence_threshold=confidence_threshold,
            )
            summaries.append(summary)
            progress.progress(1.0, text=f"Finished in {summary.elapsed_seconds:.2f} seconds")
        except ProviderError as exc:
            # Configuration problems (e.g. missing key) stop the run with a clear message.
            st.error(str(exc))
            st.stop()

    st.subheader("Elapsed-time comparison" if len(summaries) > 1 else "Elapsed time")
    comparison = compare_summaries(summaries)
    st.dataframe(comparison, width="stretch", hide_index=True)
    st.bar_chart(comparison.set_index("Provider")[["Total time (seconds)"]])
    if len(summaries) == 2 and any(summary.errors for summary in summaries):
        st.warning("Some calls failed. The timing includes those failures, so no speed ratio is reported.")
    elif len(summaries) == 2:
        # summaries[0] is OpenAI and summaries[1] is Jev (order set by provider_selection).
        openai_time, jev_time = summaries[0].elapsed_seconds, summaries[1].elapsed_seconds
        if min(openai_time, jev_time) > 0:
            faster = "OpenAI" if openai_time < jev_time else "TypeSafe Jev"
            ratio = max(openai_time, jev_time) / min(openai_time, jev_time)
            st.success(f"{faster} was {ratio:.2f}× faster in this run. Network and service load affect the result.")

    for summary in summaries:
        st.subheader(f"{summary.provider} results ({summary.model})")
        st.dataframe(_styled_results(summary.results), width="stretch", hide_index=True)
        if summary.provider == "TypeSafe Jev":
            st.caption(
                f"Human-feedback rule: assign the ticket if any field confidence is missing or below {confidence_threshold:.2f}, "
                "or its category is Needs Review / Ambiguous. "
                "The minimum and per-field confidence scores are included in the results."
            )
        st.download_button(
            f"Download {summary.provider} CSV",
            summary.results.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{summary.provider.lower().replace(' ', '_')}_ticket_results.csv",
            mime="text/csv",
            key=f"download-{summary.provider}",
        )
        if summary.errors:
            with st.expander(f"{len(summary.errors)} row-level error(s)"):
                st.write("\n".join(summary.errors))

    st.caption(
        "Timing is end-to-end wall-clock time for sequential API requests, including network latency and service overhead. "
        "Run multiple times and compare averages for a more stable demonstration."
    )
