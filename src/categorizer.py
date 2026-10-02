"""Shared ticket categorization and provider adapters for OpenAI and TypeSafe Jev."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from typesafe_sdk import Choice, TypeSafeClient

load_dotenv()

CATEGORIES = {
    "technical_issue": "Technical Issue",
    "hardware_issue": "Hardware Issue",
    "data_recovery": "Data Recovery",
    "needs_review": "Needs Review / Ambiguous",
    "out_of_scope": "Out of Scope / Unrelated",
}
DEVICES = {
    "laptop": "Laptop",
    "desktop_pc": "Desktop / PC",
    "smartphone": "Smartphone",
    "router": "Router",
    "external_hard_drive": "External Hard Drive",
    "usb_drive": "USB Drive",
    "graphics_card": "Graphics Card",
    "touchpad": "Touchpad",
    "online_account": "Online Account",
    "computer_unspecified": "Computer (unspecified)",
    "other_unclear": "Other / Unclear",
}
PROBLEM_TYPES = {
    "slow_internet": "Slow Internet Connection",
    "laptop_not_starting": "Device Won't Start",
    "deleted_files": "Deleted Files",
    "weak_wifi": "Weak Wi-Fi Signal",
    "battery_drain": "Battery Draining",
    "account_access": "Account Access / Password Reset",
    "slow_performance": "Slow Computer Performance",
    "blue_screen_crashes": "Blue Screen / Crashes",
    "drive_not_recognized": "Drive Not Recognized",
    "graphics_problem": "Graphics Card Problem",
    "formatted_drive": "Formatted Drive / Data Recovery",
    "black_screen": "Black Screen",
    "liquid_damage": "Liquid Damage",
    "physical_drive_damage": "Physically Damaged Drive",
    "touchpad_not_working": "Touchpad Not Working",
    "internet_disconnections": "Internet Disconnections",
    "software_errors_data_loss": "Software Errors / Missing Files",
    "other_unclear": "Other / Unclear",
}
IMPACTS = {"major": "Major", "moderate": "Moderate", "minor": "Minor", "not_applicable": "Not Applicable"}
PRIORITIES = {"low": "Low", "medium": "Medium", "high": "High", "not_applicable": "Not Applicable"}


class ProviderError(RuntimeError):
    """An actionable provider/configuration error for the UI."""


def _choice_descriptions(options: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in options.items()}


def _labels(raw: dict[str, Any]) -> dict[str, str]:
    fields = {
        "category": CATEGORIES,
        "device": DEVICES,
        "problem_type": PROBLEM_TYPES,
        "user_impact": IMPACTS,
        "priority": PRIORITIES,
    }
    result: dict[str, str] = {}
    for name, choices in fields.items():
        raw_value = str(raw[name]).strip()
        value = raw_value.lower().replace(" ", "_").replace("-", "_")
        if value in choices:
            result[name] = choices[value]
            continue
        match = next((label for label in choices.values() if label.casefold() == raw_value.casefold()), None)
        result[name] = match or choices.get("other_unclear", raw_value)
    return result


def _openai_client() -> tuple[OpenAI, str]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ProviderError("Set OPENAI_API_KEY in the project's .env file.")
    kwargs: dict[str, Any] = {"api_key": key}
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs), os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _openai_one(client: OpenAI, model: str, ticket: str) -> dict[str, str]:
    schema = {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": list(CATEGORIES.values())},
            "device": {"type": "string", "enum": list(DEVICES.values())},
            "problem_type": {"type": "string", "enum": list(PROBLEM_TYPES.values())},
            "user_impact": {"type": "string", "enum": list(IMPACTS.values())},
            "priority": {"type": "string", "enum": list(PRIORITIES.values())},
        },
        "required": ["category", "device", "problem_type", "user_impact", "priority"],
        "additionalProperties": False,
    }
    system = (
        "Classify this support ticket. Choose exactly one value for every field from the supplied JSON schema. "
        "Category choices are Technical Issue, Hardware Issue, Data Recovery, Needs Review / Ambiguous, "
        "or Out of Scope / Unrelated. Use Needs Review / Ambiguous when the description has conflicting or "
        "insufficient details to classify safely. Use Out of Scope / Unrelated when it is not a support issue. "
        "For out-of-scope requests, set impact and priority to Not Applicable. Impact is based on work disruption: "
        "Major means work cannot proceed, Moderate means a workaround may exist, Minor means limited disruption. "
        "Priority must be Low, Medium, or High and reflect urgency plus impact. Use Other / Unclear when needed."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": ticket}],
        response_format={"type": "json_schema", "json_schema": {"name": "ticket_classification", "strict": True, "schema": schema}},
        temperature=0,
    )
    content = response.choices[0].message.content
    if not content:
        raise ProviderError("OpenAI returned an empty classification.")
    return _labels(json.loads(content))


def _typesafe_client() -> tuple[TypeSafeClient, str]:
    key = os.getenv("TYPESAFE_API_KEY")
    if not key:
        raise ProviderError("Set TYPESAFE_API_KEY in the project's .env file.")
    # Configure the API root explicitly when supplied; otherwise the SDK's default is used.
    base_url = os.getenv("TYPESAFE_BASE_URL", "").strip() or None
    return TypeSafeClient(api_key=key, base_url=base_url), os.getenv("TYPESAFE_MODEL", "jev-latest")


def _jev_one(
    client: TypeSafeClient,
    model: str,
    ticket: str,
    confidence_threshold: float,
) -> dict[str, Any]:
    response = client.system_one(
        state={"support_ticket": ticket},
        model=model,
        questions={
            "category": Choice(instructions="Which category best describes the main issue?", criteria=_choice_descriptions(CATEGORIES)),
            "device": Choice(instructions="Which device, service, or account is involved? Choose the closest match; use other_unclear if needed.", criteria=_choice_descriptions(DEVICES)),
            "problem_type": Choice(instructions="Which problem type best matches the ticket? Choose the closest match; use other_unclear if needed.", criteria=_choice_descriptions(PROBLEM_TYPES)),
            "user_impact": Choice(instructions="How severely does this issue affect the user's ability to work or use the service?", criteria=_choice_descriptions(IMPACTS)),
            "priority": Choice(instructions="What support priority best fits the issue's impact and urgency?", criteria=_choice_descriptions(PRIORITIES)),
        },
    )
    fields = ("category", "device", "problem_type", "user_impact", "priority")
    answers = {field: response.choices[field] for field in fields}
    raw = {field: answers[field].choice for field in fields}
    confidences = {
        field: float(answers[field].confidence) if getattr(answers[field], "confidence", None) is not None else None
        for field in fields
    }
    low_confidence_fields = [
        field for field, score in confidences.items()
        if score is None or score < confidence_threshold
    ]
    result: dict[str, Any] = _labels(raw)
    result.update({f"{field}_confidence": score for field, score in confidences.items()})
    available_confidences = [score for score in confidences.values() if score is not None]
    result["minimum_confidence"] = min(available_confidences) if available_confidences else None
    result["low_confidence_fields"] = ", ".join(low_confidence_fields)
    result["review_status"] = (
        "Human Review Required" if low_confidence_fields else "No human feedback needed"
    )
    return result


@dataclass
class RunSummary:
    provider: str
    model: str
    results: pd.DataFrame
    elapsed_seconds: float
    errors: list[str]

    @property
    def average_seconds_per_ticket(self) -> float:
        return self.elapsed_seconds / len(self.results) if len(self.results) else 0.0


def run_provider(
    provider: str,
    tickets: pd.DataFrame,
    progress=None,
    jev_confidence_threshold: float = 0.70,
) -> RunSummary:
    """Run the same single-call classification task for each ticket and time it."""
    if "support_tick_id" not in tickets or "support_ticket_text" not in tickets:
        raise ValueError("CSV must contain support_tick_id and support_ticket_text columns.")
    normalized = provider.strip().lower()
    if normalized == "openai":
        client, model = _openai_client()
        classify = lambda text: _openai_one(client, model, text)
        display_name = "OpenAI"
    elif normalized in {"jev", "typesafe"}:
        client, model = _typesafe_client()
        classify = lambda text: _jev_one(client, model, text, jev_confidence_threshold)
        display_name = "TypeSafe Jev"
    else:
        raise ValueError("Provider must be OpenAI or Jev.")

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    total = len(tickets)
    for index, (_, row) in enumerate(tickets.iterrows(), start=1):
        item: dict[str, Any] = {
            "support_tick_id": str(row["support_tick_id"]),
            "support_ticket_text": str(row["support_ticket_text"]),
        }
        try:
            fields = classify(item["support_ticket_text"])
            item.update(fields)
            item["error"] = ""
        except Exception as exc:  # Keep remaining tickets running and show row-level errors.
            message = f"{item['support_tick_id']}: {exc}"
            errors.append(message)
            item.update({"category": "", "device": "", "problem_type": "", "user_impact": "", "priority": ""})
            if display_name == "TypeSafe Jev":
                item.update({
                    "category_confidence": None,
                    "device_confidence": None,
                    "problem_type_confidence": None,
                    "user_impact_confidence": None,
                    "priority_confidence": None,
                    "minimum_confidence": None,
                    "low_confidence_fields": "Model call failed",
                    "review_status": "Human Review Required",
                })
            item["error"] = str(exc)
        rows.append(item)
        if progress is not None:
            progress(index / total if total else 1.0, f"{display_name}: {index} of {total}")
    elapsed = time.perf_counter() - started
    return RunSummary(display_name, model, pd.DataFrame(rows), elapsed, errors)


def compare_summaries(summaries: list[RunSummary]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Provider": summary.provider,
            "Model": summary.model,
            "Tickets": len(summary.results),
            "Total time (seconds)": round(summary.elapsed_seconds, 3),
            "Average time per ticket (seconds)": round(summary.average_seconds_per_ticket, 3),
            "Errors": len(summary.errors),
        }
        for summary in summaries
    ])
