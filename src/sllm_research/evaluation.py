"""Score every gold example, preserving malformed and missing predictions as failures."""

from collections import Counter

from pydantic import ValidationError

from sllm_research.semantics import SemanticAction, parse_prediction


def score_example(gold: SemanticAction, raw: str | None) -> dict:
    prediction = None
    error = None
    if raw is None:
        error = "missing_prediction"
    else:
        try:
            prediction = parse_prediction(raw)
        except (ValueError, ValidationError, TypeError):
            error = "invalid_semantic_json"
    gold_slots = gold.slot_multiset()
    predicted_slots = prediction.slot_multiset() if prediction else Counter()
    true_positive = sum((gold_slots & predicted_slots).values())
    return {
        "valid": prediction is not None,
        "joint_exact_match": bool(
            prediction and prediction.intent == gold.intent and gold_slots == predicted_slots
        ),
        "intent_correct": bool(prediction and prediction.intent == gold.intent),
        "slot_true_positive": true_positive,
        "slot_false_positive": sum(predicted_slots.values()) - true_positive,
        "slot_false_negative": sum(gold_slots.values()) - true_positive,
        "raw_prediction": raw,
        "parsed_prediction": prediction.model_dump() if prediction else None,
        "error": error,
    }


def score_records(gold_records: list[dict], predictions: list[dict]) -> dict:
    if not gold_records:
        raise ValueError("Cannot score an empty gold dataset")
    gold_ids = [record["id"] for record in gold_records]
    gold_id_set = set(gold_ids)
    if len(gold_ids) != len(gold_id_set):
        raise ValueError("Duplicate gold example IDs")
    by_id = {}
    for record in predictions:
        identifier = record["id"]
        if identifier in by_id:
            raise ValueError(f"Duplicate prediction ID: {identifier}")
        if identifier not in gold_id_set:
            raise ValueError(f"Unknown prediction ID: {identifier}")
        if not isinstance(record.get("prediction"), str):
            raise ValueError("Prediction envelopes must contain a raw string under 'prediction'")
        by_id[identifier] = record["prediction"]
    rows = [
        {
            "id": record["id"],
            **score_example(
                SemanticAction.model_validate(record["target"]), by_id.get(record["id"])
            ),
        }
        for record in gold_records
    ]
    n = len(rows)
    tp = sum(row["slot_true_positive"] for row in rows)
    fp = sum(row["slot_false_positive"] for row in rows)
    fn = sum(row["slot_false_negative"] for row in rows)
    return {
        "scoring_version": "verbatim-v1",
        "summary": {
            "examples": n,
            "joint_semantic_exact_match": sum(r["joint_exact_match"] for r in rows) / n,
            "intent_accuracy": sum(r["intent_correct"] for r in rows) / n,
            "schema_validity": sum(r["valid"] for r in rows) / n,
            "slot_micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            "extra_or_invented_slot_rate": fp / (tp + fp) if tp + fp else 0.0,
            "invalid_or_missing_predictions": sum(not r["valid"] for r in rows),
        },
        "per_example": rows,
    }
