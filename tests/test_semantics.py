import json

import pytest

from sllm_research.evaluation import score_example, score_records
from sllm_research.semantics import SemanticAction, parse_prediction


def action(slots=None, intent="alarm_set"):
    return SemanticAction.model_validate({"intent": intent, "slots": slots or []})


def test_key_and_slot_order_do_not_change_semantics():
    slots = [{"type": "time", "value": "five am"}, {"type": "date", "value": "today"}]
    raw = json.dumps({"slots": list(reversed(slots)), "intent": "alarm_set"})
    assert score_example(action(slots), raw)["joint_exact_match"]


@pytest.mark.parametrize(
    "raw",
    [
        '{"intent":"alarm_set","slots":',
        '{"intent":"alarm_set"}',
        '{"intent":"alarm_set","slots":[],"extra":true}',
        '{"intent":"alarm_set","intent":"alarm_query","slots":[]}',
        '{"intent":"alarm_set","slots":[{"type":"time","value":5}]}',
        '```json\n{"intent":"alarm_set","slots":[]}\n```',
        "null",
    ],
)
def test_malformed_and_incomplete_predictions_score_zero(raw):
    result = score_example(action(), raw)
    assert not result["valid"]
    assert not result["joint_exact_match"]
    assert not result["intent_correct"]


def test_wrong_intent_with_correct_slots_still_fails_joint():
    result = score_example(action(), '{"intent":"calendar_set","slots":[]}')
    assert result["valid"] and not result["joint_exact_match"]


def test_duplicate_slots_are_not_collapsed():
    slot = {"type": "time", "value": "five am"}
    result = score_example(action([slot, slot]), action([slot]).model_dump_json())
    assert not result["joint_exact_match"]
    assert result["slot_true_positive"] == 1
    assert result["slot_false_negative"] == 1


def test_extra_slot_is_counted():
    result = score_example(action(), action([{"type": "date", "value": "today"}]).model_dump_json())
    assert not result["joint_exact_match"]
    assert result["slot_false_positive"] == 1


def test_empty_slots_and_verbatim_values():
    assert score_example(action(), action().model_dump_json())["joint_exact_match"]
    gold = action([{"type": "time", "value": "Five AM"}])
    prediction = action([{"type": "time", "value": "five am"}])
    assert not score_example(gold, prediction.model_dump_json())["joint_exact_match"]
    assert parse_prediction(gold.model_dump_json()) == gold


def test_missing_predictions_remain_in_denominator():
    gold = [
        {"id": "1", "target": action().model_dump()},
        {"id": "2", "target": action().model_dump()},
    ]
    result = score_records(gold, [{"id": "1", "prediction": action().model_dump_json()}])
    assert result["summary"]["examples"] == 2
    assert result["summary"]["joint_semantic_exact_match"] == 0.5
    assert result["per_example"][1]["error"] == "missing_prediction"


def test_invalid_output_contributes_missing_gold_slots():
    result = score_example(action([{"type": "time", "value": "five am"}]), "broken")
    assert result["slot_false_negative"] == 1


@pytest.mark.parametrize(
    "predictions",
    [
        [{"id": "other", "prediction": "{}"}],
        [{"id": "1", "prediction": "{}"}, {"id": "1", "prediction": "{}"}],
        [{"id": "1", "prediction": {}}],
    ],
)
def test_prediction_id_and_envelope_integrity(predictions):
    with pytest.raises(ValueError):
        score_records([{"id": "1", "target": action().model_dump()}], predictions)
