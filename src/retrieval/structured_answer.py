"""Map model-selected source IDs to real locations; never trust generated ranges."""
import json


def answer_schema(source_ids):
    return {
        "type":"object",
        "properties":{
            "answer":{"type":"string"},
            "source_ids":{"type":"array", "items":{"type":"string", "enum":list(source_ids)}},
            "insufficient_evidence":{"type":"boolean"},
        },
        "required":["answer", "source_ids", "insufficient_evidence"],
        "additionalProperties":False,
    }


def render_answer(raw, catalog):
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError("Model returned invalid structured answer JSON") from exc
    if not isinstance(data, dict) or not isinstance(data.get("answer"), str) or not data["answer"].strip():
        raise ValueError("Structured answer must contain nonempty answer text")
    ids = data.get("source_ids")
    if (not isinstance(ids, list) or any(not isinstance(i, str) or i not in catalog for i in ids)
            or type(data.get("insufficient_evidence")) is not bool):
        raise ValueError("Structured answer contains invalid source IDs or evidence status")
    if data["insufficient_evidence"] or not ids:
        return "Insufficient retrieved evidence to answer reliably.", True
    locations = []
    for source_id in dict.fromkeys(ids):
        source = catalog[source_id]
        if not source.raw_source:
            locations.append(f"[{source.path} (parsed excerpt)]")
        else:
            locations.append(f"[{source.path}:L{source.start_line}-L{source.end_line}]")
    return data["answer"].strip() + "\n\nSources: " + ", ".join(locations), False
