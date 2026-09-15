"""Validate copied evidence before rendering an answer; not an entailment judge."""
import json
from src.retrieval.structured_answer import render_answer


def quoted_schema(catalog):
    return {"type":"object", "properties":{
        "evidence":{"type":"array", "maxItems":3, "items":{
            "type":"object", "properties":{
                "source_id":{"type":"string", "enum":list(catalog)},
                "quote":{"type":"string", "minLength":8, "maxLength":240}},
            "required":["source_id", "quote"], "additionalProperties":False}},
        "answer":{"type":"string", "maxLength":800},
        "insufficient_evidence":{"type":"boolean"}},
        "required":["evidence", "answer", "insufficient_evidence"], "additionalProperties":False}


def render_quoted(raw, catalog):
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return 'The model returned an incomplete or malformed response; no answer was accepted.', True, {
            'status':'malformed_response', 'quotes_checked':0}
    if (not isinstance(data, dict) or not isinstance(data.get('answer'), str)
            or not data['answer'].strip() or len(data['answer']) > 800
            or type(data.get('insufficient_evidence')) is not bool
            or not isinstance(data.get('evidence'), list)):
        return 'The model returned an incomplete or malformed response; no answer was accepted.', True, {
            'status':'malformed_response', 'quotes_checked':0}
    evidence = data['evidence']
    valid = bool(evidence) and len(evidence) <= 3
    ids = []
    corrections = []
    for item in evidence:
        if not isinstance(item, dict):
            valid = False
            continue
        key, quote = item.get('source_id'), item.get('quote')
        if (not isinstance(key, str) or key not in catalog or not isinstance(quote, str)
                or not 8 <= len(quote.strip()) <= 240):
            valid = False
            continue
        normalized = ' '.join(quote.split())
        if normalized not in ' '.join(catalog[key].text.split()):
            matches = [source_id for source_id, source in catalog.items()
                       if normalized in ' '.join(source.text.split())]
            if len(matches) != 1:
                valid = False
                continue
            corrections.append({'from':key, 'to':matches[0]})
            key = matches[0]
        ids.append(key)
    if data['insufficient_evidence'] or not valid:
        return 'Insufficient verified source evidence to answer reliably.', True, {
            'status':'abstained' if data['insufficient_evidence'] else 'invalid_quote',
            'quotes_checked':len(evidence)}
    answer, abstained = render_answer(json.dumps({'answer':data['answer'],
        'source_ids':ids, 'insufficient_evidence':False}), catalog)
    return answer, abstained, {'status':'valid', 'quotes_checked':len(evidence),
                              'source_id_corrections':corrections}
