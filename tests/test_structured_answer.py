import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage, AIMessage

from src.retrieval.evidence import Evidence
from src.retrieval.structured_answer import render_answer
from src.nodes.summarize_repo_node import summarize_repo_node

CATALOG = {"S1":Evidence("src/cache.py", 20, 40, "source")}


@pytest.mark.parametrize("raw", ["not JSON", "[]", '{"answer":""}',
    '{"answer":"x","source_ids":["S1"],"insufficient_evidence":"false"}'])
def test_malformed_structured_answer_fails_explicitly(raw):
    with pytest.raises(ValueError):
        render_answer(raw, CATALOG)


def test_location_is_rendered_from_source_catalog():
    answer, abstained = render_answer(json.dumps({"answer":"Returns None.",
        "source_ids":["S1"], "insufficient_evidence":False}), CATALOG)
    assert "[src/cache.py:L20-L40]" in answer
    assert not abstained


def test_unknown_source_ids_are_rejected():
    with pytest.raises(ValueError, match="invalid source IDs"):
        render_answer(json.dumps({"answer":"result", "source_ids":["invented"],
                                  "insufficient_evidence":False}), CATALOG)


@pytest.mark.parametrize("ids,insufficient", [([], False), (["S1"], True)])
def test_no_supported_source_abstains(ids, insufficient):
    answer, abstained = render_answer(json.dumps({"answer":"draft", "source_ids":ids,
        "insufficient_evidence":insufficient}), CATALOG)
    assert abstained
    assert "draft" not in answer


def test_schema_is_used_by_real_provider_path():
    llm = AsyncMock()
    llm.supports_source_schema = True
    llm.ainvoke.return_value = AIMessage(content=json.dumps({"answer":"Returns 42.",
        "source_ids":["S1"], "insufficient_evidence":False}))
    result = asyncio.run(summarize_repo_node({"messages":[HumanMessage(content="answer")],
        "parsed_files":[{"path":"a.py", "raw":"def answer(): return 42"}], "llm":llm}))
    assert llm.ainvoke.call_args.kwargs["response_format"]["type"] == "object"
    assert result["metrics"]["citation_check"]["status"] == "valid"
    assert "a.py:L1-L1" in result["summary"]
