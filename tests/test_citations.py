from src.retrieval.citations import check_citations
from src.retrieval.evidence import Evidence


def test_uncited_answer_is_flagged():
    assert check_citations("Returns None.", []) ["status"] == "missing"


def test_known_source_range_is_valid():
    source = Evidence("src/cache.py", 20, 40, "source")
    assert check_citations("See [src/cache.py:L22-L30].", [source])["status"] == "valid"


def test_hallucinated_path_is_flagged():
    source = Evidence("src/cache.py", 20, 40, "source")
    assert check_citations("See src/invented.py:L22-L30", [source])["status"] == "invalid"


def test_out_of_context_and_reversed_ranges_are_flagged():
    source = Evidence("src/cache.py", 20, 40, "source")
    for answer in ("src/cache.py:L1-L10", "src/cache.py:L30-L22"):
        assert check_citations(answer, [source])["status"] == "invalid"


def test_parsed_lines_cannot_validate_source_citations():
    source = Evidence("src/cache.py", 20, 40, "source", raw_source=False)
    assert check_citations("src/cache.py:L22-L30", [source])["status"] == "invalid"


def test_answer_node_exposes_missing_citation_warning():
    import asyncio
    from unittest.mock import AsyncMock
    from langchain_core.messages import HumanMessage, AIMessage
    from src.nodes.summarize_repo_node import summarize_repo_node
    llm = AsyncMock()
    llm.ainvoke.return_value = AIMessage(content="Returns 42.")
    result = asyncio.run(summarize_repo_node({"messages":[HumanMessage(content="answer")],
        "parsed_files":[{"path":"a.py", "raw":"def answer(): return 42"}], "llm":llm}))
    assert "Citation warning" in result["summary"]
    assert result["metrics"]["citation_check"]["status"] == "missing"
