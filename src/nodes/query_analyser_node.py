from langchain_core.messages import HumanMessage
import re

INTENT_PATTERNS = {
    "function_usage": [
        r"where.*function",
        r"usage of",
        r"where is .* used",
        r"who calls",
        r"find usages",
    ],
    "type_lookup": [
        r"type of",
        r"what.*type",
        r"datatype of",
    ],
    "pipeline_flow": [
        r"pipeline",
        r"flow",
        r"process flow",
        r"execution flow",
        r"data flow",
    ],
    "directory_question": [
        r"what'?s inside",
        r"what is inside",
        r"show.*directory",
        r"explain.*directory",
        r"what does .* folder",
    ],
    "architecture_summary": [
        r"architecture",
        r"overall structure",
        r"design",
    ],
}

def detect_intent(query: str) -> str:
    query_lower = query.lower()
    for intent, patterns in INTENT_PATTERNS.items():
        for p in patterns:
            if re.search(p, query_lower):
                return intent
    return "high_level_summary"


def extract_keywords(query: str):
    # Simple keyword strategy – expand later
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query)
    python_keywords = {"the", "a", "to", "in", "and", "where", "what"}

    filtered = [
        t for t in tokens 
        if t.lower() not in python_keywords and len(t) > 2
    ]
    return filtered[:5]


def extract_targets(query: str):
    # Detect patterns like function(), directories, or variables
    targets = {}

    # function_name()
    fn_match = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", query)
    if fn_match:
        targets["function"] = fn_match[0]

    # directories like src/core or /models/
    dir_match = re.findall(r"([A-Za-z0-9_\-/]+/)", query)
    if dir_match:
        targets["directory"] = dir_match[0]

    # variable (simple heuristic)
    var_match = re.findall(r"type of ([A-Za-z_][A-Za-z0-9_]*)", query)
    if var_match:
        targets["variable"] = var_match[0]

    return targets


async def query_analyser_node(state: dict) -> dict:
    """
    Reads latest human message, determines intent and extracts useful info.
    """

    messages = state.get("messages", [])
    user_query = ""

    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    if not user_query:
        return {
            "intent": "high_level_summary",
            "keywords": [],
            "targets": {}
        }

    intent = detect_intent(user_query)
    keywords = extract_keywords(user_query)
    targets = extract_targets(user_query)

    return {
        "intent": intent,
        "keywords": keywords,
        "targets": targets,
    }
