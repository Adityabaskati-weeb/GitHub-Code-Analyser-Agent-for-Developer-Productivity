# state_schema.py

from typing import Annotated, Any, Optional, Sequence, TypedDict, Union, Dict, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class Agent_State(TypedDict):
    """
    LangGraph state structure for the Repo Analyzer + Summarizer Agent.

    This is the central shared memory between all nodes.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    url: Union[str, None]
    branch: Optional[str]          # Git branch to analyse; None → auto-detect
    refresh_cache: Optional[bool]  # True → bypass disk cache for this run
    repo_tree: Dict[str, Any]
    global_context: Union[str, None]
    selected_files: List[Dict[str, Any]]
    unselected_files: List[str]
    parsed_files: List[Dict[str, str]]
    intent: str
    keywords: List[str]
    targets: Dict[str, Any]
    summary: str
    llm: Any
    top_k: int
    retrieval_mode: str
    answer_protocol: str
    sources: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    model_output: str
