from src.nodes.fetch_repo_metadata_node import fetch_repo_metadata_node
from src.nodes.global_context_node import global_context_node
from src.nodes.analyze_repo_node import analyze_tree_node
from src.nodes.fetch_and_parse_node import fetch_and_parse_node
from src.nodes.summarize_repo_node import summarize_repo_node
from src.nodes.query_analyser_node import query_analyser_node

from langgraph.graph import StateGraph, END
from state_schema import Agent_State

indexing_workflow = StateGraph(Agent_State)
qa_workflow = StateGraph(Agent_State)

indexing_workflow.add_node("fetch_metadata", fetch_repo_metadata_node)
indexing_workflow.add_node("global_context", global_context_node)
indexing_workflow.add_node("analyze_tree", analyze_tree_node)
indexing_workflow.add_node("fetch_and_parse", fetch_and_parse_node)
indexing_workflow.add_node("summarize", summarize_repo_node)
indexing_workflow.add_node("query_analyser", query_analyser_node)

indexing_workflow.add_edge("fetch_metadata", "query_analyser")
indexing_workflow.add_edge("query_analyser", "global_context")
indexing_workflow.add_edge("global_context", "analyze_tree")
indexing_workflow.add_edge("analyze_tree", "fetch_and_parse")
indexing_workflow.add_edge("fetch_and_parse", "summarize")
indexing_workflow.add_edge("summarize", END)

indexing_workflow.set_entry_point("fetch_metadata")

index_app = indexing_workflow.compile()

qa_workflow.add_node("query_analyzer", query_analyser_node)
qa_workflow.add_node("analyze_tree", analyze_tree_node)
qa_workflow.add_node("fetch_and_parse", fetch_and_parse_node)
qa_workflow.add_node("summarize", summarize_repo_node)

qa_workflow.set_entry_point("query_analyzer")
qa_workflow.add_edge("query_analyzer", "analyze_tree")
qa_workflow.add_edge("analyze_tree", "fetch_and_parse")
qa_workflow.add_edge("fetch_and_parse", "summarize")
qa_workflow.add_edge("summarize", END)

qa_app = qa_workflow.compile()

__all__ = ["index_app", "qa_app"]
