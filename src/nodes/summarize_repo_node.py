# nodes/summarize_repo_node.py
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.config.settings import MAX_CHUNKS, TOP_K, RETRIEVAL_MODE
from src.retrieval.chunker import chunk_text
from src.retrieval.ranker import rank_chunks

async def summarize_repo_node(state: dict) -> dict:
    """
    Final summarization node.

    1. Splits each parsed file's text into overlapping word-level chunks.
    2. Ranks all chunks by relevance to the user query (lexical by default;
       semantic if sentence-transformers is available and RETRIEVAL_MODE=semantic).
    3. Feeds only the top-k most relevant chunks to the LLM, reducing token
       usage while improving answer quality.
    """

    llm = state.get("llm")
    global_context = state.get("global_context", "")
    parsed_files = state.get("parsed_files", [])
    intent = state.get("intent")
    keywords = state.get("keywords", [])
    targets = state.get("targets", {})
    selected_files = state.get("selected_files", [])

    # Allow per-run override via state (set by CLI --top-k flag)
    top_k: int = state.get("top_k") or TOP_K

    if not llm:
        return {"messages": state.get("messages", []) + [
            SystemMessage(content="LLM not found in state, cannot summarize.")
        ]}

    user_query = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    if not user_query:
        user_query = "Provide a summary of this repository."

    selected_paths = [f.get("path") for f in selected_files if isinstance(f, dict)]

    # ------------------------------------------------------------------
    # Build all chunks from parsed files
    # ------------------------------------------------------------------
    all_chunks: list[str] = []
    chunk_labels: list[str] = []  # tracks which file each chunk came from

    for f in parsed_files:
        path = f.get("path", "<unknown>")
        parsed_text = f.get("parsed", "")
        if not parsed_text:
            continue
        file_chunks = chunk_text(parsed_text)
        for i, c in enumerate(file_chunks):
            all_chunks.append(c)
            chunk_labels.append(f"### File: {path} (chunk {i + 1})\n")

    # ------------------------------------------------------------------
    # Rank and select top-k
    # ------------------------------------------------------------------
    if all_chunks:
        ranked_chunks = rank_chunks(user_query, all_chunks, top_k=top_k)
        print(
            f"[retrieval] {len(all_chunks)} total chunks -> "
            f"top {len(ranked_chunks)} selected (mode={RETRIEVAL_MODE})"
        )
        merged_text = "\n".join(ranked_chunks)
    else:
        # Fallback: truncated raw merge (original behaviour)
        merged_chunks = [
            f"\n### File: {f.get('path', '<unknown>')}\n{f.get('parsed', '')}\n"
            for f in parsed_files
        ]
        merged_text = "\n".join(merged_chunks[:MAX_CHUNKS])

    system_msg = SystemMessage(content=(
        "You are an expert software engineer and code analysis assistant. "
        "You receive:\n"
        "- A high-level repository context\n"
        "- A list of selected relevant files\n"
        "- Parsed content from those files\n"
        "- The user's question\n"
        "You must provide a precise, technically accurate answer.\n\n"
        "Requirements:\n"
        "- Use the parsed files and global context as primary ground truth.\n"
        "- If the user asks about a function, variable, directory, or pipeline, "
        "  focus on those elements specifically.\n"
        "- When describing locations, mention file names and (if available) roles "
        "  or responsibilities of those files.\n"
        "- If information is not present in the provided context, say so explicitly "
        "  instead of hallucinating.\n"
    ))

    human_msg = HumanMessage(content=f"""
            User Query:
            {user_query}

            Detected Intent: {intent}
            Keywords: {keywords}
            Targets: {targets}

            Global Repository Context:
            {global_context}

            Selected Files (preview):
            {selected_paths}

            Parsed File Content (top-{top_k} most relevant chunks):
            {merged_text}

            Now, based on the above information, answer the user's question as clearly and concretely as possible.
            If the intent is:
            - function_usage: explain where the function is defined and where it is used.
            - type_lookup: infer the variable type and show where it is defined/assigned.
            - pipeline_flow: describe the logical execution flow and main entry points.
            - directory_question: describe the purpose and contents of the directory.
            - architecture_summary/high_level_summary: explain the architecture and major components.

            If something cannot be determined from the provided context, clearly state the limitation.
""")

    response = await llm.ainvoke([system_msg, human_msg])
    new_ai_msg = AIMessage(content=response.content)

    return {
        "summary": response.content,
        "messages": [new_ai_msg],
    }
