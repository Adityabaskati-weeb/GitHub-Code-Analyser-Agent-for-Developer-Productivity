# nodes/summarize_repo_node.py
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.config.settings import MAX_CHUNKS

async def summarize_repo_node(state: dict) -> dict:
    """
    Final summarization node.
    Combines global context + parsed files + user query
    into a multi-chat answer.
    """

    # print("Initializing Summarize Repo Node...") 

    llm = state.get("llm")
    global_context = state.get("global_context", "")
    parsed_files = state.get("parsed_files", [])
    intent = state.get("intent")
    keywords = state.get("keywords", [])
    targets = state.get("targets", {})
    selected_files = state.get("selected_files", [])


    if not llm:
        return {"messages": state.get("messages", []) + [
            SystemMessage(content="LLM not found in state, cannot summarize.")
        ]}

    user_query = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            # print(user_query)
            break
    
    if not user_query:
        user_query = "Provide a summary of this repository."

    # Reduce parsed files to a manageable token length
    merged_chunks = []

    selected_paths = [f.get("path") for f in selected_files if isinstance(f, dict)]
    # selected_paths_preview = "\n".join(f"- {p}" for p in selected_paths[:20])

    for f in parsed_files: #parsed_files[:MAX_CHUNKS]:
        path = f.get("path", "<unknown>")
        print(f"these are the files which are used by summarizer node: {path}")
        parsed_text = f.get("parsed", "")
        merged_chunks.append(f"\n### File: {path}\n{parsed_text}\n")

    merged_text = "\n".join(merged_chunks[:8000])

    system_msg = SystemMessage(content="""
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
""")

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

            Parsed File Content (truncated to {MAX_CHUNKS} files):
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
    # print(f"Response Variable: {response}")
    new_ai_msg = AIMessage(content=response.content)

    return {
        "summary": response.content,
        "messages": [new_ai_msg],
    }
