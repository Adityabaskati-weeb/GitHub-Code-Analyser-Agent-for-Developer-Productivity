"""Run the small synthetic answerability smoke set with local inference."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from langchain_core.messages import HumanMessage
from evaluation.run import HERE
from src.config.settings import get_llm
from src.nodes.summarize_repo_node import summarize_repo_node


async def generate(output):
    if output.exists():
        raise ValueError('Choose a new output path; existing reports are not overwritten')
    raw = (HERE / 'challenges.json').read_text(encoding='utf-8')
    dataset = json.loads(raw)
    llm = get_llm('ollama')
    report = {'scope':dataset['scope'], 'dataset_sha256':hashlib.sha256(raw.encode()).hexdigest(),
              'model':llm.model, 'retrieval_mode':'code', 'answer_protocol':'quoted_v2',
              'reviewer_type':'unassigned', 'rows':[]}
    output.parent.mkdir(parents=True, exist_ok=True)
    for question in dataset['questions']:
        result = await summarize_repo_node({'parsed_files':dataset['files'],
            'messages':[HumanMessage(content=question['question'])], 'llm':llm, 'retrieval_mode':'code'})
        report['rows'].append({**question, 'reference_answer':question['answer'],
            'answer':result['summary'], 'sources':result['sources'],
            'model_output':result['model_output'], 'metrics':result['metrics'],
            'score':None, 'reviewer_reason':''})
        pending = output.with_suffix('.json.tmp')
        pending.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
        pending.replace(output)
        print(f'Saved {question["id"]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/challenges.json'))
    asyncio.run(generate(parser.parse_args().output))
