"""Gold-evidence diagnostic on baseline failures, NOT a retrieval benchmark.

Deliberately replaces retrieval using labeled file/anchor information. The model
still receives only the question and actual source, never the reference answer.
"""
import argparse
import asyncio
import json
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import HumanMessage
from evaluation.run import HERE, load_dataset, dataset_fingerprint
from src.config.settings import get_llm
from src.retrieval.code_search import syntax_chunks
from src.nodes.summarize_repo_node import summarize_repo_node


def oracle_chunks(question, files):
    candidates = [c for c in syntax_chunks(files)
                  if c.path == question['path'] and question['anchor'] in c.text]
    if not candidates:
        raise ValueError(f'No gold excerpt for {question["id"]}')
    return [min(candidates, key=lambda c:len(c.text))]


async def generate(output):
    if output.exists():
        raise ValueError('Choose a new output path; existing reports are not overwritten')
    corpus, questions = load_dataset()
    baseline = json.loads((HERE / 'live_answers.json').read_text(encoding='utf-8'))
    ids = {r['id'] for r in baseline['rows'] if r['score']==0}
    llm = get_llm('ollama')
    report = {'scope':'Gold file/anchor retrieval bypass on the 11 baseline failures. Diagnostic only; not deployable retrieval or held-out accuracy.',
              'model':llm.model, 'answer_protocol':'source_ids_v1', 'evidence_mode':'oracle',
              'dataset_sha256':dataset_fingerprint(), 'reviewer_type':'unassigned', 'rows':[]}
    output.parent.mkdir(parents=True, exist_ok=True)
    for q in questions:
        if q['id'] not in ids:
            continue
        selected = oracle_chunks(q, corpus['files'])
        # Patch only retrieval in this isolated diagnostic process, never the model.
        with patch('src.nodes.summarize_repo_node.retrieve', return_value=selected):
            result = await summarize_repo_node({'parsed_files':[], 'llm':llm,
                'messages':[HumanMessage(content=q['question'])],
                'retrieval_mode':'lexical', 'answer_protocol':'source_ids_v1'})
        report['rows'].append({'id':q['id'], 'question':q['question'], 'reference_answer':q['answer'],
            'answer':result['summary'], 'model_output':result['model_output'],
            'sources':result['sources'], 'metrics':result['metrics'], 'score':None, 'reviewer_reason':''})
        pending = output.with_suffix('.json.tmp')
        pending.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8', newline='\n')
        pending.replace(output)
        print(f'Saved {q["id"]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/oracle.json'))
    asyncio.run(generate(parser.parse_args().output))
