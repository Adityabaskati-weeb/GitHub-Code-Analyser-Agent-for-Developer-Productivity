"""Dependency-free, identifier-aware BM25 over bounded Python syntax units.

Only parses source; never imports or executes the inspected repository.
Non-Python and invalid Python use the original line-window fallback.
"""
import ast
from collections import Counter
import math
import re

from src.retrieval.evidence import Evidence, build_chunks

STOP = set('a an the is are was were be been being how what which when where does do did '
           'to of in on for from by with and or it its this that each'.split())


def tokens(text):
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    words = re.findall(r'[a-z]+|[0-9]+', text.lower())
    result = []
    for word in words:
        if word in STOP:
            continue
        if len(word) > 4:
            for suffix in ('ing', 'ed', 'es', 's'):
                if word.endswith(suffix) and not word.endswith('ss'):
                    word = word[:-len(suffix)].rstrip('e')
                    break
        result.append(word)
    return result


def syntax_chunks(files, max_lines=100):
    if max_lines <= 8:
        raise ValueError('max_lines must exceed the eight-line overlap')
    chunks = []
    for file in files:
        if file.get('error'):
            continue
        raw = file.get('raw')
        if not isinstance(raw, str) or not file['path'].endswith('.py'):
            chunks.extend(build_chunks([file]))
            continue
        try:
            tree = ast.parse(raw)
        except (SyntaxError, ValueError, RecursionError):
            chunks.extend(build_chunks([file]))
            continue
        lines = raw.splitlines()
        spans = []
        def visit(nodes):
            for node in nodes:
                if isinstance(node, ast.ClassDef):
                    visit(node.body)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                    spans.append((start, node.end_lineno))
        visit(tree.body)
        # Keep module/class-level material, including comments, in gap windows.
        cursor = 1
        ranges = []
        for start, end in sorted(spans):
            if cursor < start:
                ranges.append((cursor, start - 1, 40))
            ranges.append((start, end, max_lines))
            cursor = end + 1
        if cursor <= len(lines):
            ranges.append((cursor, len(lines), 40))
        for start, end, limit in ranges:
            step = limit - 8
            for low in range(start, end + 1, step):
                high = min(low + limit - 1, end)
                text = '\n'.join(line[:1000] for line in lines[low-1:high])
                if text.strip():
                    chunks.append(Evidence(file['path'], low, high, text))
                if high == end:
                    break
    return chunks


def search(query, files, top_k=8):
    if top_k <= 0:
        raise ValueError('top_k must be positive')
    chunks = syntax_chunks(files)
    documents = [tokens(c.text) for c in chunks]
    if not documents:
        return []
    df = Counter(t for doc in documents for t in set(doc))
    n = len(documents)
    avg = sum(map(len, documents)) / n or 1
    query_words = tokens(query)
    terms = set(query_words)
    # Match natural-language acronym phrases to code identifiers (e.g. IDF).
    terms.update(''.join(w[0] for w in query_words[i:i+width])
                 for width in (2, 3, 4) for i in range(len(query_words)-width+1))
    ranked = []
    for chunk, doc in zip(chunks, documents):
        freq = Counter(doc)
        score = 0.0
        for term in terms:
            count = freq[term]
            if count:
                idf = math.log(1 + (n - df[term] + .5) / (df[term] + .5))
                score += idf * count * 2.2 / (count + 1.2 * (.25 + .75 * len(doc) / avg))
        if score > 0:
            ranked.append((score, chunk))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in ranked[:top_k]]
