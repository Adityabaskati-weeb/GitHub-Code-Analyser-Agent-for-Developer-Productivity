"""Check cited locations against retrieved excerpts, not factual correctness."""
import re

LOCATION = re.compile(r"(?P<path>[\w./-]+):L(?P<start>\d+)(?:-L(?P<end>\d+))?")


def check_citations(answer, sources):
    citations = list(LOCATION.finditer(answer))
    if not citations:
        return {"status":"missing", "checked":0, "invalid":[]}
    invalid = []
    for citation in citations:
        start = int(citation["start"])
        end = int(citation["end"] or citation["start"])
        supported = any(source.raw_source and source.path == citation["path"]
                        and source.start_line <= start <= end <= source.end_line
                        for source in sources)
        if not supported:
            invalid.append(citation.group())
    return {"status":"invalid" if invalid else "valid", "checked":len(citations), "invalid":invalid}
