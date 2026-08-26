from typing import List, Dict, Any, Tuple

def filter_by_split(rows: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    """Filters dataset rows preserving official split names ('train', 'dev', 'test')."""
    return [r for r in rows if r.get("split") == split_name]
