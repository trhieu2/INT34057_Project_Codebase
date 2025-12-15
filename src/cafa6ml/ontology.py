from __future__ import annotations
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple
import pandas as pd

def parse_obo(obo_path: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    go_parents = defaultdict(list)
    go_children = defaultdict(list)

    current = None
    parents = []

    with open(obo_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                current = None
                parents = []
                continue
            if line.startswith("id: GO:"):
                current = line.split("id: ")[1].strip()
                continue
            if current is None:
                continue
            if line.startswith("is_a: GO:"):
                p = line.split("is_a: ")[1].split(" !")[0].strip()
                parents.append(p)
            elif line.startswith("relationship: part_of GO:"):
                p = line.split("relationship: part_of ")[1].split(" !")[0].strip()
                parents.append(p)
            elif line == "":
                for p in parents:
                    go_parents[current].append(p)
                    go_children[p].append(current)
                current = None
                parents = []

    return dict(go_parents), dict(go_children)

def get_all_ancestors(term: str, go_parents: Dict[str, List[str]], cache: Dict[str, Set[str]]) -> Set[str]:
    if term in cache:
        return cache[term]
    seen: Set[str] = set()
    q = deque([term])
    while q:
        t = q.popleft()
        for p in go_parents.get(t, []):
            if p not in seen:
                seen.add(p)
                q.append(p)
    cache[term] = seen
    return seen

def get_all_descendants(term: str, go_children: Dict[str, List[str]], cache: Dict[str, Set[str]]) -> Set[str]:
    if term in cache:
        return cache[term]
    seen: Set[str] = set()
    q = deque([term])
    while q:
        t = q.popleft()
        for ch in go_children.get(t, []):
            if ch not in seen:
                seen.add(ch)
                q.append(ch)
    cache[term] = seen
    return seen

def propagate_predictions(predictions_df: pd.DataFrame, go_parents: Dict[str, List[str]]) -> pd.DataFrame:
    """
      - iterate per pid
      - term_scores dict with max
      - add ancestors with same score
    """
    from tqdm import tqdm
    print("Propagating predictions to ancestor GO terms...")
    ancestor_cache: Dict[str, Set[str]] = {}
    propagated_rows = []

    for pid, group in tqdm(predictions_df.groupby("pid"), desc="Propagating"):
        term_scores = {}

        for _, row in group.iterrows():
            term, score = row["term"], float(row["p"])
            term_scores[term] = max(term_scores.get(term, 0.0), score)

            for anc in get_all_ancestors(str(term), go_parents, ancestor_cache):
                term_scores[anc] = max(term_scores.get(anc, 0.0), score)

        for term, score in term_scores.items():
            propagated_rows.append({"pid": pid, "term": term, "p": score})

    propagated_df = pd.DataFrame(propagated_rows)
    print(f"Before: {len(predictions_df)}, After: {len(propagated_df)} predictions")
    return propagated_df
