from __future__ import annotations
import re

def to_acc(x: str) -> str:
    s = str(x).strip()
    # sp|A0A0C5B5G6|NAME  -> A0A0C5B5G6
    if "|" in s:
        parts = s.split("|")
        if len(parts) >= 2 and re.fullmatch(r"[A-Z0-9]+", parts[1]):
            return parts[1]
    return s

def load_fasta_ids(path: str) -> list[str]:
    ids = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(">"):
                header = line[1:].strip()
                pid_raw = header.split()[0]
                ids.append(to_acc(pid_raw))
    return ids
