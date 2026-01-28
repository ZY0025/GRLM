import json
import sys

path = 'toys_summaries_with_similarity.jsonl'
with open(path, encoding='utf-8') as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f'line {line_no} not valid: {e}')
            continue

        words = data.get('summary_words', [])
        if len(words) != 5 or "" in words:
            print(f"id={data.get('id')} No.{line_no} len(summary_words) ={len(words)} ≠5")
            print(line)