# -*- coding: utf-8 -*-
"""Print a readable digest of one failed-question JSON for stage analysis.

Usage: python show_case.py <path-to-json>
"""
import json
import sys


def main():
    path = sys.argv[1]
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    ocr = d.get('ocr_data') or {}
    lines = []
    lines.append('TASK_ID: %s  (dataset=%s, difficulty=%s)' % (
        d.get('task_id'), d.get('dataset'), d.get('difficulty')))
    lines.append('FAILURE RESULT: %s' % (d.get('result'),))
    lines.append('=' * 30 + ' PROMPT ' + '=' * 30)
    lines.append(d.get('prompt', ''))
    lines.append('=' * 30 + ' ILR ' + '=' * 30)
    lines.append(d.get('ilr') or '(none)')
    lines.append('=' * 30 + ' GENERATED CODE ' + '=' * 30)
    lines.append(d.get('code', ''))
    lines.append('=' * 30 + ' TEST (first 800 chars) ' + '=' * 30)
    lines.append((d.get('test') or '')[:800])
    lines.append('=' * 30 + ' OCR RAW TEXTS ' + '=' * 30)
    texts = []
    for n in ocr.get('nodes', []):
        texts.append('#%s [%s] %r' % (n.get('node_id'), n.get('shape_type'), n.get('raw_text')))
    for e in ocr.get('edges', []) or []:
        texts.append('edge %s -> %s' % (e.get('from_node') or e.get('from'), e.get('to_node') or e.get('to')))
    lines.append('\n'.join(texts) if texts else '(no ocr_data)')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
