# -*- coding: utf-8 -*-
"""Print a readable digest of one qwen3 failed-question JSON, with prompt/test pulled from the source results file.

Usage: python show_qwen_case.py <json-filename-or-path>
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    arg = sys.argv[1]
    path = arg if os.path.isabs(arg) else os.path.join(HERE, arg)
    if not path.endswith('.json'):
        path += '.json'
    d = json.load(open(path, encoding='utf-8'))
    tid, ds = d['task_id'], d['dataset']
    src = f'F:/科研/flowchart/flowchart2code/output/{ds}/qwen3-vl-8b-instruct-qwen3-vl-8b-instruct/samples.jsonl_results.jsonl'
    rec = None
    with open(src, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r['task_id'] == tid:
                    rec = r
                    break
    print('TASK_ID: %s (dataset=%s)' % (tid, ds))
    print('FAILURE RESULT:', d.get('result'))
    ilr = {k: v for k, v in d.items() if k not in ('task_id', 'dataset', 'result', 'code')}
    print('=' * 30 + ' ILR (from JSON) ' + '=' * 30)
    print(json.dumps(ilr, ensure_ascii=False, indent=1))
    print('=' * 30 + ' GENERATED CODE ' + '=' * 30)
    print(d.get('code', ''))
    if rec:
        print('=' * 30 + ' PROMPT ' + '=' * 30)
        print(rec.get('prompt', ''))
        print('=' * 30 + ' TEST (first 1000 chars) ' + '=' * 30)
        print((rec.get('test') or '')[:1000])
        print('=' * 30 + ' OCR RAW TEXTS ' + '=' * 30)
        ocr = rec.get('ocr_data') or {}
        texts = []
        for nd in ocr.get('nodes', []):
            texts.append('#%s [%s] %r' % (nd.get('node_id'), nd.get('shape_type'), nd.get('raw_text')))
        for e in ocr.get('edges', []) or []:
            texts.append('edge %s -> %s' % (e.get('from_node', e.get('from')), e.get('to_node', e.get('to'))))
        print('\n'.join(texts) if texts else '(no ocr_data)')


if __name__ == '__main__':
    main()
