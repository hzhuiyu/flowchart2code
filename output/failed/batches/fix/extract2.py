import json, os, sys, re

BASE = r'F:\科研\flowchart\flowchart2code'
BATCH = os.path.join(BASE, 'output/failed/batches/batch_2.json')
FIX = os.path.join(BASE, 'output/failed/batches/fix')

def slug(t):
    return re.sub(r'[^A-Za-z0-9]+', '-', t).strip('-')

def extract(task_id):
    d = json.load(open(BATCH, encoding='utf-8'))
    s = [x for x in d if x['task_id'] == task_id][0]
    os.makedirs(FIX, exist_ok=True)
    p = os.path.join(FIX, f'batch_2_{slug(task_id)}.sample.json')
    json.dump(s, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('wrote', p)

if __name__ == '__main__':
    for t in sys.argv[1:]:
        extract(t)
