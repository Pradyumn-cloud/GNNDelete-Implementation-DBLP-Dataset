"""Summarise results/**/*.json into mean +- std tables (markdown), grouped over seeds.

    python collect_results.py            # prints and writes results_summary.md
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def ms(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return '-'
    return f'{np.mean(v):.3f} ± {np.std(v):.3f}' if len(v) > 1 else f'{v[0]:.3f}'


def main():
    # Windows consoles default to cp1252, which cannot print the lambda / arrows below.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    rows = [json.load(open(p)) for p in glob.glob(os.path.join(ROOT, 'results', '*', '*', '*', '*.json'))]
    out = ['# 3DSSG results (mean ± std over seeds)\n']

    base = defaultdict(list)
    for r in rows:
        if r['method'] == 'original':
            base[(r['edge_set'], r['gnn'])].append(r)
    out += ['## Base models (test link-prediction AUROC)\n', '| edge set | gnn | seeds | Et AUROC | Et AUPR |', '|---|---|---|---|---|']
    for (es, gnn), v in sorted(base.items()):
        out.append(f'| {es} | {gnn} | {len(v)} | {ms([x["et_auroc"] for x in v])} | {ms([x["et_aupr"] for x in v])} |')

    groups = defaultdict(list)
    for r in rows:
        if r['method'] != 'original' or r.get('df'):
            if r.get('df'):
                groups[(r['edge_set'], r['gnn'], r['df'], r['df_size'], r['method'], r.get('alpha'))].append(r)
    cur = None
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], k[3], k[4], k[5] or 0)):
        es, gnn, df, size, method, alpha = key
        head = (es, gnn, df, size)
        if head != cur:
            cur = head
            out += [f'\n## {es} | {gnn} | delete {df} {size:g}\n',
                    '| method | λ | seeds | Et AUROC ↑ | Ed AUROC | MI ratio ↑ | time (s) | trainable params |',
                    '|---|---|---|---|---|---|---|---|']
        v = groups[key]
        out.append(f'| {method} | {"" if alpha is None else f"{alpha:g}"} | {len(v)} | {ms([x["et_auroc"] for x in v])} | '
                   f'{ms([x["ed_auroc"] for x in v])} | {ms([x.get("mi_ratio") for x in v])} | '
                   f'{ms([x.get("train_time_sec") for x in v]).replace(" ± ", "±")} | {int(np.mean([x["trainable_params"] for x in v])):,} |')
    text = '\n'.join(out) + '\n'
    print(text)
    with open(os.path.join(ROOT, 'results_summary.md'), 'w', encoding='utf-8') as f:
        f.write(text)


if __name__ == '__main__':
    main()
