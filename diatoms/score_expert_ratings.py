"""Score expert answers returned from the rating form built by make_expert_packet.py.

  python diatoms/score_expert_ratings.py --key expert_packet/answer_key.json \
      --answers expert_packet/returned/*.json --out results/expert_ratings.txt

Part 1 (realism): per rater and pooled, accuracy at telling real from generated (50% =
cannot tell them apart; binomial test), and the share of generated images called real.
Part 2 (species): for counterfactual items, share judged "B" (the target), "A", "neither";
the same for the hidden non-robust controls (expected "A"); Fisher test between the two;
mean plausibility; how often each kind of change was ticked. Agreement between raters:
Fleiss' kappa on the species answers (needs >= 2 raters).
"""
import argparse
import collections
import json

import numpy as np
from scipy.stats import binomtest, fisher_exact


def fleiss_kappa(table):
    """table: items x categories counts (same number of raters per item)."""
    table = np.asarray(table, float)
    n = table.sum(1)[0]
    p_j = table.sum(0) / table.sum()
    p_i = ((table ** 2).sum(1) - n) / (n * (n - 1))
    pe = (p_j ** 2).sum()
    return (p_i.mean() - pe) / (1 - pe) if pe < 1 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', required=True)
    ap.add_argument('--answers', nargs='+', required=True)
    ap.add_argument('--out')
    args = ap.parse_args()
    key = json.load(open(args.key))
    raters = [json.load(open(p)) for p in args.answers]
    L = [f'{len(raters)} rater(s): ' + ', '.join(f"{r.get('rater') or '?'} ({r.get('experience')})" for r in raters), '']

    # part 1
    L.append('PART 1: real photograph or generated?')
    pooled = [0, 0]
    gen_called_real = [0, 0]
    for r in raters:
        a = r['answers']
        ok = n = 0
        for iid, k in key.items():
            if k['part'] != 'realism' or f'{iid}_real' not in a:
                continue
            n += 1
            ok += int(a[f'{iid}_real'] == k['truth'])
            if k['truth'] == 'generated':
                gen_called_real[1] += 1
                gen_called_real[0] += int(a[f'{iid}_real'] == 'real')
        pooled[0] += ok
        pooled[1] += n
        if n:
            L.append(f"  {r.get('rater') or '?':12s} accuracy {ok / n:.0%} ({ok}/{n}), binomial p vs 50% = {binomtest(ok, n, .5).pvalue:.2g}")
    if pooled[1]:
        L.append(f'  pooled       accuracy {pooled[0] / pooled[1]:.0%} ({pooled[0]}/{pooled[1]}), p = {binomtest(*pooled, .5).pvalue:.2g}; '
                 f'generated images called real: {gen_called_real[0]}/{gen_called_real[1]}')
    L.append('')

    # part 2
    L.append('PART 2: which species does the modified image look like?')
    counts = {kind: collections.Counter() for kind in ('counterfactual', 'control_nonrobust')}
    plaus = collections.defaultdict(list)
    changed = {kind: collections.Counter() for kind in counts}
    n_items = collections.Counter()
    per_item = collections.defaultdict(collections.Counter)
    for r in raters:
        a = r['answers']
        for iid, k in key.items():
            if k['part'] != 'species' or f'{iid}_species' not in a:
                continue
            kind = k['kind']
            counts[kind][a[f'{iid}_species']] += 1
            per_item[iid][a[f'{iid}_species']] += 1
            n_items[kind] += 1
            if f'{iid}_plausible' in a:
                plaus[kind].append(int(a[f'{iid}_plausible']))
            for c in a.get(f'{iid}_changed', []):
                changed[kind][c] += 1
    for kind, c in counts.items():
        tot = sum(c.values())
        if not tot:
            continue
        L.append(f'  {kind:18s} n={tot:3d}  B {c["B"] / tot:5.0%}  A {c["A"] / tot:5.0%}  neither {c["neither"] / tot:5.0%}  '
                 f"can't tell {c['unsure'] / tot:5.0%}  | plausibility {np.mean(plaus[kind]) if plaus[kind] else float('nan'):.2f}/5")
        L.append('    changes ticked: ' + ', '.join(f'{k} {v / tot:.0%}' for k, v in changed[kind].most_common()))
    cf, ct = counts['counterfactual'], counts['control_nonrobust']
    if sum(cf.values()) and sum(ct.values()):
        tab = [[cf['B'], sum(cf.values()) - cf['B']], [ct['B'], sum(ct.values()) - ct['B']]]
        L.append(f'  "B" for counterfactuals vs controls: Fisher p = {fisher_exact(tab)[1]:.2g}')
    if len(raters) >= 2:
        cats = ['A', 'B', 'neither', 'unsure']
        full = [it for it, c in per_item.items() if sum(c.values()) == len(raters)]
        if full:
            L.append(f"  Fleiss' kappa (species answers, {len(full)} items rated by all): "
                     f'{fleiss_kappa([[per_item[it][c] for c in cats] for it in full]):.2f}')
    txt = '\n'.join(L)
    print(txt)
    if args.out:
        open(args.out, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
