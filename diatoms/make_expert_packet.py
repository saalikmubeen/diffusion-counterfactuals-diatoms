"""Build a blind rating packet for diatom experts (one self-contained HTML file).

Part 1 (realism): single images, half counterfactuals and half real test photos of the
same target species, shuffled. Question: real photo or generated?
Part 2 (species judgement): an original specimen of species A, a modified version, and
real reference photos of A and of B (same scale). Questions: which species does the
modified image resemble, what changed, could it be a real specimen? A few items are
hidden controls: counterfactuals of a NON-robust classifier, which look almost
unchanged (expected answer: still A).

Outputs in --out_dir: rating_form.html (send this), answer_key.json (keep private),
items/ (the images). Score the returned answers with diatoms/score_expert_ratings.py.

  python diatoms/make_expert_packet.py --out_dir expert_packet
"""
import argparse
import base64
import csv
import io
import json
import os
import random
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_classifier import DATA_NATIVE  # noqa: E402

S = 'L2_0_l1_0_l1.0_0.15_classifier_3_-1_-1_0.1_reg_lpips_0_example_200_steps_skip_100_start_p_sample_deg_0_s_1'
FULL = 'DiatlasVCEs/clf_r50madry_random_eps1.0__diff_ft_v2_nativesize/' + S
NONROB = 'DiatlasVCEs/clf_independent__diff_ft_v2_nativesize/' + S


def load(prefix, n0, n1):
    ds = [torch.load(f'{prefix}_wid_{w}_2_dvces/{n}.pth', map_location='cpu') for w, n in ((0, n0), (1, n1))]
    return dict(x=torch.cat([d['gt_imgs'].float() for d in ds]).mean(1).numpy(),
                cf=torch.cat([d['counterfactuals'][:, 0, 0].float().clamp(0, 1) for d in ds]).mean(1).numpy(),
                y=torch.cat([d['gt_labels'][:len(d['gt_imgs'])] for d in ds]).numpy(),
                t=torch.cat([d['targets'] for d in ds]).numpy())


def png_b64(a):
    buf = io.BytesIO()
    Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).save(buf, 'PNG')
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='expert_packet')
    ap.add_argument('--n_real', type=int, default=20)
    ap.add_argument('--n_species', type=int, default=20)
    ap.add_argument('--n_controls', type=int, default=4)
    ap.add_argument('--ckpt', default='DiatlasModels/r50madry_random_eps1.0/best.pt')
    ap.add_argument('--taxa', default='/mnt/diatoms/datasets/diatlas/taxonomic_code.csv')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    classes = torch.load(args.ckpt, map_location='cpu')['classes']
    names = {}
    with open(args.taxa) as f:
        for r in csv.DictReader(f, delimiter=';'):
            names[r['taxonomic_code']] = r['infra_species'] or r['species'] or r['genus']
    name = lambda code: f'{names.get(code, code)} ({code})'
    with open(os.path.join(DATA_NATIVE, 'splits_random.csv')) as f:
        rows = list(csv.DictReader(f))
    full, nonrob = load(FULL, 302, 300), load(NONROB, 50, 50)
    with open('evaluation/vce_pairs_paper_all.json') as f:
        source_paths = {e['path'] for e in json.load(f)['entries']}

    def photos(code, split_ok, k, exclude=()):
        pool = sorted(r['path'] for r in rows if r['species_code'] == code and r['split_page'] in split_ok
                      and r['path'] not in source_paths and r['path'] not in exclude)
        pick = rng.sample(pool, min(k, len(pool)))
        return [np.asarray(Image.open(os.path.join(DATA_NATIVE, q)).convert('L'), np.float32) / 255 for q in pick], pick

    # counterfactual items: one per source, random sources, split between the two parts
    sources = list(range(len(full['t']) // 2))
    rng.shuffle(sources)
    realism_src = sources[:args.n_real]
    species_src = sources[args.n_real:args.n_real + args.n_species]

    key, realism, species = {}, [], []
    used_real = set()
    for s in realism_src:
        k = 2 * s + rng.randint(0, 1)
        tgt = classes[full['t'][k]]
        real, picked = photos(tgt, ('test',), 1, used_real)
        if not real:
            real, picked = photos(tgt, ('val', 'train'), 1, used_real)
        used_real.update(picked)
        for kind, img in (('generated', full['cf'][k]), ('real', real[0])):
            iid = f'R{len(realism):02d}'
            realism.append(dict(id=iid, img=png_b64(img)))
            key[iid] = dict(part='realism', truth=kind, species=tgt, cf_index=int(k) if kind == 'generated' else None,
                            real_path=picked[0] if kind == 'real' else None)
    rng.shuffle(realism)

    def species_item(run, k, kind):
        a, b = classes[run['y'][k]], classes[run['t'][k]]
        ra, _ = photos(a, ('train',), 2)
        rb, _ = photos(b, ('train',), 2)
        return dict(a=name(a), b=name(b), orig=png_b64(run['x'][k]), mod=png_b64(run['cf'][k]),
                    ref_a=[png_b64(i) for i in ra], ref_b=[png_b64(i) for i in rb]), dict(source=a, target=b, kind=kind, cf_index=int(k))
    items = []
    for s in species_src:
        items.append(species_item(full, 2 * s + rng.randint(0, 1), 'counterfactual'))
    for k in rng.sample(range(len(nonrob['t'])), args.n_controls):
        items.append(species_item(nonrob, k, 'control_nonrobust'))
    rng.shuffle(items)
    for i, (item, info) in enumerate(items):
        iid = f'S{i:02d}'
        item['id'] = iid
        species.append(item)
        key[iid] = dict(part='species', **info)

    html = HTML.replace('__DATA__', json.dumps(dict(realism=realism, species=species)))
    with open(os.path.join(args.out_dir, 'rating_form.html'), 'w') as f:
        f.write(html)
    with open(os.path.join(args.out_dir, 'answer_key.json'), 'w') as f:
        json.dump(key, f, indent=1)
    print(f'{len(realism)} realism items, {len(species)} species items ({args.n_controls} hidden controls) -> '
          f'{args.out_dir}/rating_form.html ({os.path.getsize(os.path.join(args.out_dir, "rating_form.html")) / 1e6:.1f} MB); '
          f'answer key: {args.out_dir}/answer_key.json (do not send)')


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Diatom image rating</title>
<style>
:root { --bg:#fbfbf9; --fg:#1c1c1c; --muted:#666; --card:#fff; --line:#ddd; --accent:#2c5d8a; }
@media (prefers-color-scheme: dark) { :root { --bg:#161616; --fg:#eee; --muted:#aaa; --card:#202020; --line:#3a3a3a; --accent:#7fb0dc; } }
body { background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; margin:0; padding:16px; }
main { max-width: 980px; margin: 0 auto; }
h1 { font-size: 1.5rem; } h2 { margin-top: 2.5rem; border-bottom: 1px solid var(--line); padding-bottom: .3rem; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 14px; margin: 14px 0; }
.row { display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-start; }
figure { margin: 0; text-align: center; font-size: .82rem; color: var(--muted); width: 202px; flex: 0 0 auto; }
.big figure { width: 258px; }
.row > div { flex: 1 1 260px; min-width: 0; }
img { width: 200px; height: 200px; image-rendering: auto; border: 1px solid var(--line); background: #888; }
.big img { width: 256px; height: 256px; }
@media (max-width: 600px) { img, .big img { width: 150px; height: 150px; } figure, .big figure { width: 152px; } }
label { margin-right: 14px; white-space: nowrap; }
.q { margin-top: 10px; } .muted { color: var(--muted); font-size: .9rem; }
textarea, input[type=text] { width: 100%; box-sizing: border-box; font: inherit; padding: 6px; background: var(--bg); color: var(--fg); border: 1px solid var(--line); border-radius: 4px; }
button { font: inherit; padding: 10px 18px; background: var(--accent); color: #fff; border: 0; border-radius: 6px; cursor: pointer; }
#status { margin-left: 12px; }
</style></head><body><main>
<h1>Diatom image rating</h1>
<div class="card">
<p>Thank you for helping. This form takes about 25–30 minutes. Some images are real light-microscopy
photographs of diatoms from published atlases; others were generated by a computer model. Please judge
each image on its own, from your experience, without looking anything up. All images of a part are shown at the
same pixel scale, so size differences are real.</p>
<p>When you are done, click <b>Download answers</b> at the bottom and email the downloaded file back.
Your answers are only stored in this browser window until then.</p>
<p class="q">Your name or initials: <input type="text" id="rater" style="max-width:300px"></p>
<p class="q">Your experience with diatom identification:
<label><input type="radio" name="experience" value="expert"> regular identification work</label>
<label><input type="radio" name="experience" value="some"> some experience</label>
<label><input type="radio" name="experience" value="little"> little</label></p>
</div>

<h2>Part 1 · Real photograph or generated?</h2>
<p class="muted">For each image: is it a real photograph or generated by a computer model, and how sure are you?</p>
<div id="part1"></div>

<h2>Part 2 · Which species does the modified image look like?</h2>
<p class="muted">Each item shows a real specimen of species A, a <b>modified</b> version of it, and real reference photos of
species A and species B. Judge the modified image only. Some modified images may barely differ from the original.</p>
<div id="part2"></div>

<div class="card"><p class="q">Any general comments?</p><textarea id="general" rows="3"></textarea>
<p class="q"><button id="dl">Download answers</button><span id="status" class="muted"></span></p></div>
</main>
<script>
const DATA = __DATA__;
const img = b => `<img alt="" src="data:image/png;base64,${b}">`;
const radios = (n, opts) => opts.map(([v, l]) => `<label><input type="radio" name="${n}" value="${v}"> ${l}</label>`).join('');
const checks = (n, opts) => opts.map(([v, l]) => `<label><input type="checkbox" name="${n}" value="${v}"> ${l}</label>`).join('');
document.getElementById('part1').innerHTML = DATA.realism.map((it, i) => `
<div class="card"><div class="row big"><figure>${img(it.img)}<figcaption>Image ${i + 1} of ${DATA.realism.length}</figcaption></figure>
<div><p class="q">This image is:<br>${radios(it.id + '_real', [['real', 'a real photograph'], ['generated', 'generated']])}</p>
<p class="q">How sure?<br>${radios(it.id + '_conf', [['1', 'guessing'], ['2', 'fairly sure'], ['3', 'certain']])}</p></div></div></div>`).join('');
document.getElementById('part2').innerHTML = DATA.species.map((it, i) => `
<div class="card"><p><b>Item ${i + 1} of ${DATA.species.length}</b> · A = <i>${it.a}</i> · B = <i>${it.b}</i></p>
<div class="row"><figure>${img(it.orig)}<figcaption>original (species A)</figcaption></figure>
<figure>${img(it.mod)}<figcaption><b>modified</b></figcaption></figure>
${it.ref_a.map(r => `<figure>${img(r)}<figcaption>reference: A</figcaption></figure>`).join('')}
${it.ref_b.map(r => `<figure>${img(r)}<figcaption>reference: B</figcaption></figure>`).join('')}</div>
<p class="q">The <b>modified</b> image looks most like:<br>${radios(it.id + '_species', [['A', 'species A'], ['B', 'species B'], ['neither', 'neither / another species'], ['unsure', "can't tell"]])}</p>
<p class="q">What changed between original and modified? (all that apply)<br>${checks(it.id + '_changed', [['outline', 'outline / overall shape'], ['apices', 'apices (ends)'], ['size', 'size (length or width)'], ['striae', 'striae (density or pattern)'], ['central', 'central area / raphe'], ['nothing', 'nothing visible'], ['other', 'other']])}</p>
<p class="q">Could the modified image be a real specimen?<br>${radios(it.id + '_plausible', [['1', '1 clearly not'], ['2', '2'], ['3', '3'], ['4', '4'], ['5', '5 fully plausible']])}</p>
<p class="q">Comment (optional): <input type="text" name="${it.id}_comment"></p></div>`).join('');

function collect() {
  const out = { rater: document.getElementById('rater').value, general: document.getElementById('general').value, answers: {} };
  const exp = document.querySelector('input[name=experience]:checked'); out.experience = exp ? exp.value : null;
  document.querySelectorAll('input[type=radio]:checked').forEach(e => { if (e.name !== 'experience') out.answers[e.name] = e.value; });
  const ch = {}; document.querySelectorAll('input[type=checkbox]:checked').forEach(e => (ch[e.name] = ch[e.name] || []).push(e.value));
  Object.assign(out.answers, ch);
  document.querySelectorAll('input[type=text][name]').forEach(e => { if (e.value) out.answers[e.name] = e.value; });
  return out;
}
const KEY = 'diatom-rating-draft';
document.addEventListener('change', () => { try { localStorage.setItem(KEY, JSON.stringify(collect())); } catch (e) {} });
try { const d = JSON.parse(localStorage.getItem(KEY) || 'null'); if (d) {
  document.getElementById('rater').value = d.rater || ''; document.getElementById('general').value = d.general || '';
  for (const [k, v] of Object.entries(d.answers || {})) {
    if (Array.isArray(v)) v.forEach(x => { const e = document.querySelector(`input[name="${k}"][value="${x}"]`); if (e) e.checked = true; });
    else { const e = document.querySelector(`input[name="${k}"][value="${v}"]`); if (e) e.checked = true;
           else { const t = document.querySelector(`input[type=text][name="${k}"]`); if (t) t.value = v; } } }
  if (d.experience) { const e = document.querySelector(`input[name=experience][value="${d.experience}"]`); if (e) e.checked = true; } } } catch (e) {}
document.getElementById('dl').onclick = () => {
  const out = collect(); out.saved = new Date().toISOString();
  const nAns = Object.keys(out.answers).filter(k => k.endsWith('_real') || k.endsWith('_species')).length;
  const nItems = DATA.realism.length + DATA.species.length;
  const blob = new Blob([JSON.stringify(out, null, 1)], { type: 'application/json' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = `diatom_ratings_${(out.rater || 'anonymous').replace(/[^a-z0-9]+/gi, '_')}.json`; a.click();
  document.getElementById('status').textContent = `Saved (${nAns} of ${nItems} main questions answered). Please email the downloaded file.`;
};
</script></body></html>'''

if __name__ == '__main__':
    main()
