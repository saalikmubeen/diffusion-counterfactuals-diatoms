# Selected DVCE results (full run, 301 diatoms)

Rules: **best** = both targets reached with p >= 0.9, smallest change (LPIPS), independent agreement first; **typical** = both reached, LPIPS closest to the median; **worst** = 5 lowest target probability + 5 largest changes.
Full figures (with real reference photos): `figures/NNN.png`.

## best (`best.png`)

| figure | mean LPIPS | min p(target) | independent agreements (of 4) |
|---|---|---|---|
| `figures/291.png` | 0.058 | 0.97 | 4 |
| `figures/062.png` | 0.092 | 1.00 | 4 |
| `figures/149.png` | 0.105 | 1.00 | 4 |
| `figures/154.png` | 0.125 | 1.00 | 4 |
| `figures/145.png` | 0.129 | 1.00 | 4 |
| `figures/081.png` | 0.136 | 1.00 | 4 |
| `figures/292.png` | 0.145 | 1.00 | 4 |
| `figures/112.png` | 0.145 | 1.00 | 4 |
| `figures/067.png` | 0.148 | 1.00 | 4 |
| `figures/024.png` | 0.149 | 0.99 | 4 |

## typical (`typical.png`)

| figure | mean LPIPS | min p(target) | independent agreements (of 4) |
|---|---|---|---|
| `figures/244.png` | 0.164 | 1.00 | 2 |
| `figures/282.png` | 0.165 | 1.00 | 0 |
| `figures/164.png` | 0.164 | 0.98 | 2 |
| `figures/276.png` | 0.164 | 0.99 | 2 |
| `figures/011.png` | 0.165 | 1.00 | 4 |
| `figures/284.png` | 0.165 | 0.98 | 2 |
| `figures/055.png` | 0.166 | 0.99 | 0 |
| `figures/041.png` | 0.163 | 0.94 | 0 |
| `figures/085.png` | 0.166 | 0.98 | 0 |
| `figures/131.png` | 0.163 | 0.98 | 2 |

## worst (`worst.png`)

| figure | mean LPIPS | min p(target) | independent agreements (of 4) |
|---|---|---|---|
| `figures/191.png` | 0.233 | 0.00 | 0 |
| `figures/281.png` | 0.130 | 0.00 | 0 |
| `figures/294.png` | 0.165 | 0.00 | 0 |
| `figures/082.png` | 0.292 | 0.00 | 0 |
| `figures/155.png` | 0.143 | 0.00 | 0 |
| `figures/204.png` | 0.413 | 0.38 | 0 |
| `figures/235.png` | 0.315 | 0.83 | 0 |
| `figures/156.png` | 0.291 | 1.00 | 0 |
| `figures/200.png` | 0.285 | 1.00 | 3 |
| `figures/061.png` | 0.278 | 1.00 | 1 |

