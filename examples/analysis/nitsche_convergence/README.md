# Nitsche convergence

The leaf includes its comparison implementation, config, and reference cache.
Each run writes one `results.csv` table.

- Smoke (two levels, so a rate is measurable): `python run.py --depths 3 4 --ref-n 64`
- Reference: `python run.py --depths 3 4 5 6 --ref-n 96`
