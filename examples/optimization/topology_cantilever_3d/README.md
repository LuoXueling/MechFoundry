# 3-D topology optimization

- Smoke: `python run.py --n-iters 1 --nx 16 --ny 25 --nz 4`
- Reference validation: `python run.py --n-iters 1`
- Full reference trajectory: `python run.py`

The load occupies `y <= 0.044`, so `ny` must be at least 23; coarser values
select only a collinear node row and cannot define the traction surface.
