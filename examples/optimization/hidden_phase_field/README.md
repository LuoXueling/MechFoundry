# Hidden phase-field optimization

Demonstrates interior void creation without the duplicate void-nucleation
example.

- Smoke: `python run.py --n-iters 2 --phi-start-iter 0 --depth 7`
- Full reference trajectory: `python run.py`

The smoke run uses the reference depth and activates hidden phi immediately;
the first step initializes the persistent field and the second exercises its
interior update. The reference schedule enables it at iteration 35 and runs 91
iterations so a void can form and continue evolving.
