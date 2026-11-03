# Four-support shape optimization

This is a distinct cylindrical body with four supports, a fixed lower patch,
and a distributed top load. It calls the production optimizer directly.

- Reference-aligned smoke: `python run.py --n-iters 1 --depth 7`
- Full reference trajectory: `python run.py`

Both retain the illustration's depth-7 mesh; the smoke run only limits the
number of optimization iterations.
