# Latent shape optimization

This is the sole example that uses `LatentShapeOpt`.

- Reference-aligned smoke: `python run.py --n-iters 1 --init-octree-depth 4 --octree-depth 6 --latent-lr 5e-2 --latent-optimizer adam --disable-pullback`
- Full reference trajectory: use the same command without `--n-iters 1`
