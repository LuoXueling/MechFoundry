# Surface mesh to octree

Run `python prepare.py`, then `python run.py --init-depth 3 --depth 6`.
The second command uses the production Warp BVH/octree path and writes
`octree.vtk`; use `--depth 7` for the finer reference mesh.
