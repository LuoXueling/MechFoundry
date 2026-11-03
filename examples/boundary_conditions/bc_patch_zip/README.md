# BC patch zipper

The leaf includes the table CAD asset and procedural bracket.

- Smoke/branch-coverage case: `python run.py --case bracket`
- Reference case: `python run.py --case table`

Each command writes a face-colored `stitched.glb` and `stitched.npz`, whose
`face_labels` array retains the production zipper labels.
