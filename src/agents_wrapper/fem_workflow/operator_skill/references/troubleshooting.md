# Troubleshooting: message → cause → fix

`ok:false` with `restored: "card restored…"`: nothing is lost; fix the answer and send it
again. A card back with `api_error`, `preview_error` or `solve_error`: the step ran and
reported a problem. The newest tool's stderr is `tool_receipts/<key>.stderr.log`.

## Driver and CLI

| Message | Cause | Fix |
|---|---|---|
| `no idle GPU after waiting N s` | every GPU busy (foreign memory ≥ 1 GiB or utilization ≥ 10 %) | wait and retry (`python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py gpu` shows who); `--gpu N` / `--allow-busy-gpu` only with the user's permission |
| `another driver is working on this session` | a second driver on the session | wait for the first (`python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py status SID`) |
| `no card is waiting; pending step [...]: use retry` | a step without a card failed or was interrupted | `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py retry SID`; failing the same way, rewind to the card before it |
| `unknown regions [...]` / `components must be distinct integers` | the answer does not match the card | the card's region names; components 0/1/2 |
| lint `needs its own revised prompt` | CAD text not revised | keep `revise_prompt: true` |
| `case.json writes to …, outside this session` (`solve-backend`) | a copied session | answer the settings card once in the copy |
| `show`: `pending_node ['solve']` after the driver was killed | a solve interrupted mid-step | `rewind --kind shape_opt_settings` and answer again; not `retry` |
| `API probe failed` at start | missing key, or the provider is down | check the key; DeepSeek is the default |
| warning `the chosen GPU … barely moved while GPU(s) … grew` | a worker from another front end served the step unpinned | `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py gpu --stop-workers`, re-run the step |
| warning `another job appeared on this GPU` | a foreign process arrived mid-step | re-check the result; re-run if it looks off |

## Geometry

| Message | Cause | Fix |
|---|---|---|
| `inspect_surface failed: surface is not watertight` | an upload with open edges | fill the holes and upload again |
| `TripoSG produced no usable watertight surface component` | text mode fails outright on some prompts and seeds | change the seed or reword; then an SDXL drawing (geometry.md) |
| the drawing shows the object twice, or several objects | SDXL draws a set or several views at some seeds | never generated from: `"drawing_crop": "largest"`, else a new `image_seed` (geometry.md) |
| `SDXL is not available: …` | the Hunyuan3D env or its SDXL weights are missing | upload an image, or generate from text |
| `… did not produce a closed solid` (CAD-Coder) | an inconsistent revised description (a hole larger than its wall, parts that only touch) | correct the inconsistent part in `geometry.description` |
| a CAD part with other features or proportions than the brief | the generator does not control them (geometry.md) | the route that keeps coordinates, or a built section |
| a generated load-bearing part missing or fused | a weak prompt, a straight-on drawing, or `erosion_radius` above 0 | name the part and add "solid"; `erosion_radius` 0; another seed |
| `… is not ready here` | the optional generator's env or weights are missing | use TripoSG, or set up the env |

## Boundary conditions

| Message | Cause | Fix |
|---|---|---|
| `api_error: ValidationError: …` from BC synthesis | the model's selector JSON was invalid | retry once, then reword more simply: positional regions, or verified segment ids |
| `preview_error: BC 'x' selected no faces` | a selector matched nothing, often a zero-thickness box | reword with explicit spans; name the facing and segment |
| the region covers the wrong part | a mislabelled segment, or ambiguous words | describe it positionally; re-segment |
| a load region with low normal coherence | it wraps round several facings | add "only the faces whose normal points +y"; split into regions |
| `unknown segment ids` / `supplied segmentation has N labels for M faces` | stale segment ids | preview the segmentation again, then use the new names |
| `cannot spread a force over 'x': its area is unknown` | the region's record has no area | preview again, or give the traction yourself |
| `values name none of the approved regions` | the inferred names don't match | apply values by hand with the card's names |
| `held pieces the brief's contacts do not account for` | the held region's pieces are not the contact's patches | a surface split is already one contact; else the real contact is several patches (`pieces` + `pieces_why`) or one (revise the region) |

## Material

| Message | Cause | Fix |
|---|---|---|
| lint `implausible uniform material, likely a unit slip` | E or ρ in the wrong units | convert to the session's units |
| `density is required: the load case has a nonzero acceleration` | gravity without density | add the density |
| `the surface has no colour; texture_surface first` | parts or VoMP on a grey surface | paint it, or use uniform |
| `api_error … values look like a unit slip` (parts) | the part values are off by orders of magnitude | retry once, or set the parts with `apply_parts` |

## Paint

| Message | Cause | Fix |
|---|---|---|
| `NOT PAINTED: the painter crashed or failed on this surface …` (`texture_fallback.kind: plain_colour`) | the painter crashed or raised an error at the plan's seed and one more; the body got one flat colour | nothing to fix: go on to the materials; no seed or words paint it, never repaint. The report says so |
| `texture_fallback.kind: reseeded` | the plan's seed failed; the next seed painted | judge the painting as usual |
| `the painting failed: the painted surface carries no colour` | a painting that never reached the surface | change `texture_seed` (geometry.md, Texture) |

## Solve

| Symptom in `solve_error` / the log | Cause | Fix (one change per attempt) |
|---|---|---|
| `Newton solver did not converge in 20 iterations`, rel flat from iteration 1, `strain_estimate` below `strain_floor` | strain below the float32 floor | `use_mixed_precision: true` (the default) |
| same, rel growing | divergence: a large load, or missing supports | check the supports and load; `load_steps: 5`, `line_search_enabled: true` |
| `NaN residual at iteration k` | an inverted element: an absurd load or material | check units and magnitudes; `load_steps` |
| `the solve failed: its field is zero everywhere` | a failed solve, never "below resolution" | check the loads and holds reached the regions; solve again, else "could not be run" |
| `Linear solver failed …: JAX-AMG … stalled … ill-conditioned`, ‖b − Kx‖/‖b‖ under 1 | an ill-conditioned system: penalty-dominated, thin members, or a part contrast | the runner runs `solve-backend --backend pyamgx` itself; a converged Newton is the result. Else its `next` names the one change: first the setting the message names (`beta_max_over_e: 500`, a uniform material only; a free retry), else a triage cause (below), kept only if ‖b − Kx‖/‖b‖ drops. The scaling: last rung only (solver_settings.md §2) |
| `… it was still converging and ran out of iterations …` | a slow linear solve | the runner runs `solve-backend --backend jaxamg --max-iters 4000` itself; then `judge` and `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py finish SID` |
| the same for every setting, ‖b − Kx‖/‖b‖ around 1e-2 to 1e-3 | members a few leaves thick (known_issues.md #1) | a deeper depth if VRAM allows, or `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py solve-backend SID --backend pyamgx`; report thin members as under-resolved |
| the same with ‖b − Kx‖/‖b‖ of 1 or more, unscaled | a singular or nearly singular system (no fallback runs at 10 or more) | Supports, components and parts, below: one change, never a solver setting |
| the same with `scale_system_to_unit_diagonal` on | the residual is the scaled system's | remove the scaling (`plan.settings.scale_system_to_unit_diagonal: null`); after the last rung, report it |
| `Linear solver produced invalid solution … singular matrix` | no support, or a detached body | fix the supports; check `components` |
| `wp.MarchingCubes produced an empty surface` | the SDF has no zero crossing | check the scale card and the geometry |
| `BC zip init failed: …` or `BC zip bridge-merge: …` (optimization, `zip`) | the BC patches are not clean disks | disk-like regions about 8 leaves across; lower `gd_lr`; too complex: `mask` |
| `Did not converge in N iterations` (optimization) | informational | judge the trajectory |
| the run takes far longer than expected | a cold start (Warp JIT, model load), GPU contention, or depth 8 | `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py status SID`; `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py gpu` |
| `Newton solver did not converge` during optimization only | the design thinned badly | lower `gd_lr` (or `gd_lr_auto`); raise `target_volume_fraction` |

### Supports, components and parts

The causes a solver setting does not remove. The runner checks them before the first solve (a
`triage` stop, shown once, not an attempt) and names the one change in its `next` after a failed
one. ‖b − Kx‖/‖b‖ of 1 or more on the unscaled system means the system is singular or nearly so,
not ill-conditioned (that stalls at 1e-2 to 1e-4). One change per attempt, the first row that
applies; every solve and fallback counts against the solve limit.

| Cause | How it shows | The one change |
|---|---|---|
| a near rigid-body mode: a bulky part hanging from the holds by members 8 leaves thick or fewer | ‖b − Kx‖/‖b‖ of 1 or more on every backend; Newton's rel jumps by 1e2 to 1e5 | the members between the holds and the loads over 8 leaves thick (the section stand-in, objectives/routes.md §3.4, when the real ones are thinner); or, only where each loaded part is held in use, every loaded part held as well |
| a part nearly incompressible: ν 0.48 or more | a stall (1e-2 to 1e-1) or worse, the same for every setting | ν 0.45 on those parts (`plan.material.parts`, or `plan.material`): a marked simplification the report states |
| a part-stiffness contrast of 1e3 or more (`parts`) | a stall or worse, the fallback's Newton oscillating | the soft parts' E raised to the stiffest / 100, in the same correction: a marked simplification; or one material when the soft parts carry no load the brief asks about |
| a hold of some components only, or a piece no hold reaches | ‖b − Kx‖/‖b‖ of 1 or more; `singular matrix` | hold every component; hold or join each piece |

A run whose materials were changed this way is "a simplified case was run: …"; the report lists
each change.

## Inspector and render

| Symptom | Cause | Fix |
|---|---|---|
| "predates the newest run" | the newest solve failed | fix the solve; quote nothing from stale files |
| render shows an older run | `render` draws the newest finished surface | re-run the solve |
| `matplotlib is missing and no mechfoundry-warp117 interpreter was found` | env discovery failed | pass `--compute-env` |

## Region words (what the preview shows → what to change in the words)

| Symptom at the regions stop | Cause | Fix |
|---|---|---|
| a support whose `selector` is `highest` for "lowest" words, or a load's `lowest` | the band read upside-down | add the facing ("faces whose normal points down (−y)") |
| a support of a few dozen faces on a pointed or curved bottom | "the lowest N %" caught only the tip | name the segment and a height in metres ("all faces below y = <h> m") |
| `preview_error: selected no faces` for a part named by noun | the noun matched no segment | "selected by position alone with no segment: all faces within <position>" |
| a region on a different part than the noun | a neighbouring segment, or a height band crossing into it | position only, a verified segment id, or a disc about an axis |
| a load's `normal_coherence` below what the face predicts | faces of another facing caught | add the facing cone or an exclusion ("not the <neighbour>") |
| a zero-thickness box, selecting nothing | a box at a face's height with no span | give all three spans |
| a support that covers most of the surface | the words named a part instead of its contact patch | name the contact patch: its facing and position |
| a support held in all components moves a visible fraction of the peak | the region did not hold in the solve; often a bound at the body's extreme | correct that selector in `plan.region_revisions` (boundary_conditions.md §2), before any setting |
