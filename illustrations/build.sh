#!/usr/bin/env bash
# Rebuild every paper figure in this folder from its source.
# Usage:  ./build.sh            # build all
#         ./build.sh octree     # SVG octree schematic
#         ./build.sh workflow   # octree-workflow (2-D bunny field) figure
#         ./build.sh bunny      # ParaView 3-D bunny-SDF figure
#         ./build.sh sdfoctree  # uniform-grid vs octree SDF-cube figure
#         ./build.sh bcregion   # BC-region identification figure
#         ./build.sh bcautobunny # automatic shape-diameter BC workflow
#         ./build.sh pfbunny   # PartField feature field + segmentation
#         ./build.sh nitsche    # diffused-Nitsche boundary-region figure
#         ./build.sh voidnuc    # hidden-phase-field void-nucleation figure
#         ./build.sh shapeopt   # forward+backward shape-optimization loop figure
#         ./build.sh hiddenphi  # hidden-phi optimization-loop figure
#         ./build.sh shapeoptmerged # THE shape-optimisation figure: one loop, optional hidden phi
#         ./build.sh soframes   # shape-opt design-evolution frame series (+movie)
#         ./build.sh hpframes   # hidden-phi design-evolution frame series (+movie)
#         ./build.sh bczip      # 7-step BC-restoration (zip) workflow figure
#         ./build.sh bczipx     # the BC-restoration branches bczip never reaches
#         ./build.sh nitscheconv # twisting-box mesh-convergence study
#         ./build.sh nitscheconvuniform # same study on fixed-depth octrees
#         ./build.sh nitscheconvuniformsharp # fixed-depth, sharp volume
#         ./build.sh nitscheconvsharp # adaptive octree, sharp volume
#         ./build.sh overview   # Figure 1: the whole loop on the bunny (needs ovassets once)
#         ./build.sh ovassets   # Figure 1's generative inputs, opt-in (see build_ovassets)
#         ./build.sh agenticresults # agentic FEM runs, one row each (AR_RUNS=...)
#         ./build.sh gallery        # the 50 Opus objects in one amphitheater
#         ./build.sh agenticcompare # Haiku / Sonnet / Opus side by side, letter pages (ARC_*=...)
#         ./build.sh showcase  # variation figure: shapes / materials / loadings (SC_*_RUNS=...)
#         ./build.sh challenge # intro figure: the set-up's challenges for engineer / agent, and Figure 2's answers
#         ./build.sh introoverview # challenge_overview stacked above method_overview, one vector page
#         ./build.sh operator  # the AI agent operator's loop: steps, gates, stops, driver calls
#         ./build.sh demovideo # 28.8 s demo movie: CLI -> park bench through every look, optimised -> 21-object grid
#
# LAYOUT: each figure's scripts + regenerable intermediates live in its own
# subfolder (octree/, workflow/, ...); every target runs there (so the scripts'
# cwd-relative and __file__-relative paths resolve inside it) and then moves the
# FINAL .pdf/.png deliverables back up to this root, where the paper includes
# them.  One shared input: bunny/prep_bunny_sdf.py, which supplies the SAME
# bunny to bunny, sdfoctree, bcregion and nitsche.  _scratch/ holds orphan
# experiments; cad/ (Poly Haven table) lives under bczip/.
# See AGENTS.md for the toolchains and conventions.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"

DPI=300
CONDA="conda run -n mechfoundry-warp117 python"
# core/configs live here; hp_prep.py's own ../src path only resolved pre-reorg
# (when it ran from illustrations/), so its invocations set PYTHONPATH explicitly.
SRC=/home/xl2254/implicit_optimize/src/warp_wrapper
REPO="${SRC%/src/warp_wrapper}"
PV_ENV=(env
  __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
  VTK_DEFAULT_EGL_DEVICE_INDEX="${PV_GPU:-0}")

# --- SVG schematics (Inkscape): NAME.svg -> NAME.pdf + NAME.png (in cwd) ------
svg_render() {
  local name="$1"
  [ -f "${name}.svg" ] || { echo "skip: ${name}.svg not found"; return 0; }
  inkscape "${name}.svg" --export-type=pdf --export-text-to-path \
    --export-area-drawing --export-filename="${name}.pdf"
  inkscape "${name}.svg" --export-type=png --export-dpi="${DPI}" \
    --export-area-drawing --export-filename="${name}.png"
}

# --- octree schematic: generator emits the SVG master, Inkscape renders -------
build_octree() {
  ( cd octree
    $CONDA gen_octree_nitsche.py
    svg_render octree_nitsche
    mv -f octree_nitsche.pdf octree_nitsche.png "$ROOT"/ )
  echo "built: octree_nitsche.pdf octree_nitsche.png"
}

# --- octree-workflow figure: real 2-D bunny field -> multi-panel matplotlib --
build_workflow() {
  ( cd workflow
    $CONDA wf_bunny_field.py      # writes bunny_slice.npz (real bunny SDF profile)
    $CONDA wf_figure.py           # writes octree_step1..4.{png,pdf}
    rm -f bunny_slice.npz         # intermediate; regenerable
    mv -f octree_step1.pdf octree_step1.png octree_step2.pdf octree_step2.png \
          octree_step3.pdf octree_step3.png octree_step4.pdf octree_step4.png "$ROOT"/ )
  echo "built: octree_step1..4.{png,pdf}"
}

# --- ParaView 3-D figure: prep (conda) -> render (pvbatch, GPU) --------------
build_bunny() {
  ( cd bunny
    $CONDA prep_bunny_sdf.py                       # writes bunny.stl, bunny_sdf.vtk, .meta
    "${PV_ENV[@]}" pvbatch render_bunny_sdf.py      # writes bunny_sdf_pv.png
    rm -f bunny.stl bunny_sdf.vtk bunny_sdf.meta    # regenerable
    mv -f bunny_sdf_pv.png "$ROOT"/ )
  echo "built: bunny_sdf_pv.png"
}

# --- SDF-cube panels: uniform grid vs octree hex mesh (both cut in half) -------
# Reuses bunny/prep_bunny_sdf.py (shared with the bunny figure) for bunny.stl.
build_sdfoctree() {
  ( cd sdfoctree
    $CONDA ../bunny/prep_bunny_sdf.py                 # bunny.stl + bunny_sdf.vtk + .meta
    $CONDA prep_octree_sdf.py                         # octree.vtk (Warp GPU octree gen)
    "${PV_ENV[@]}" pvbatch render_sdf_panel.py uniform
    "${PV_ENV[@]}" pvbatch render_sdf_panel.py octree
    $CONDA compose_sdf_panels.py                      # -> bunny_sdf_octree.{png,pdf}
    rm -f bunny.stl bunny_sdf.vtk bunny_sdf.meta octree.vtk octree_stats.txt \
          panel_uniform.png panel_octree.png
    mv -f bunny_sdf_octree.pdf bunny_sdf_octree.png "$ROOT"/ )
  echo "built: bunny_sdf_octree.{png,pdf}"
}

# --- BC-region figure: same bunny + same octree as sdfoctree and nitsche -------
# No TripoSG: bc_bunny_select.py asserts the mesh matches bunny_sdf_octree's.
# bcregion/bc_bunny_geom.py (the decoder path that used to feed this figure) is
# kept in the tree as the record of how that geometry was made, but nothing in
# build.sh runs it any more, and nothing reads _shared/bc_bunny_geom.npz.
build_bcregion() {
  ( cd bcregion
    $CONDA ../bunny/prep_bunny_sdf.py                     # bunny.stl + bunny_sdf.meta
    $CONDA bc_bunny_select.py                             # production SurfaceSelectors -> STL/VTK
    "${PV_ENV[@]}" pvbatch render_bc_bunny.py             # -> bc_panel_{a,b,c}.png
    $CONDA bc_bunny_figure.py                             # -> bc_region_id.{pdf,png}
    rm -f bc_full.stl bc_seed.stl bc_ears.stl bc_rest_seed.stl bc_rest_ears.stl \
          bc_ear_context.stl bc_ear_focus.stl \
          bunny.stl bunny_sdf.vtk bunny_sdf.meta
    mv -f bc_region_id.pdf bc_region_id.png "$ROOT"/ )
  echo "built: bc_region_id.pdf bc_region_id.png"
}

# --- Automatic BC workflow: separate from the established selector figure ---
build_bcautobunny() {
  ( cd bcautobunny
    $CONDA ../bunny/prep_bunny_sdf.py                  # retained signed-distance grid
    CUDA_VISIBLE_DEVICES="${BC_GPU:-1}" $CONDA auto_bunny_prep.py
    env "${PV_ENV[@]:1}" VTK_DEFAULT_EGL_DEVICE_INDEX="${BC_GPU:-1}" \
        pvbatch render_auto_bunny.py
    $CONDA auto_bunny_figure.py
    rm -f bunny.stl bunny_sdf.vtk bunny_sdf.meta auto_surface.vtk \
          auto_seed*.stl auto_selected*.stl previous_selected*.stl \
          auto_selected_context.vtk \
          auto_panel_*.png
    mv -f automatic_bc_bunny.pdf automatic_bc_bunny.png "$ROOT"/ )
  echo "built: automatic_bc_bunny.{pdf,png}"
}

# --- PartField bunny: the learned feature field and the segments it clusters to
# The counterpart of bcautobunny's shape-diameter panels, on the SAME bunny and
# the same camera.  PartField is an optional install living in its own conda
# env, so this target is not part of `all`; it resolves the interpreter through
# the production readiness check rather than hard-coding a path, and fails loud
# when the model is absent.
build_pfbunny() {
  # Reads the PartField run Figure 1 shows (overview/assets/seg: its features and
  # k = 8 parts, made by the production segment.py in `ovassets segments` and
  # sha-checked): PartField is not seeded, so running it again here would give
  # this figure other parts than the overview's.
  [ -f overview/assets/MANIFEST.json ] || { echo "no overview/assets: run ./build.sh ovassets first"; return 1; }
  ( cd pfbunny
    set -e
    $CONDA ../bunny/prep_bunny_sdf.py                  # the shared bunny surface
    $CONDA pf_surface.py                               # -> pf_surface.stl + topology
    CUDA_VISIBLE_DEVICES="${PF_GPU:-1}" $CONDA pf_prep.py
    $CONDA pf_palette_check.py                         # gate, before any pixel
    env "${PV_ENV[@]:1}" VTK_DEFAULT_EGL_DEVICE_INDEX="${PF_GPU:-1}" \
        pvbatch render_pf_bunny.py
    $CONDA pf_figure.py
    rm -f bunny.stl bunny_sdf.vtk bunny_sdf.meta pf_surface.stl pf_topology.npz \
          pf_surface.vtk pf_panel_*.png pf_palette.txt
    mv -f partfield_feature.pdf partfield_feature.png \
          partfield_segmentation.pdf partfield_segmentation.png "$ROOT"/ )
  echo "built: partfield_feature.{pdf,png} partfield_segmentation.{pdf,png}"
}

# --- diffused-Nitsche figure: what the method needs, on bunny_sdf_octree's mesh
# Same geometry AND same octree call as sdfoctree (so the two figures print the
# same mesh) — hence the same prep_bunny_sdf line build_sdfoctree runs.  No
# TripoSG anywhere in this target; dn_prep asserts the mesh matches.
build_nitsche() {
  ( cd nitsche
    $CONDA ../bunny/prep_bunny_sdf.py                     # bunny.stl + bunny_sdf.meta
    $CONDA dn_prep.py                                     # -> dn_full/trim + surface + stats
    "${PV_ENV[@]}" pvbatch render_dn.py                   # -> dn_panel_{a,b,c,d,e}.png
    DN_SOLVE_PREFIX=dn_solve CUDA_VISIBLE_DEVICES="${DN_GPU:-0}" \
        $CONDA ../overview/ov_bunny_solve.py               # -> dn_solve.vtu + stats
    env "${PV_ENV[@]:1}" VTK_DEFAULT_EGL_DEVICE_INDEX="${DN_GPU:-0}" \
        DN_SOLVE_PREFIX=dn_solve DN_SOLVE_RENDER=dn_panel_f.png \
        pvbatch ../overview/render_ov_solve.py              # -> dn_panel_f.png
    $CONDA dn_figure.py                                   # -> diffused_nitsche.{pdf,png}
    rm -f dn_full.vtk dn_trim.vtk dn_surf_*.stl \
          bunny.stl bunny_sdf.vtk bunny_sdf.meta dn_solve.vtu dn_panel_f.png
    mv -f diffused_nitsche.pdf diffused_nitsche.png "$ROOT"/ )
  echo "built: diffused_nitsche.pdf diffused_nitsche.png"
}

# --- focused void-nucleation figure from the canonical hidden-phi capture -----
# This is a second composition of hiddenphi's ONE numerical run, not another
# optimizer execution.  Reusing the hp fields/surfaces makes every quoted count
# and every pixel refer to the same captured nucleation event.
build_voidnuc() {
  build_hiddenphi
  ( cd voidnuc
    $CONDA render_vn.py                                  # aliases hiddenphi panels
    $CONDA vn_figure.py                                   # -> void_nucleation.{pdf,png}
    mv -f void_nucleation.pdf void_nucleation.png "$ROOT"/ )
  echo "built: void_nucleation.pdf void_nucleation.png"
}

# --- shape-optimization loop figure: real ShapeOptTrimmed pipeline -----------
# so_prep runs the actual optimizer on a cylinder (top load, 4 fixed square feet,
# 50% volume target) and captures every forward/backward intermediate at the
# compliance-turn iteration; render_so draws the panels; so_figure composes the
# loop.  Runs on GPU ${SO_GPU:-1} (GPU 0 is often shared); ~4 min total.
#
# The optimiser run is CACHED: so_prep is skipped when its outputs are present, so
# iterating on the render/compose does not re-run the ~10-min optimisation.  The
# fields/masks/trajectory it writes are KEPT (they are the cache); only the large
# prep-internal files nothing downstream reads are cleaned up.  Force a fresh run
# with `rm shapeopt/so_design.vtk` (or `./build.sh shapeopt` after deleting it).
build_shapeopt() {
  ( cd shapeopt
    [ -f so_design.vtk ] || XLA_PYTHON_CLIENT_PREALLOCATE=false \
      CUDA_VISIBLE_DEVICES="${SO_GPU:-1}" $CONDA so_prep.py  # cached optimiser run
    $CONDA so_split_bc.py                                     # -> so_body + so_bc_dir + so_bc_neu (panel g)
    $CONDA ../_shared/table_load_arrows.py so_ so_load_arrows.stl  # Geometry + BC's load arrows
    # dJ/du for shape_opt_merged: computed on the CPU from the cached run's own VTU of
    # the captured iteration (so_dJdu.py).  so_dJdu.vtk is a cache; after a fresh
    # so_prep run, point SO_RUN_VTU at that run's vtu/solution_t00<capture>.vtu.
    [ -f so_dJdu.vtk ] || pvpython so_dJdu.py "${SO_RUN_VTU:-$SRC/output/shape_opt_cantilever-2026-09-24-08-27-05/vtu/solution_t0040.vtu}"
    env "${PV_ENV[@]:1}" VTK_DEFAULT_EGL_DEVICE_INDEX="${SO_GPU:-1}" \
        pvbatch render_so.py                                  # -> panel_so_{a,b,d,e,g}.png
    $CONDA so_figure.py                                       # -> shape_opt_loop.{png,pdf}
    $CONDA so_objective.py                                    # -> so_objective.{png,pdf}
    # keep the cache (so_{design,solve,grad,updated}.vtk, so_bc_masks.npz,
    # so_evo_*.stl, so_{stats,traj}.txt) + the regenerable panels; drop only the
    # big prep-internal files that render/split/compose never read.
    rm -f so_bg.vtk so_new.vtk so_mc_raw.stl so_mc_clean.stl so_stitched.stl so_bcnodes.npy
    mv -f shape_opt_loop.pdf shape_opt_loop.png so_objective.pdf so_objective.png "$ROOT"/ )
  echo "built: shape_opt_loop.{pdf,png} so_objective.{pdf,png}"
}

# --- hidden-phi loop figure: shape_opt_loop's structure, phase-field content --
# hp_prep runs the same optimiser in phase-field mode until a void nucleates,
# hp_cut sections the surfaces (CPU trimesh), render_hp draws the panels, and
# hp_figure composes the loop.  Same GPU handling as shapeopt.
#
# The optimiser run is CACHED (~15 min): hp_prep is skipped when hp_pre.vtk is
# present, so iterating on cut/render/compose never re-runs it.  Everything
# hp_prep writes is reused downstream, so nothing is cleaned up.  Force a fresh
# run with `rm hiddenphi/hp_pre.vtk`.
build_hiddenphi() {
  ( cd hiddenphi
    [ -f hp_pre.vtk ] || XLA_PYTHON_CLIENT_PREALLOCATE=false \
      PYTHONPATH="$SRC" CUDA_VISIBLE_DEVICES="${HP_GPU:-1}" \
      $CONDA hp_prep.py  # cached optimiser run
    # rtree/libspatialindex needs the conda C++ runtime when build.sh is
    # launched outside an activated environment.
    env LD_LIBRARY_PATH="/home/xl2254/anaconda3/envs/mechfoundry-warp117/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        SPATIALINDEX_C_LIBRARY=/home/xl2254/anaconda3/envs/mechfoundry-warp117/lib/libspatialindex_c.so \
        $CONDA hp_cut.py                                    # -> hp_e_*/hp_evo_*_{shell,face,ghost}.stl (CPU)
    $CONDA ../_shared/table_load_arrows.py hp_full_ hp_load_arrows.stl  # Geometry + BC's load arrows
    env "${PV_ENV[@]:1}" VTK_DEFAULT_EGL_DEVICE_INDEX="${HP_GPU:-1}" \
        pvbatch render_hp.py                                 # -> hp_panel_*.png
    $CONDA hp_figure.py                                      # -> hidden_phi_loop.{png,pdf}
    $CONDA hp_objective.py                                   # -> hp_objective.{png,pdf}
    mv -f hidden_phi_loop.pdf hidden_phi_loop.png hp_objective.pdf hp_objective.png "$ROOT"/ )
  echo "built: hidden_phi_loop.{pdf,png} hp_objective.{pdf,png}"
}

# --- the merged shape-optimisation figure (the paper's main deliverable) -----
# One loop in Figure 1's visual language with the hidden phase field as an
# OPTIONAL dashed branch, and both runs' design evolution on the right.  It reads
# the render caches of both loop targets (their *_m_* slots, rendered onto the
# block tints), so it runs them first; neither re-runs an optimiser while its
# cache (so_design.vtk / hp_pre.vtk) exists.
build_shapeoptmerged() {
  build_shapeopt
  build_hiddenphi
  # b's own frames (§5j round 11, ../_shared/evo_style.py): fainter near half, voids
  # darkened by openness (void_open.py, CPU warp, ~1 s a mesh).  The shape run's
  # panel_so_m_evo_*.png are a committed cache, re-made when absent or with SOM_B_RENDER=1;
  # the hidden run's are git-ignored like the rest of hiddenphi's panels.
  ( cd shapeopt
    if [ -n "${SOM_B_RENDER:-}" ] || ! ls panel_so_m_evo_*.png >/dev/null 2>&1; then
      for it in 0 40 100; do
        $CONDA ../_shared/void_open.py so_evo_$it.stl so_evo_${it}_shell.stl \
          so_evo_${it}_ghost.stl so_evo_$it
      done
      env "${PV_ENV[@]:1}" VTK_DEFAULT_EGL_DEVICE_INDEX="${PV_GPU:-1}" \
        pvbatch render_so.py m_evo                           # -> panel_so_m_evo_<it>.png
    fi )
  ( cd hiddenphi
    for it in 20 90; do
      $CONDA ../_shared/void_open.py hp_evo_$it.stl hp_evo_${it}_shell.stl \
        hp_evo_${it}_ghost.stl hp_evo_$it
    done
    $CONDA ../_shared/void_open.py hp_bc_masks.npz \
      hp_e_body.stl,hp_e_dir.stl,hp_e_neu.stl,hp_e_void.stl hp_e_ghost.stl hp_e
    env "${PV_ENV[@]:1}" VTK_DEFAULT_EGL_DEVICE_INDEX="${HP_GPU:-1}" \
      pvbatch render_hp.py m_evo_20 m_evo_cap m_evo_90 )    # -> hp_panel_m_evo_*.png
  ( cd shapeopt
    # row c, the geometry clean-up (experiments/geometry_cleanup): its panels are a
    # cache (panel_som_cu_*.png + som_cleanup.json, committed), re-made only when absent
    # or with SOM_CU_RENDER=1, from the clean-up's OWN outputs in output/geometry_cleanup
    # (sessions, render/it_<k>.vtk, results.json) -- nothing is re-run.  The prep is
    # CPU-only (warp's ray queries on the CPU); the render checks the change map
    # against the clean-up's contact-sheet tile first.
    if [ -n "${SOM_CU_RENDER:-}" ] || ! ls panel_som_cu_*_change.png >/dev/null 2>&1; then
      CUDA_VISIBLE_DEVICES="" $CONDA som_cleanup_prep.py      # -> _cleanup/*.vtk + som_cleanup.json
      env "${PV_ENV[@]:1}" VTK_DEFAULT_EGL_DEVICE_INDEX="${PV_GPU:-1}" \
        pvbatch render_som_cleanup.py --check crocodile      # change map == the clean-up's tile
      env "${PV_ENV[@]:1}" VTK_DEFAULT_EGL_DEVICE_INDEX="${PV_GPU:-1}" \
        pvbatch render_som_cleanup.py                        # -> panel_som_cu_<object>_{bc,change}.png
      rm -rf _cleanup
    fi
    $CONDA som_figure.py                                     # -> shape_opt_merged.{pdf,png}
    mv -f shape_opt_merged.pdf shape_opt_merged.png "$ROOT"/ )
  echo "built: shape_opt_merged.{pdf,png}"
}

# --- design-evolution frame series: one frame per optimiser iteration --------
# so_prep/hp_prep are re-run ONCE in FRAMES mode (SO_FRAMES/HP_FRAMES=1): they
# skip the loop-figure captures and instead dump every iteration's octree+d and
# design surface (so_fr_*/hp_fr_*), plus a cumulative optimiser wall clock.  The
# loop cache in shapeopt/ and hiddenphi/ is NOT touched (those files are never
# written in FRAMES mode).  Each frame is then sectioned + rendered with the same
# panel code the loop uses and composed side by side with its iteration number
# and wall-clock time.  Heavy raw frames (so_fr_*.vtk/.stl) are dropped after
# render; the panel PNGs + times are kept as the cache, so re-composing is cheap.
# Re-render from scratch with `rm shapeopt/panel_so_frbody_*.png`.
build_soframes() {
  ( cd shapeopt
    if ! ls panel_so_frbody_*.png >/dev/null 2>&1; then
      XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$SRC" \
        SO_FRAMES=1 SO_SAVE_EVERY=0 CUDA_VISIBLE_DEVICES="${SO_GPU:-1}" \
        $CONDA so_prep.py                                   # per-iter frames + times
      $CONDA so_fr_cut.py                                   # -> so_fr_*_{shell,face,ghost}
      env "${PV_ENV[@]:1}" VTK_DEFAULT_EGL_DEVICE_INDEX="${SO_GPU:-1}" \
        pvbatch render_so_frames.py                         # -> panel_so_fr_* + frbody_*
      rm -f so_fr_*.vtk so_fr_*.stl                         # heavy; panels+times are the cache
    fi )
  $CONDA fr_figure.py so                                    # -> shape_opt_evolution.{gif,mp4}+frames
  echo "built: shape_opt_evolution.gif (+ .mp4) + shape_opt_evolution_frames/"
}

build_hpframes() {
  ( cd hiddenphi
    if ! ls hp_panel_frbody_*.png >/dev/null 2>&1; then
      XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$SRC" \
        HP_FRAMES=1 HP_SAVE_EVERY=0 CUDA_VISIBLE_DEVICES="${HP_GPU:-1}" \
        $CONDA hp_prep.py
      # hp_fr_cut uses trimesh proximity queries through rtree.  Match the
      # static hidden-phi path so every frame can load libspatialindex when
      # build.sh is launched outside an activated conda environment.
      env LD_LIBRARY_PATH="/home/xl2254/anaconda3/envs/mechfoundry-warp117/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
          SPATIALINDEX_C_LIBRARY=/home/xl2254/anaconda3/envs/mechfoundry-warp117/lib/libspatialindex_c.so \
          $CONDA hp_fr_cut.py
      env "${PV_ENV[@]:1}" VTK_DEFAULT_EGL_DEVICE_INDEX="${HP_GPU:-1}" \
        pvbatch render_hp_frames.py
      rm -f hp_fr_*.vtk hp_fr_*.stl
    fi )
  $CONDA fr_figure.py hp
  echo "built: hidden_phi_evolution.gif (+ .mp4) + hidden_phi_evolution_frames/"
}

# --- BC-restoration workflow: 7-step zip figure (Y-bracket, GPU + ParaView) --
# Reads the Poly Haven table model from bczip/cad/ (bz_prep uses {HERE}/cad/).
build_bczip() {
  ( cd bczip
    PYTHONPATH="$SRC" $CONDA bz_prep.py     # -> bz_p*.vtk + bz_capture.npz + bz_stats.txt
    env "${PV_ENV[@]:1}" pvbatch render_bz.py   # -> bz_panel_{1..7}.png + *z.png
    $CONDA bz_figure.py                     # -> bc_zip_workflow.{png,pdf}
    rm -f bz_p*.vtk bz_patch_all.vtk bz_capture.npz bz_panel_*.png
    mv -f bc_zip_workflow.pdf bc_zip_workflow.png "$ROOT"/ )
  echo "built: bc_zip_workflow.pdf bc_zip_workflow.png"
}

# --- BC-restoration special cases: the branches bczip never reaches -----------
# Two fabricated mini-geometries (bored bar / split trunk), each the smallest
# thing that forces one branch of the paper's enumeration.  Same production
# calls as bczip, ~30 s total.
build_bczipx() {
  ( cd bczipx
    PYTHONPATH="$SRC" $CONDA bzx_prep.py     # -> bzx_*.vtk + bzx_stats.txt
    env "${PV_ENV[@]:1}" pvbatch render_bzx.py  # -> bzx_panel_{a..e}.png
    $CONDA bzx_figure.py                     # -> bc_zip_special_cases.{png,pdf}
    rm -f bzx_*.vtk bzx_panel_*.png
    mv -f bc_zip_special_cases.pdf bc_zip_special_cases.png "$ROOT"/ )
  echo "built: bc_zip_special_cases.pdf bc_zip_special_cases.png"
}

# --- twisting-box convergence: body-fitted FEM vs diffused Nitsche ------------
# nc_prep.py runs the sweep through its local comparison_helpers.py solvers:
# 12 solves in ~4 min, plus the body-fitted 96^3 reference at ~13 min.  BOTH are
# CACHED and both are gitignored: the reference as nc_ref_96.npz, the sweep as
# nc_data.txt + the nc_*.vtk it wrote.  With the cache warm this target is ~20 s
# (render + compose only), which is what iterating on the figure costs.  Re-run
# the solves with NC_FORCE=1; drop the reference by deleting the .npz.  Do force
# it before trusting a from-scratch rebuild.
# nc_palette_check.py runs FIRST as a gate: it exits non-zero if a series colour
# fails the CVD/contrast checks, and set -e stops the build there.  The sweep is
# the only GPU step and runs on GPU ${NC_GPU:-0} — panel (g) plots wall clock, so
# it has to be the same card every time for the numbers to mean anything.
build_nitscheconv() {
  ( cd nitscheconv
    $CONDA nc_palette_check.py                 # gate: CVD/contrast on the 3 series
    CUDA_VISIBLE_DEVICES="${NC_GPU:-0}" \
      $CONDA nc_prep.py                        # -> nc_data.txt + nc_stats.txt + nc_*.vtk
    env "${PV_ENV[@]:1}" pvbatch render_nc.py  # -> nc_panel_{std,nit,rot,*_phi}.png
    $CONDA nc_figure.py                        # -> nitsche_convergence.{png,pdf}
    rm -f nc_panel_*.png                       # the VTKs STAY: they are the
                                               # sweep's cache (see below)
    mv -f nitsche_convergence.pdf nitsche_convergence.png "$ROOT"/ )
  echo "built: nitsche_convergence.pdf nitsche_convergence.png"
}

build_nitscheconvuniform() {
  local work="$ROOT/nitscheconv_uniform"
  mkdir -p "$work"
  ( cd nitscheconv
    $CONDA nc_palette_check.py
    NC_WORKDIR="$work" NC_UNIFORM=1 CUDA_VISIBLE_DEVICES="${NC_GPU:-0}" \
      $CONDA nc_prep.py
    env "${PV_ENV[@]:1}" NC_WORKDIR="$work" pvbatch render_nc.py
    NC_WORKDIR="$work" NC_UNIFORM=1 NC_FIGURE_NAME=nitsche_convergence_uniform \
      $CONDA nc_figure.py
    rm -f "$work"/nc_panel_*.png
    mv -f "$work"/nitsche_convergence_uniform.pdf \
          "$work"/nitsche_convergence_uniform.png "$ROOT"/ )
  echo "built: nitsche_convergence_uniform.pdf nitsche_convergence_uniform.png"
}

build_nitscheconvuniformsharp() {
  local work="$ROOT/nitscheconv_uniform_sharp"
  mkdir -p "$work"
  ( cd nitscheconv
    $CONDA nc_palette_check.py
    NC_WORKDIR="$work" NC_UNIFORM=1 NC_SHARP_VOLUME=1 NC_SCALE_DIAG=1 NC_AMG_MAX_ITERS=4000 \
      CUDA_VISIBLE_DEVICES="${NC_GPU:-0}" $CONDA nc_prep.py
    env "${PV_ENV[@]:1}" NC_WORKDIR="$work" pvbatch render_nc.py
    NC_WORKDIR="$work" NC_UNIFORM=1 NC_SHARP_VOLUME=1 \
      NC_FIGURE_NAME=nitsche_convergence_uniform_sharp $CONDA nc_figure.py
    rm -f "$work"/nc_panel_*.png
    mv -f "$work"/nitsche_convergence_uniform_sharp.pdf \
          "$work"/nitsche_convergence_uniform_sharp.png "$ROOT"/ )
  echo "built: nitsche_convergence_uniform_sharp.{pdf,png}"
}

build_nitscheconvsharp() {
  local work="$ROOT/nitscheconv_sharp"
  mkdir -p "$work"
  ( cd nitscheconv
    $CONDA nc_palette_check.py
    NC_WORKDIR="$work" NC_SHARP_VOLUME=1 NC_SCALE_DIAG=1 NC_AMG_MAX_ITERS=4000 \
      CUDA_VISIBLE_DEVICES="${NC_GPU:-0}" $CONDA nc_prep.py
    env "${PV_ENV[@]:1}" NC_WORKDIR="$work" pvbatch render_nc.py
    NC_WORKDIR="$work" NC_SHARP_VOLUME=1 \
      NC_FIGURE_NAME=nitsche_convergence_sharp $CONDA nc_figure.py
    rm -f "$work"/nc_panel_*.png
    mv -f "$work"/nitsche_convergence_sharp.pdf \
          "$work"/nitsche_convergence_sharp.png "$ROOT"/ )
  echo "built: nitsche_convergence_sharp.{pdf,png}"
}

# --- whole-pipeline overview (Figure 1): one forward/backward loop on the bunny
# Stage 1 is OPT-IN and never part of `all`: SDXL + Hunyuan-Paint, PartField and
# RMBG (the input cut-outs) are generative, the CAD part is downloaded; none is
# bit-reproducible, so `ovassets` makes them once into overview/assets/ (seeds,
# prompts and sha256 in assets/MANIFEST.json) and `overview` only READS them.
#   ./build.sh ovassets [input paint segments cad cadmodel cadimage cutout]
#   OV_ASSETS_FORCE=1 ./build.sh ovassets paint        # redo one step
build_ovassets() {
  ( cd overview
    $CONDA ../bunny/prep_bunny_sdf.py                     # bunny.stl + bunny_sdf.meta
    XLA_PYTHON_CLIENT_PREALLOCATE=false OV_GPU="${OV_GPU:-1}" \
      $CONDA ov_assets.py "$@" )
  echo "built: overview/assets/ (see assets/MANIFEST.json)"
}

# Stages 2-4, deterministic, every build (~2 min on GPU ${OV_GPU:-1}): the solve
# the figure shows, the adjoint of the same problem, the assets made renderable,
# every panel from one camera, then the compose.  Reads no other figure.
build_overview() {
  ( cd overview
    set -e
    $CONDA ov_palette_check.py                            # gate: CVD on identity colours
    $CONDA ov_style.py                                    # gate: d / phi ramps == render_dn
    [ -f assets/MANIFEST.json ] || { echo "no overview/assets: run ./build.sh ovassets first"; exit 1; }
    $CONDA ../bunny/prep_bunny_sdf.py                     # bunny.stl + bunny_sdf.meta
    XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES="${OV_GPU:-1}" \
      $CONDA ov_bunny_loop.py                             # dn octree, BCs, solve, dJ/dd, dJ/dphi
    CUDA_VISIBLE_DEVICES="${OV_GPU:-1}" $CONDA ov_surfaces.py      # -> ov_s_*.vtk
    env "${PV_ENV[@]:1}" VTK_DEFAULT_EGL_DEVICE_INDEX="${OV_GPU:-1}" \
        pvbatch render_ov_panels.py                       # -> ov_p_*.png
    # the solve is diffused_nitsche's panel (f), same script, framed on the display
    # bunny as the other bunnies are (so it is their size), onto the FEA tint
    local tint
    tint=$(python3 -c "import ov_style as S; print(','.join(f'{c:.6f}' for c in S.fill_rgb('fea')))")
    env "${PV_ENV[@]:1}" VTK_DEFAULT_EGL_DEVICE_INDEX="${OV_GPU:-1}" \
        DN_SOLVE_PREFIX=ov_loop_solve DN_SOLVE_RENDER=ov_p_solve.png \
        DN_SOLVE_FIT=assets/bunny_display.stl DN_SOLVE_DOLLY=1.02 DN_SOLVE_BG="$tint" \
        pvbatch render_ov_solve.py                        # -> ov_p_solve.png
    $CONDA ov_figure.py                                   # -> method_overview.{png,pdf}
    rm -f bunny_sdf.vtk ov_p_*.png ov_s_*.vtk ov_grad_full.vtk ov_trim_bc.vtk \
          ov_loop_solve.vtu ov_bc_*.stl ov_bc_faces.npz \
          dn_full.vtk dn_trim.vtk dn_surf_rest.stl dn_surf_ears.stl dn_surf_base.stl
    mv -f method_overview.pdf method_overview.png "$ROOT"/ )
  echo "built: method_overview.pdf method_overview.png"
}

# --- Agentic FEM results: one row per operator-skill run, method_overview's look --
# Runs: AR_RUNS="output/agentic_batch/run1 output/agentic_batch/run2 ..." (batch
# folders: workflow/ + claude_session/ + metrics.json, or bare session folders),
# paths relative to the repo root; default the development list ar_dev_runs.json
# (or AR_RUNS_JSON).  Not in `all`: its input is whatever batch is current.
# GPU ${AR_GPU:-1} for both the interpolation and pvbatch.  Keeps ar_stats.json.
build_agenticresults() {
  ( cd agentic_results
    set -e
    local name="${AR_NAME:-agentic_results}" args=()
    if [ -n "${AR_RUNS:-}" ]; then read -r -a args <<< "$AR_RUNS"
    else args=(--runs-json "${AR_RUNS_JSON:-ar_dev_runs.json}"); fi
    python3 ar_style.py                                   # gate: look == overview's source
    rm -rf _work
    CUDA_VISIBLE_DEVICES="${AR_GPU:-1}" $CONDA ar_prep.py "${args[@]}" --work _work
    # ground under each object: the UI viewer's grid (the user's choice); AR_GROUND=
    # shadow (key-light shadow) or none for the alternatives
    env "${PV_ENV[@]:1}" VTK_DEFAULT_EGL_DEVICE_INDEX="${AR_GPU:-1}" AR_GROUND="${AR_GROUND:-grid}" \
        pvbatch render_ar.py _work                        # -> _work/ar_<i>_<panel>[_g].png
    $CONDA ar_figure.py _work --out "$name" --ground "${AR_GROUND:-grid}"
    cp -f _work/ar_stats.json ar_stats.json               # the record behind every number
    # AR_KEEP_WORK=1 keeps the renders, so a layout change reruns only ar_figure.py
    [ "${AR_KEEP_WORK:-0}" = 1 ] || rm -rf _work
    mv -f "$name.pdf" "$name.png" "$ROOT"/ )
  echo "built: ${AR_NAME:-agentic_results}.{pdf,png}"
}

# --- The gallery: the 50 Opus objects in one amphitheater (gallery/) ------------
# gl_prep (the agentic-results last column's surface, |u| and xN, gated against the
# Opus category records) -> am_layout (the user-approved schematic's rules: seats on
# arcs, smallest in front, tiers, the eye on the stage, fitted FOV) -> render_gl (pvbatch,
# 5 passes) -> am_figure (labels on the floor with the collision rule, gates, contact
# sheet).  GL_BATCH_ROOT (output/agentic_batch50_opus, read only), GL_GPU (1),
# GL_WORK (gallery/_work), GL_KEEP_WORK=1, GL_FORCE=1 re-runs the prep; GL_P (0.25),
# GL_LABEL_PT (4.5), GL_SHORTEN_WITH (1: trailing " with ..." -> " …"), GL_PAGE_H
# (101.25 = 16:9), GL_SEAT_SWAPS ("a skyscraper:a lighthouse", the user's choice;
# "" for none), GL_SEARCH_PROCS (6 parallel layout candidates), GL_OVERRIDES
# (gallery/placement_overrides.json: revision 8's kept parameters + the user's per-row /
# per-object arrangement of 2026-10-19, each with its reason; "" runs the search instead).  The build fails if any file in the batch root is newer than its start.
build_gallery() {
  ( cd gallery
    set -e
    local name="${GL_NAME:-agentic_gallery}" work="${GL_WORK:-_work}" gpu="${GL_GPU:-1}"
    local batch="${GL_BATCH_ROOT:-$REPO/output/agentic_batch50_opus}"
    python3 ../agentic_results/ar_style.py               # gate: look == overview's source
    mkdir -p "$work"
    local stamp="$work/.build_start"; touch "$stamp"
    if [ "${GL_FORCE:-0}" = 1 ] || [ ! -f "$work/gl_objects.json" ]; then
      CUDA_VISIBLE_DEVICES="$gpu" $CONDA gl_prep.py --batch-root "$batch" --work "$work"
    fi
    # the user's seat swap (2026-10-17): the one named exception to rule-only placement
    CUDA_VISIBLE_DEVICES="$gpu" GL_SEAT_SWAPS="${GL_SEAT_SWAPS-a skyscraper:a lighthouse}" \
        GL_OVERRIDES="${GL_OVERRIDES-placement_overrides.json}" \
        $CONDA am_layout.py --work "$work"
    env "${PV_ENV[@]:1}" VTK_DEFAULT_EGL_DEVICE_INDEX="$gpu" pvbatch render_gl.py "$work" am_scene.json
    # the compose places the labels on the render's finer raster; a label that fails there
    # widens its seat by the same repair rule (am_layout --refine), up to GL_REFINE times
    local tries=0
    until CUDA_VISIBLE_DEVICES="$gpu" $CONDA am_figure.py "$work" --out "$name"; do
      tries=$((tries + 1))
      [ "$tries" -le "${GL_REFINE:-3}" ] && [ -s "$work/label_failed.json" ] && \
        [ "$(cat "$work/label_failed.json")" != "[]" ] || exit 1
      CUDA_VISIBLE_DEVICES="$gpu" GL_SEAT_SWAPS="${GL_SEAT_SWAPS-a skyscraper:a lighthouse}" \
        GL_OVERRIDES="${GL_OVERRIDES-placement_overrides.json}" \
          $CONDA am_layout.py --work "$work" --refine
      env "${PV_ENV[@]:1}" VTK_DEFAULT_EGL_DEVICE_INDEX="$gpu" pvbatch render_gl.py "$work" am_scene.json
    done
    if [ -n "$(find "$batch" -newer "$stamp" -print -quit)" ]; then
      echo "the batch root was written to during the build:"; find "$batch" -newer "$stamp" | head
      exit 1
    fi
    cp -f "$work/gl_stats.json" gl_stats.json
    [ "${GL_KEEP_WORK:-0}" = 1 ] || [ "$work" != _work ] || rm -rf _work
    mv -f "$name.pdf" "$name.png" "$ROOT"/ )
  echo "built: ${GL_NAME:-agentic_gallery}.{pdf,png}"
}

# --- Agentic results, model comparison: three batch roots side by side ---------
# ARC_HAIKU / ARC_SONNET / ARC_OPUS = the three batch roots (absolute, or relative to
# the repo root); every object of experiments/agentic_batch/objects_50.txt must have a
# run in each, or the plan step fails naming the missing ones.  ARC_ONLY="slug ..."
# restricts to categories; ARC_ROWS = rows per page at most (17); ARC_GPU (1) for the
# prep and pvbatch; ARC_GROUND (grid); ARC_KEEP_WORK=1 keeps _cwork (renders) so a
# layout change reruns only the compose.  One prep over all runs: one E scale and one
# cost scale for every page.  -> ${ARC_NAME:-agentic_results_compare}_p<k>.{pdf,png}
build_agenticcompare() {
  ( cd agentic_results
    set -e
    : "${ARC_HAIKU:?set ARC_HAIKU to the Haiku batch root}" "${ARC_SONNET:?set ARC_SONNET}" \
      "${ARC_OPUS:?set ARC_OPUS}"
    local name="${ARC_NAME:-agentic_results_compare}" work=_cwork gpu="${ARC_GPU:-1}" only=() runs=()
    [ -z "${ARC_ONLY:-}" ] || only=(--only $ARC_ONLY)
    python3 ar_style.py                                   # gate: look == overview's source
    rm -rf "$work"
    python3 ar_compare.py plan --haiku "$ARC_HAIKU" --sonnet "$ARC_SONNET" --opus "$ARC_OPUS" \
        --work "$work" "${only[@]}"                       # -> compare_plan.json, runs.txt
    read -r -a runs < "$work/runs.txt"
    CUDA_VISIBLE_DEVICES="$gpu" AR_FLOOR="${AR_FLOOR:-warn}" \
        $CONDA ar_prep.py "${runs[@]}" --work "$work" --no-print-check   # -> ar_stats.json + ar_<i>_*.vtk
    python3 ar_compare.py restrict "$work"                # shape, BCs, E: no deformed panel
    env "${PV_ENV[@]:1}" VTK_DEFAULT_EGL_DEVICE_INDEX="$gpu" AR_GROUND="${ARC_GROUND:-grid}" \
        pvbatch render_ar.py "$work"                      # -> ar_<i>_<panel>[_g].png
    $CONDA ar_compare.py compose "$work" --out "$name" --ground "${ARC_GROUND:-grid}" \
        --rows "${ARC_ROWS:-17}"                          # -> NAME_p<k>.{pdf,png} + tables
    cp -f "$work/ar_stats.json" "ar_stats_$name.json"     # the record behind every number
    [ "${ARC_KEEP_WORK:-0}" = 1 ] || rm -rf "$work"
    mv -f "$name"_p*.pdf "$name"_p*.png "$ROOT"/ )
  echo "built: ${ARC_NAME:-agentic_results_compare}_p<k>.{pdf,png}"
}

# --- the AI agent operator's loop (Supplementary Methods, "Operation by a coding agent") ---
build_operator() {
  ( cd operator && python3 op_figure.py ../operator_loop )
  echo "built: operator_loop.{pdf,png}"
}

# --- showcase (variation) figure: three rows of agentic runs ------------------
# Rows come from SC_SHAPE_RUNS / SC_MATERIAL_RUNS / SC_LOADING_RUNS (run folders or
# globs, 'folder::label' for a label) or SC_ROWS_JSON; default sc_rows_placeholder.json
# (PLACEHOLDER rows built from output/agentic_batch).  Reuses agentic_results' prep
# (sc_prep.py imports ar_prep), its renderer (render_ar.py) and its compose helpers.
# ch_prep.py cuts the bunny's silhouette with GEOS (shapely: buffers, booleans,
# topology-preserving simplify), and mechfoundry-warp117 has no shapely.  A port to
# manifold3d/Clipper2 (which the env has) matches the areas to 1e-7 but not the
# vertices (the inward offset gets 308 vertices, not GEOS's 300), so the meshes
# drawn from them would change.  So ch_prep runs, in order of preference, under
# $CH_PREP_PYTHON; under mechfoundry-warp117 itself once it has shapely; else under a
# private venv layered on mechfoundry-warp117 (--system-site-packages) that adds ONLY
# shapely $CH_SHAPELY (its wheel bundles its own GEOS, so the pin fixes the
# geometry), made once in the user cache, never in a shared conda env.
CH_SHAPELY=2.0.7
ch_prep_python() {
  if [ -n "${CH_PREP_PYTHON:-}" ]; then echo "$CH_PREP_PYTHON"; return; fi
  local env
  env="$(conda run -n mechfoundry-warp117 python -c 'import sys; print(sys.prefix)')"
  if "$env/bin/python" -c 'import shapely' 2>/dev/null; then echo "$CONDA"; return; fi
  local venv="${XDG_CACHE_HOME:-$HOME/.cache}/implicit_optimize/venv-shapely-$CH_SHAPELY"
  if ! "$venv/bin/python" -c "import shapely; assert shapely.__version__ == '$CH_SHAPELY'" 2>/dev/null; then
    echo "challenge: making $venv (mechfoundry-warp117 + shapely $CH_SHAPELY, once)" >&2
    "$env/bin/python" -m venv --system-site-packages "$venv" >&2
    "$venv/bin/python" -m pip install --quiet --no-deps "shapely==$CH_SHAPELY" >&2
  fi
  # the env's own libstdc++ (as conda run would put first), not the system's older one
  echo "env LD_LIBRARY_PATH=$env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} $venv/bin/python"
}

build_challenge() {
  # the introductory figure before method_overview: one request, run by an
  # engineer and by an LLM agent, the four set-up steps that hold each back,
  # and MechFoundry's answer to each.  CPU only, every pictogram drawn from
  # the bunny's silhouette (no rendered panel).
  local prep_py
  prep_py="$(ch_prep_python)"
  ( cd challenge
    set -e
    $CONDA ch_palette_check.py                            # gate: CVD on the one new colour
    $prep_py ch_prep.py                                   # -> ch_geom.npz, ch_stats.json (needs shapely, above)
    $CONDA ch_figure.py                                   # -> challenge_overview.{pdf,png}
    rm -f ch_geom.npz
    mv -f challenge_overview.pdf challenge_overview.png "$ROOT"/ )
  echo "built: challenge_overview.pdf challenge_overview.png"
}

build_introoverview() {
  # the paper's opening figure: challenge_overview stacked ABOVE method_overview, both
  # 180 mm wide, 2 mm apart, embedded as PDF forms by plain pdfTeX (vector stays vector,
  # no rescale).  Reads the two committed root PDFs and rebuilds neither: run
  # ./build.sh challenge / overview first when either changes.  PNG at 600 dpi.
  ( cd intro_overview
    set -e
    for f in ../challenge_overview.pdf ../method_overview.pdf; do
      [ -f "$f" ] || { echo "introoverview: $f missing (build it first)"; exit 1; }
    done
    $CONDA io_letters.py                                  # -> io_letters.pdf (panel letters a, b; gated clear of ink)
    pdftex -interaction=nonstopmode -halt-on-error -jobname=intro_overview io_compose.tex \
      | grep -E '^intro_overview:|^Output' || true
    [ -f intro_overview.pdf ] || { echo "introoverview: pdftex failed, see intro_overview/intro_overview.log"; exit 1; }
    pdftocairo -png -r 600 -singlefile intro_overview.pdf intro_overview
    rm -f intro_overview.log io_letters.pdf
    mv -f intro_overview.pdf intro_overview.png "$ROOT"/ )
  echo "built: intro_overview.pdf intro_overview.png"
}

build_showcase() {
  ( cd showcase
    set -e
    local name="${SC_NAME:-showcase}"
    python3 ../agentic_results/ar_style.py                # gate: look == overview's source
    rm -rf _work
    CUDA_VISIBLE_DEVICES="${SC_GPU:-1}" $CONDA sc_prep.py --work _work
    env "${PV_ENV[@]:1}" VTK_DEFAULT_EGL_DEVICE_INDEX="${SC_GPU:-1}" AR_GROUND="${SC_GROUND:-grid}" \
        pvbatch ../agentic_results/render_ar.py _work     # -> _work/ar_<i>_<panel>[_g].png
    $CONDA sc_figure.py _work --out "$name" --ground "${SC_GROUND:-grid}"
    cp -f _work/ar_stats.json sc_stats.json               # the record behind every label/number
    python3 sc_record.py "$REPO"                          # each row's authoring prompt -> experiments/agentic_batch/
    [ "${SC_KEEP_WORK:-0}" = 1 ] || rm -rf _work
    mv -f "$name.pdf" "$name.png" "$ROOT"/ )
  echo "built: ${SC_NAME:-showcase}.{pdf,png}"
}

# --- The demo video: one park bench through every look, then 20 more (demo_video/) --
# dv_cleanup (cached GPU stage, ~12 min once: the bench's geometry clean-up re-run with
# the solver's output kept at every iteration, gated against the archived traj.tsv, and a
# per-part forward solve of its final shape; DV_CLEANUP_FORCE=1 redoes both) ->
# dv_prep (agentic_results' own prep, imported: surfaces, |u|, x N gated against the
# figure records; one orbit camera per object) -> render_dv (pvbatch, one long-lived
# view per look, camera turned per frame) -> dv_compose (PIL: crossfades, titles, CLI,
# colour bar, grid) -> ffmpeg (H.264 CRF 20, yuv420p, faststart).  DV_GPU (1) for the
# prep and pvbatch; DV_DRAFT=1 renders half size, every 2nd frame, at 15 fps;
# DV_KEEP_WORK=1 keeps _work (renders + frames), so a 2-D change reruns only the compose.
# demo_video.{mp4,png} are tracked deliverables.  Reads output/agentic_batch50_opus and
# output/geometry_cleanup/park-bench (extract both from output/archives first, AGENTS.md
# "Archived inputs": the latter with  zstd -dc output/archives/geometry_cleanup.tar.zst |
# tar -x -C output geometry_cleanup/park-bench); writes its cache to
# output/geometry_cleanup_demovideo/ (kept across builds).
build_demovideo() {
  ( cd demo_video
    set -e
    local gpu="${DV_GPU:-1}" scale=1 step=1 fps=30 outs=1
    if [ "${DV_DRAFT:-0}" = 1 ]; then scale=0.5; step=2; fps=15; outs=0.5; fi
    python3 ../agentic_results/ar_style.py               # gate: look == overview's source
    rm -rf _work
    CUDA_VISIBLE_DEVICES="$gpu" $CONDA dv_cleanup.py      # cached: does only what is missing
    CUDA_VISIBLE_DEVICES="$gpu" $CONDA dv_prep.py --work _work
    # the bench and the tiles in two processes on the same GPU (~6 / ~2.5 min at full size)
    env "${PV_ENV[@]:1}" VTK_DEFAULT_EGL_DEVICE_INDEX="$gpu" \
        pvbatch render_dv.py _work --scale "$scale" --step "$step" --part bench &
    env "${PV_ENV[@]:1}" VTK_DEFAULT_EGL_DEVICE_INDEX="$gpu" \
        pvbatch render_dv.py _work --scale "$scale" --step "$step" --part tiles
    wait
    $CONDA dv_compose.py _work --step "$step" --out-scale "$outs"
    local ff="$HOME/anaconda3/envs/ffmpeg/bin/ffmpeg" i=0
    # renumber (a draft's frames are every 2nd number) so the image2 pattern reads them all
    mkdir -p _work/enc
    for f in _work/out/f*.png; do ln -sf "../out/$(basename "$f")" "$(printf '_work/enc/e%04d.png' "$i")"; i=$((i + 1)); done
    "$ff" -y -loglevel error -framerate "$fps" -i _work/enc/e%04d.png -c:v libx264 -crf 20 \
      -preset slow -pix_fmt yuv420p -movflags +faststart "$ROOT/demo_video.mp4"
    # the poster: the full grid at 24.9 s (frame 747 = the 23 s cut's 573 + the 5.8 s
    # optimisation beat, dv_timeline.OPT_DT; the draft has only even frames)
    cp -f _work/out/f0747.png "$ROOT/demo_video.png"
    cp -f _work/dv_stats.json dv_stats.json               # the record: runs, x N, cameras
    [ "${DV_KEEP_WORK:-0}" = 1 ] || rm -rf _work )
  echo "built: demo_video.{mp4,png}"
}

case "${1:-all}" in
  octree)    build_octree ;;
  workflow)  build_workflow ;;
  bunny)     build_bunny ;;
  sdfoctree) build_sdfoctree ;;
  bcregion)  build_bcregion ;;
  bcautobunny) build_bcautobunny ;;
  pfbunny)   build_pfbunny ;;
  nitsche)   build_nitsche ;;
  voidnuc)   build_voidnuc ;;
  shapeopt)  build_shapeopt ;;
  hiddenphi) build_hiddenphi ;;
  shapeoptmerged) build_shapeoptmerged ;;
  soframes)  build_soframes ;;
  hpframes)  build_hpframes ;;
  bczip)     build_bczip ;;
  bczipx)    build_bczipx ;;
  nitscheconv) build_nitscheconv ;;
  nitscheconvuniform) build_nitscheconvuniform ;;
  nitscheconvuniformsharp) build_nitscheconvuniformsharp ;;
  nitscheconvsharp) build_nitscheconvsharp ;;
  overview)  build_overview ;;
  agenticresults) build_agenticresults ;;
  agenticcompare) build_agenticcompare ;;
  showcase)  build_showcase ;;
  challenge) build_challenge ;;
  introoverview) build_introoverview ;;
  operator)  build_operator ;;
  gallery)   build_gallery ;;
  demovideo) build_demovideo ;;
  ovassets)  shift; build_ovassets "$@" ;;
  all)       build_octree; build_workflow; build_bunny; build_sdfoctree; \
             build_bcregion; build_nitsche; build_shapeopt; build_voidnuc; \
             build_bczip; build_bczipx; build_nitscheconv; \
             build_nitscheconvuniform; \
             build_overview; build_challenge; \
             build_introoverview ;;   # needs overview/assets (./build.sh ovassets, opt-in)
  *) echo "unknown target: $1 (use octree|workflow|bunny|sdfoctree|bcregion|bcautobunny|pfbunny|nitsche|voidnuc|shapeopt|hiddenphi|shapeoptmerged|soframes|hpframes|bczip|bczipx|nitscheconv|nitscheconvuniform|nitscheconvuniformsharp|nitscheconvsharp|overview|ovassets|agenticresults|agenticcompare|showcase|challenge|introoverview|operator|gallery|demovideo|all)"
     exit 1 ;;
esac
