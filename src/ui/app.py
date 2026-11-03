"""Chat-style Streamlit UI for the API-assisted FEM workflow."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import html
import io
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
from PIL import Image

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st_canvas = None

from src.agents_wrapper import conda_environments, probe_compute_environment
from src.agents_wrapper.fem_workflow.monitor import ToolMonitor
from src.agents_wrapper.fem_workflow.graph import needs_cad_revision
from src.agents_wrapper.fem_workflow.models import (
    BC_FIX_METHODS, LINEAR_SOLVER_LABELS, LINEAR_SOLVERS,
)
from src.agents_wrapper.fem_workflow import (
    FLOAT32_EPSILON,
    FLOAT32_RESIDUAL_FACTOR,
    UNIT_SYSTEM_DETAILS,
    UNIT_SYSTEMS,
)
from src.agents_wrapper.fem_workflow.assets import (
    FIELD_COMPONENTS,
    TENSOR_COMPONENTS,
    field_asset_request,
    load_step_surfaces,
    newest_surface_vtu,
    prefetch_solution_assets,
    solution_prefetch_requests,
)
from src.agents_wrapper.fem_workflow.assets import (
    surface_point_fields as read_surface_point_fields,
)
from src.ui.hover import surface_hover
from src.ui.browse import (
    ARCHIVE_LIMIT,
    PREVIEW_TEXT_LIMIT,
    archive_name,
    breadcrumb,
    build_archive,
    download_mime,
    human_size,
    listing,
    needs_archive,
    preview_kind,
    preview_text,
    resolve_within,
    selection_bytes,
    viewer_surface,
)
from src.ui.telemetry import (
    DEFORMATION_REACH,
    awaiting_solve,
    last_solve_failure,
    solver_failure_notes,
    DEFORMATION_STEPS,
    deformation_scales,
    chosen_iterations,
    contour_shows,
    iteration_numbers,
    next_live_frame,
    optimization_saves,
    surface_deformation,
    residual_chart,
    token_bar_markup,
    LoadSampler,
    newest_solve_log,
    optimization_shown,
    read_backend_events,
    read_solver_progress,
    gd_lr_auto_note,
)
from src.agents_wrapper.fem_workflow.runtime_files import RENAMED_ENVIRONMENTS
from src.ui.controller import (
    DEFAULT_COMPUTE_ENV,
    WorkflowController,
    WorkflowEvent,
    day_heading,
    interrupt_marker,
    faces_slider_options,
    material_field_root,
    preferred_visualization,
    visualization_key,
    restored_interrupt,
    segmentation_method_choices,
)

# FEM_WORKFLOW_OUTPUT_ROOT opens another folder of sessions -- a command-line
# run's --output-root, say.  Resident workers are shared per root, so a page on
# its own root also keeps its workers from serving (and being served by) others.
DEFAULT_OUTPUT_ROOT = (REPO_ROOT / "output" / "fem_workflow").resolve()
OUTPUT_ROOT = Path(os.environ.get(
    "FEM_WORKFLOW_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT
)).expanduser().resolve()
controller = WorkflowController(OUTPUT_ROOT)
# Standard gravity in each unit system's own length units, so a total force can be
# shown as the mass it corresponds to without leaving the selected system.
STANDARD_GRAVITY = {
    "SI (m-kg-s)": 9.80665,  # m/s^2   -> mass in kg
    "SI-mm (mm-tonne-s)": 9806.65,  # mm/s^2  -> mass in tonne
    "US-inch (in-lbf-s)": 386.0886,  # in/s^2  -> mass in lbf*s^2/in
}

# Text-only and image-bearing calls are probed and selected separately, because a
# strong text model is not necessarily multimodal.
# DeepSeek leads, so a new conversation starts on it.
PROVIDER_TEXT_MODELS = {
    "DeepSeek": ("deepseek-flash",),
    "Groq": ("openai/gpt-oss-120b", "qwen/qwen3.8-27b"),
    "Gemini": ("gemini-2.5-flash",),
}
PROVIDER_VISION_MODELS = {
    "DeepSeek": ("deepseek-flash",),
    "Groq": ("qwen/qwen3.8-27b",),
    "Gemini": ("gemini-2.5-flash",),
}
SEGMENTATION_LABELS = {
    "cad_faces": "CAD faces", "partfield": "PartField",
    "shape_diameter": "Shape diameter",
}
GENERATOR_LABELS = {
    "triposg": "TripoSG", "text2cad": "Text2CAD (text to CAD)",
    "cadcoder": "CAD-Coder (text to CAD)",
    "hunyuan3d": "Hunyuan3D-2.1 (image, with colour)",
}
# The generators offered for each input; TripoSG takes every input.
INPUT_GENERATORS = {
    "text": ("triposg", "text2cad", "cadcoder"),
    "image": ("triposg", "hunyuan3d"),
}
# Text-only generators that build CAD parts; each reports its readiness.
CAD_GENERATORS = ("text2cad", "cadcoder")
# The optional generators, each installed in its own environment.
OPTIONAL_GENERATORS = (*CAD_GENERATORS, "hunyuan3d")
# Where an image input comes from: an upload, or SDXL drawing it from text.
IMAGE_SOURCES = ("Upload", "Generate with SDXL")

st.set_page_config(
    page_title="Implicit FEM Agent",
    page_icon="◫",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
<style>
  [data-testid="stHeader"], [data-testid="stToolbar"],
  [data-testid="stSidebarHeader"], [data-testid="stSidebarCollapseButton"],
  [data-testid="stExpandSidebarButton"] {
    display: none !important;
  }
  /* The same gap on both sides: the gutters used to be the theme's own generous
     left padding plus an extra 0.75rem added on the right, so the list sat
     off-centre in a column wider than the titles needed.  The width itself is
     set in the handle script, not here -- Streamlit's own resize handle writes
     an inline width, and an !important rule would silently kill it. */
  [data-testid="stSidebarContent"], [data-testid="stSidebarUserContent"] {
    height: 100% !important;
    max-height: 100% !important;
    overflow: hidden !important;
    box-sizing: border-box;
  }
  [data-testid="stSidebarContent"] {
    padding-left: 0 !important;
    padding-right: 0 !important;
  }
  [data-testid="stSidebarUserContent"] {
    padding-left: 0.6rem !important;
    padding-right: 0.6rem !important;
  }
  html.fem-workflow-busy [class*="st-key-workflow_panel"] button {
    pointer-events: none !important;
    cursor: wait !important;
    opacity: 0.5 !important;
    filter: saturate(0.55);
  }
  [data-testid="stSidebarUserContent"] {
    position: relative;
  }
  [class*="st-key-sidebar_controls"] {
    flex: 0 0 auto;
    padding-top: 0.75rem;
  }
  [class*="st-key-sidebar_controls"] h2 {
    margin-top: 0;
    padding-top: 0;
    height: 2.5rem;
    line-height: 2.5rem;
  }
  [class*="st-key-sidebar_controls"] [data-testid="stHeadingWithActionElements"] {
    height: 2.5rem;
    display: flex;
    align-items: center;
  }
  [class*="st-key-sidebar_controls"] [data-testid="stButton"] {
    margin-top: 0.5rem;
  }
  [class*="st-key-conversation_list"] {
    position: absolute !important;
    top: 7.75rem;
    bottom: 0;
    /* An absolute box is placed against its parent's padding box, so the
       0.6rem the controls above get from stSidebarUserContent never reaches
       this list; its side gaps are the insets here plus the scrollbar gutter.
       The gutter alone used to be the gap, which held only where scrollbars
       take room: macOS draws overlay scrollbars, the gutter is 0px there, and
       the rows ran into both edges.  The inset makes up the difference, so
       the rows line up with the controls whichever kind the browser draws.
       The bootstrap script measures the gutter into --sidebar-scrollbar;
       until it has, the fallback keeps the inset at 0, the old layout.
       Streamlit gives the block a width of its own, which would win over
       `right`, so the width is left to the two insets. */
    left: max(0px, calc(0.6rem - var(--sidebar-scrollbar, 0.6rem)));
    right: max(0px, calc(0.6rem - var(--sidebar-scrollbar, 0.6rem)));
    width: auto !important;
    min-height: 0 !important;
    height: auto !important;
    max-height: none !important;
    /* Reserved on both edges, so the list stays centred whether or not the
       scrollbar is showing; a one-sided gutter is the asymmetry it was there
       to prevent, moved rather than removed. */
    scrollbar-gutter: stable both-edges;
    scrollbar-width: thin;
    gap: 0 !important;
  }
  [class*="st-key-conversation_list"] [data-testid="stButton"] button {
    height: 1.75rem;
    min-height: 1.75rem;
    padding: 0.1rem 0.4rem;
    border-color: transparent !important;
    background: transparent !important;
    box-shadow: none !important;
    text-align: left !important;
    justify-content: flex-start !important;
    align-items: center;
  }
  [class*="st-key-conversation_list"] [data-testid="stButton"] button > div,
  [class*="st-key-conversation_list"] [data-testid="stButton"] button > span {
    width: 100% !important;
    display: flex !important;
    justify-content: flex-start !important;
    text-align: left !important;
  }
  [class*="st-key-conversation_list"] [data-testid="stButton"] button:hover {
    border-color: transparent !important;
    background: rgba(128, 128, 128, 0.13) !important;
  }
  [class*="st-key-conversation_list"] [data-testid="stVerticalBlock"] {
    gap: 0 !important;
  }
  [class*="st-key-conversation_list"] [data-testid="stElementContainer"] {
    margin: 0 !important;
    padding: 0 !important;
  }
  [class*="st-key-conversation_list"] [data-testid="stButton"] button p {
    display: block;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
    line-height: 1.1rem;
    text-align: left !important;
    width: 100%;
    margin: 0;
  }
  .conversation-header {
    /* The panel below reserves a scrollbar gutter, so its contents stop short of
       the column edge.  Reserving the same here puts the two right ends level
       without hard-coding a scrollbar width. */
    overflow-y: auto;
    scrollbar-gutter: stable;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    min-width: 0;
    height: 2.5rem;
    margin: 0 0 0.5rem;
    white-space: nowrap;
  }
  /* Left of the session id: what the conversation has spent, input and output
     in different colours, against a scale that doubles rather than a bar that
     grows. */
  .session-meta {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    min-width: 0;
    flex: 0 1 auto;
  }
  .token-meter {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    flex: 0 0 auto;
  }
  .token-bar {
    display: flex;
    width: 5.5rem;
    height: 0.45rem;
    border-radius: 0.25rem;
    background: rgba(100, 116, 139, 0.22);
    overflow: hidden;
  }
  .token-fill { height: 100%; }
  .token-in { background: #2563eb; }
  .token-out { background: #16a34a; }
  .token-label {
    color: #64748b;
    font-size: 0.72rem;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  .conversation-heading {
    flex: 0 0 auto;
    font-size: 1.5rem;
    font-weight: 600;
    line-height: 2.5rem;
  }
  html, body, #root, [data-testid="stAppViewContainer"] {
    position: fixed;
    inset: 0;
    width: 100%;
    height: 100%;
    overflow: hidden !important;
  }
  [data-testid="stMain"] {
    height: 100% !important;
    max-height: 100% !important;
    overflow: hidden !important;
  }
  [data-testid="stMainBlockContainer"] {
    position: relative;
    height: 100% !important;
    max-height: 100% !important;
    width: min(100%, 2400px) !important;
    max-width: 2400px !important;
    margin-left: auto;
    margin-right: auto;
    overflow: hidden !important;
    padding: 0 !important;
  }
  [class*="st-key-workspace"] {
    position: absolute !important;
    inset: 0.75rem 1rem 0.5rem 1rem !important;
    height: auto !important;
    max-height: none !important;
    width: auto !important;
    max-width: none !important;
    overflow: hidden !important;
  }
  [class*="st-key-workspace"] > div {
    height: 100% !important;
    max-height: 100% !important;
    overflow: hidden !important;
    min-width: 0;
  }
  [data-testid="stHorizontalBlock"]:has([class*="st-key-workflow_panel"]):has([class*="st-key-visualization_panel"]) {
    height: 100% !important;
    max-height: 100% !important;
    overflow: hidden !important;
  }
  [data-testid="stHorizontalBlock"]:has([class*="st-key-workflow_panel"]):has([class*="st-key-visualization_panel"])
    > [data-testid="stColumn"] {
    height: 100% !important;
    max-height: 100% !important;
    overflow: hidden !important;
    min-width: 0;
  }
  [class*="st-key-workflow_panel"],
  [class*="st-key-visualization_panel"],
  [class*="st-key-manager_widget"] {
    border-top: 0 !important;
  }
  /* Streamlit wraps each block child in an stLayoutWrapper whose flex-basis carries
     the container's fixed height (flex: 0 0 700px), so no rule on the panel itself
     can make it grow.  Stretch the wrapper instead: the column's vertical block is
     already a flex column, so the panel then takes exactly what is left below the
     header, whatever the header and the block gap happen to measure. */
  [data-testid="stLayoutWrapper"]:has(> [class*="st-key-workflow_panel"]) {
    flex: 1 1 auto !important;
    height: auto !important;
    min-height: 0 !important;
  }
  /* Zero basis, so the viewer takes the share left over rather than a share
     that grows with what is inside it.  With a content basis, adding the
     component selector made this panel claim height back from the tool window,
     and the pair no longer matched the viewer's height in other views. */
  [data-testid="stLayoutWrapper"]:has(> [class*="st-key-visualization_panel"]) {
    flex: 1 1 0 !important;
    height: auto !important;
    min-height: 0 !important;
  }
  /* The tool window takes the bottom of the right column and gives the rest back
     to the viewer, shrinking rather than pushing it off screen when short. */
  [data-testid="stLayoutWrapper"]:has(> [class*="st-key-tool_panel"]) {
    flex: 0 1 var(--tools-height, 24rem) !important;
    height: auto !important;
    min-height: 6rem !important;
  }
  /* The two splits are driven by custom properties so the drag handles only
     ever set a value; nothing has to re-derive the layout when Streamlit
     rebuilds the tree underneath them. */
  [data-testid="stHorizontalBlock"]:has([class*="st-key-workflow_panel"]):has([class*="st-key-visualization_panel"])
    > [data-testid="stColumn"]:first-child {
    flex: 0 0 var(--workspace-split, 57.5%) !important;
    width: var(--workspace-split, 57.5%) !important;
  }
  [data-testid="stHorizontalBlock"]:has([class*="st-key-workflow_panel"]):has([class*="st-key-visualization_panel"])
    > [data-testid="stColumn"]:last-child {
    flex: 1 1 0 !important;
    width: auto !important;
  }
  /* Out of the flow but still rendered: a display:none frame is not reliably
     loaded, and this one exists only to run its script. */
  [class*="st-key-split_handles"] {
    position: absolute !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
  }
  .workspace-split {
    position: fixed;
    z-index: 90;
    border-radius: 3px;
    background: transparent;
    transition: background 120ms ease;
  }
  .workspace-split:hover, .workspace-split[data-dragging="1"] {
    background: rgba(128, 138, 152, 0.45);
  }
  .workspace-split[data-axis="x"] { cursor: col-resize; }
  .workspace-split[data-axis="y"] { cursor: row-resize; }
  /* The list never scrolls sideways; a row that does not fit is shortened. */
  [class*="st-key-conversation_list"] {
    overflow-x: hidden !important;
  }
  /* The row is the positioning context for the trash that sits on its end. */
  [class*="st-key-history_row_"] {
    position: relative !important;
  }
  /* The mask is the container, not the button.  Pinning the container to all
     four edges covers the row's full height whatever it measures, while the
     icon stays its own size and centred inside it -- stretching the button to
     the row height instead made it a tall slab that read as misplaced. */
  [class*="st-key-history_row_"] [class*="st-key-delete_"] {
    position: absolute !important;
    top: 0;
    bottom: 0;
    right: 0;
    height: auto !important;
    width: 2rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background:
      linear-gradient(rgba(128, 128, 128, 0.13), rgba(128, 128, 128, 0.13)),
      var(--secondary-background-color, #f0f2f6);
    z-index: 2;
  }
  [class*="st-key-history_row_"] [class*="st-key-delete_"] [data-testid="stButton"] {
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
  }
  /* The title has the whole width -- nothing is reserved for the trash, which
     overlaps it instead and only appears on hover.  What sits under it is the
     tail of a name that is already trailing off, and the fade below keeps the
     two from running together. */
  [class*="st-key-history_row_"] [class*="st-key-delete_"]::before {
    content: "";
    position: absolute;
    right: 100%;
    top: 0;
    bottom: 0;
    width: 1.5rem;
    pointer-events: none;
    background:
      linear-gradient(to right, rgba(128, 128, 128, 0), rgba(128, 128, 128, 0.13)),
      linear-gradient(
        to right,
        rgba(240, 242, 246, 0),
        var(--secondary-background-color, #f0f2f6)
      );
  }
  /* The title keeps to one line and trails off; the whole of it is in the
     tooltip, and a wrapping title would make the rows different heights. */
  [class*="st-key-history-"] button p {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    display: block;
    text-align: left;
  }
  /* Irreversible, so it is coloured like it rather than left as the theme's
     ordinary primary, which is the colour of every other way forward. */
  [class*="st-key-dialog_confirm"] button {
    background: #d32f2f !important;
    border-color: #d32f2f !important;
    color: #ffffff !important;
  }
  [class*="st-key-dialog_confirm"] button:hover,
  [class*="st-key-dialog_confirm"] button:focus {
    background: #b71c1c !important;
    border-color: #b71c1c !important;
    color: #ffffff !important;
  }
  /* The trash keeps its place in the row and only fades in when pointed at, so
     nothing moves under the cursor. */
  [class*="st-key-history_row_"] [class*="st-key-delete_"] {
    opacity: 0;
    /* Unclickable while invisible: it sits over the item's right end, and
       deleting a conversation is not something to do by accident on a control
       that cannot be seen. */
    pointer-events: none;
    transition: opacity 120ms ease;
  }
  [class*="st-key-history_row_"]:hover [class*="st-key-delete_"],
  [class*="st-key-history_row_"] [class*="st-key-delete_"]:focus-within {
    opacity: 1;
    pointer-events: auto;
  }
  /* The container carries the fill; the button is only the icon, so it keeps
     its natural size and sits centred rather than being stretched to the row. */
  [class*="st-key-history_row_"] [class*="st-key-delete_"] button {
    border: 0 !important;
    background: transparent !important;
    border-radius: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    height: 1.4rem !important;
    min-height: 0 !important;
    justify-content: center !important;
  }
  /* The window itself never scrolls; it is a fixed frame whose contents fit the
     room they are given.  Declaring it a size container lets the rules below ask
     how much room that is. */
  [class*="st-key-tool_panel"] {
    height: 100% !important;
    max-height: 100% !important;
    max-width: 100% !important;
    overflow: hidden !important;
    box-sizing: border-box;
    container-type: size;
    container-name: tools;
  }
  /* The header and the section buttons keep their height; the section below
     takes the rest, and its chart is sized by that rather than by a constant. */
  [data-testid="stLayoutWrapper"]:has(> [class*="st-key-tool_body_"]) {
    flex: 1 1 auto !important;
    height: auto !important;
    min-height: 0 !important;
  }
  [class*="st-key-tool_body_"] {
    height: 100% !important;
    max-height: 100% !important;
    min-height: 0;
  }
  [class*="st-key-tool_body_fixed"] { overflow: hidden !important; }
  /* Only the backend backlog scrolls. */
  [class*="st-key-tool_body_scroll"] {
    overflow-y: auto !important;
    overflow-x: hidden !important;
  }
  /* A chart asked to fill its container needs a height to fill, and Streamlit
     wraps it in five boxes that all size to their content by default.  Each one
     down to the chart is told to take the space instead. */
  [class*="st-key-tool_body_fixed"] > [data-testid="stLayoutWrapper"] {
    flex: 1 1 auto !important;
    min-height: 0 !important;
  }
  [class*="st-key-tool_body_fixed"] [data-testid="stVerticalBlock"] {
    min-height: 0 !important;
  }
  [class*="st-key-tool_body_fixed"]
    [data-testid="stElementContainer"]:has([data-testid="stVegaLiteChart"]) {
    display: flex;
    flex-direction: column;
    flex: 1 1 auto !important;
    min-height: 0 !important;
  }
  /* The listing fills the section the same way, for the same reason: asking
     it to stretch needs a parent that has a height to stretch into. */
  [class*="st-key-tool_body_fixed"]
    [data-testid="stElementContainer"]:has([data-testid="stDataFrame"]) {
    display: flex;
    flex-direction: column;
    flex: 1 1 auto !important;
    min-height: 0 !important;
  }
  /* Where the browser is.  Written as its own line rather than a caption,
     because captions in this panel are dropped when it gets short and the path
     is the one thing that has to survive that. */
  .browser-path, .browser-note {
    color: #64748b;
    font-size: 0.875rem;
    padding: 0 0 0.25rem;
  }
  /* A path too long for its line loses its beginning, not its end: the end is
     the folder actually being looked at, and the session it started from is
     the part that can be guessed.  Laying the box out right-to-left is what
     moves the ellipsis to the front; the text inside keeps its own direction,
     so the separators stay where they were written. */
  .browser-path {
    direction: rtl;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .browser-path > span {
    direction: ltr;
    unicode-bidi: embed;
  }
  .browser-note { padding: 0.25rem 0 0; }
  /* The path and the three things that can be done share one line.  The
     columns are laid out here rather than by their weights, because a weight
     is a share of whatever width the panel happens to have and these buttons
     want to be one size: square, at the right end, whatever is left over going
     to the path. */
  [class*="st-key-browser_heading"] [data-testid="stHorizontalBlock"] {
    align-items: center;
    flex-wrap: nowrap;
    gap: 0.25rem;
  }
  [class*="st-key-browser_heading"] [data-testid="stColumn"]:first-child {
    flex: 1 1 auto;
    /* Without this the column refuses to shrink below its text and the path
       runs off the panel instead of losing its beginning. */
    min-width: 0;
  }
  [class*="st-key-browser_heading"] [data-testid="stColumn"]:not(:first-child) {
    flex: 0 0 2rem;
    width: 2rem;
  }
  /* The listing brings a floating toolbar of its own -- a CSV download, a
     search, a fullscreen -- which appears over the table's top corner, lands
     on the three buttons sitting there, and offers a "download" that means the
     table rather than the files.  It has no place in this panel. */
  [class*="st-key-tool_body_"] [data-testid="stElementToolbar"] {
    display: none !important;
  }
  [class*="st-key-browser_up"] button,
  [class*="st-key-browser_refresh"] button,
  [class*="st-key-browser_download"] button {
    height: 2rem;
    min-height: 2rem;
    padding: 0;
    width: 100%;
  }
  [class*="st-key-tool_body_fixed"] [data-testid="stFullScreenFrame"],
  [class*="st-key-tool_body_fixed"] [data-testid="stFullScreenFrame"] > div,
  [class*="st-key-tool_body_fixed"] [data-testid="stVegaLiteChart"] {
    display: block;
    height: 100% !important;
    min-height: 0 !important;
  }
  [class*="st-key-tool_panel"] [role="radiogroup"],
  [class*="st-key-tool_panel"] [data-testid="stButtonGroup"] {
    flex: 0 0 auto;
  }
  /* The charts report; they are not instruments to pan, zoom or hover. */
  [class*="st-key-tool_panel"] [data-testid="stVegaLiteChart"],
  [class*="st-key-tool_panel"] [data-testid="stArrowVegaLiteChart"],
  [class*="st-key-tool_panel"] [class*="stVegaLiteChart"] {
    pointer-events: none !important;
  }
  /* Too little room for everything: the curves and residuals are the point, so
     the headline figures go first, then the captions under them. */
  @container tools (max-height: 23rem) {
    [class*="st-key-tool_panel"] [data-testid="stMetric"] { display: none !important; }
  }
  @container tools (max-height: 17rem) {
    [class*="st-key-tool_panel"] [data-testid="stCaptionContainer"] {
      display: none !important;
    }
  }
  [class*="st-key-tool_panel"] [data-testid="stExpander"] pre {
    max-height: 18rem;
    overflow: auto;
  }
  [class*="st-key-workflow_panel"] {
    flex: 1 1 auto;
    height: 100% !important;
    max-height: 100% !important;
    min-height: 0 !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
    box-sizing: border-box;
    scrollbar-gutter: stable;
  }
  [class*="st-key-visualization_panel"] {
    height: 100% !important;
    max-height: 100% !important;
    max-width: 100% !important;
    overflow: hidden !important;
    box-sizing: border-box;
  }
  /* Give the viewer the room the panel has left.  Streamlit sizes an iframe to
     the pixel height Python passed, which cannot know the panel's height, so
     the frame is stretched here and the shell inside follows it. */
  /* The selector and the viewer live in one gapless box that stands exactly
     where the viewer alone stands for other views, so the pair is the same
     height as the viewer is elsewhere. */
  [data-testid="stLayoutWrapper"]:has(> [class*="st-key-field_view"]) {
    flex: 1 1 0 !important;
    height: auto !important;
    min-height: 0 !important;
  }
  [class*="st-key-field_view"] {
    height: 100% !important;
    max-height: 100% !important;
    min-height: 0;
  }
  [class*="st-key-visualization_panel"]
    [data-testid="stElementContainer"]:has([data-testid="stIFrame"]) {
    display: flex;
    flex-direction: column;
    flex: 1 1 auto !important;
    min-height: 0 !important;
  }
  [class*="st-key-visualization_panel"] [data-testid="stIFrame"] {
    flex: 1 1 auto !important;
    height: 100% !important;
    min-height: 0 !important;
  }
  [class*="st-key-visualization_panel"] [data-testid="stImage"] img {
    width: 100% !important;
    max-height: calc(100vh - 13.5rem) !important;
    object-fit: contain !important;
  }
  [class*="st-key-workflow_panel"] * ,
  [class*="st-key-visualization_panel"] * {
    box-sizing: border-box;
    min-width: 0;
  }
  /* A card spans exactly the assistant's text column: on the left the message
     spends its 1rem inset, the 2.5rem avatar track and the 0.55rem column gap
     before the text starts, and on the right both run to the panel edge, which
     is where the Conversation header ends. */
  [class*="st-key-manager_widget"] {
    margin-left: 4.05rem;
    margin-right: 0;
    width: calc(100% - 4.05rem);
    max-width: calc(100% - 4.05rem);
    overflow-x: hidden !important;
  }
  [class*="st-key-manager_widget"] [data-testid="stCustomComponentV1"] {
    width: min(100%, 20rem) !important;
    max-width: 100% !important;
    overflow: hidden !important;
    border: 0 !important;
    border-radius: 0.75rem !important;
    background: #f0f2f6 !important;
    box-shadow: none !important;
  }
  [class*="st-key-manager_widget"] [data-testid="stCustomComponentV1"] iframe {
    display: block !important;
    width: 100% !important;
    max-width: 100% !important;
    border: 0 !important;
    border-radius: 0.75rem !important;
    background: #f0f2f6 !important;
    box-shadow: none !important;
  }
  .env-label {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.25rem;
    font-size: 0.875rem;
  }
  .env-checking { color: #64748b; }
  /* The ui_bridge_* carriers draw nothing and should take no room.  Zeroing the
     keyed block alone is not enough: Streamlit wraps it in a stLayoutWrapper and
     gives it flex: 1 1 0%, so it grows back to whatever the wrapper's content
     asks for.  The iframe bridges ask for ~0 px and got away with it, but the
     page_drawn text_input asks for 40 px and showed as a full-width grey box
     above the setup card.  Collapse the direct wrapper too; the ">" keeps an
     outer panel that merely contains a bridge (the Files browser) intact.  The
     scroll and busy bridges are unkeyed because they are written more than once
     a run; BRIDGE_STYLE, inside each frame, keeps those from tinting a strip. */
  [class*="st-key-ui_bridge_"],
  [data-testid="stLayoutWrapper"]:has(> [class*="st-key-ui_bridge_"]),
  [data-testid="stElementContainer"]:has([class*="st-key-ui_bridge_"]) {
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    background: transparent !important;
    border: 0 !important;
  }
  [class*="st-key-ui_bridge_"] iframe {
    background: transparent !important;
    border: 0 !important;
  }
  .env-ready { color: #16a34a; font-weight: 600; }
  .env-failed { color: #dc2626; font-weight: 600; }
  [class*="st-key-workflow_panel"] [data-testid="stChatMessage"] {
    display: grid !important;
    grid-template-columns: 2.5rem minmax(0, 1fr);
    column-gap: 0.55rem;
    align-items: start;
    width: 100%;
    margin: 0 !important;
    padding-top: 0.3rem !important;
    padding-bottom: 0.3rem !important;
    /* Declared rather than inherited, so the cards below can line up with the
       text column instead of guessing at Streamlit's own inset. */
    padding-left: 1rem !important;
    padding-right: 0 !important;
  }
  [class*="st-key-workflow_panel"] [data-testid="stChatMessageAvatarAssistant"] {
    grid-column: 1;
  }
  [class*="st-key-workflow_panel"] [data-testid="stChatMessage"]
    [data-testid="stChatMessageContent"] {
    grid-column: 2;
    width: 100%;
    max-width: 100%;
    text-align: left !important;
  }
  [class*="st-key-workflow_panel"] [data-testid="stChatMessageAvatarUser"] {
    grid-column: 3;
    grid-row: 1;
    justify-self: end;
  }
  [class*="st-key-workflow_panel"]
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
  }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] {
    grid-column: 2;
    grid-row: 1;
    justify-self: end;
    display: inline-flex !important;
    flex-direction: column;
    align-items: flex-start;
    inline-size: fit-content !important;
    width: fit-content !important;
    min-inline-size: min(9rem, 100%);
    min-width: min(9rem, 100%);
    max-inline-size: 100% !important;
    max-width: 100% !important;
    margin-left: auto !important;
    margin-right: 0 !important;
    overflow-x: clip;
    overflow-wrap: anywhere;
    white-space: normal;
    text-align: left !important;
    padding: 0.5rem 0.75rem !important;
    border-radius: 0.75rem;
    background: var(--secondary-background-color, #f0f2f6) !important;
    box-shadow: none !important;
  }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stMarkdownContainer"],
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stMarkdownContainer"] p {
    width: auto !important;
    inline-size: auto !important;
    max-width: 100% !important;
    max-inline-size: 100% !important;
    min-width: 0;
    margin-left: 0 !important;
    margin-right: 0 !important;
    white-space: normal;
    word-break: break-word;
    overflow-wrap: anywhere;
    text-align: left !important;
  }
  [class*="st-key-history_step_"] {
    position: relative;
    margin: 0 !important;
    padding: 0 !important;
  }
  .axis-key {
    display: block;
    color: #64748b;
    font-size: 0.82rem;
    line-height: 1.5rem;
    min-height: 1.5rem;
    margin: -0.55rem 0 0;
    padding: 0 0.15rem;
    position: relative;
    z-index: 10;
    overflow: visible;
  }
  .axis-x { color:#dc2626; font-weight:700; }
  .axis-y { color:#16a34a; font-weight:700; }
  .axis-z { color:#2563eb; font-weight:700; }
  .session-id {
    flex: 1 1 auto;
    min-width: 0;
    color: #64748b;
    font-size: 0.8rem;
    text-align: right;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
  }
</style>
""",
    unsafe_allow_html=True,
)

# These carry scripts and draw nothing, but an iframe paints its own document's
# background, which no rule outside the frame can reach -- so each one says it is
# transparent from the inside.
BRIDGE_STYLE = "<style>html,body{background:transparent;margin:0;overflow:hidden}</style>"

# Reopen a sidebar that the browser collapsed before hiding its controls.  Keyed
# like the other script carriers so it does not leave a 1px strip on the page.
with st.container(key="ui_bridge_bootstrap"):
    st.iframe(
        BRIDGE_STYLE
        + f"""
        <script>
          // A unique marker forces Streamlit to execute this bootstrap after reruns.
          const pageRun = '{time.time_ns()}';
          const reopen = () => {{
            // The busy state is asserted from Python, which knows whether a step
            // is actually running.  Guessing it here -- disable on click, then
            // watch for a spinner to come and go -- stuck fast the moment the
            // work moved off the script thread and the spinner went away, and
            // stuck anyway on a click that never started a step.
            //
            // The clear runs once per page, not once per rerun: this script is
            // re-executed on every rerun, so an unguarded clear would undo the
            // state Python had just set and leave the controls live while a step
            // was still running.  Once per load still recovers a stuck session
            // on refresh.
            if (!window.parent.__femWorkflowBusyReset) {{
              window.parent.__femWorkflowBusyReset = true;
              const root = window.parent.document.documentElement;
              root.classList.remove('fem-workflow-busy');
            }}
            // How much room the history list's scrollbar gutter takes on each
            // edge: about 10px where scrollbars take room, 0 where they
            // overlay the content (macOS).  The list's side insets are
            // computed from it.  The probe reserves its gutter the way the
            // list does; an empty overflow:scroll box draws no scrollbar on
            // this page, so it would read 0 everywhere.
            const doc = window.parent.document;
            const probe = doc.createElement('div');
            probe.style.cssText = (
              'position:absolute;top:-9999px;left:-9999px;width:100px;'
              + 'height:100px;overflow-y:auto;scrollbar-width:thin;'
              + 'scrollbar-gutter:stable both-edges'
            );
            doc.body.appendChild(probe);
            doc.documentElement.style.setProperty(
              '--sidebar-scrollbar',
              (probe.offsetWidth - probe.clientWidth) / 2 + 'px'
            );
            probe.remove();
            const button = doc.querySelector(
              '[data-testid="stExpandSidebarButton"]'
            );
            if (button) button.click();
          }};
          requestAnimationFrame(() => setTimeout(reopen, 20));
        </script>
        """,
            height=1,
        )
# JavaScript-only updates are rendered here, outside conversation cards and both
# independently scrolling workspace panels.
SCROLL_BRIDGE = st.empty()
BUSY_BRIDGE = st.empty()

# Streamlit can lose a finished rerun on its way to the browser.  A timed
# fragment asking to run just as a full run ends, before the event loop has
# forwarded that run's messages, gets a new script runner, and whatever the old
# one still had to send is dropped as coming from a runner no longer current --
# the race Streamlit's own ``AppSession.request_rerun`` describes.  Every card
# here has timed fragments beside it (the tool window's panels, the viewer's
# live bridge), and two of seventeen skipped texture cards in testing landed
# with the next card never drawn: the page sat on "Manager is working…" until
# something else was clicked.  So each full run numbers itself and leaves that
# number, last of all, in a hidden widget.  The browser reports the widget's
# value with every request, so a page still showing an older run gives itself
# away and is drawn again.  The grace covers the tick or two in which a page
# that did arrive has not echoed it yet.  Its clock restarts with every full
# run, because this fragment also runs inline in one, before the number is
# written, and must not redraw a run that is still drawing.
REDRAW_GRACE = 3.0
st.session_state["page_generation"] = st.session_state.get("page_generation", 0) + 1
st.session_state["page_drawn_at"] = time.time()


@st.fragment(run_every=1.0)
def redraw_if_lost() -> None:
    """Draw the page again once the browser is seen still showing an older run."""
    shown = st.session_state.get("page_drawn")
    if not shown or shown == str(st.session_state["page_generation"]):
        return
    if time.time() - st.session_state["page_drawn_at"] > REDRAW_GRACE:
        st.rerun(scope="app")


# Placed before anything drawn conditionally, so it is the same fragment in
# every run -- the one a page left behind by a lost run is still calling.
with st.container(key="ui_bridge_redraw"):
    redraw_if_lost()

SETUP_MESSAGE = (
    "Choose the API, optional model, and compute environment to "
    "start this FEM conversation. I will verify the API and model before continuing."
)


@st.cache_resource
def environment_probe_pool() -> ThreadPoolExecutor:
    # Environment, text model, and vision model are probed concurrently; a pool
    # smaller than that serialises them behind the slow environment check.
    return ThreadPoolExecutor(max_workers=4, thread_name_prefix="fem-env-probe")


def render_environment_probe(name: str) -> None:
    future = st.session_state.get("environment_probe_future")
    if st.session_state.get("environment_probe_name") != name or future is None:
        future = environment_probe_pool().submit(probe_compute_environment, name)
        st.session_state.environment_probe_name = name
        st.session_state.environment_probe_future = future
        st.session_state.environment_status = None
        st.session_state.environment_probe_completed = None
    if not future.done():
        st.markdown(
            '<div class="env-label"><span>Compute environment</span>'
            '<span class="env-checking">Checking…</span></div>',
            unsafe_allow_html=True,
        )
        return
    try:
        status = future.result()
    except Exception as exc:
        status = None
        st.session_state.environment_probe_error = f"{type(exc).__name__}: {exc}"
    st.session_state.environment_status = status
    if status and status.ready:
        st.markdown(
            f'<div class="env-label"><span>Compute environment</span>'
            f'<span class="env-ready" title="{html.escape(status.summary)}">Ready</span></div>',
            unsafe_allow_html=True,
        )
    else:
        detail = (
            status.summary
            if status
            else st.session_state.get(
                "environment_probe_error", "environment probe failed"
            )
        )
        st.markdown(
            f'<div class="env-label"><span>Compute environment</span>'
            f'<span class="env-failed" title="{html.escape(detail)}">Unavailable</span></div>',
            unsafe_allow_html=True,
        )


def render_model_probe(provider: str, model: str, role: str, label: str) -> None:
    """Probe one role's model; both must pass before a conversation can start."""
    selection = (provider, model)
    selection_key = f"{role}_probe_selection"
    future_key = f"{role}_probe_future"
    ready_key = f"{role}_probe_ready"
    future = st.session_state.get(future_key)
    if st.session_state.get(selection_key) != selection or future is None:
        future = environment_probe_pool().submit(
            WorkflowController.probe, provider, model
        )
        st.session_state[selection_key] = selection
        st.session_state[future_key] = future
        st.session_state[ready_key] = None
        st.session_state[f"{role}_probe_error"] = None
    if not future.done():
        st.markdown(
            f'<div class="env-label"><span>{label}</span>'
            '<span class="env-checking">Checking…</span></div>',
            unsafe_allow_html=True,
        )
        return
    try:
        future.result()
        ready = True
        detail = "API and model responded successfully"
    except Exception as exc:
        ready = False
        detail = f"{type(exc).__name__}: {exc}"
    st.session_state[ready_key] = ready
    st.session_state[f"{role}_probe_error"] = None if ready else detail
    status_class = "env-ready" if ready else "env-failed"
    status_text = "Ready" if ready else "Unavailable"
    st.markdown(
        f'<div class="env-label"><span>{label}</span><span class="{status_class}" '
        f'title="{html.escape(detail)}">{status_text}</span></div>',
        unsafe_allow_html=True,
    )


# The settings step reappears after every solve, successful or not.  Recording
# it again would stack a fresh card per run, so it refreshes the one already
# there instead.
SINGLE_CARD_KINDS = frozenset({"shape_opt_settings"})


def set_event(
    event: WorkflowEvent,
    previous_answer: Any | None = None,
    previous_checkpoint: str | None = None,
) -> None:
    outgoing = st.session_state.get("event")
    st.session_state.session_id = event.session_id
    st.session_state.event = event
    # Bank the step just answered exactly as the user left it: the payload has
    # grown since it was recorded (previews, inferred values, API errors), and
    # the answer is what they typed into it.  A card redrawn later reads both,
    # so it shows what was sent rather than what first appeared.  The step
    # answered is not always the one on screen -- answering an earlier card
    # rolls the transcript back to it first -- so its checkpoint says which.
    answered = previous_checkpoint or getattr(outgoing, "checkpoint_id", None)
    if answered and (outgoing is None or outgoing.session_id == event.session_id):
        history = controller.history(event.session_id)
        if history and history[-1].get("checkpoint_id") == answered:
            fields = {}
            if (
                outgoing is not None
                and outgoing.interrupt
                and outgoing.checkpoint_id == answered
            ):
                fields["interrupt"] = outgoing.interrupt
            if previous_answer is not None:
                fields["answer"] = previous_answer
            if fields:
                controller.update_history_entry(
                    event.session_id, len(history) - 1, **fields
                )
    if event.interrupt:
        marker = interrupt_marker(event)
        if st.session_state.get("last_interrupt") != marker:
            message = event.interrupt.get(
                "message", event.interrupt.get("kind", "Continue")
            )
            history = controller.history(event.session_id)
            last = history[-1] if history else {}
            same_turn = (
                last.get("role") == "assistant"
                and last.get("content") == message
                and last.get("checkpoint_id") == event.checkpoint_id
            )
            kind = event.interrupt.get("kind")
            repeats_last = (
                kind in SINGLE_CARD_KINDS
                and last.get("role") == "assistant"
                and (last.get("interrupt") or {}).get("kind") == kind
            )
            if same_turn:
                controller.update_history_entry(
                    event.session_id, len(history) - 1, interrupt=event.interrupt
                )
            else:
                controller.append_history(
                    event.session_id,
                    "assistant",
                    message,
                    event.checkpoint_id,
                    interrupt=event.interrupt,
                    replace_last=repeats_last,
                )
            st.session_state.last_interrupt = marker
            # A new card opens at its top, so what the manager says is the
            # first thing in view; the end is for watching work.
            st.session_state.scroll_conversation = "card"
    elif event.complete:
        st.session_state.scroll_conversation = "bottom"


@dataclass(frozen=True)
class ConversationCard:
    """One recorded step, drawn as the card the user acted on."""

    index: int
    role: str
    message: str
    interrupt: dict[str, Any] | None
    checkpoint_id: str | None
    active: bool

    @property
    def key(self) -> str:
        return str(self.index)

    def _rollback(self) -> int | None:
        """How much of the transcript this card's answer replaces.

        Acting on the newest card continues the conversation.  Acting on an
        earlier one re-decides that step, so everything the old answer led to
        is dropped and the workflow runs forward again from here.
        """
        return None if self.active else self.index + 1

    def submit(self, answer: Any) -> None:
        submit(answer, checkpoint_id=self.checkpoint_id, truncate_to=self._rollback())

    def resume_in_place(self, answer: dict[str, Any], status: str) -> None:
        resume_in_place(
            answer,
            status,
            checkpoint_id=self.checkpoint_id,
            truncate_to=self._rollback(),
        )


def conversation_cards(event: WorkflowEvent) -> list[ConversationCard]:
    history = controller.history(event.session_id)
    newest = max(
        (
            index
            for index, entry in enumerate(history)
            if entry.get("role") == "assistant"
        ),
        default=None,
    )
    cards = []
    for index, entry in enumerate(history):
        # The newest step, and only while the event in hand is that step: during
        # a step that is still running the transcript and the event belong to
        # different points, and pairing them drew one card's message above
        # another card's contents.
        live = (
            index == newest
            and event.interrupt is not None
            and entry.get("checkpoint_id") == event.checkpoint_id
        )
        cards.append(
            ConversationCard(
                index=index,
                role=entry.get("role", "assistant"),
                message=entry.get("content", ""),
                interrupt=event.interrupt if live else restored_interrupt(entry),
                checkpoint_id=(
                    event.checkpoint_id if live else entry.get("checkpoint_id")
                ),
                active=live,
            )
        )
    return cards


def scroll_conversation(target: str = "bottom", delay_ms: int = 0) -> None:
    """Scroll the conversation pane without moving the surrounding workspace.

    ``target`` is "bottom", or the key of a card's container to bring to the
    top of the pane.  The place is held for five seconds while images above it
    load; scrolling, clicking or typing in the pane in that time lets go.
    """
    selector = json.dumps(f".st-key-{target}" if target != "bottom" else "")
    with SCROLL_BRIDGE.container():
        st.iframe(
            BRIDGE_STYLE
            + f"""
            <script>
              // {time.time_ns()}: a new document each time, so a repeat scrolls.
              const doc = window.parent.document;
              const selector = {selector};
              function scroller() {{
                const panel = doc.querySelector('[class*="st-key-workflow_panel"]');
                if (!panel) return null;
                return [panel, ...panel.querySelectorAll('*')]
                  .filter(node => node.scrollHeight > node.clientHeight + 2)
                  .sort((a, b) =>
                    (b.scrollHeight - b.clientHeight) - (a.scrollHeight - a.clientHeight)
                  )[0] || null;
              }}
              // Pictures in earlier cards keep loading after the page is drawn
              // and push the target down, so it is placed again until the page
              // settles -- unless the user scrolls first, which ends it.
              let cancelled = false;
              function target(box) {{
                if (!selector) return box.scrollHeight - box.clientHeight;
                const card = doc.querySelector(selector);
                if (!card) return null;
                return box.scrollTop + card.getBoundingClientRect().top
                       - box.getBoundingClientRect().top - 8;
              }}
              function place(behavior) {{
                const box = scroller();
                if (!box || cancelled) return;
                const top = target(box);
                if (top !== null && Math.abs(box.scrollTop - top) > 4) {{
                  box.scrollTo({{top, behavior}});
                }}
              }}
              requestAnimationFrame(() => setTimeout(() => {{
                const box = scroller();
                if (!box) return;
                for (const name of ['wheel', 'touchstart', 'keydown', 'mousedown']) {{
                  box.addEventListener(name, () => {{ cancelled = true; }},
                                       {{once: true, passive: true}});
                }}
                place('smooth');
                // Once the smooth scroll has had time to finish, hold the place.
                setTimeout(() => {{
                  const settle = setInterval(() => place('auto'), 150);
                  setTimeout(() => clearInterval(settle), 5000);
                }}, 700);
              }}, {delay_ms}));
            </script>
            """,
            height=1,
        )


def set_workflow_controls_busy(busy: bool) -> None:
    """Block conversation actions without inserting content into their cards."""
    value = "true" if busy else "false"
    update_id = time.time_ns()
    with BUSY_BRIDGE.container():
        st.iframe(
            BRIDGE_STYLE
            + f"""
            <script>
              const busyUpdate = '{update_id}';
              const root = window.parent.document.documentElement;
              root.classList.toggle('fem-workflow-busy', {value});
              const panel = window.parent.document.querySelector(
                '[class*="st-key-workflow_panel"]'
              );
              if (panel) panel.setAttribute('aria-busy', '{value}');
            </script>
            """,
            height=1,
        )



@st.cache_resource
def workflow_pool() -> ThreadPoolExecutor:
    """Runs one workflow step at a time, off the thread drawing the page.

    A forward solve holds its call for minutes.  Run on the script thread it
    freezes the whole app -- the tool window included, which is the one part
    whose job is to show what is happening while you wait.
    """
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="fem-workflow")


@st.cache_resource
def prefetch_pool() -> ThreadPoolExecutor:
    """Builds result assets ahead of the click that asks for them.

    One worker, and a pool of its own.  Prefetching is speculative, so it must
    never run two exports at once, and the step the user is actually waiting on
    must never be queued behind it.
    """
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="fem-prefetch")


def _start_job(
    mode: str,
    answer: Any,
    *,
    status: str = "Manager is working…",
    checkpoint_id: str | None = None,
    truncate_to: int | None = None,
) -> None:
    """Hand a workflow step to the worker and return to drawing the page."""
    session_id = st.session_state.session_id
    event = st.session_state.event
    answered = checkpoint_id or event.checkpoint_id
    # Generation holds the GPU for anything from ten seconds to minutes, and
    # what follows it -- segmentation, then a BC resolve per confirmed set of
    # regions -- needs torch and the warp kernel modules imported.  That is CPU
    # work, so it can happen while the GPU is busy rather than in front of the
    # first call that wants it.
    if (event.interrupt or {}).get("kind") == "geometry_input":
        with contextlib.suppress(OSError):
            controller.preload_tool_worker(event.workflow.get("compute_env"), "cuda")
    st.session_state.job_error = None
    # The status line of the work just started is at the end of the conversation.
    st.session_state.scroll_conversation = "bottom"
    st.session_state.pending_job = {
        "mode": mode,
        "status": status,
        "answer": answer,
        "checkpoint": answered,
        "truncate_to": truncate_to,
        "session_id": session_id,
        "future": workflow_pool().submit(
            controller.resume, session_id, answer, answered
        ),
    }
    st.rerun()


def collect_finished_job() -> None:
    """Apply a finished step's result, on the thread that owns session state."""
    pending = st.session_state.get("pending_job")
    if pending is None or not pending["future"].done():
        return
    st.session_state.pop("pending_job")
    session_id = pending["session_id"]
    try:
        result = pending["future"].result()
    except Exception as exc:
        st.session_state.job_error = f"{type(exc).__name__}: {exc}"
        return
    # Roll the transcript back only now the step has landed.  Doing it when the
    # work was handed over left the conversation describing a state that had
    # not happened yet, and a failed step would have discarded the steps it was
    # meant to replace.
    if pending["truncate_to"] is not None:
        controller.truncate_history(session_id, pending["truncate_to"])
    # A finished solve is the one moment the next several clicks are known in
    # advance, so it is where the viewer's assets are worth building early.
    start_solution_prefetch(result)
    restore_geometry_worker(result)
    if pending["mode"] == "submit":
        set_event(
            result,
            previous_answer=pending["answer"],
            previous_checkpoint=pending["checkpoint"],
        )
        return
    st.session_state.event = result
    # Refining in place forks the thread, so the step now lives at a new
    # checkpoint.  Leaving the old one recorded would send a later edit of this
    # same card back to before the refinement.
    history = controller.history(session_id)
    if history and result.interrupt is not None:
        controller.update_history_entry(
            session_id,
            len(history) - 1,
            checkpoint_id=result.checkpoint_id,
            interrupt=result.interrupt,
        )
    st.session_state.last_interrupt = interrupt_marker(result)
    # Acting on an earlier card rewound the conversation to it: open it at its
    # top, as a new card.  Work on the newest card keeps the end in view.
    st.session_state.scroll_conversation = (
        "card" if pending["truncate_to"] is not None else "bottom")


def first_surface_landed(pending: dict) -> bool:
    """Whether the running solve has written a surface the viewer could show.

    This run's, not the last one's: a session that has solved before still has
    the previous result on disk, and taking that as this run's first shape would
    redraw the page around a picture that is one solve out of date.
    """
    if not isinstance(pending.get("answer"), dict):
        return False
    if pending["answer"].get("action") != "run":
        return False
    root = controller.session_dir(pending["session_id"])
    newest = newest_surface_vtu(root / "solve")
    if newest is None:
        return False
    requested = st.session_state.get("solve_requested", {}).get(
        pending["session_id"]
    )
    try:
        return not requested or newest.stat().st_mtime >= requested
    except OSError:
        return False


def solve_in_flight() -> bool:
    """Whether the step now running is the solve itself.

    The one step that writes while it runs: a forward ramp leaves a surface per
    load step and a shape optimization one per saved iteration, so the viewer
    has something new to show long before the step returns.  Every other step
    produces its artifacts at the end, where the page's own rerun picks them up.
    """
    pending = st.session_state.get("pending_job")
    if not isinstance(pending, dict):
        return False
    answer = pending.get("answer")
    return isinstance(answer, dict) and answer.get("action") == "run"


def describing_segments() -> bool:
    """Whether the step now running is waiting on the segment description.

    The graph segments and then describes within one resume, so there is no
    checkpoint between them to read.  What separates them is the montage: the
    segment step deletes it before it starts and the describe step renders it
    immediately before the call, so its presence marks the boundary.
    """
    pending = st.session_state.get("pending_job")
    if not isinstance(pending, dict):
        return False
    answer = pending.get("answer")
    if not isinstance(answer, dict) or answer.get("action") != "preview_segmentation":
        return False
    montage = (
        controller.session_dir(pending["session_id"])
        / "boundary_conditions" / "segments.png"
    )
    return montage.exists()


@st.fragment(run_every=1.0)
def watch_pending_job() -> None:
    """Redraw the page once the worker is finished, without polling it here."""
    pending = st.session_state.get("pending_job")
    if pending is None:
        return
    if pending["future"].done():
        st.rerun(scope="app")
        return
    # The page is frozen for the length of a step, so a phase boundary the card
    # wants to show has to ask for the one redraw that shows it.  Segmentation
    # handing over to the vision model is one such boundary, and it is worth a
    # redraw because the two waits are minutes and seconds apart.
    if not pending.get("shown_describing") and describing_segments():
        pending["shown_describing"] = True
        st.rerun(scope="app")
    # A solve writing its first surface is the other: until one exists the panel
    # has no result to offer and is showing the geometry, and the live bridge
    # has no viewer to push the rest into.  One redraw hands it both.
    if not pending.get("shown_first_surface") and first_surface_landed(pending):
        pending["shown_first_surface"] = True
        st.rerun(scope="app")


def submit(
    answer, *, checkpoint_id: str | None = None, truncate_to: int | None = None
) -> None:
    """Send an answer for one step and continue from there.

    ``truncate_to`` drops the steps an earlier card's previous answer produced,
    so re-deciding a step rebuilds what follows instead of appending a second
    branch of it.
    """
    _start_job("submit", answer, checkpoint_id=checkpoint_id, truncate_to=truncate_to)


def render_total_force(traction, area, units: dict[str, str], unit_system: str,
                       loaded_area=None) -> None:
    """Show what a traction adds up to over its region, under the field it sets.

    The total is traction times the *loaded* area when it is known: the solver
    loads a region along its one average normal, weighting each face by the
    cosine to it, so a region that faces several ways receives less than
    traction times its whole area -- and the caption says how much less.
    """
    if not area:
        st.caption("Region area unavailable; total force cannot be shown.")
        return
    loaded = float(loaded_area) if loaded_area else float(area)
    magnitude = math.sqrt(sum(float(component) ** 2 for component in traction))
    total = magnitude * loaded
    summary = (
        f"Region area {float(area):.4g} {units['length']}² · "
        f"total force **{total:.4g} {units['force']}**"
    )
    if loaded < 0.995 * float(area):
        summary += (
            f" (the region faces several ways, so only {loaded / float(area):.0%} "
            "of it is loaded; narrow it to one facing to load all of it)"
        )
    gravity = STANDARD_GRAVITY.get(unit_system)
    if gravity:
        summary += f" ≈ {total / gravity:.4g} {units['mass']} at standard gravity"
    st.caption(summary)


def render_texture_controls(key: str, interrupt: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Controls for painting the surface with Hunyuan3D-2.1: a description, or a reference image.

    Returns whether the paint button was pressed, and the answer fields to send.
    """
    painter = interrupt.get("texture_status") or {}
    coloured = interrupt.get("surface_has_colour")
    textured = interrupt.get("textured_surface")
    texture_prompt = st.text_area(
        "Describe its look",
        value=interrupt.get("texture_prompt", ""),
        placeholder="For example: black leather cushions on a polished chrome frame.",
        key=f"texture-prompt-{key}",
    )
    reference_upload = st.file_uploader(
        "Reference image (optional; followed instead of the description)",
        type=["png", "jpg", "jpeg", "webp"],
        key=f"texture-image-{key}",
    )
    use_input = bool(interrupt.get("texture_input_image")) and st.checkbox(
        "Follow the image this surface was generated from",
        key=f"texture-use-input-{key}",
    )
    # A repaint starts from the next seed, so pressing the button again gives
    # a new painting.
    texture_seed = st.number_input(
        "Seed", min_value=0, step=1,
        value=int(interrupt.get("texture_seed") or 0) + (1 if textured else 0),
        key=f"texture-seed-{key}",
    )
    by_image = bool(reference_upload) or use_input
    if not painter.get("ready"):
        st.caption(f"Painting from a description is unavailable: "
                   f"{painter.get('summary', 'unknown')}")
    pressed = st.button(
        "Re-generate texture" if textured else "Generate texture",
        type="secondary" if textured or coloured else "primary",
        disabled=not (painter.get("ready") or by_image),
        key=f"paint-texture-{key}", width="stretch",
    )
    answer: dict[str, Any] = {"texture_prompt": texture_prompt, "texture_seed": int(texture_seed)}
    if reference_upload:
        answer["texture_image"] = reference_upload
    elif use_input:
        answer["texture_image"] = interrupt["texture_input_image"]
    return pressed, answer


def render_texture_previews(interrupt: dict[str, Any], painted_label: str) -> None:
    """The reference image beside six views of the painted surface, once there is one."""
    if not interrupt.get("textured_surface"):
        return
    for column, (label, field) in zip(st.columns(2), (
        ("Reference image", "texture_reference"), (painted_label, "texture_views"),
    )):
        path = interrupt.get(field)
        if path and Path(path).is_file():
            column.image(path, caption=label, width="stretch")


def texture_answer_with_upload(event, answer: dict[str, Any]) -> dict[str, Any]:
    """Save an uploaded reference image into the session, sending its path instead."""
    image = answer.get("texture_image")
    if image is not None and not isinstance(image, str):
        answer = {**answer, "texture_image": str(controller.save_upload(
            event.session_id, image.name, image.getvalue()))}
    return answer


def render_api_error(interrupt: dict[str, Any], source: str) -> None:
    """Show a failed API call under the button that triggered it.

    The card keeps whatever the user typed, so the next click retries with any
    edits they made in between.
    """
    if interrupt.get("api_error") and interrupt.get("api_error_source") == source:
        st.error(interrupt["api_error"], icon=":material/error:")


def resume_in_place(
    answer: dict[str, Any],
    status: str,
    *,
    checkpoint_id: str | None = None,
    truncate_to: int | None = None,
) -> None:
    """Resume work without creating a new visible conversation turn."""
    _start_job(
        "in_place", answer, status=status, checkpoint_id=checkpoint_id,
        truncate_to=truncate_to,
    )


def render_start_conversation(
    provider: str, model: str, compute_env: str, vision_model: str,
    vision_provider: str, text_label: str, vision_label: str,
) -> None:
    # Each model and the environment must have passed its own check, against the
    # selection now on screen rather than an earlier one.  Every field shows its
    # own Ready/Unavailable status, so the button carries no explanation.
    ready = all(
        st.session_state.get(f"{role}_probe_selection") == (provider, chosen)
        and st.session_state.get(f"{role}_probe_ready") is True
        for role, chosen in (("text_model", model), ("vision_model", vision_model))
    )
    environment_status = (
        st.session_state.get("environment_status")
        if st.session_state.get("environment_probe_name") == compute_env
        else None
    )
    started = st.button(
        "Start conversation",
        type="primary",
        disabled=not (
            ready and environment_status is not None and environment_status.ready
        ),
    )
    if not started:
        return
    try:
        with st.spinner("Starting conversation…"):
            # This callback runs inside the setup polling fragment. Streamlit
            # forbids a fragment from writing into the page-level busy bridge;
            # the submitted Start button is already locked by Streamlit itself.
            created = controller.create(
                provider, model, compute_env, vision_model, vision_provider
            )
        chosen_model = created.workflow.get("model") or model
        chosen_vision = created.workflow.get("vision_model") or vision_model
        # Recorded as a card like every other step, so the choices stay on
        # screen instead of collapsing into a line of prose.  It carries no
        # checkpoint: the conversation is built on these, so they are shown
        # rather than offered for editing.
        controller.append_history(
            created.session_id,
            "assistant",
            SETUP_MESSAGE,
            interrupt={
                "kind": "setup",
                "provider": text_label,
                "vision_provider": vision_label,
                "model": chosen_model,
                "vision_model": chosen_vision,
                "compute_env": compute_env,
            },
        )
        st.session_state.last_interrupt = None
        set_event(created)
        st.rerun(scope="app")
    except Exception as exc:
        st.error(f"API/model unavailable: {type(exc).__name__}: {exc}")


@st.fragment(run_every=0.25)
def render_setup_widget() -> None:
    def role_picker(catalogue, role, label):
        """Pick a provider and model for one role, and probe what was picked.

        The roles choose independently: the strongest text model and the one
        that reads pictures best need not be on the same service.  Probe exactly
        what the selectboxes return, not what session state held before they
        ran -- the start button compares the probe's subject against these
        values, so any divergence leaves it disabled forever.  The status line
        is reserved first so it still renders above the dropdowns.
        """
        status = st.empty()
        columns = st.columns(2)
        chosen_label = columns[0].selectbox(
            f"{label} API",
            tuple(catalogue),
            key=f"selected_{role}_provider",
            label_visibility="collapsed",
        )
        chosen_provider = chosen_label.lower()
        chosen_model = columns[1].selectbox(
            label,
            catalogue[chosen_label],
            key=f"selected_{role}_{chosen_provider}",
            label_visibility="collapsed",
        )
        with status.container():
            render_model_probe(chosen_provider, chosen_model, role, label)
        return chosen_label, chosen_provider, chosen_model

    text_label, provider, model = role_picker(
        PROVIDER_TEXT_MODELS, "text_model", "Text model"
    )
    vision_label, vision_provider, vision_model = role_picker(
        PROVIDER_VISION_MODELS, "vision_model", "Vision model"
    )
    try:
        available_envs = conda_environments()
    except Exception as exc:
        available_envs = ()
        st.error(f"Could not detect Conda environments: {exc}")
    if available_envs:
        # The MechFoundry environment, else its pre-rename name on a machine not yet migrated.
        preferred_envs = [DEFAULT_COMPUTE_ENV, *(
            old for old, new in RENAMED_ENVIRONMENTS.items() if new == DEFAULT_COMPUTE_ENV)]
        default_env = next(
            (available_envs.index(name) for name in preferred_envs if name in available_envs), 0
        )
        selected_env = st.session_state.get(
            "selected_compute_environment", available_envs[default_env]
        )
        if selected_env not in available_envs:
            selected_env = available_envs[default_env]
        with st.container(key="environment_field"):
            render_environment_probe(selected_env)
            compute_env = st.selectbox(
                "Compute environment",
                available_envs,
                index=default_env,
                key="selected_compute_environment",
                label_visibility="collapsed",
            )
    else:
        compute_env = ""
    render_start_conversation(
        provider, model, compute_env, vision_model, vision_provider,
        text_label, vision_label,
    )


@st.cache_resource
def load_sampler() -> LoadSampler:
    """One sampler per server process, shared by every session that asks."""
    return LoadSampler()


# One minute of history: long enough to see a solve spin the GPU up, short enough
# that the curve still shows the last few seconds at a readable scale.
RESOURCE_WINDOW = 60
TOOL_SECTIONS = ("Resources", "Solver", "Backend", "Files")


def resource_series(sample) -> list[tuple[str, float | None]]:
    """Named readings for one sample, GPUs first and one entry per card.

    The GPUs are what a solve is waiting on, so they lead the legend, and a
    second card gets its own pair rather than being averaged into the first.
    """
    cards = sample.gpus if sample else ()
    named: list[tuple[str, float | None]] = []
    for index, (utilisation, memory) in enumerate(cards):
        label = "GPU" if len(cards) == 1 else f"GPU {index}"
        named.append((label, utilisation))
        named.append((f"{label} MEM", memory))
    named.append(("CPU", sample.cpu if sample else None))
    named.append(("RAM", sample.memory if sample else None))
    return named


def _fill_height(chart: alt.Chart) -> alt.Chart:
    """Let the chart take whatever height its container has.

    A fixed height cannot honour "adapt to the space available": the page knows
    the pixels, Python does not.
    """
    return chart.properties(height="container", width="container")


@st.fragment(run_every=2.0)
def render_resource_tab() -> None:
    now = time.time()
    history = [
        sample
        for sample in load_sampler().history()
        if now - sample.timestamp <= RESOURCE_WINDOW
    ]
    latest = history[-1] if history else None
    readings = resource_series(latest)
    for column, (label, value) in zip(st.columns(len(readings)), readings):
        column.metric(label, "—" if value is None else f"{value:.0f}%")
    rows = [
        {"age": sample.timestamp - now, "series": label, "percent": value}
        for sample in history
        for label, value in resource_series(sample)
        if value is not None
    ]
    frame = pd.DataFrame(rows, columns=["age", "series", "percent"])
    st.altair_chart(
        _fill_height(
            alt.Chart(frame)
            .mark_line()
            .encode(
                # Both axes are pinned: the window is always the last minute and
                # the scale always the full range, so a glance reads the same way
                # every time instead of re-scaling around whatever arrived.
                x=alt.X(
                    "age:Q",
                    title=None,
                    scale=alt.Scale(domain=[-RESOURCE_WINDOW, 0], nice=False),
                    axis=alt.Axis(labelExpr="datum.value + 's'", tickCount=4),
                ),
                y=alt.Y(
                    "percent:Q",
                    title=None,
                    scale=alt.Scale(domain=[0, 100], nice=False, clamp=True),
                    axis=alt.Axis(
                        values=[0, 25, 50, 75, 100], labelExpr="datum.value + '%'"
                    ),
                ),
                color=alt.Color(
                    "series:N", title=None, sort=[label for label, _ in readings]
                ),
            )
        ),
        width="stretch",
    )


@st.fragment(run_every=2.0)
def render_solver_tab(session_dir: Path | None, session_id: str | None = None) -> None:
    log_path = newest_solve_log(session_dir) if session_dir else None
    requested = st.session_state.get("solve_requested", {}).get(session_id)
    # A failed solve is reported as one, until another is asked for.  Its log
    # never says "finished", so it read as running for good; and one that
    # failed before writing a log left the panel waiting for it forever.
    failure = last_solve_failure(session_dir) if session_dir else None
    failed = failure is not None and not (requested and requested > failure["written"])
    if failed:
        # No error box here: the settings card reports the error, and the panel
        # only says the run failed and what it left (solver_failure_notes).
        for note in solver_failure_notes(failure):
            st.caption(note)
        failed_log = Path(failure["log"]) if failure.get("log") else None
        if failed_log is None or not failed_log.is_file():
            return
        log_path = failed_log
    # A solve that has been asked for but has not written anything yet must not
    # show the last one's progress: the run takes a moment to reach its first
    # iteration, and until it does the panel was still reporting a finished
    # solve, complete with its residuals.
    elif awaiting_solve(log_path, requested):
        st.progress(0.0, text="Starting the solve…")
        st.caption("Waiting for the solver to write its first iteration…")
        return
    if log_path is None:
        st.caption("No solve has run in this conversation yet.")
        return
    try:
        progress = read_solver_progress(
            log_path.read_text(encoding="utf-8", errors="replace")
        )
    except OSError as exc:
        st.caption(f"Could not read the solve log: {exc}")
        return
    state = "failed" if failed else "complete" if progress.finished else "running"
    if progress.optimizing:
        # The outer loop is what a shape optimization is waiting on; its load
        # step is always one of one and says nothing.  Iterations are logged
        # from zero, so the bar counts the one in progress as the first.
        total = f"/{progress.opt_iterations}" if progress.opt_iterations else ""
        label = f"Optimization iteration {progress.opt_iteration + 1}{total}"
    else:
        label = f"Load step {progress.load_step}/{progress.load_steps}"
    st.progress(progress.fraction, text=f"{label} · {state}")
    auto_note = gd_lr_auto_note(progress)
    if auto_note:
        st.caption(auto_note)
    if not progress.iterations:
        st.caption("Waiting for the first Newton iteration…")
        return
    latest = progress.iterations[-1]
    columns = st.columns(4 if progress.objective else 3)
    columns[0].metric("Absolute ‖R‖", f"{latest['residual']:.3e}")
    columns[1].metric("Relative", f"{latest['relative']:.3e}")
    columns[2].metric("Newton iterations", len(progress.iterations))
    if progress.objective:
        columns[3].metric("Objective", f"{progress.objective[-1]['value']:.4e}")
    st.altair_chart(
        _fill_height(residual_chart(progress.iterations, progress.objective)),
        width="stretch",
    )


@st.fragment(run_every=3.0)
def render_backend_tab(session_dir: Path | None) -> None:
    if session_dir is None:
        st.caption("Start a conversation to see backend output.")
        return
    events = read_backend_events(session_dir)
    log_path = newest_solve_log(session_dir)
    if log_path is not None:
        try:
            events.append(
                {
                    "timestamp": log_path.stat().st_mtime,
                    "kind": "solver",
                    "title": f"FEA solve · {log_path.parent.name}",
                    "body": log_path.read_text(encoding="utf-8", errors="replace")[
                        -8000:
                    ],
                    "detail": "",
                }
            )
        except OSError:
            pass
    if not events:
        st.caption("Nothing has run in this conversation yet.")
        return
    # Newest first, because the reason to open this is what just happened.
    for position, event in enumerate(reversed(events)):
        stamp = time.strftime("%H:%M:%S", time.localtime(event["timestamp"]))
        label = f"{stamp} · {event['title']}"
        if event["detail"]:
            label += f" · {event['detail']}"
        with st.expander(label, expanded=position == 0):
            st.code(event["body"], language=None)


# The scalar people look at first, so it leads and is what a field opens on.
COMPONENT_LABELS = {"von_mises": "von Mises", "safety_factor": "Yield safety factor"}


@st.cache_data(show_spinner=False)
def surface_point_fields(path: str, mtime: float) -> tuple[str, ...]:
    """Which point fields a surface VTU carries, keyed so a new solve re-reads.

    The reading itself is shared with the command line's prefetch; only the
    caching around it belongs to the page.
    """
    return read_surface_point_fields(path)


def _file_stamp(path: Path) -> float | None:
    """Modification time of a file, or None when it is not there."""
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def component_asset(
    session_dir: str, compute_env: str, vtu: str, mtime: float,
    field: str, component: str, deform_scale: float = 0.0, step: int = 0,
    total_steps: int = 1, reference_vtu: str = "",
) -> str:
    """Colour one field, building the asset the first time it is asked for.

    Every combination of component, deformation and load step is its own small
    asset, named for all three so that moving a slider back lands on something
    already built rather than rebuilding it -- and so that one the prefetch
    already built is found rather than made again.

    The page cache below is keyed on the asset and its ``.values.bin`` hover
    sidecar as they sit on disk, not only on the solve.  Keyed on the solve
    alone it kept answering with a path whose files had since gone -- an asset
    exported before the sidecar existed, or one a rewind took back -- so the
    viewer lost its hover readout and safety view until the server restarted.
    A missing or rewritten file now changes the key and the monitor is asked
    again; its own receipt cache makes that a stat and a digest look-up.
    """
    output, _ = field_asset_request(
        session_dir, vtu, field, component, deform_scale, step, total_steps,
        reference_vtu,
    )
    target = Path(output)
    return _cached_component_asset(
        session_dir, compute_env, vtu, mtime, field, component, deform_scale,
        step, total_steps, reference_vtu, _file_stamp(target),
        _file_stamp(target.with_suffix(target.suffix + ".values.bin")),
    )


@st.cache_data(show_spinner=False)
def _cached_component_asset(
    session_dir: str, compute_env: str, vtu: str, mtime: float,
    field: str, component: str, deform_scale: float, step: int,
    total_steps: int, reference_vtu: str,
    asset_mtime: float | None, sidecar_mtime: float | None,
) -> str:
    """The export behind ``component_asset``; the two stamps only key the cache."""
    output, arguments = field_asset_request(
        session_dir, vtu, field, component, deform_scale, step, total_steps,
        reference_vtu,
    )
    ToolMonitor(Path(session_dir), compute_env).execute(
        "export_surface_field", arguments
    )
    return output


def start_solution_prefetch(event: WorkflowEvent) -> None:
    """Colour the finished solve while the user is still reading it."""
    if not (event.interrupt or {}).get("solve_complete"):
        return
    session_dir = controller.session_dir(event.session_id)
    newest = newest_surface_vtu(session_dir / "solve")
    if newest is None:
        return
    # A repeat solve writes its own run directory, so the surface it produced
    # identifies it; the mtime covers a run that reused the path.
    solved = (event.session_id, str(newest), newest.stat().st_mtime)
    done = st.session_state.get("prefetched_solves") or set()
    if solved in done:
        return
    st.session_state.prefetched_solves = done | {solved}
    requests = solution_prefetch_requests(session_dir)
    if requests:
        prefetch_pool().submit(
            prefetch_solution_assets, str(session_dir),
            event.workflow.get("compute_env", DEFAULT_COMPUTE_ENV), requests,
        )


def restore_geometry_worker(event: WorkflowEvent) -> None:
    """Load TripoSG again after the solve it made room for.

    The solve unloads it so the GPU is its own; reloading now means that going
    back to the geometry card to try another shape -- the card people iterate on
    most -- does not wait for the model a second time.
    """
    if not (event.interrupt or {}).get("solve_complete"):
        return
    with contextlib.suppress(OSError):
        controller.preload_geometry_model(event.session_id)


def keep_tools_warm(event: WorkflowEvent | None) -> None:
    """Keep a tool worker up, warm to the depth this conversation has reached.

    Called on every rerun, which is what makes losing a worker invisible: one
    that idled out or was killed is back by the next time the page draws, and
    the only cost is the cold start of whichever call found it gone.  Before a
    session exists, the environment is the one the setup card has probed.
    """
    if event is None:
        status = st.session_state.get("environment_status")
        compute_env = status.name if status is not None and status.ready else None
        tier = "light"
    else:
        compute_env = event.workflow.get("compute_env")
        # Up to and including the geometry card nothing has reached a GPU tool;
        # from the next card on, everything does.
        tier = ("light" if (event.interrupt or {}).get("kind") == "geometry_input"
                else "cuda")
    if not compute_env:
        return
    with contextlib.suppress(OSError):
        controller.preload_tool_worker(compute_env, tier)
        # The settings card is the one a solve is started from, and the one a
        # finished solve returns to: its process starts while it is read.
        if event is not None and (event.interrupt or {}).get("kind") == "shape_opt_settings":
            controller.preload_solver_worker(compute_env)


# Where the browser is looking, as a path under the session folder; which rows
# are ticked; which file the window is showing; and a move asked for on one run
# and made on the next -- deferred because the table cannot be told where it is
# on the run that draws it.
BROWSER_AT = "file_browser_at"
BROWSER_ROWS = "file_browser_rows"
BROWSER_PREVIEW = "file_preview"
BROWSER_MOVE = "file_browser_move"
# The query string is how a double-click in the page reaches Python: it carries
# the row, and a rerun brings whatever is in the URL along with it.  The script
# puts the row there and presses refresh, so the two arrive together.
BROWSER_OPEN = "browse_open"
# The listing's row height is pinned rather than left to the theme, because the
# script below reads a row from where the pointer was and the two have to agree.
ROW_HEIGHT = 35


# Turning a double-click on a row into something Python can act on.  Nothing
# else in the table reports one: the grid draws itself on a canvas, and the only
# thing it tells Python is which rows are ticked.  So the click is answered by
# ticking the row it landed on -- which is also what makes this survive sorting,
# since the grid maps a tick back to the right entry however the columns are
# ordered, and a row counted off in pixels would not.
BROWSER_SCRIPT = """
<script>
  (() => {
    const w = window.parent, doc = w.document;
    const ROW = %(row)d, HEADER = %(row)d, TICKS = 48;

    // A previous copy of this frame may still be listening.  The page keeps
    // running while the script behind it changes -- the frame is replaced, the
    // page is never reloaded -- so whatever the last copy registered is taken
    // off before this one registers anything.  Without that, a change here
    // would not reach a page that was already open, and two copies would
    // answer the same click twice.
    for (const [type, fn] of (w.__femBrowse || [])) {
      doc.removeEventListener(type, fn, true);
    }
    const wired = [];
    const listen = (type, fn) => {
      doc.addEventListener(type, fn, true);
      wired.push([type, fn]);
    };
    w.__femBrowse = wired;

    // Which row the pointer is over, counted from the top of the listing and
    // through however far it has been scrolled.  The listing is found when the
    // click happens rather than now: this frame loads while the page is still
    // being built, and the listing is a fresh element after some reruns.
    const spot = (event) => {
      const host = doc.querySelector(
        '[class*="st-key-tool_body_"] [data-testid="stDataFrame"]'
      );
      if (!host || !host.contains(event.target)) return null;
      // The grid's own toolbar -- search, fullscreen -- is its own business.
      if (event.target.closest('[data-testid="stElementToolbar"]')) return null;
      const box = host.getBoundingClientRect();
      const x = event.clientX - box.left, y = event.clientY - box.top;
      // The tick column is the one place a row may be picked out, heading
      // included, so that "all of them" still works.
      if (x < TICKS) return null;
      const scroller = host.querySelector('.dvn-scroller');
      const scrolled = scroller ? scroller.scrollTop : 0;
      return {heading: y < HEADER,
              index: Math.floor((y - HEADER + scrolled) / ROW)};
    };

    // Everything else in the table is inert.  A click on a name drew the
    // grid's own outline around the cell -- painted on the canvas, so no
    // stylesheet reaches it, and never reported to Python, but reading as
    // "this row is chosen" beside ticks that mean exactly that.  A click on a
    // heading sorted the rows, which would put the row under the pointer and
    // the row Python counts to out of step.  Only propagation is stopped and
    // never the default, so the scrollbar still drags and the wheel is
    // untouched.
    for (const type of ['pointerdown', 'mousedown', 'click']) {
      listen(type, (event) => { if (spot(event)) event.stopPropagation(); });
    }

    listen('dblclick', (event) => {
      const at = spot(event);
      if (!at || at.heading || at.index < 0) return;
      // The row goes in the URL and the refresh button is pressed to carry it
      // over: a rerun brings whatever is in the URL with it.  Nothing in the
      // table is touched on the way -- an earlier version said which row by
      // ticking it and unticking it again, and a tick that appears because
      // something was opened is a tick nobody asked for.
      w.history.replaceState(null, '', '?%(open)s=' + at.index);
      const refresh = doc.querySelector(
        '[class*="st-key-browser_refresh"] button'
      );
      if (refresh) refresh.click();
    });
  })();
</script>
""" % {"row": ROW_HEIGHT, "open": BROWSER_OPEN}


@st.cache_data(show_spinner=False, max_entries=256)
def folder_bytes(path: str, mtime: float) -> int:
    """What a folder weighs, remembered against the folder as it was.

    Every row wants a number: Streamlit writes "None" where a number column has
    none, and a listing of folders each reporting None reads as a fault rather
    than as an absence.  The only true figure for a folder is a walk of it, so
    it is walked -- once, keyed on the folder's own modification time, which
    moves whenever anything is added to it or taken out of it.

    A file already in there growing does not move that time, so the figure can
    sit slightly behind a solve still writing.  That is the right trade for a
    column that only describes: what the size cap is decided on is worked out
    afresh, uncached, every time a selection changes.
    """
    return selection_bytes([Path(path)])


def browse_to(root: Path, target: Path) -> None:
    """Ask for the browser to be somewhere else on the next run."""
    st.session_state[BROWSER_MOVE] = (
        "" if target == root else str(target.relative_to(root))
    )
    st.rerun()


def forget_preview() -> None:
    """Let go of the file the window was showing.

    The window can be dismissed without pressing anything in it -- the corner
    cross, or a click outside -- and until this ran on that path, the file was
    still recorded as the one being shown, so the next thing clicked anywhere
    on the page redrew the window that had just been closed.
    """
    st.session_state.pop(BROWSER_PREVIEW, None)


@st.dialog("File preview", width="large", on_dismiss=forget_preview)
def preview_file(path_text: str) -> None:
    """One file in a window over the page.

    The title is fixed where the dialog is declared, so the file names itself
    in the body -- the same shape the delete prompt uses.
    """
    path = Path(path_text)
    st.write(f"**{path.name}**")
    try:
        info = path.stat()
    except OSError:
        st.write("This file is no longer on disk.")
        if st.button("Close", key="preview_close", width="stretch"):
            forget_preview()
            st.rerun()
        return
    st.caption(
        f"{human_size(info.st_size)} · "
        f"{time.strftime('%b %d, %H:%M', time.localtime(info.st_mtime))}"
    )
    kind = preview_kind(path)
    if kind == "image":
        st.image(str(path), width="stretch")
    elif kind == "text":
        text, size = preview_text(path)
        st.code(text, language=None)
        if size > PREVIEW_TEXT_LIMIT:
            st.caption(
                f"Showing the first {human_size(PREVIEW_TEXT_LIMIT)} of "
                f"{human_size(size)}. Download it to read the rest."
            )
    else:
        st.caption(
            "There is no preview for this kind of file. It can still be "
            "downloaded."
        )
    take, close = st.columns(2)
    take.download_button(
        "Download", data=lambda: path.open("rb"), file_name=path.name,
        mime=download_mime(path), key="preview_download", width="stretch",
    )
    if close.button("Close", key="preview_close", width="stretch"):
        forget_preview()
        st.rerun()


def render_files_tab(session_dir: Path | None) -> None:
    """Everything this conversation has written, to look at and to take away.

    Not a timed fragment like the three sections beside it.  Those report a
    quantity that moves on its own; this one shows a selection somebody is in
    the middle of making, and redrawing underneath a half-made one would be the
    wrong thing to do.  The refresh is a button.
    """
    if session_dir is None:
        st.caption("Start a conversation to browse its files.")
        return
    root = session_dir.resolve()
    # Both acted on here, before the table exists, so that what it is told about
    # its own selection is settled before it draws itself.
    move = st.session_state.pop(BROWSER_MOVE, None)
    if move is not None:
        st.session_state[BROWSER_AT] = move
    # Re-resolved every draw: a folder deleted underneath the browser puts it
    # back at the session root rather than leaving it pointed at nothing.
    here = resolve_within(root, st.session_state.get(BROWSER_AT, "")) or root
    if not here.is_dir():
        here = root
    # The table is named after the folder it is showing, so entering a
    # directory builds a new one instead of inheriting the last one's ticks.
    # Ticks are reported by position and three rows down in one folder is a
    # different file in the next, so carrying them over would hand the download
    # button things nobody pointed at.  Making that impossible is worth more
    # than remembering to clear them: clearing is one line that has to run on
    # every route out of a folder, and this is none.
    rows_key = f"{BROWSER_ROWS}_{hashlib.sha1(str(here).encode()).hexdigest()[:12]}"
    for stale in [key for key in st.session_state
                  if key.startswith(BROWSER_ROWS) and key != rows_key]:
        del st.session_state[stale]
    entries = listing(here)
    heading = st.container(key="browser_heading")
    # A real integer and a real timestamp rather than text already formatted:
    # the table sorts what it is given, and "3.8 MB" sorts alphabetically.
    frame = pd.DataFrame({
        "Name": pd.Series(
            [("📁 " if entry.is_dir else "📄 ") + entry.name for entry in entries],
            dtype="string",
        ),
        "Size": pd.Series(
            [folder_bytes(str(entry.path), entry.mtime) if entry.is_dir
             else entry.size for entry in entries],
            dtype="int64",
        ),
        "Modified": pd.Series(
            [datetime.fromtimestamp(entry.mtime) for entry in entries],
            dtype="datetime64[ns]",
        ),
    })
    event = st.dataframe(
        frame,
        key=rows_key,
        on_select="rerun",
        selection_mode="multi-row",
        hide_index=True,
        width="stretch",
        height="stretch",
        row_height=ROW_HEIGHT,
        column_config={
            "Name": st.column_config.TextColumn("Name"),
            "Size": st.column_config.NumberColumn("Size", format="bytes"),
            "Modified": st.column_config.DatetimeColumn(
                "Modified", format="MMM D, HH:mm"
            ),
        },
    )
    rows = sorted(event.selection["rows"]) if event and event.selection else []
    # A double-click landed on a row, and the script put which one in the URL.
    # Nothing about the ticks is read or written here: opening something is not
    # a way of choosing what to download, and a double-click that left a tick
    # behind would be making that choice on the reader's behalf.
    opened = st.query_params.get(BROWSER_OPEN)
    if opened is not None:
        del st.query_params[BROWSER_OPEN]
        index = int(opened) if opened.isdigit() else -1
        # The listing can have changed under a run that was still writing since
        # the pointer went down, so the row is checked against it rather than
        # trusted.
        if 0 <= index < len(entries):
            target = entries[index]
            if target.is_dir:
                browse_to(root, target.path)
            else:
                st.session_state[BROWSER_PREVIEW] = str(target.path)
                st.rerun()

    chosen = [entries[row].path for row in rows if row < len(entries)]
    total = selection_bytes(chosen)
    if needs_archive(chosen):
        # One thing names itself -- a folder picked on its own is the common
        # case, and "1 items" is not what anybody meant.
        subject = chosen[0].name if len(chosen) == 1 else f"{len(chosen)} items"
        detail = f"Download {subject} · {human_size(total)} as a .zip"
        payload = lambda: build_archive(here, chosen)
        name, mime = archive_name(root, here, chosen), "application/zip"
    elif chosen:
        subject = chosen[0].name
        detail = f"Download {subject} · {human_size(total)}"
        payload = lambda: chosen[0].open("rb")
        name, mime = chosen[0].name, download_mime(chosen[0])
    else:
        subject = ""
        detail, name, mime = "Tick something to download it", "nothing", ""
        payload = lambda: b""
    # The finished download is held in the server's own memory until the
    # browser has taken it, so a whole session -- gigabytes of it -- would go
    # into the process running the conversation.  The total is worked out
    # before the button is drawn, so the figure shown is the one that decided.
    oversize = total > ARCHIVE_LIMIT
    # The heading was reserved above the table and is filled in now: what it has
    # to say about the selection is only known once the table has reported one.
    with heading:
        # The path on the left, the three things that can be done on the right,
        # all on one line -- opening is a double-click on the row itself and
        # needs no button.
        where, up, again, take = st.columns(
            4, vertical_alignment="center", gap="small"
        )
        summary = f" · {subject} · {human_size(total)}" if chosen else ""
        where.markdown(
            '<div class="browser-path"><span>'
            f"{html.escape(breadcrumb(root, here))}{html.escape(summary)}"
            "</span></div>",
            unsafe_allow_html=True,
        )
        if up.button("↑", key="browser_up", width="stretch",
                     disabled=here == root, help="The folder above this one"):
            browse_to(root, here.parent)
        if again.button("↻", key="browser_refresh", width="stretch",
                        help="Read this folder again"):
            st.rerun()
        take.download_button(
            "⬇", data=payload, file_name=name, mime=mime or None,
            key="browser_download", width="stretch", help=detail,
            disabled=oversize or not chosen,
        )
    if oversize:
        st.markdown(
            f'<div class="browser-note">{human_size(total)} is more than the '
            f"{human_size(ARCHIVE_LIMIT)} a download may weigh. Pick fewer "
            "things, or go into the folder and take part of it.</div>",
            unsafe_allow_html=True,
        )
    # Last, so the table it reaches for has been drawn.
    with st.container(key="ui_bridge_browser"):
        st.iframe(BRIDGE_STYLE + BROWSER_SCRIPT, height=1)
    pending = st.session_state.get(BROWSER_PREVIEW)
    if pending:
        preview_file(pending)


def render_tool_window(event: WorkflowEvent | None) -> None:
    st.markdown(
        '<div class="conversation-header"><div class="conversation-heading">'
        "Tools</div></div>",
        unsafe_allow_html=True,
    )
    session_dir = controller.session_dir(event.session_id) if event else None
    # A keyed control rather than st.tabs: the choice is then a widget value that
    # survives a rerun, so opening an earlier conversation cannot land the window
    # on a section nobody asked for, and the active one is plainly marked.
    section = (
        st.segmented_control(
            "Tool section",
            TOOL_SECTIONS,
            default=TOOL_SECTIONS[0],
            key="tool_section",
            label_visibility="collapsed",
        )
        or TOOL_SECTIONS[0]
    )
    # Only the backlog of backend output scrolls; the other two fit their space.
    scrolls = "scroll" if section == "Backend" else "fixed"
    with st.container(border=False, key=f"tool_body_{scrolls}"):
        if section == "Solver":
            render_solver_tab(session_dir, event.session_id if event else None)
        elif section == "Backend":
            render_backend_tab(session_dir)
        elif section == "Files":
            render_files_tab(session_dir)
        else:
            render_resource_tab()


# Every step is embedded in the frame, so the payload grows with the ramp.  Six
# is roughly where that stops being reasonable; longer ramps show their last
# step alone rather than loading tens of megabytes into the page.
MAX_EMBEDDED_STEPS = 6


# Bounded rather than unbounded: each entry is a whole surface, several
# megabytes of it, and an optimization has a hundred to scrub through.
ENCODED_ASSETS = 12


@st.cache_data(show_spinner=False, max_entries=ENCODED_ASSETS)
def asset_data_url(path: str, mtime: float) -> str:
    """One asset as a data URI, kept so a redraw does not re-encode it.

    The viewer redraws itself every couple of seconds while a solve is writing,
    and a surface runs to megabytes; reading and encoding one on every pass
    would cost more than the rest of the page together.  Keyed on the file's own
    modification time, so an asset rebuilt in place is encoded again.
    """
    return "data:model/gltf-binary;base64," + base64.b64encode(
        Path(path).read_bytes()
    ).decode("ascii")


@st.cache_data(show_spinner=False)
def asset_range(path: str, mtime: float) -> dict[str, float] | None:
    """The extremes the exporter recorded beside one asset, if it wrote them."""
    try:
        meta = json.loads(
            Path(path + ".json").read_text(encoding="utf-8")
        )
        return {"minimum": float(meta["minimum"]), "maximum": float(meta["maximum"])}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def embedded_frame(asset: Path, scales: list[float],
                   name: str | None = None) -> dict[str, Any]:
    """One shape, as the viewer takes it: asset, displacements, and its scale.

    ``name`` is what the slider calls this position -- the iteration that wrote
    it, for an optimization.  The colour range travels with the frame because an
    optimization colours each shape over its own extremes, so the bar beside the
    picture has to change when the picture does.
    """
    payload = surface_deformation(asset, scales)
    return {
        "glb": asset_data_url(str(asset), asset.stat().st_mtime),
        "disp": payload["data"] if payload else None,
        "name": name,
        "range": asset_range(str(asset), asset.stat().st_mtime),
        "hover": surface_hover(asset),
    }


def render_glb(
    path: Path | None,
    height: int = 360,
    legend: list[dict[str, str]] | None = None,
    colorbar: dict[str, Any] | None = None,
    deformation: dict[str, Any] | None = None,
    steps: list[dict[str, Any]] | None = None,
    length_unit: str | None = None,
    step_label: str = "Load step",
    live: bool = False,
    view_key: str | None = None,
) -> None:
    """Show one asset, with the controls that belong to the picture itself.

    ``deformation`` carries the surface's displacements and how far they may be
    exaggerated.  Its slider lives in here rather than in the page because a
    Streamlit widget costs a rerun and an asset rebuild per step: the shape has
    to be re-exported, re-encoded and re-loaded before anything moves.  With the
    displacements in hand the viewer moves the vertices itself, so dragging is
    immediate.

    ``length_unit`` is the session's own -- metres, millimetres, inches -- and
    is what the ground grid says its squares measure.  None means this asset has
    no physical size to report, which is true of a generated surface until it is
    scaled, and the grid then carries no figure rather than an invented one.

    ``step_label`` names what the frames are -- load steps for a ramp,
    iterations for a shape optimization -- and ``live`` says the run is still
    writing, so the control is drawn even for the single shape that exists so
    far and shapes pushed in later have somewhere to land.

    ``view_key`` (controller.visualization_key) is which visualization this is.
    The view spins until the user first drags, wheels or touches it, and spins
    again only when the key changes; None spins on every build.
    """
    if path is not None and not path.exists():
        return
    source_attribute = ""
    if path is not None:
        source = asset_data_url(str(path), path.stat().st_mtime)
        source_attribute = f' src="{source}"'
    legend_rows = []
    for item in legend or []:
        color = str(item.get("color", "#64748b"))
        if not (len(color) == 7 and color.startswith("#")):
            color = "#64748b"
        legend_rows.append(
            '<div class="legend-row"><span class="legend-swatch" '
            f'style="background:{color}"></span>{html.escape(str(item.get("label", "")))}</div>'
        )
    legend_markup = (
        f'<div class="viewer-legend">{"".join(legend_rows)}</div>'
        if legend_rows
        else ""
    )
    colorbar_markup = ""
    if colorbar:
        minimum = float(colorbar["minimum"])
        maximum = float(colorbar["maximum"])
        unit = html.escape(str(colorbar.get("unit", "")))
        # Two lines, so the bar stays narrow whatever the field is called.
        colorbar_title = "<br>".join(
            html.escape(word)
            for word in str(colorbar.get("title", "Displacement magnitude")).split(" ", 1)
        )
        middle = (
            math.sqrt(minimum * maximum)
            if colorbar.get("scale") == "logarithmic" and minimum > 0.0
            else (minimum + maximum) / 2
        )
        colorbar_markup = (
            '<div class="viewer-colorbar">'
            f'<div class="colorbar-title">{colorbar_title}</div>'
            '<div class="colorbar-content">' +
            ('<div class="colorbar-gradient" style="background:linear-gradient(to top,#cd232d 0%,#cd232d 19%,#f5cd37 20%,#37be5f 40%,#37be5f 100%)"></div>'
             if colorbar.get("safety_factor") else '<div class="colorbar-gradient"></div>') +
            '<div class="colorbar-labels">'
            f"<span>{maximum:.4g} {unit}</span>"
            f"<span>{middle:.4g} {unit}</span>"
            f"<span>{minimum:.4g} {unit}</span>"
            "</div></div></div>"
        )
    # Every load step is carried into the frame together, so stepping through a
    # ramp is a change of which one is shown rather than a rebuild: the page
    # would otherwise re-export, re-encode and reload an asset per step.
    frames = list(steps or [])
    if not frames and path is not None:
        frames = [{
            "glb": source,
            "disp": (deformation or {}).get("data"),
            "hover": surface_hover(path),
        }]
    frames_payload = json.dumps(frames)
    step_markup = ""
    if len(frames) > 1 or (live and frames):
        last = len(frames) - 1
        named = frames[last].get("name")
        step_markup = (
            '<div class="viewer-step">'
            f'<label for="step-range">{html.escape(step_label)} '
            f'<span id="step-value">'
            f'{html.escape(str(named) if named is not None else f"{len(frames)}/{len(frames)}")}'
            '</span></label>'
            f'<input id="step-range" type="range" min="0" max="{last}" '
            f'step="1" value="{last}">'
            "</div>"
        )
    deform_markup = ""
    deform_payload = "null"
    if deformation and deformation.get("data"):
        top = float(deformation["max_scale"])
        stops = max(int(deformation.get("stops", 10)), 2)
        deform_payload = json.dumps({
            "data": deformation["data"], "max": top, "step": top / (stops - 1),
        })
        deform_markup = (
            '<div class="viewer-deform">'
            '<label for="deform-range">Deformation ×<span id="deform-value">0</span></label>'
            f'<input id="deform-range" type="range" min="0" max="{top:.6g}" '
            f'step="{top / (stops - 1):.6g}" value="0">'
            "</div>"
        )
    safety_enabled = bool((colorbar or {}).get("safety_factor"))
    safety_unit = (colorbar or {}).get("stress_unit", "Pa").split(" ", 1)[0]
    safety_scale = 1e6 if safety_unit == "Pa" else 1.0
    safety_display_unit = "MPa" if safety_unit == "Pa" else safety_unit
    peak_stress = 0.0
    for frame in frames:
        hover = frame.get("hover") or {}
        if hover.get("kind") == "scalar" and hover.get("data"):
            values = np.frombuffer(base64.b64decode(hover["data"]), dtype="<f8")
            if len(values):
                peak_stress = max(peak_stress, float(np.max(values)))
    strength_max = max(1000.0 if safety_display_unit == "MPa" else 1.0,
                       peak_stress / safety_scale * 10)
    safety_markup = (
        '<div class="viewer-safety">'
        f'<label for="strength-range">Yield strength [{html.escape(safety_display_unit)}]</label>'
        '<div class="safety-row">'
        f'<input id="strength-range" type="range" min="0" max="{strength_max:.6g}" step="any" value="0">'
        '<input id="strength-number" aria-label="Yield strength" type="number" min="0" step="any" value="0"></div>'
        '<label for="target-range">Target safety factor</label><div class="safety-row">'
        '<input id="target-range" type="range" min="1" max="10" step="0.1" value="2">'
        '<input id="target-number" aria-label="Target safety factor" type="number" min="1" step="0.1" value="2"></div>'
        '</div>'
    ) if safety_enabled else ""
    safety_payload = json.dumps({
        "enabled": safety_enabled, "scale": safety_scale,
        "unit": safety_display_unit,
        "storage": str(path.parent) if path else "safety-view",
    })
    controls_markup = (
        f'<div class="viewer-controls">{step_markup}{deform_markup}{safety_markup}</div>'
        if (step_markup or deform_markup or safety_markup) else ""
    )
    unit_payload = json.dumps(length_unit)
    hover_title = json.dumps((colorbar or {}).get("title", "Surface value"))
    hover_unit = json.dumps((colorbar or {}).get("unit", ""))
    hover_legend = json.dumps(legend or [])
    markup = f"""
    <style>
      html,body{{margin:0;padding:0;height:100%;overflow:hidden;background:#f4f6f8}}
      /* Fills the frame rather than standing at a fixed height, so everything
         pinned to its edges -- the gizmo at the bottom, the legend at the top --
         travels with the viewport.  No minimum: one would stop the shell
         shrinking past it and strand the gizmo again.  A frame that is never
         stretched still gives this its own height, which is what Python asked
         for. */
      .viewer-shell{{position:relative;width:100%;height:100%;
        border-radius:12px;overflow:hidden}}
      #main-view{{display:block;width:100%;height:100%}}
      /* The safety controls are one more card in the slider column, dressed
         like the deformation slider above them: a bare <label> or <input>
         here falls back to the browser's serif default. */
      .viewer-safety{{
        padding:0.35rem 0.6rem;border-radius:0.55rem;
        background:rgba(255,255,255,0.84);color:#334155;
        font:500 12px/1.3 sans-serif;
      }}
      .viewer-safety label{{display:block;margin:0.3rem 0 0.15rem}}
      .viewer-safety label:first-child{{margin-top:0}}
      .safety-row{{display:flex;gap:0.5rem;align-items:center}}
      .safety-row input[type=range]{{width:8rem;accent-color:#2659f2;cursor:pointer}}
      .safety-row input[type=number]{{width:4.5rem;box-sizing:border-box;
        padding:0.1rem 0.3rem;border:1px solid rgba(51,65,85,0.3);border-radius:0.35rem;
        background:#fff;color:#334155;font:inherit}}
      /* The top-left corner holds the ground scale and, in the safety view, the
         summary under it: one column, so the two can never sit on each other
         whatever height either takes. */
      .viewer-corner{{
        position:absolute;left:0.65rem;top:0.65rem;max-width:48%;
        display:flex;flex-direction:column;align-items:flex-start;gap:0.3rem;
        pointer-events:none;
      }}
      .safety-summary{{
        padding:0.35rem 0.55rem;border-radius:0.55rem;
        background:rgba(255,255,255,0.84);color:#334155;
        font:500 11px/1.4 sans-serif;white-space:pre-line;
      }}
      .viewer-hover{{position:absolute;display:none;pointer-events:none;z-index:10;
        padding:7px 9px;background:rgba(255,255,255,.95);color:#334155;
        border:1px solid #cbd5e1;border-radius:6px;font:12px/1.5 sans-serif;
        white-space:pre-line;max-width:calc(100% - 20px)}}
      #axes-view{{
        /* The gizmo's origin sits at the centre of its own square, so half the
           canvas is empty between the corner and the axes.  A small inset keeps
           it off the edge without adding to that. */
        position:absolute;left:0.15rem;bottom:0.15rem;
        background:transparent;box-shadow:none;pointer-events:none;
      }}
      .viewer-legend{{
        position:absolute;right:0.65rem;top:0.65rem;max-width:48%;padding:0.45rem 0.55rem;
        border-radius:0.55rem;background:rgba(255,255,255,0.84);color:#334155;
        font:500 12px/1.3 sans-serif;pointer-events:none;
      }}
      /* Stacked in the bottom-right corner: the gizmo owns the bottom-left and
         the colour bar the top-right, and one column keeps the two sliders
         aligned without either of them naming an offset for the other. */
      .viewer-controls{{
        position:absolute;right:0.65rem;bottom:0.65rem;
        display:flex;flex-direction:column;align-items:stretch;gap:0.3rem;
      }}
      .viewer-deform, .viewer-step{{
        padding:0.35rem 0.6rem;border-radius:0.55rem;
        background:rgba(255,255,255,0.84);color:#334155;
        font:500 12px/1.3 sans-serif;
        display:flex;align-items:center;justify-content:space-between;gap:0.5rem;
      }}
      .viewer-deform input, .viewer-step input{{
        width:8rem;accent-color:#2659f2;cursor:pointer;
      }}
      .legend-row{{display:flex;align-items:center;gap:0.4rem;white-space:nowrap}}
      .legend-swatch{{width:0.75rem;height:0.75rem;border-radius:0.18rem;flex:0 0 auto}}
      .viewer-colorbar{{
        position:absolute;right:0.65rem;top:0.65rem;padding:0.5rem 0.6rem;
        border-radius:0.55rem;background:rgba(255,255,255,0.84);color:#334155;
        font:500 12px/1.2 sans-serif;pointer-events:none;
      }}
      .colorbar-title{{margin-bottom:0.4rem;white-space:normal;line-height:1.25}}
      .colorbar-content{{display:flex;height:150px;gap:0.42rem}}
      .colorbar-gradient{{
        width:16px;border:1px solid rgba(51,65,85,0.3);
        background:linear-gradient(to top,#2341b4,#1eafd2,#37be5f,#f5cd37,#cd232d);
      }}
      .colorbar-labels{{display:flex;flex-direction:column;justify-content:space-between;white-space:nowrap}}
      /* The free corner: the gizmo owns the bottom-left, the colour bar and the
         segment legend the top-right, the sliders the bottom-right. */
      .viewer-scale{{
        padding:0.3rem 0.5rem;
        border-radius:0.5rem;background:rgba(255,255,255,0.84);color:#334155;
        font:500 11px/1.2 sans-serif;pointer-events:none;display:none;
      }}
      .viewer-error{{
        position:absolute;left:0.65rem;top:0.65rem;right:0.65rem;padding:0.5rem 0.6rem;
        border-radius:0.55rem;background:rgba(127,29,29,0.92);color:#fff;
        font:500 11px/1.35 ui-monospace,monospace;white-space:pre-wrap;display:none;
      }}
    </style>
    <script type="importmap">
      {{"imports":{{"three":"https://unpkg.com/three@0.180.0/build/three.module.js"}}}}
    </script>
    <div class="viewer-shell">
      <canvas id="main-view"></canvas>
      <div class="viewer-hover" id="viewer-hover" role="status"></div>
      <canvas id="axes-view" aria-label="World axes: X red, Y green, Z blue"></canvas>
      {legend_markup}
      {colorbar_markup}
      {controls_markup}
      <div class="viewer-corner">
        <div class="viewer-scale" id="viewer-scale"></div>
        <div class="safety-summary" id="safety-summary" style="display:none"></div>
      </div>
      <div class="viewer-error" id="viewer-error"></div>
    </div>
    <script type="module">
      import * as THREE from 'three';
      import {{OrbitControls}} from 'https://unpkg.com/three@0.180.0/examples/jsm/controls/OrbitControls.js';
      import {{GLTFLoader}} from 'https://unpkg.com/three@0.180.0/examples/jsm/loaders/GLTFLoader.js';

      const srgbToLinear = (c) =>
        c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);

      // A throw inside the module leaves the renderer's black clear colour on
      // screen, which looks exactly like a shading bug.  Surface it instead.
      const reportViewerError = (label, error) => {{
        const box = document.querySelector('#viewer-error');
        if (!box) return;
        box.textContent = label + ': ' + (error && error.message ? error.message : error);
        box.style.display = 'block';
      }};
      window.addEventListener('error', (event) => reportViewerError('script', event.error || event.message));
      window.addEventListener('unhandledrejection', (event) => reportViewerError('load', event.reason));

      const shell = document.querySelector('.viewer-shell');
      const canvas = document.querySelector('#main-view');
      const renderer = new THREE.WebGLRenderer({{canvas, antialias:true, alpha:false}});
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setClearColor(0xf4f6f8, 1);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      // Khronos PBR Neutral rather than ACES: it leaves a base colour as it
      // is up to ~0.76 and only rolls off above, so a swatch lit head-on
      // shows the hex it was written with.  three's ACES also scales its
      // input by 1/0.6 and drifts hues (orange to yellow, red to pink) on the
      // way to white.
      renderer.toneMapping = THREE.NeutralToneMapping;
      renderer.toneMappingExposure = 1.0;
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(35, 1, 0.001, 10000);
      camera.position.set(2.4, 1.8, 2.4);
      const controls = new OrbitControls(camera, canvas);
      controls.enableDamping = true;
      controls.autoRotateSpeed = 1.25;
      // Spin until the user takes hold of the view, and again on every switch
      // to another visualization -- but not on a rerun that rebuilds this
      // frame for the one already on screen, which would undo the user's
      // drag every few seconds while a solve is writing.  The frame dies with
      // each rebuild, so what survives it is a record in the host tab's
      // sessionStorage: the visualization last shown and whether its spin was
      // stopped.  Landing on any other key rewrites the record, so going back
      // to an earlier visualization counts as a switch and spins too.
      const VIEW_KEY = {json.dumps(view_key)};
      const SPIN_RECORD = 'implicit:viewer-spin';
      let spinStore = null;
      try {{ spinStore = window.parent.sessionStorage; }} catch (error) {{}}
      const readSpin = () => {{
        try {{ return JSON.parse(spinStore.getItem(SPIN_RECORD) || 'null'); }} catch (error) {{ return null; }}
      }};
      const writeSpin = (stopped) => {{
        if (VIEW_KEY === null) return;
        try {{ spinStore.setItem(SPIN_RECORD, JSON.stringify({{view: VIEW_KEY, stopped}})); }} catch (error) {{}}
      }};
      const lastSpin = readSpin();
      const sameView = VIEW_KEY !== null && lastSpin !== null && lastSpin.view === VIEW_KEY;
      controls.autoRotate = !(sameView && lastSpin.stopped === true);
      if (!sameView) writeSpin(false);
      // OrbitControls raises 'start' for a pointer drag, a wheel zoom and a
      // touch alike -- every way of taking hold of the view.
      controls.addEventListener('start', () => {{
        controls.autoRotate = false;
        writeSpin(true);
      }});
      controls.target.set(0, 0, 0);

      // Intensities are physical: a light of intensity I adds I*cos/pi of a
      // matte face's colour.  The rig used to be 1.35 / 0.22 / 3.2 / 1.35 /
      // 1.8, which put 1.6-1.9x the stored colour on a face turned toward the
      // lights, before ACES's 1/0.6 -- so every BC region, segment and contour
      // came out several steps lighter than its hex (#1f77b4 rendered as
      // #75b7dd, #f00000 as #ff725a, #ff8f00 as #facf77, measured on a sphere
      // facing the camera).  Same directions and ratios, scaled so the
      // brightest face gets exactly 1x; ambient raised a little so faces
      // turned away from every light keep ~13 % instead of going black.
      // Measured after: #1f77b4 -> #0b6ca6 facing the camera, #0d6fab
      // between camera and key light; #f00000 -> #e00303 / #e50404.
      scene.add(new THREE.HemisphereLight(0xffffff, 0x475569, 0.6));
      scene.add(new THREE.AmbientLight(0xffffff, 0.35));
      const key = new THREE.DirectionalLight(0xffffff, 1.4);
      key.position.set(3, 5, 4);
      key.castShadow = true;
      key.shadow.mapSize.set(2048, 2048);
      key.shadow.bias = -0.0002;
      key.shadow.normalBias = 0.025;
      scene.add(key);
      scene.add(key.target);
      const fill = new THREE.DirectionalLight(0xb9d7ff, 0.6);
      fill.position.set(-4, 1, -3);
      scene.add(fill);
      scene.add(fill.target);
      const rim = new THREE.DirectionalLight(0xffffff, 0.8);
      rim.position.set(-2, 4, 5);
      scene.add(rim);
      scene.add(rim.target);
      // No distance decay: the camera sits at 2.35x the model's span, so with
      // the physical 1/d^2 a 0.3 m part got ~0.45 of its colour added and a
      // 2 m one ~0.01 -- brightness depended on the object's size.
      const headlight = new THREE.PointLight(0xffffff, 0.26, 0, 0);
      headlight.position.set(0, 0, 0);
      camera.add(headlight);
      scene.add(camera);

      const axesCanvas = document.querySelector('#axes-view');
      const axesRenderer = new THREE.WebGLRenderer({{canvas:axesCanvas, antialias:true, alpha:true}});
      axesRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      // A fixed square: the gizmo is a constant-size key, not something that
      // scales with the view.  It holds its corner because the shell it is
      // pinned to now follows the panel.  setSize updates the canvas style too,
      // so the drawing buffer and the displayed size cannot drift apart.
      axesRenderer.setSize(104, 104);
      axesRenderer.setClearColor(0xffffff, 0);
      axesRenderer.outputColorSpace = THREE.SRGBColorSpace;
      const axesScene = new THREE.Scene();
      const axesCamera = new THREE.PerspectiveCamera(40, 1, 0.1, 20);
      axesScene.add(new THREE.AxesHelper(1));

      function axisLabel(text, color, position) {{
        const label = document.createElement('canvas');
        label.width = 64; label.height = 64;
        const context = label.getContext('2d');
        context.font = 'bold 44px sans-serif';
        context.textAlign = 'center'; context.textBaseline = 'middle';
        context.fillStyle = color; context.fillText(text, 32, 34);
        const material = new THREE.SpriteMaterial({{
          map:new THREE.CanvasTexture(label), transparent:true, depthTest:false
        }});
        const sprite = new THREE.Sprite(material);
        sprite.position.copy(position); sprite.scale.set(0.34, 0.34, 0.34);
        axesScene.add(sprite);
      }}
      axisLabel('X', '#dc2626', new THREE.Vector3(1.22, 0, 0));
      axisLabel('Y', '#16a34a', new THREE.Vector3(0, 1.22, 0));
      axisLabel('Z', '#2563eb', new THREE.Vector3(0, 0, 1.22));

      function resize() {{
        const width = Math.max(shell.clientWidth, 1);
        const height = Math.max(shell.clientHeight, 1);
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
      }}
      new ResizeObserver(resize).observe(shell);
      resize();

      // Declared above the loader that fills them in: a `const` used before its
      // declaration throws at load and takes the whole viewer down.
      const safetySettings = {safety_payload};
      const strengthRange = document.getElementById('strength-range');
      const strengthNumber = document.getElementById('strength-number');
      const targetRange = document.getElementById('target-range');
      const targetNumber = document.getElementById('target-number');
      const safetySummary = document.getElementById('safety-summary');
      const weakMarker = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 8),
        new THREE.MeshBasicMaterial({{color:0x8b00ff, depthTest:false}}));
      weakMarker.renderOrder = 10;
      weakMarker.visible = false;
      scene.add(weakMarker);
      let weakMesh = null, weakIndex = -1;
      let lastHoverTime = 0;

      const groundUnit = {unit_payload};
      const deformation = {deform_payload};
      const deformRange = document.getElementById('deform-range');
      const deformValue = document.getElementById('deform-value');
      let deformMesh = null;
      let restPositions = null;
      let deformDeltas = null;

      function decodeDeltas(encoded) {{
        const binary = atob(encoded);
        const bytes = new Uint8Array(binary.length);
        for (let index = 0; index < binary.length; index += 1) {{
          bytes[index] = binary.charCodeAt(index);
        }}
        return new Float32Array(bytes.buffer);
      }}

      function applyDeformation() {{
        if (!deformMesh || !restPositions || !deformDeltas) return;
        const scale = Number(deformRange.value);
        const positions = deformMesh.geometry.getAttribute('position');
        for (let index = 0; index < restPositions.length; index += 1) {{
          positions.array[index] = restPositions[index] + deformDeltas[index] * scale;
        }}
        positions.needsUpdate = true;
        // Lighting follows the deformed shape, and the bounding sphere has to
        // keep up or the surface is culled once it moves past the original one.
        deformMesh.geometry.computeVertexNormals();
        deformMesh.geometry.computeBoundingSphere();
        deformValue.textContent = scale.toPrecision(3);
        positionWeakPoint();
      }}

      if (deformRange) deformRange.addEventListener('input', applyDeformation);

      const frames = {frames_payload};
      const stepRange = document.getElementById('step-range');
      const stepValue = document.getElementById('step-value');
      const loadedFrames = new Array(frames.length).fill(null);
      let cameraFramed = false;

      function prepareModel(gltf, encodedDisplacement, hover) {{
        const model = gltf.scene;
        let mesh = null;
        let rest = null;
        let deltas = null;
        if (encodedDisplacement) {{
          try {{
            deltas = decodeDeltas(encodedDisplacement);
          }} catch (error) {{
            reportViewerError('deformation', error);
            deltas = null;
          }}
        }}
        model.traverse((node) => {{
          if (!node.isMesh) return;
          node.castShadow = true; node.receiveShadow = true;
          if (deltas && !mesh) {{
            // The surface the displacements belong to is the one with a vertex
            // for each of them; anything else in the scene is left alone.
            const positions = node.geometry.getAttribute('position');
            if (positions && positions.count * 3 === deltas.length) {{
              mesh = node;
              rest = Float32Array.from(positions.array);
            }}
          }}
            const vertexColors = node.geometry.getAttribute('color');
            if (vertexColors) {{
              // COLOR_0 holds sRGB bytes -- segment swatches and contour ramp
              // alike are authored as display colours -- but glTF declares that
              // attribute linear, so the renderer's sRGB output encode is applied
              // to a value whose encode was never removed and every colour comes
              // out far lighter than its swatch (#1f77b4 reads as ~#62b6db).
              // Undo it into a fresh float attribute: writing back into the
              // source would store raw floats in its normalized uint8 array.
              // On any failure keep the original colours rather than risk a
              // black surface -- too light still beats unreadable.
              try {{
                const items = vertexColors.itemSize >= 4 ? 4 : 3;
                const linear = new Float32Array(vertexColors.count * items);
                for (let index = 0; index < vertexColors.count; index += 1) {{
                  const at = index * items;
                  linear[at] = srgbToLinear(vertexColors.getX(index));
                  linear[at + 1] = srgbToLinear(vertexColors.getY(index));
                  linear[at + 2] = srgbToLinear(vertexColors.getZ(index));
                  if (items === 4) linear[at + 3] = vertexColors.getW(index);
                }}
                node.geometry.setAttribute(
                  'color', new THREE.BufferAttribute(linear, items)
                );
              }} catch (error) {{
                reportViewerError('colour', error);
              }}
            }}
            const materials = Array.isArray(node.material) ? node.material : [node.material];
            for (const material of materials) {{
              if (material?.color && !node.geometry.getAttribute('color') && !material.map) material.color.multiplyScalar(0.78);
              // A vertex-coloured asset (BC regions, segments, contours) is a
              // key: a glossy sheen on its upward faces reads as a lighter
              // colour, so it is kept nearly matte.
              if ('roughness' in material) material.roughness = node.geometry.getAttribute('color') ? 0.85 : 0.58;
              if ('metalness' in material) material.metalness = Math.min(material.metalness, 0.08);
            }}
        }});
        return {{model: model, mesh: mesh, rest: rest, deltas: deltas,
          hover: hover ? {{...hover, values: decodeSamples(hover)}} : null}};

      }}

      // The ground the model stands on, drawn as a grid so a surface floating in
      // an empty frame gets a floor to be read against.  It lies on the x-z
      // plane through the model's lowest point rather than through y=0: a
      // placed model does sit on the ground -- every solved surface here starts
      // within a centimetre of zero -- but what the geometry card shows is
      // TripoSG's own output, which is centred on the origin, so a grid at y=0
      // would saw it exactly in half.  The lowest point is the floor in both
      // cases.
      //
      // Everything here is in metres, so the cell is rounded to a 1/2/5 decade
      // and the lines snapped to multiples of it -- then the spacing is a scale
      // bar rather than an arbitrary subdivision of whatever this model happens
      // to measure.  The grid carries no highlighted centre line: it is centred
      // on the model, not on the origin, and a bright line through it would
      // read as a world axis it is not.  The gizmo is where orientation comes
      // from.  Nothing is filled in: lines alone, writing no depth, so orbiting
      // under the floor looks up through it at the model rather than at a lid.
      // How far the grid reaches, in multiples of the model, and how dark its
      // lines are directly under it.
      const GROUND_REACH = 2.2;
      const GROUND_OPACITY = 0.8;
      let ground = null;

      // The nearest 1/2/5 of a decade, not the next one up: rounding 0.23 up to
      // 0.5 leaves two cells under a model that wanted five.  Nearest in the
      // ratio, since that is how a cell twice too big and one half too big
      // compare to the eye.
      // Reported in the session's own length unit and never converted into a
      // tidier one: a millimetre session's cell of 50 measures 50 mm, and
      // calling that 5 cm would be the one number on screen not in the units
      // every other number here is quoted in.  toPrecision drops the float
      // noise a decade of halving leaves behind.
      function cellLabel(cell) {{
        return String(Number(cell.toPrecision(3))) + ' ' + groundUnit;
      }}

      function roundCell(raw) {{
        const decade = Math.pow(10, Math.floor(Math.log10(raw)));
        let best = decade;
        for (const step of [1, 2, 5, 10]) {{
          const candidate = step * decade;
          if (Math.abs(Math.log(raw / candidate)) < Math.abs(Math.log(raw / best))) {{
            best = candidate;
          }}
        }}
        return best;
      }}

      function placeGround(center, floor, span) {{
        if (ground) {{
          scene.remove(ground);
          ground.geometry.dispose();
          ground.material.dispose();
        }}
        const cell = roundCell(span / 5);
        const rings = Math.max(Math.ceil(GROUND_REACH * span / cell), 3);
        const reach = rings * cell;
        const originX = Math.round(center.x / cell) * cell;
        const originZ = Math.round(center.z / cell) * cell;
        // Drawn cell by cell rather than as whole lines so the fade can follow
        // the distance from the model: a line's colour interpolates between its
        // two ends, and one span of it from edge to edge would fade along its
        // own length instead of outwards.
        const positions = [];
        const colours = [];
        const corner = (x, z) => {{
          positions.push(originX + x, floor, originZ + z);
          const fade = Math.max(0, 1 - Math.hypot(x, z) / reach);
          colours.push(
            srgbToLinear(0.58), srgbToLinear(0.64), srgbToLinear(0.72),
            Math.pow(fade, 1.4) * GROUND_OPACITY
          );
        }};
        for (let line = -rings; line <= rings; line += 1) {{
          for (let piece = -rings; piece < rings; piece += 1) {{
            corner(line * cell, piece * cell);
            corner(line * cell, (piece + 1) * cell);
            corner(piece * cell, line * cell);
            corner((piece + 1) * cell, line * cell);
          }}
        }}
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute(
          'position', new THREE.Float32BufferAttribute(positions, 3)
        );
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colours, 4));
        // Lines under a surface stay behind it, but they write no depth of
        // their own: a grid line has no thickness to occlude anything with, and
        // writing it would let the far half of the grid punch through the near.
        // The tone mapping is for the model's shading, not for a flat line on a
        // flat background, so the grid opts out and keeps the colour asked for.
        const material = new THREE.LineBasicMaterial({{
          vertexColors: true, transparent: true, depthWrite: false,
          toneMapped: false,
        }});
        ground = new THREE.LineSegments(geometry, material);
        scene.add(ground);
        // Saying what one square measures is what turns the grid from a
        // backdrop into something the model can be read off.
        const scaleBox = document.querySelector('#viewer-scale');
        if (scaleBox && groundUnit) {{
          scaleBox.textContent = cellLabel(cell) + ' per square';
          scaleBox.style.display = 'block';
        }}
      }}

      function frameCamera(model) {{
          const box = new THREE.Box3().setFromObject(model);
          const center = box.getCenter(new THREE.Vector3());
          const size = box.getSize(new THREE.Vector3());
          const span = Math.max(size.x, size.y, size.z, 1e-6);
          controls.target.copy(center);
          placeGround(center, box.min.y, span);
          camera.position.copy(center).add(new THREE.Vector3(1.5, 1.05, 1.5).normalize().multiplyScalar(span * 2.35));
          camera.near = Math.max(span / 1000, 1e-5);
          camera.far = span * 100;
          camera.updateProjectionMatrix();
          key.target.position.copy(center);
          key.position.copy(center).add(new THREE.Vector3(3, 5, 4).normalize().multiplyScalar(span * 3));
          key.shadow.camera.left = -span * 1.2;
          key.shadow.camera.right = span * 1.2;
          key.shadow.camera.top = span * 1.2;
          key.shadow.camera.bottom = -span * 1.2;
          key.shadow.camera.near = Math.max(span * 0.05, 1e-5);
          key.shadow.camera.far = span * 8;
          key.shadow.camera.updateProjectionMatrix();
          fill.target.position.copy(center);
          fill.position.copy(center).add(new THREE.Vector3(-4, 1, -3).normalize().multiplyScalar(span * 3));
          rim.target.position.copy(center);
          rim.position.copy(center).add(new THREE.Vector3(-2, 4, 5).normalize().multiplyScalar(span * 3));
          controls.update();
      }}

      // What the slider calls each position.  A ramp counts its steps; an
      // optimization names the iteration that wrote the shape, which is not the
      // position when the run saved every tenth one or when the live bridge
      // skipped past an iteration it could not keep up with.
      function frameLabel(index) {{
        const frame = frames[index];
        if (frame && frame.name !== null && frame.name !== undefined) {{
          return String(frame.name);
        }}
        return (index + 1) + '/' + frames.length;
      }}

      // Each shape of an optimization is coloured over its own extremes, so the
      // bar has to change when the picture does; a ramp, coloured against one
      // reference, simply hands every frame the same numbers.
      const colorbarLabels = document.querySelectorAll(
        '.viewer-colorbar .colorbar-labels span'
      );
      const colorbarUnit = colorbarLabels.length === 3
        ? (colorbarLabels[0].textContent.split(' ').slice(1).join(' ') || '') : '';

      // Four significant figures, switching to an exponent exactly where
      // Python's own %.4g does, so a bar rewritten here reads the same as the
      // one the page drew.
      function significant(value) {{
        if (!isFinite(value)) return String(value);
        if (value === 0) return '0';
        const exponent = Math.floor(Math.log10(Math.abs(value)));
        if (exponent < -4 || exponent >= 4) {{
          return value.toExponential(3).replace(/e([+-])(\d)$/, 'e$10$2');
        }}
        return String(Number(value.toPrecision(4)));
      }}

      function showRange(range) {{
        if (!range || colorbarLabels.length !== 3) return;
        const text = (value) => {{
          const rounded = significant(value);
          return (colorbarUnit ? rounded + ' ' + colorbarUnit : rounded);
        }};
        colorbarLabels[0].textContent = text(range.maximum);
        colorbarLabels[1].textContent = text((range.minimum + range.maximum) / 2);
        colorbarLabels[2].textContent = text(range.minimum);
      }}

      function showFrame(index) {{
        for (let other = 0; other < loadedFrames.length; other += 1) {{
          if (loadedFrames[other]) {{
            loadedFrames[other].model.visible = other === index;
          }}
        }}
        const entry = loadedFrames[index];
        if (!entry) return;
        // The deformation follows whichever step is on screen: each one has its
        // own displacements, and a ramp's last step moves furthest.
        deformMesh = entry.mesh;
        restPositions = entry.rest;
        deformDeltas = entry.deltas;
        if (stepValue) {{
          stepValue.textContent = frameLabel(index);
        }}
        if (!safetySettings.enabled) showRange(frames[index] && frames[index].range);
        applyDeformation();
        updateSafety();
      }}

      function selectedFrame() {{
        return stepRange ? Number(stepRange.value) : frames.length - 1;
      }}

      function loadFrame(frame, index) {{
        new GLTFLoader().load(frame.glb, (gltf) => {{
          try {{
            const entry = prepareModel(gltf, frame.disp, frame.hover);
            loadedFrames[index] = entry;
            entry.model.visible = false;
            scene.add(entry.model);
            if (!cameraFramed) {{
              frameCamera(entry.model);
              cameraFramed = true;
            }}
            // Steps arrive in whatever order they decode; show the selected one
            // as soon as it is among them rather than waiting for the rest.
            if (loadedFrames[selectedFrame()]) showFrame(selectedFrame());
          }} catch (error) {{
            reportViewerError('scene', error);
            throw error;
          }}
        }}, undefined, (error) => reportViewerError('fetch', error));
      }}

      function positionWeakPoint() {{
        if (!weakMesh || weakIndex < 0) return;
        weakMesh.updateWorldMatrix(true, false);
        weakMarker.position.copy(weakMesh.localToWorld(new THREE.Vector3().fromBufferAttribute(
          weakMesh.geometry.getAttribute('position'), weakIndex)));
      }}
      function updateSafety() {{
        if (!safetySettings.enabled) return;
        weakMarker.visible = false;
        weakMesh = null; weakIndex = -1;
        const entry = loadedFrames[selectedFrame()];
        if (!entry?.hover?.values) return;
        const values = entry.hover.values;
        const strength = Number(strengthNumber.value) * safetySettings.scale;
        const target = Number(targetNumber.value);
        const valid = Number.isFinite(strength) && strength > 0 && Number.isFinite(target) && target >= 1;
        let peak = 0, belowOne = 0, belowTarget = 0;
        for (let i = 0; i < values.length; i++) {{
          if (values[i] > peak) {{peak = values[i]; weakIndex = i;}}
          const factor = values[i] > 0 ? strength / values[i] : Infinity;
          if (factor < 1) belowOne++;
          if (factor < target) belowTarget++;
        }}
        entry.model.traverse(node => {{
          if (!node.isMesh || node.geometry.getAttribute('position').count !== values.length) return;
          // Written in the same linear space prepareModel decodes COLOR_0 into:
          // THREE.Color.set parses the sRGB hex and, with three's colour
          // management on (the default since r152), stores it linear, so the
          // renderer's one sRGB output encode lands each band on its legend hex
          // under the same lit, matte shading as every other contour.
          const colors = new Float32Array(values.length * 3);
          for (let i = 0; i < values.length; i++) {{
            const factor = values[i] > 0 ? strength / values[i] : Infinity;
            const color = new THREE.Color();
            if (!valid) color.set('#94a3b8');
            else if (factor < 1) color.set('#cd232d');
            else if (factor >= target) color.set('#37be5f');
            else color.set('#f5cd37');
            colors.set(color.toArray(), i * 3);
          }}
          node.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
          if (weakIndex >= 0) weakMesh = node;
        }});
        if (weakMesh) {{
          const size = new THREE.Box3().setFromObject(entry.model).getSize(new THREE.Vector3());
          weakMarker.scale.setScalar(Math.max(size.x, size.y, size.z) * 0.008);
          weakMarker.visible = true;
          positionWeakPoint();
        }}
        const minimum = peak > 0 ? strength / peak : Infinity;
        safetySummary.style.display = 'block';
        safetySummary.textContent = valid
          ? 'Minimum surface factor: ' + (minimum === Infinity ? '∞' : significant(minimum))
            + '\\nBelow 1: ' + significant(100 * belowOne / values.length) + '% of surface samples'
            + '\\nBelow target ' + significant(target) + ': ' + significant(100 * belowTarget / values.length) + '%'
            + (peak > 0 ? '\\nPurple marker: lowest factor (highest stress)' : '\\nZero surface stress')
          : 'Set the material yield strength to calculate safety factors.'
            + (peak > 0 ? '\\nPurple marker: highest-stress surface point' : '');
        const cap = Math.max(5, target * 2);
        showRange({{minimum:0, maximum:cap}});
        const gradient = document.querySelector('.colorbar-gradient');
        if (gradient && !valid) gradient.style.background = '#94a3b8';
        else if (gradient) gradient.style.background = 'linear-gradient(to top,#cd232d 0%,#cd232d '
          + (100 / cap) + '%,#f5cd37 ' + (100 / cap) + '%,#f5cd37 '
          + (target * 100 / cap) + '%,#37be5f ' + (target * 100 / cap) + '%,#37be5f 100%)';
        lastHoverTime = 0;
      }}
      if (safetySettings.enabled) {{
        try {{
          const saved = JSON.parse(localStorage.getItem('manual-safety:' + safetySettings.storage));
          if (saved && Number.isFinite(saved.strength) && saved.strength >= 0) {{
            strengthNumber.value = saved.strength;
            strengthRange.max = Math.max(Number(strengthRange.max), saved.strength);
            strengthRange.value = saved.strength;
          }}
          if (saved && Number.isFinite(saved.target) && saved.target >= 1) {{
            targetNumber.value = saved.target;
            targetRange.max = Math.max(10, saved.target);
            targetRange.value = saved.target;
          }}
        }} catch (error) {{ /* Storage may be disabled in an embedded viewer. */ }}
        for (const [range, number] of [[strengthRange,strengthNumber],[targetRange,targetNumber]]) {{
          range.addEventListener('input', () => {{number.value = range.value; updateSafety(); saveSafety();}});
          number.addEventListener('input', () => {{
            const value = Number(number.value);
            if (!Number.isFinite(value) || value < Number(number.min)) {{updateSafety(); return;}}
            range.max = Math.max(Number(range.max), value);
            range.value = value; updateSafety(); saveSafety();
          }});
        }}
      }}
      function saveSafety() {{
        try {{
          localStorage.setItem('manual-safety:' + safetySettings.storage, JSON.stringify({{
            strength:Number(strengthNumber.value), target:Number(targetNumber.value),
          }}));
        }} catch (error) {{ /* Controls remain usable without storage. */ }}
      }}

      frames.forEach(loadFrame);

      function discardFrame(index) {{
        const entry = loadedFrames[index];
        if (entry) {{
          scene.remove(entry.model);
          entry.model.traverse((node) => {{
            if (node.geometry) node.geometry.dispose();
            if (node.material) node.material.dispose();
          }});
        }}
        frames.splice(index, 1);
        loadedFrames.splice(index, 1);
      }}

      // Shapes written after this frame was built arrive here rather than by
      // rebuilding it: a rebuild would reload every shape already in hand and
      // put the camera back where it started, several times a minute, for the
      // whole of a run.
      window.addEventListener('message', (event) => {{
        const message = event.data;
        if (!message || message.femFrame !== true) return;
        try {{
          const following = !stepRange
            || Number(stepRange.value) >= frames.length - 1;
          if (message.drop !== null && message.drop !== undefined) {{
            const at = stepRange ? Number(stepRange.value) : -1;
            discardFrame(message.drop);
            // Everything above the discarded shape has moved down a place, so
            // a viewer pinned to one of them follows the shape rather than
            // staying on the position it used to occupy.
            if (stepRange && at > message.drop) {{
              stepRange.value = String(at - 1);
            }}
          }}
          frames.push({{glb: message.glb, disp: message.disp,
                       name: message.name, range: message.range, hover: message.hover}});
          const last = frames.length - 1;
          if (stepRange) {{
            stepRange.max = String(last);
            // Following the run unless the user has scrubbed back to look at a
            // shape, which is the one moment the view must not be moved.
            if (following) stepRange.value = String(last);
            stepRange.parentElement.hidden = false;
          }}
          loadFrame(frames[last], last);
        }} catch (error) {{
          reportViewerError('frame', error);
        }}
      }});

      if (stepRange) {{
        stepRange.addEventListener('input', () => showFrame(selectedFrame()));
      }}

      const tooltip = document.getElementById('viewer-hover');
      const hoverTitle = {hover_title};
      const hoverUnit = {hover_unit};
      const hoverLegend = {hover_legend};
      const raycaster = new THREE.Raycaster();
      const pointer = new THREE.Vector2();
      let hoverPointer = null;
      let dragging = false;
      function decodeSamples(hover) {{
        const binary = atob(hover.data);
        const bytes = Uint8Array.from(binary, c => c.charCodeAt(0));
        return hover.kind === 'segment' ? new Int32Array(bytes.buffer) : new Float64Array(bytes.buffer);
      }}
      canvas.addEventListener('pointermove', event => {{
        const box = canvas.getBoundingClientRect();
        hoverPointer = {{x:event.clientX - box.left, y:event.clientY - box.top}};
        pointer.set(hoverPointer.x / box.width * 2 - 1, 1 - hoverPointer.y / box.height * 2);
      }});
      canvas.addEventListener('pointerleave', () => {{hoverPointer = null; tooltip.style.display = 'none';}});
      controls.addEventListener('start', () => {{dragging = true; tooltip.style.display = 'none';}});
      controls.addEventListener('end', () => {{dragging = false;}});
      function updateHover() {{
        // Bound picking work on dense surfaces while the model auto-rotates.
        const now = performance.now();
        if (now - lastHoverTime < 100) return;
        lastHoverTime = now;
        tooltip.style.display = 'none';
        if (!hoverPointer || dragging) return;
        const entry = loadedFrames[selectedFrame()];
        if (!entry) return;
        raycaster.setFromCamera(pointer, camera);
        const hit = raycaster.intersectObject(entry.model, true).find(item => item.object.isMesh);
        if (!hit || !hit.face) return;
        const local = hit.object.worldToLocal(hit.point.clone());
        const positions = hit.object.geometry.getAttribute('position');
        const ids = [hit.face.a, hit.face.b, hit.face.c];
        const corners = ids.map(index => new THREE.Vector3().fromBufferAttribute(positions, index));
        const bary = THREE.Triangle.getBarycoord(local, ...corners, new THREE.Vector3());
        if (!bary) return;
        const weights = [bary.x, bary.y, bary.z];
        const lines = [];
        const data = entry.hover;
        if (data?.kind === 'segment' && data.values.length === (hit.object.geometry.index
            ? hit.object.geometry.index.count / 3 : positions.count / 3)) {{
          lines.push('Segment ' + data.values[hit.faceIndex]);
        }} else if (data?.kind === 'scalar' && data.values.length === positions.count) {{
          let value = ids.reduce((sum, index, i) => sum + weights[i] * data.values[index], 0);
          if (safetySettings.enabled) {{
            const strength = Number(strengthNumber.value) * safetySettings.scale;
            lines.push('von Mises stress: ' + significant(value / safetySettings.scale) + ' ' + safetySettings.unit);
            value = strength > 0 ? (value > 0 ? strength / value : Infinity) : NaN;
          }} else if (data.component === 'safety_factor') value = value > 0 ? data.yield_strength / value : Infinity;
          lines.push(hoverTitle + ': ' + (Number.isNaN(value) ? 'set yield strength' : value === Infinity ? '∞' : significant(value)) + (hoverUnit ? ' ' + hoverUnit : ''));
        }} else if (hoverLegend.length) {{
          // Region colors are discrete and flat across each triangle.
          const colors = hit.object.geometry.getAttribute('color');
          if (colors) {{
            const observed = new THREE.Color().setRGB(colors.getX(ids[0]), colors.getY(ids[0]), colors.getZ(ids[0]));
            let closest = null, distance = Infinity;
            for (const item of hoverLegend) {{
              const swatch = new THREE.Color(item.color);
              const d = observed.toArray().reduce((sum, v, i) => sum + (v - swatch.toArray()[i]) ** 2, 0);
              if (d < distance) {{distance = d; closest = item.label;}}
            }}
            if (distance < 0.005 && closest) lines.push(closest);
          }}
        }}
        lines.push('Position: ' + hit.point.toArray().map(significant).join(', ') + (groundUnit ? ' ' + groundUnit : ' (model units)'));
        tooltip.textContent = lines.join('\\n');
        tooltip.style.display = 'block';
        tooltip.style.left = Math.max(4, Math.min(hoverPointer.x + 12, shell.clientWidth - tooltip.offsetWidth - 4)) + 'px';
        tooltip.style.top = Math.max(4, Math.min(hoverPointer.y + 12, shell.clientHeight - tooltip.offsetHeight - 4)) + 'px';
      }}

      function animate() {{
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
        updateHover();
        const direction = camera.position.clone().sub(controls.target).normalize();
        axesCamera.position.copy(direction.multiplyScalar(4.2));
        axesCamera.up.copy(camera.up);
        axesCamera.lookAt(0, 0, 0);
        axesRenderer.render(axesScene, axesCamera);
      }}
      animate();
    </script>
    """
    st.iframe(markup, height=height + 8)


SPLIT_HANDLES = """
<script>
(() => {
  // The handles live in the host document, not in this frame, because the
  // gaps they sit over belong to Streamlit's layout.  This frame is rebuilt on
  // every rerun and its listeners die with it, so each run clears the previous
  // pair and makes a live one rather than trying to reuse dead handles.
  let doc, root, store;
  try {
    doc = window.parent.document;
    root = doc.documentElement;
    store = window.parent.localStorage;
  } catch (err) {
    return;  // Cross-origin: no handles rather than a broken page.
  }

  const SPLIT = 'implicit:workspace-split';
  const TOOLS = 'implicit:tools-height';
  const ASIDE = 'implicit:sidebar-width';
  const read = (name) => { try { return store.getItem(name); } catch (err) { return null; } };
  const write = (name, value) => { try { store.setItem(name, value); } catch (err) {} };
  const restore = (name, key) => {
    const value = read(key);
    if (value) root.style.setProperty(name, value);
  };
  restore('--workspace-split', SPLIT);
  restore('--tools-height', TOOLS);

  // A narrower default than the theme's, applied as an inline width so that
  // Streamlit's own resize handle still wins when dragged.  Whatever width the
  // sidebar ends up at is remembered, so a drag survives the next load.
  const aside = doc.querySelector('[data-testid="stSidebar"]');
  if (aside) {
    const stored = read(ASIDE);
    if (stored) {
      aside.style.width = stored;
    } else if (!aside.dataset.implicitSized) {
      aside.style.width = '16rem';
      aside.dataset.implicitSized = '1';
    }
  }
  let asideWidth = aside ? aside.style.width : '';
  const rememberAside = () => {
    if (!aside || !aside.style.width || aside.style.width === asideWidth) return;
    asideWidth = aside.style.width;
    write(ASIDE, asideWidth);
  };

  doc.querySelectorAll('.workspace-split').forEach((node) => node.remove());

  const block = () => doc.querySelector(
    '[data-testid="stHorizontalBlock"]' +
    ':has([class*="st-key-workflow_panel"])' +
    ':has([class*="st-key-visualization_panel"])'
  );
  const wrapper = (key) => doc.querySelector(
    '[data-testid="stLayoutWrapper"]:has(> [class*="st-key-' + key + '"])'
  );

  const make = (axis, title) => {
    const handle = doc.createElement('div');
    handle.className = 'workspace-split';
    handle.dataset.axis = axis;
    handle.title = title;
    handle.style.display = 'none';
    doc.body.appendChild(handle);
    return handle;
  };
  const THICKNESS = 8;
  const vertical = make('x', 'Drag to resize · double-click to reset');
  const horizontal = make('y', 'Drag to resize · double-click to reset');

  const place = () => {
    const frame = block();
    if (!frame) {
      vertical.style.display = horizontal.style.display = 'none';
      return;
    }
    const columns = frame.querySelectorAll(':scope > [data-testid="stColumn"]');
    const outer = frame.getBoundingClientRect();
    if (columns.length === 2) {
      const left = columns[0].getBoundingClientRect();
      const right = columns[1].getBoundingClientRect();
      Object.assign(vertical.style, {
        display: '',
        left: ((left.right + right.left) / 2 - THICKNESS / 2) + 'px',
        top: outer.top + 'px',
        width: THICKNESS + 'px',
        height: outer.height + 'px',
      });
    } else {
      vertical.style.display = 'none';
    }
    const viewer = wrapper('visualization_panel');
    const tools = wrapper('tool_panel');
    if (viewer && tools) {
      const above = viewer.getBoundingClientRect();
      const below = tools.getBoundingClientRect();
      Object.assign(horizontal.style, {
        display: '',
        top: ((above.bottom + below.top) / 2 - THICKNESS / 2) + 'px',
        left: below.left + 'px',
        width: below.width + 'px',
        height: THICKNESS + 'px',
      });
    } else {
      horizontal.style.display = 'none';
    }
  };

  const clamp = (value, low, high) => Math.min(Math.max(value, low), high);

  const draggable = (handle, property, key, measure) => {
    handle.addEventListener('pointerdown', (event) => {
      event.preventDefault();
      handle.setPointerCapture(event.pointerId);
      handle.dataset.dragging = '1';
      const selection = doc.body.style.userSelect;
      doc.body.style.userSelect = 'none';
      const move = (moving) => {
        const value = measure(moving);
        if (value === null) return;
        root.style.setProperty(property, value);
        write(key, value);
        place();
      };
      const stop = (ending) => {
        try { handle.releasePointerCapture(ending.pointerId); } catch (err) {}
        delete handle.dataset.dragging;
        doc.body.style.userSelect = selection;
        handle.removeEventListener('pointermove', move);
        handle.removeEventListener('pointerup', stop);
        handle.removeEventListener('pointercancel', stop);
      };
      handle.addEventListener('pointermove', move);
      handle.addEventListener('pointerup', stop);
      handle.addEventListener('pointercancel', stop);
    });
    handle.addEventListener('dblclick', () => {
      root.style.removeProperty(property);
      try { store.removeItem(key); } catch (err) {}
      place();
    });
  };

  draggable(vertical, '--workspace-split', SPLIT, (event) => {
    const frame = block();
    if (!frame) return null;
    const outer = frame.getBoundingClientRect();
    if (!outer.width) return null;
    const share = clamp((event.clientX - outer.left) / outer.width, 0.25, 0.78);
    return (share * 100).toFixed(2) + '%';
  });

  draggable(horizontal, '--tools-height', TOOLS, (event) => {
    const frame = block();
    if (!frame) return null;
    const outer = frame.getBoundingClientRect();
    // Measured from the bottom, so the viewer above keeps whatever is left and
    // the pair still fills the column exactly.
    return Math.round(clamp(outer.bottom - event.clientY, 96, outer.height - 160)) + 'px';
  });

  place();
  rememberAside();
  // Streamlit reflows constantly -- a rerun, a fragment, an expanding card --
  // and the handles have to follow without watching every mutation, so they
  // re-measure on a slow tick and immediately on a resize.
  const timer = window.setInterval(() => { place(); rememberAside(); }, 150);
  window.parent.addEventListener('resize', place);
  window.addEventListener('pagehide', () => {
    window.clearInterval(timer);
    window.parent.removeEventListener('resize', place);
  });
})();
</script>
"""


@st.dialog("Delete conversation")
def confirm_delete(session_id: str) -> None:
    """Ask before removing a session and everything it produced."""
    workflow = controller.read_workflow(session_id)
    title = workflow.get("geometry_description") or "New FEM analysis"
    st.write(f"Delete **{title}**?")
    st.caption(
        "Its geometry, boundary conditions and solve results are removed from "
        "disk. This cannot be undone."
    )
    keep, remove = st.columns(2)
    if keep.button("Keep", key="dialog_keep", width="stretch"):
        st.session_state.pop("pending_delete", None)
        st.rerun()
    if remove.button("Delete", key="dialog_confirm", type="primary",
                     width="stretch"):
        st.session_state.pop("pending_delete", None)
        try:
            controller.delete_session(session_id)
        except OSError as exc:
            st.session_state.delete_error = f"Could not delete: {exc}"
        else:
            if st.session_state.get("session_id") == session_id:
                st.session_state.pop("event", None)
                st.session_state.pop("session_id", None)
                st.session_state.last_interrupt = None
        st.rerun()


with st.sidebar:
    with st.container(key="sidebar_controls"):
        st.header("Conversations")
        if st.button("New conversation", type="primary", width="stretch"):
            st.session_state.pop("event", None)
            st.session_state.pop("session_id", None)
            st.session_state.last_interrupt = None
            st.rerun()
    with st.container(height=500, border=False, key="conversation_list"):
        if OUTPUT_ROOT != DEFAULT_OUTPUT_ROOT:
            # Opened on another root (FEM_WORKFLOW_OUTPUT_ROOT): say which, since
            # the list and the resident workers are that root's.
            st.caption(f"Sessions in `{OUTPUT_ROOT.name}`", help=str(OUTPUT_ROOT))
        # Dividers are drawn from the sessions that are left, so a day whose
        # last conversation is deleted takes its heading with it.
        drawn_day = None
        for session_id, touched in controller.dated_sessions():
            day = day_heading(touched)
            if day != drawn_day:
                st.caption(day)
                drawn_day = day
            workflow = controller.read_workflow(session_id)
            title = workflow.get("geometry_description") or "New FEM analysis"
            active = st.session_state.get("session_id") == session_id
            row = st.container(key=f"history_row_{session_id}", gap=None)
            # No column split: splitting the row shortened every title to make
            # room.  Both live in the row and the trash is lifted out of the
            # flow onto its right end, so the item keeps the width it had and
            # the space it needs is taken from its own padding.
            if row.button(
                f"{'● ' if active else ''}{title}",
                key=f"history-{session_id}",
                width="stretch",
                help=f"{title}\n{session_id} · {workflow.get('stage', 'new')}",
            ):
                try:
                    st.session_state.last_interrupt = None
                    set_event(controller.resume(session_id))
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not resume: {type(exc).__name__}: {exc}")
            if row.button(
                "", icon=":material/delete:", key=f"delete_{session_id}",
                help="Delete this conversation",
            ):
                st.session_state.pending_delete = session_id
                st.rerun()
    # Asked in a real dialog rather than in the row.  The list flattens what it
    # holds -- gap zero, margins stripped, every button clamped to the height of
    # one line -- so a prompt built inside it kept coming out with its buttons
    # sitting on its text.  A dialog is not in the list and inherits none of it.
    pending = st.session_state.get("pending_delete")
    if pending:
        confirm_delete(pending)
    problem = st.session_state.pop("delete_error", None)
    if problem:
        st.error(problem)

# Take delivery of any finished step here, on the thread that owns session
# state, before the page is drawn from it.
collect_finished_job()
# Asserted every pass from the one fact that decides it: whether a step is still
# running.  A run that collects a job clears it here, so the controls cannot be
# left disabled by a handover that never arrives.
set_workflow_controls_busy(bool(st.session_state.get("pending_job")))
event: WorkflowEvent | None = st.session_state.get("event")
keep_tools_warm(event)
workspace = st.container(key="workspace")
left, right = workspace.columns((1.15, 0.85), gap="medium")

with left:
    session_markup = ""
    if event:
        session_markup = (
            f'<div class="session-meta">{token_bar_markup(event.workflow)}'
            f'<div class="session-id" title="Session {html.escape(event.session_id)}">'
            f"Session {html.escape(event.session_id)}</div></div>"
        )
    st.markdown(
        f'<div class="conversation-header"><div class="conversation-heading">'
        f"Conversation</div>{session_markup}</div>",
        unsafe_allow_html=True,
    )

conversation_panel = left.container(height=700, border=False, key="workflow_panel")
active_card_key = None  # the card waiting on an answer, once drawn
with conversation_panel:
    if not event:
        with st.chat_message("assistant"):
            st.markdown(SETUP_MESSAGE)
        with st.container(border=False, key="manager_widget"):
            render_setup_widget()
    else:
        # Every step stays on screen as the card it was, still editable.  Sending
        # an earlier one re-decides that step and rebuilds what follows.
        for card in conversation_cards(event):
            if card.active:
                active_card_key = f"history_step_{card.index}"
            with st.container(key=f"history_step_{card.index}"):
                if card.role == "assistant":
                    with st.chat_message("assistant"):
                        st.markdown(card.message)
                else:
                    # Transcripts recorded before the cards were kept summarised
                    # each submission as a line of text; the card now says it.
                    st.markdown(card.message)
            if card.interrupt is None:
                continue
            interrupt = card.interrupt
            kind = interrupt.get("kind")
            with st.container(border=False, key=f"manager_widget_{card.key}"):
                if kind == "setup":
                    # Each role beside the service that serves it: the two can
                    # differ, so one API field would not say which is which.
                    for role, provider_key, model_key in (
                        ("Text", "provider", "model"),
                        ("Vision", "vision_provider", "vision_model"),
                    ):
                        role_cols = st.columns(2)
                        role_cols[0].selectbox(
                            f"{role} API",
                            (
                                interrupt.get(provider_key)
                                or interrupt.get("provider", ""),
                            ),
                            disabled=True,
                            key=f"setup_{role.lower()}_api_{card.key}",
                        )
                        role_cols[1].selectbox(
                            f"{role} model",
                            (interrupt.get(model_key) or "default",),
                            disabled=True,
                            key=f"setup_{role.lower()}_model_{card.key}",
                        )
                    st.selectbox(
                        "Compute environment",
                        (interrupt.get("compute_env", ""),),
                        disabled=True,
                        key=f"setup_env_{card.key}",
                    )
                elif kind == "geometry_input":
                    # Launches a detached worker; model loading continues while the
                    # user describes the geometry and adjusts generation settings.
                    # Only the step being worked on needs it -- the card stays on
                    # screen afterwards and must not restart the worker on every
                    # rerun for the rest of the session.
                    asset_options = ("text", "image", "scribble", "surface")
                    sent_kind = interrupt.get("asset_kind") or "text"
                    asset_kind = st.selectbox(
                        "Input",
                        asset_options,
                        index=(
                            asset_options.index(sent_kind)
                            if sent_kind in asset_options
                            else 0
                        ),
                    )
                    # The CAD generators build parts from text alone and
                    # Hunyuan3D-2.1 from an image; TripoSG takes every input.
                    generator_status = {
                        name: interrupt.get(name) or {} for name in OPTIONAL_GENERATORS
                    }
                    generator = "triposg"
                    offered = INPUT_GENERATORS.get(asset_kind)
                    if offered:
                        sent_generator = (
                            (interrupt.get("generation_settings") or {}).get("generator")
                            or interrupt.get("generator")
                            or "triposg"
                        )
                        generator = st.selectbox(
                            "Generator",
                            offered,
                            index=offered.index(sent_generator)
                            if sent_generator in offered else 0,
                            format_func=GENERATOR_LABELS.get,
                            key=f"geometry_generator_{asset_kind}_{card.key}",
                        )
                        if (generator in OPTIONAL_GENERATORS
                                and not generator_status[generator].get("ready")):
                            st.caption(
                                f"{GENERATOR_LABELS[generator].split(' (')[0]} is unavailable: "
                                f"{generator_status[generator].get('summary', 'not installed')}"
                            )
                    if card.active:
                        try:
                            controller.preload_geometry_model(
                                event.session_id, generator=generator
                            )
                        except OSError as exc:
                            st.warning(
                                f"Could not preload {GENERATOR_LABELS[generator]}; "
                                f"generation will load it on demand: {exc}"
                            )
                    # An uploaded mesh needs no prompt to exist, but saying what
                    # it is still pays: it is what the vision stage is told the
                    # shape is meant to be, and that reading goes on to set the
                    # scale, choose the material, and name the segments the
                    # supports and loads attach to.
                    uploaded_surface = asset_kind == "surface"
                    description_key = (
                        f"geometry_description_{event.session_id}_{card.key}"
                    )
                    suggested_description = str(interrupt.get("description", ""))
                    suggestion_marker = (card.checkpoint_id, suggested_description)
                    applied_key = f"{description_key}_applied_suggestion"
                    if (
                        suggested_description
                        and st.session_state.get(applied_key) != suggestion_marker
                    ):
                        st.session_state[description_key] = suggested_description
                        st.session_state[applied_key] = suggestion_marker
                    description = st.text_area(
                        "What is it? (optional)" if uploaded_surface
                        else "Describe the geometry",
                        placeholder=(
                            "a single-seat sofa" if uploaded_surface
                            else "solid bracket with two mounting feet"
                        ),
                        key=description_key,
                        help=(
                            "Not used to build anything -- the mesh is already "
                            "made. It tells the later steps what they are looking "
                            "at, which is what lets them call a part a seat or a "
                            "leg rather than a slab."
                            if uploaded_surface else None
                        ),
                    )
                    revise_requested = False
                    # The CAD generators do not know object names, so the
                    # revision is a required step for them, not a polish.
                    unrevised_cad = needs_cad_revision(
                        generator, description, interrupt.get("cad_revision")
                    )
                    if not uploaded_surface:
                        if generator in CAD_GENERATORS:
                            cad_label = GENERATOR_LABELS[generator].split(" (")[0]
                            if unrevised_cad:
                                st.warning(
                                    f"**{cad_label} needs a revised prompt.** Click "
                                    "**Revise prompt** first: it rewrites your "
                                    f"description into the geometry {cad_label} can "
                                    "build. It does not know object names, so an "
                                    "unrevised prompt usually comes back a box. "
                                    "Review the rewrite, then Send.",
                                    icon=":material/priority_high:",
                                )
                            else:
                                st.caption(
                                    f"Revised for {cad_label}: review it, then Send."
                                )
                        revise_requested = st.button(
                            "Revise prompt", disabled=not description.strip(),
                            type="primary" if unrevised_cad else "secondary",
                        )
                        render_api_error(interrupt, "revise_geometry_prompt")
                        render_api_error(interrupt, "revise_cad_prompt")
                        render_api_error(interrupt, "revise_cadcoder_prompt")
                    elif not description.strip():
                        st.caption(
                            "Optional, but worth it: naming the object is what "
                            "lets the later steps read the shape as a seat or a "
                            "leg rather than a slab, and that reading sets the "
                            "scale, the material, and the boundary regions."
                        )
                    upload = None
                    canvas_result = None
                    condition = True
                    draw_requested = False
                    image_source = IMAGE_SOURCES[0]
                    drawn_image = interrupt.get("drawn_image")
                    if asset_kind == "image":
                        # SDXL draws the image from text, when it is installed:
                        # an input for either image generator, beside an upload.
                        sdxl = interrupt.get("sdxl") or {}
                        if sdxl.get("ready"):
                            image_source = st.radio(
                                "Image source", IMAGE_SOURCES,
                                index=1 if drawn_image else 0, horizontal=True,
                                key=f"geometry_image_source_{card.key}",
                            )
                        if image_source == IMAGE_SOURCES[0]:
                            upload = st.file_uploader(
                                "Input image",
                                type=["png", "jpg", "jpeg", "webp"],
                                key=f"geometry_image_{card.key}",
                            )
                        else:
                            # The description above is what SDXL draws: one box
                            # says what the object is, whatever the input.
                            st.caption(
                                "SDXL draws the description above as one whole object "
                                "on a plain background, and the drawing is cut out from "
                                "its background. Ask for thick, joined parts: the shape "
                                "is built from what the picture shows."
                            )
                            # Bottom-aligned, so the button lines up with the seed's
                            # box rather than its label.
                            draw_cols = st.columns([1, 1], vertical_alignment="bottom")
                            image_seed = draw_cols[0].number_input(
                                "Image seed", min_value=0, max_value=2**32 - 1,
                                value=int(interrupt.get("image_seed") or 0), step=1,
                                help="Another seed draws another picture of the same words.",
                                key=f"geometry_image_seed_{card.key}",
                            )
                            draw_requested = draw_cols[1].button(
                                "Generate image", disabled=not description.strip(),
                                type="secondary" if drawn_image else "primary",
                                key=f"geometry_draw_{card.key}", width="stretch",
                            )
                            if drawn_image and Path(drawn_image).is_file():
                                st.image(
                                    drawn_image, width=320,
                                    caption=(f"SDXL, seed {interrupt.get('image_seed', 0)}: "
                                             f"{interrupt.get('drawn_image_prompt', '')}"),
                                )
                                if interrupt.get("drawn_image_dropped"):
                                    # SDXL reads 77 tokens; the drawing tool reports the rest.
                                    st.warning(
                                        "SDXL read only the first 77 tokens of this prompt and never "
                                        f"read: \"{interrupt['drawn_image_dropped']}\". Shorten the "
                                        "description (about 30 words) so its single-object style "
                                        "is read too, and draw again.")
                                st.caption(
                                    "Check that the whole object is there and nothing "
                                    "else; draw again with another seed or other words "
                                    "if not. Send builds the shape from this image."
                                )
                    elif asset_kind == "surface":
                        upload = st.file_uploader(
                            "Surface mesh",
                            type=["glb", "stl", "obj", "ply"],
                            key=f"geometry_surface_{card.key}",
                        )
                    elif asset_kind == "scribble":
                        condition = st.checkbox("Condition scribble on text", True)
                        if st_canvas is None:
                            st.error(
                                "Drawing support is unavailable; install "
                                "streamlit-drawable-canvas in mechfoundry-agents."
                            )
                        else:
                            canvas_result = st_canvas(
                                fill_color="rgba(255,255,255,0)",
                                stroke_width=5,
                                stroke_color="#000000",
                                background_color="#F0F2F6",
                                height=220,
                                width=320,
                                drawing_mode="freedraw",
                                return_image_data=True,
                                key=f"geometry_scribble_canvas_{card.key}",
                            )

                    settings = {
                        "faces": -1,
                        "init_octree_depth": 4,
                        "octree_depth": 7,
                        "seed": 42,
                        "inference_steps": 50 if asset_kind == "image" else 16,
                        "guidance_scale": 7.0,
                        "prompt_confidence": 1.0,
                        "scribble_confidence": 0.4,
                        "erosion_radius": 0,
                        # A card that was already sent redraws with what it sent.
                        **(interrupt.get("generation_settings") or {}),
                    }
                    settings["generator"] = generator
                    if generator in CAD_GENERATORS:
                        with st.expander(
                            "Advanced geometry settings",
                            key=f"advanced_geometry_{card.key}",
                        ):
                            cad_cols = st.columns(2)
                            settings["faces"] = cad_cols[0].select_slider(
                                "Target surface faces",
                                options=faces_slider_options(settings["faces"])[0],
                                value=faces_slider_options(settings["faces"])[1],
                                format_func=lambda value: (
                                    "Default" if value == -1 else f"{value:,}"
                                ),
                                help="Bounds edge length when the CAD tessellation is refined.",
                                key=f"cad_faces_{card.key}",
                            )
                            settings["seed"] = cad_cols[1].slider(
                                "Random seed", 0, 100, int(settings["seed"]), 1,
                                help="Used by the sampled retries after a failed greedy decode.",
                                key=f"cad_seed_{card.key}",
                            )
                    elif generator == "hunyuan3d":
                        with st.expander(
                            "Advanced geometry settings",
                            key=f"advanced_geometry_{card.key}",
                        ):
                            # A card last sent to another generator carries its
                            # step count; Hunyuan3D starts from its own 30.
                            sent_hunyuan3d = (
                                (interrupt.get("generation_settings") or {}).get("generator")
                                == "hunyuan3d"
                            )
                            shape_cols = st.columns(2)
                            settings["inference_steps"] = shape_cols[0].slider(
                                "Inference steps", 1, 100,
                                int(settings["inference_steps"]) if sent_hunyuan3d else 30,
                                help="Shape diffusion steps. 30 matched 50 on the tested "
                                     "images at about half the time.",
                                key=f"hunyuan3d_steps_{card.key}",
                            )
                            settings["hunyuan3d_octree_resolution"] = shape_cols[1].select_slider(
                                "Octree resolution", options=[256, 384],
                                value=int(settings.get("hunyuan3d_octree_resolution", 384)),
                                help="Marching-cubes grid of the shape stage; 384 keeps finer detail.",
                                key=f"hunyuan3d_octree_{card.key}",
                            )
                            settings["hunyuan3d_flashvdm"] = st.toggle(
                                "Fast shape decoding (FlashVDM)",
                                bool(settings.get("hunyuan3d_flashvdm", True)),
                                help="Upstream's faster volume decoding: about half the "
                                     "shape-stage time, with no visible change on the tested images.",
                                key=f"hunyuan3d_flashvdm_{card.key}",
                            )
                            settings["texture"] = st.toggle(
                                "Colour the surface", bool(settings.get("texture", True)),
                                help="Runs the paint stage and keeps the object's colour "
                                     "on the mesh. Off gives the shape only, faster.",
                                key=f"hunyuan3d_texture_{card.key}",
                            )
                            paint_cols = st.columns(2)
                            settings["hunyuan3d_texture_size"] = paint_cols[0].select_slider(
                                "Texture size", options=[1024, 2048, 4096],
                                value=int(settings.get("hunyuan3d_texture_size", 1024)),
                                format_func=lambda value: f"{value}²",
                                help="The surface keeps one colour per vertex, so a larger "
                                     "texture mostly costs time: 1024² paints in about half "
                                     "the time of 4096².",
                                disabled=not settings["texture"],
                                key=f"hunyuan3d_texture_size_{card.key}",
                            )
                            settings["hunyuan3d_upscale"] = paint_cols[1].toggle(
                                "Upscale views (RealESRGAN)",
                                bool(settings.get("hunyuan3d_upscale", False)),
                                help="Enlarges each painted view 4x before baking. Sharper "
                                     "texture, a few seconds slower.",
                                disabled=not settings["texture"],
                                key=f"hunyuan3d_upscale_{card.key}",
                            )
                            mesh_cols = st.columns(2)
                            settings["faces"] = mesh_cols[0].select_slider(
                                "Target surface faces",
                                options=faces_slider_options(settings["faces"])[0],
                                value=faces_slider_options(settings["faces"])[1],
                                format_func=lambda value: (
                                    "Default (40,000)" if value == -1 else f"{value:,}"
                                ),
                                key=f"hunyuan3d_faces_{card.key}",
                            )
                            settings["seed"] = mesh_cols[1].slider(
                                "Random seed", 0, 100, int(settings["seed"]), 1,
                                key=f"hunyuan3d_seed_{card.key}",
                            )
                    elif asset_kind != "surface":
                        with st.expander(
                            "Advanced geometry settings",
                            key=f"advanced_geometry_{card.key}",
                        ):
                            depth_cols = st.columns(2)
                            settings["init_octree_depth"] = depth_cols[0].slider(
                                "Initial octree depth",
                                1,
                                8,
                                int(settings["init_octree_depth"]),
                                1,
                                help=(
                                    "Starting grid used to extract the generated shape. "
                                    "Each depth adds a factor of two in resolution along "
                                    "each axis. A higher initial depth can capture thin "
                                    "parts or small features that a coarse starting grid "
                                    "might miss, but uses more time and memory. Later "
                                    "refinement may not recover features missed here."
                                ),
                            )
                            settings["octree_depth"] = depth_cols[1].slider(
                                "Final octree depth", 2, 9, int(settings["octree_depth"]), 1,
                                help=(
                                    "Finest grid used to extract the generated surface. "
                                    "Each additional depth halves the smallest cell size, "
                                    "allowing finer detail at a higher time and memory "
                                    "cost. Detail also depends on the initial grid "
                                    "detecting the feature. These settings generate the "
                                    "surface; the solver has its own octree depths."
                                ),
                            )
                            inference_cols = st.columns(2)
                            settings["inference_steps"] = inference_cols[0].slider(
                                "Inference steps",
                                1,
                                100,
                                int(settings["inference_steps"]),
                                key=f"geometry_steps_{asset_kind}_{card.key}",
                            )
                            if asset_kind == "image":
                                settings["guidance_scale"] = inference_cols[1].slider(
                                    "Image guidance scale",
                                    1.0,
                                    15.0,
                                    float(settings["guidance_scale"]),
                                    0.1,
                                )
                            elif asset_kind in ("text", "scribble"):
                                settings["prompt_confidence"] = inference_cols[1].slider(
                                    "Prompt confidence",
                                    0.0,
                                    1.0,
                                    float(settings["prompt_confidence"]),
                                    0.05,
                                )
                            if asset_kind == "scribble":
                                settings["scribble_confidence"] = st.slider(
                                    "Scribble confidence",
                                    0.0,
                                    1.0,
                                    float(settings["scribble_confidence"]),
                                    0.05,
                                )
                            mesh_cols = st.columns(2)
                            settings["faces"] = mesh_cols[0].select_slider(
                                "Target surface faces",
                                options=faces_slider_options(settings["faces"])[0],
                                value=faces_slider_options(settings["faces"])[1],
                                format_func=lambda value: (
                                    "Unlimited" if value == -1 else f"{value:,}"
                                ),
                            )
                            settings["seed"] = mesh_cols[1].slider(
                                "Random seed", 0, 100, int(settings["seed"]), 1
                            )
                            settings["erosion_radius"] = st.slider(
                                "Component filter erosion (voxels)", 0, 4,
                                int(settings.get("erosion_radius", 0)), 1,
                                help="TripoSG keeps one body of what it generates. With "
                                     "erosion, pieces joined only by something thinner "
                                     "than twice this many voxels are cut apart first, so "
                                     "thin members (slats, spindles, rails) and whatever "
                                     "hangs from them are dropped. 0 keeps every part "
                                     "joined to the body; floating fragments still go.",
                                key=f"geometry_erosion_{card.key}",
                            )
                    sent = st.button(
                        "Send", type="secondary" if unrevised_cad else "primary",
                        disabled=(generator in OPTIONAL_GENERATORS
                                  and not generator_status[generator].get("ready")),
                    )
                    if draw_requested:
                        card.resume_in_place(
                            {
                                "action": "draw_image",
                                "description": description,
                                "image_seed": int(image_seed),
                                "asset_kind": "image",
                                "generation_settings": settings,
                            },
                            "Drawing the image with SDXL…",
                        )
                    elif sent or revise_requested:
                        path = None
                        source_label = description
                        if asset_kind == "image" and image_source == IMAGE_SOURCES[1]:
                            if drawn_image and Path(drawn_image).is_file():
                                path = Path(drawn_image)
                                source_label = description or "SDXL image"
                        elif upload:
                            path = controller.save_upload(
                                event.session_id, upload.name, upload.getvalue()
                            )
                            source_label = (
                                description or f"Uploaded {html.escape(upload.name)}"
                            )
                        elif asset_kind == "scribble" and canvas_result is not None:
                            pixels = canvas_result.image_data
                            if (
                                pixels is not None
                                and np.asarray(pixels)[..., :3].min() < 64
                            ):
                                buffer = io.BytesIO()
                                Image.fromarray(
                                    np.asarray(pixels).astype(np.uint8), "RGBA"
                                ).save(buffer, format="PNG")
                                path = controller.save_upload(
                                    event.session_id, "scribble.png", buffer.getvalue()
                                )
                                source_label = description or "Drawn scribble"
                        if (
                            asset_kind in ("image", "surface", "scribble")
                            and path is None
                        ):
                            st.error(
                                "Generate an image first."
                                if image_source == IMAGE_SOURCES[1] and asset_kind == "image"
                                else f"Provide a {asset_kind} input before sending."
                            )
                        else:
                            answer = {
                                "description": description,
                                "asset_path": str(path) if path else None,
                                "asset_kind": (
                                    None if asset_kind == "text" else asset_kind
                                ),
                                "condition_on_text": condition,
                                "generation_settings": settings,
                                "revise_prompt": revise_requested,
                            }
                            if revise_requested:
                                card.resume_in_place(answer, "Revising prompt…")
                            else:
                                card.submit(answer)
                elif kind == "geometry_texture":
                    # Optional: paint a bare surface now, so the views every
                    # later step reads (scaling, boundary conditions, material)
                    # carry its colours; or skip straight on to scaling.
                    textured = interrupt.get("textured_surface")
                    if not textured and interrupt.get("surface_views") and Path(
                            interrupt["surface_views"]).is_file():
                        st.image(interrupt["surface_views"], caption="Generated surface (no colour)",
                                 width="stretch")
                    st.caption(
                        "This surface has no colour. Painting it now lets the geometry "
                        "reading, boundary-condition synthesis and VoMP see its materials; "
                        "the shape is unchanged. You can also skip this and paint it later "
                        "in the VoMP material card."
                    )
                    render_api_error(interrupt, "texture_geometry")
                    paint_texture, texture_answer = render_texture_controls(card.key, interrupt)
                    render_texture_previews(interrupt, "Painted surface")
                    choices = st.columns(2)
                    use_texture = choices[0].button(
                        "Continue with this texture", type="primary", disabled=not textured,
                        key=f"use-texture-{card.key}", width="stretch",
                    )
                    skip_texture = choices[1].button(
                        "Skip texturing", key=f"skip-texture-{card.key}", width="stretch",
                    )
                    if paint_texture:
                        card.resume_in_place(
                            {"action": "texture_surface",
                             **texture_answer_with_upload(event, texture_answer)},
                            "Painting the surface with Hunyuan3D-2.1…",
                        )
                    if use_texture:
                        card.submit({"action": "use_texture"})
                    if skip_texture:
                        card.submit({"action": "skip_texture"})
                elif kind == "geometry_physics":
                    axis_options = ("z", "y", "x")
                    unit_options = list(UNIT_SYSTEMS)
                    current_axis = interrupt.get("up_axis") or axis_options[0]
                    current_units = interrupt.get("unit_system") or unit_options[0]
                    current_extent = float(interrupt.get("physical_extent") or 1.0)
                    with st.form(f"geometry_physics_{card.key}"):
                        units = st.selectbox(
                            "Unit system",
                            unit_options,
                            index=(
                                unit_options.index(current_units)
                                if current_units in unit_options
                                else 0
                            ),
                        )
                        unit_detail = UNIT_SYSTEM_DETAILS[units]
                        st.caption(
                            f"Length/displacement: {unit_detail['length']} · "
                            f"Force: {unit_detail['force']} · Mass: {unit_detail['mass']} · "
                            f"Time: {unit_detail['time']} · Stress/modulus/traction: "
                            f"{unit_detail['stress_modulus_traction']} · Body force: "
                            f"{unit_detail['body_force']} · Acceleration: {unit_detail['acceleration']} · "
                            f"Density: {unit_detail['density']} · "
                            f"Energy: {unit_detail['energy']}"
                        )
                        scale_description = st.text_area(
                            "Describe the real-world size (optional)",
                            value=interrupt.get("scale_description", ""),
                            placeholder="For example: an armchair about 90 cm tall.",
                        )
                        infer_scale = st.form_submit_button(
                            "Infer axis and extent with API",
                            help=(
                                "Uses the geometry prompt, this text, and a low-resolution "
                                "view of the input. Both values stay editable below."
                            ),
                        )
                        render_api_error(interrupt, "infer_scale")
                        scale_cols = st.columns(2)
                        axis = scale_cols[0].selectbox(
                            "Reference axis",
                            axis_options,
                            index=(
                                axis_options.index(current_axis)
                                if current_axis in axis_options
                                else 0
                            ),
                        )
                        extent = scale_cols[1].number_input(
                            "Physical extent",
                            min_value=1e-12,
                            value=current_extent,
                            format="%.6g",
                        )
                        measured = interrupt.get("measured_extents") or {}
                        if measured:
                            # The extent above is applied whatever it says, so
                            # show what the mesh actually measures.  The default
                            # is the reference axis's own size, which leaves an
                            # already-correct mesh untouched; switching axis
                            # means typing that axis's figure.
                            sizes = " · ".join(
                                f"{axis} {value:.6g}" for axis, value in measured.items()
                            )
                            st.caption(
                                f"Uploaded mesh measures {sizes} in its own units. "
                                "Scaling maps the reference axis to the extent above."
                            )
                        sent = st.form_submit_button("Apply scaling", type="primary")
                    if infer_scale:
                        card.resume_in_place(
                            {
                                "action": "infer_scale",
                                "scale_description": scale_description,
                                "unit_system": units,
                            },
                            "Inferring reference axis and physical extent…",
                        )
                    if sent:
                        card.submit(
                            {
                                "action": "apply_scaling",
                                "up_axis": axis,
                                "physical_extent": extent,
                                "unit_system": units,
                                "scale_description": scale_description,
                            }
                        )
                elif kind == "bc_input":
                    defaults = interrupt["segmentation"]
                    with st.container(border=True, key=f"bc_input_widget_{card.key}"):
                        if interrupt.get("preview_error"):
                            st.error(interrupt["preview_error"])
                        partfield = interrupt.get("partfield") or {}
                        methods, current_method = segmentation_method_choices(
                            bool(partfield.get("ready")), defaults.get("method"),
                            bool(interrupt.get("cad_faces")),
                        )
                        method = st.selectbox(
                            "Segmentation method", methods,
                            index=(methods.index(current_method)
                                   if current_method in methods else 0),
                            format_func=SEGMENTATION_LABELS.get,
                            key=f"segmentation_method_{card.key}",
                        )
                        if not partfield.get("ready"):
                            st.caption(
                                "PartField is optional and unavailable: "
                                + str(partfield.get("summary", "not installed"))
                            )
                        elif method == "partfield" and card.active:
                            with contextlib.suppress(OSError):
                                controller.preload_partfield_worker(
                                    interrupt.get("compute_env", DEFAULT_COMPUTE_ENV)
                                )
                        with st.expander(
                            "Advanced segmentation settings",
                            key=f"advanced_segmentation_{card.key}",
                        ):
                            segmentation_cols = st.columns(2)
                            num_clusters = int(defaults["num_clusters"])
                            if method == "cad_faces":
                                st.caption(
                                    "One segment per face of the CAD model; "
                                    "there is nothing to tune."
                                )
                            else:
                                num_clusters = segmentation_cols[0].slider(
                                    "Clusters", 2, 20, num_clusters, 1
                                )
                            num_rays = int(defaults["num_rays"])
                            smoothing = float(defaults["smoothing"])
                            cone_angle = float(defaults["cone_angle_deg"])
                            if method == "shape_diameter":
                                num_rays = segmentation_cols[1].slider(
                                    "Rays per face", 1, 100, num_rays, 1
                                )
                                smoothing = segmentation_cols[0].slider(
                                    "Smoothing", 0.0, 1.0, smoothing, 0.05
                                )
                                cone_angle = segmentation_cols[1].slider(
                                    "Cone angle (degrees)", 1.0, 180.0,
                                    cone_angle, 1.0,
                                )
                        segmentation = {
                            "method": method,
                            "num_clusters": num_clusters,
                            "num_rays": num_rays,
                            "smoothing": smoothing,
                            "cone_angle_deg": cone_angle,
                        }
                        preview_segments = st.button(
                            "Preview segmentation",
                            width="stretch",
                            help=(
                                "Segments locally, then asks the vision model what "
                                "each segment is."
                            ),
                        )
                        # The box is for the wait on the vision model, not for
                        # the segmentation that precedes it: it appears once the
                        # montage has been rendered -- the last thing that
                        # happens before the call -- and is replaced by the
                        # answer when it arrives.  Any other pending job on this
                        # card is somebody else's wait and shows nothing here.
                        if card.active and describing_segments():
                            st.info("Describing segments…", icon=":material/hourglass:")
                        elif interrupt.get("segment_description"):
                            st.success(
                                interrupt["segment_description"],
                                icon=":material/image_search:",
                            )
                        render_api_error(interrupt, "describe_segments")
                        description = st.text_area(
                            "Supports and loads",
                            value=interrupt.get("description", ""),
                            placeholder="Fix the bottom face; load the top face downward.",
                        )
                        revise_bc_prompt = st.button(
                            "Revise prompt with API",
                            width="stretch",
                            disabled=not description.strip(),
                            help=(
                                "Rewrites this description into a compact supports-and-loads "
                                "prompt, keeping every region, direction, magnitude, and unit. "
                                "You can edit the result before previewing."
                            ),
                        )
                        render_api_error(interrupt, "revise_prompt")
                        preview_regions = st.button(
                            "Preview BC regions",
                            width="stretch",
                            help=(
                                "Automatically runs segmentation with the selected parameters when needed, "
                                "then uses the API to convert the description into selectors."
                            ),
                        )
                        render_api_error(interrupt, "preview_regions")
                        settings_dirty = segmentation != defaults
                        description_dirty = (
                            description.strip()
                            != interrupt.get("description", "").strip()
                        )
                        approve_regions = st.button(
                            "Approve regions",
                            type="primary",
                            width="stretch",
                            disabled=(
                                not interrupt.get("has_regions", False)
                                or settings_dirty
                                or description_dirty
                            ),
                            help="Preview again after changing segmentation or text.",
                        )
                    if preview_segments:
                        if method == "partfield":
                            with contextlib.suppress(OSError):
                                controller.preload_partfield_worker(
                                    interrupt.get("compute_env", DEFAULT_COMPUTE_ENV)
                                )
                        card.resume_in_place(
                            {
                                "action": "preview_segmentation",
                                "segmentation": segmentation,
                                "description": description,
                            },
                            "Updating segmentation preview…",
                        )
                    if revise_bc_prompt:
                        card.resume_in_place(
                            {
                                "action": "revise_prompt",
                                "segmentation": segmentation,
                                "description": description,
                            },
                            "Revising supports-and-loads prompt…",
                        )
                    if preview_regions:
                        if not description.strip():
                            st.error(
                                "Describe at least one support or applied-load region."
                            )
                        else:
                            if method == "partfield":
                                with contextlib.suppress(OSError):
                                    controller.preload_partfield_worker(
                                        interrupt.get(
                                            "compute_env", DEFAULT_COMPUTE_ENV
                                        )
                                    )
                            card.resume_in_place(
                                {
                                    "action": "preview_regions",
                                    "segmentation": segmentation,
                                    "description": description,
                                },
                                "Generating boundary-condition preview…",
                            )
                    if approve_regions:
                        card.submit({"action": "approve_regions"})
                elif kind == "bc_values":
                    bc_units = interrupt["units"]
                    axis_names = {0: "X", 1: "Y", 2: "Z"}
                    value_signature = hashlib.sha1(
                        json.dumps(
                            {
                                "dirichlet": interrupt.get("dirichlet", []),
                                "neumann": interrupt.get("neumann", []),
                                "body_force": interrupt.get("body_force", []),
                                "acceleration": interrupt.get("acceleration", []),
                            },
                            sort_keys=True,
                        ).encode()
                    ).hexdigest()[:10]
                    with st.form(f"bc_values_{card.key}"):
                        st.caption(f"Unit system: {interrupt['unit_system']}")
                        value_description = st.text_area(
                            "Describe boundary-condition values (optional)",
                            value=interrupt.get("value_description", ""),
                            placeholder=(
                                "For example: fully fix the support; apply 2 MPa downward "
                                "traction and gravity downward."
                            ),
                        )
                        infer_values = st.form_submit_button(
                            "Infer values with API",
                            help=(
                                "Uses the approved regions, previous BC description, unit system, "
                                "and this text. You can review all inferred values before continuing."
                            ),
                        )
                        render_api_error(interrupt, "infer_values")
                        dirichlet_values = {}
                        for index, item in enumerate(interrupt.get("dirichlet", [])):
                            st.markdown(f"**Dirichlet BC region — {item['name']}**")
                            components = st.multiselect(
                                "Constrained components",
                                (0, 1, 2),
                                default=item.get("components", [0, 1, 2]),
                                format_func=lambda component: axis_names[component],
                                key=f"dirichlet-components-{card.key}-{value_signature}-{index}",
                            )
                            value = st.number_input(
                                f"Prescribed displacement [{bc_units['displacement']}]",
                                value=float(item.get("value", 0.0)),
                                format="%.6g",
                                key=f"dirichlet-value-{card.key}-{value_signature}-{index}",
                                help="One shared displacement value is applied to the selected components.",
                            )
                            dirichlet_values[item["name"]] = {
                                "components": components,
                                "value": value,
                            }
                        neumann_values = {}
                        for index, item in enumerate(interrupt.get("neumann", [])):
                            st.markdown(f"**Neumann BC region — {item['name']}**")
                            traction_cols = st.columns(3)
                            defaults = item.get("traction", [0.0, 0.0, 0.0])
                            neumann_values[item["name"]] = [
                                traction_cols[axis].number_input(
                                    f"Traction {axis_names[axis]} [{bc_units['stress_modulus_traction']}]",
                                    value=float(defaults[axis]),
                                    format="%.6g",
                                    key=f"neumann-{card.key}-{value_signature}-{index}-{axis}",
                                )
                                for axis in range(3)
                            ]
                            # Traction is per unit area; the load people reason about
                            # is the total force it applies over the region.
                            render_total_force(
                                neumann_values[item["name"]],
                                item.get("area"),
                                bc_units,
                                interrupt["unit_system"],
                                item.get("loaded_area"),
                            )
                        st.markdown("**Body force**")
                        body_cols = st.columns(3)
                        body_defaults = interrupt.get("body_force", [0.0, 0.0, 0.0])
                        body_force = [
                            body_cols[axis].number_input(
                                f"Body force {axis_names[axis]} [{bc_units['body_force']}]",
                                value=float(body_defaults[axis]),
                                format="%.6g",
                                key=f"body-force-{card.key}-{value_signature}-{axis}",
                            )
                            for axis in range(3)
                        ]
                        st.markdown("**Acceleration (gravity / self-weight)**")
                        acceleration_cols = st.columns(3)
                        acceleration_defaults = interrupt.get("acceleration", [0.0, 0.0, 0.0])
                        acceleration = [
                            acceleration_cols[axis].number_input(
                                f"Acceleration {axis_names[axis]} [{bc_units['acceleration']}]",
                                value=float(acceleration_defaults[axis]),
                                format="%.6g",
                                key=f"acceleration-{card.key}-{value_signature}-{axis}",
                                help="Density multiplies this acceleration to produce force per volume.",
                            )
                            for axis in range(3)
                        ]
                        sent = st.form_submit_button("Set BC values", type="primary")
                    if infer_values:
                        if not value_description.strip():
                            st.error("Describe the values you want the API to infer.")
                        else:
                            card.resume_in_place(
                                {
                                    "action": "infer_values",
                                    "value_description": value_description,
                                },
                                "Inferring boundary-condition values…",
                            )
                    if sent:
                        if any(
                            not value["components"]
                            for value in dirichlet_values.values()
                        ):
                            st.error(
                                "Each Dirichlet region must constrain at least one component."
                            )
                        else:
                            card.submit(
                                {
                                    "action": "apply_values",
                                    "dirichlet_values": dirichlet_values,
                                    "neumann_values": neumann_values,
                                    "body_force": body_force,
                                    "acceleration": acceleration,
                                    "value_description": value_description,
                                }
                            )
                elif kind == "material_input":
                    material_units = interrupt["units"]
                    # Re-key the inputs when inference changes them, or Streamlit
                    # keeps showing the previous entry.
                    material_signature = hashlib.sha1(
                        json.dumps(
                            {
                                "e": interrupt.get("youngs_modulus"),
                                "yield_strength": interrupt.get("yield_strength"),
                                "nu": interrupt.get("poisson_ratio"),
                                "rho": interrupt.get("density"),
                                "mode": interrupt.get("material_mode", "uniform"),
                            },
                            sort_keys=True,
                        ).encode()
                    ).hexdigest()[:10]
                    # Outside the form, like segmentation method and solver
                    # mode: changing assignment redraws the card immediately
                    # instead of waiting for an unrelated submit button.
                    st.caption(f"Unit system: {interrupt['unit_system']}")
                    material_modes = ("uniform", "parts", "vomp")
                    material_mode = st.selectbox(
                        "Material assignment",
                        material_modes,
                        index=material_modes.index(interrupt.get("material_mode", "uniform")),
                        format_func=lambda value: {
                            "uniform": "Uniform material",
                            "parts": "Material per part (vision)",
                            "vomp": "VoMP spatial material field",
                        }[value],
                        key=f"material-mode-{card.key}",
                    )
                    infer_material = infer_vomp = infer_parts = apply_parts = False
                    sent = paint_texture = False
                    edited_parts: list[dict[str, Any]] = []
                    texture_answer: dict[str, Any] = {}
                    material_description = interrupt.get("material_description", "")
                    young = poisson = density = None
                    if material_mode == "uniform":
                        with st.form(f"material_uniform_{card.key}"):
                            material_description = st.text_area(
                                "Describe the material (optional)",
                                value=material_description,
                                placeholder="For example: oak frame, or leave blank to infer "
                                "from the geometry.",
                            )
                            infer_material = st.form_submit_button(
                                "Infer material with API",
                                help=(
                                    "Uses the geometry prompt, this text, and a low-resolution "
                                    "view of the object, against a reference table of common "
                                    "materials. Values remain editable below."
                                ),
                            )
                            render_api_error(interrupt, "infer_material")
                            young = st.number_input(
                                    f"Young's modulus [{material_units['stress_modulus_traction']}]",
                                    min_value=1e-12,
                                    value=float(interrupt.get("youngs_modulus") or 1e6),
                                    format="%.6g",
                                    key=f"young-{card.key}-{material_signature}",
                            )
                            poisson = st.number_input(
                                    "Poisson ratio", min_value=-0.99, max_value=0.49,
                                    # ν = 0 is a valid answer (cork), not a missing one.
                                    value=float(0.3 if interrupt.get("poisson_ratio") is None
                                                else interrupt["poisson_ratio"]),
                                    key=f"poisson-{card.key}-{material_signature}",
                            )
                            density = st.number_input(
                                    f"Density [{material_units['density']}] (optional without acceleration)",
                                    min_value=0.0,
                                    value=float(interrupt.get("density") or 0.0),
                                    format="%.6g",
                                    key=f"density-{card.key}-{material_signature}",
                            )
                            if interrupt.get("material_inferred"):
                                st.caption(
                                    "Values above were inferred from the description and "
                                    "geometry. They can coincide with the defaults; edit "
                                    "them if wrong."
                                )
                            sent = st.form_submit_button("Set parameters", type="primary")
                    elif material_mode == "parts":
                        with st.container(border=True, key=f"material_parts_{card.key}"):
                            # The parts' materials are read off their colour, so
                            # a bare surface is painted first, as for VoMP.
                            coloured = interrupt.get("surface_has_colour")
                            textured = interrupt.get("textured_surface")
                            needs_texture = coloured is False and not textured
                            if needs_texture:
                                st.warning(
                                    "Step 1: this surface has no colour, and each part's "
                                    "material is read off its colour. Paint it first."
                                )
                            with st.expander(
                                "Generate texture (required)" if needs_texture
                                else "Re-generate texture (optional)",
                                expanded=needs_texture,
                            ):
                                paint_texture, texture_answer = render_texture_controls(
                                    card.key, interrupt)
                            render_texture_previews(interrupt, "Painted surface (read for materials)")
                            material_description = st.text_area(
                                "Describe the materials (optional)",
                                value=material_description,
                                placeholder="For example: chrome base, black leather seat.",
                                key=f"parts-description-{card.key}",
                            )
                            parts = interrupt.get("material_parts") or []
                            infer_parts = st.button(
                                "Assign materials again" if parts else "Assign materials by part",
                                type="secondary" if parts else "primary",
                                disabled=needs_texture,
                                help=(
                                    "Takes the parts the boundary conditions were picked on "
                                    "(PartField parts or CAD faces; otherwise splits the "
                                    "surface with PartField), then a vision "
                                    "model gives each part E, Poisson ratio and density from "
                                    "its colour, its role, and the reference table used for "
                                    "uniform materials."
                                ),
                                key=f"infer-parts-{card.key}", width="stretch",
                            )
                            render_api_error(interrupt, "infer_parts")
                            artifact = interrupt.get("parts_artifact")
                            if parts:
                                montage = interrupt.get("parts_montage")
                                if montage and Path(montage).exists():
                                    st.image(montage, caption="Colour (top) and parts (bottom), same views")
                                if interrupt.get("material_inferred"):
                                    st.caption("Values below were inferred per part; edit any that are wrong.")
                                unit = material_units["stress_modulus_traction"]
                                parts_signature = hashlib.sha1(
                                    json.dumps(parts, sort_keys=True).encode()).hexdigest()[:10]
                                with st.form(f"material_parts_{card.key}_{parts_signature}"):
                                    header = st.columns(4)
                                    for column, title in zip(header, (
                                        "Part", f"E [{unit}]", "Poisson ratio",
                                        f"Density [{material_units['density']}]",
                                    )):
                                        column.caption(title)
                                    for part in parts:
                                        row = st.columns(4)
                                        row[0].markdown(f"part {part['id']}")
                                        edited_parts.append({
                                            "id": part["id"],
                                            "youngs_modulus": row[1].number_input(
                                                "E", min_value=1e-12, format="%.4g",
                                                value=float(part["youngs_modulus"]),
                                                label_visibility="collapsed",
                                                key=f"part-e-{card.key}-{part['id']}-{parts_signature}"),
                                            "poisson_ratio": row[2].number_input(
                                                "nu", min_value=-0.99, max_value=0.49,
                                                value=float(part["poisson_ratio"]),
                                                label_visibility="collapsed",
                                                key=f"part-nu-{card.key}-{part['id']}-{parts_signature}"),
                                            "density": row[3].number_input(
                                                "rho", min_value=1e-30, format="%.4g",
                                                value=float(part["density"]),
                                                label_visibility="collapsed",
                                                key=f"part-rho-{card.key}-{part['id']}-{parts_signature}"),
                                        })
                                    apply_parts = st.form_submit_button("Apply edited values")
                                if artifact:
                                    surface_path = event.workflow.get("surface_path")
                                    from src.vomp_wrapper.visualize import surface_visualizations_current
                                    if surface_path and not surface_visualizations_current(artifact):
                                        try:
                                            from src.vomp_wrapper.artifact import load_artifact
                                            from src.vomp_wrapper.visualize import write_surface_visualizations
                                            write_surface_visualizations(
                                                load_artifact(artifact), surface_path
                                            )
                                        except Exception as exc:
                                            st.warning(
                                                "Could not prepare material-field previews: "
                                                f"{type(exc).__name__}: {exc}"
                                            )
                            else:
                                st.info(
                                    "Each part of the surface gets its own E, Poisson ratio and "
                                    "density; its volume carries them into the solve."
                                )
                            sent = st.button(
                                "Approve per-part material",
                                type="primary", disabled=not bool(artifact),
                                key=f"approve-parts-{card.key}", width="stretch",
                            )
                    else:
                        if card.active:
                            with contextlib.suppress(OSError):
                                controller.preload_vomp_worker(
                                    event.workflow.get("compute_env", DEFAULT_COMPUTE_ENV)
                                )
                        with st.container(border=True, key=f"material_vomp_{card.key}"):
                            young = poisson = density = None
                            # VoMP reads colour as well as shape: a bare surface
                            # can be painted first (Hunyuan3D-2.1's paint stage,
                            # from a description or a reference image), and a
                            # painted one repainted.  The geometry is unchanged.
                            coloured = interrupt.get("surface_has_colour")
                            textured = interrupt.get("textured_surface")
                            # Required on a bare surface, optional once it has colour.
                            needs_texture = coloured is False and not textured
                            if needs_texture:
                                st.warning(
                                    "Step 1: this surface has no colour, and VoMP reads colour "
                                    "as well as shape. Paint it before running VoMP."
                                )
                            with st.expander(
                                "Generate texture (required)" if needs_texture
                                else "Re-generate texture (optional)",
                                expanded=needs_texture,
                            ):
                                paint_texture, texture_answer = render_texture_controls(
                                    card.key, interrupt)
                            render_texture_previews(interrupt, "Painted surface (VoMP reads this)")
                            artifact = interrupt.get("material_artifact")
                            if artifact:
                                # Backfill previews for a cached field produced
                                # by an older wrapper.  This is a vectorized
                                # surface lookup only; it never invokes VoMP.
                                surface_path = event.workflow.get("surface_path")
                                from src.vomp_wrapper.visualize import surface_visualizations_current
                                if surface_path and not surface_visualizations_current(artifact):
                                    try:
                                        from src.vomp_wrapper.artifact import load_artifact
                                        from src.vomp_wrapper.visualize import write_surface_visualizations
                                        write_surface_visualizations(
                                            load_artifact(artifact), surface_path
                                        )
                                    except Exception as exc:
                                        st.warning(
                                            "Could not prepare material-field previews: "
                                            f"{type(exc).__name__}: {exc}"
                                        )
                                st.success("VoMP field is ready for review and approval.")
                                st.code(artifact)
                                summary = interrupt.get("material_summary") or {}
                                ranges = summary.get("ranges") or {}
                                if ranges:
                                    unit = material_units["stress_modulus_traction"]
                                    density_unit = material_units["density"]
                                    st.caption(
                                        f"E: {ranges['youngs_modulus'][0]:.4g}–"
                                        f"{ranges['youngs_modulus'][1]:.4g} {unit} · "
                                        f"ν: {ranges['poisson_ratio'][0]:.4g}–"
                                        f"{ranges['poisson_ratio'][1]:.4g} · "
                                        f"ρ: {ranges['density'][0]:.4g}–"
                                        f"{ranges['density'][1]:.4g} {density_unit}"
                                    )
                                if summary.get("source_voxels") is not None:
                                    st.caption(
                                        f"{summary['source_voxels']:,} source voxels; "
                                        f"{summary.get('occupied_voxels', 0):,} occupied grid cells "
                                        "before nearest fill."
                                    )
                                infer_vomp = st.button(
                                    "Run VoMP again", key=f"rerun-vomp-{card.key}",
                                    disabled=needs_texture, width="stretch",
                                )
                            else:
                                st.info(
                                    "VoMP predicts E, Poisson ratio, and density on a "
                                    "64³ volumetric grid and projects each field onto the surface."
                                )
                                infer_vomp = st.button(
                                    "Run VoMP inference", type="primary",
                                    disabled=needs_texture,
                                    help=("Paint the surface above first." if needs_texture else None),
                                    key=f"run-vomp-{card.key}", width="stretch",
                                )
                            sent = st.button(
                                "Approve spatial material",
                                type="primary", disabled=not bool(artifact),
                                key=f"approve-vomp-{card.key}", width="stretch",
                            )
                    if infer_material:
                        card.resume_in_place(
                            {
                                "action": "infer_material",
                                "material_description": material_description,
                            },
                            "Inferring material parameters…",
                        )
                    if paint_texture:
                        card.resume_in_place(
                            {"action": "texture_surface",
                             "material_description": material_description,
                             **texture_answer_with_upload(event, texture_answer)},
                            "Painting the surface with Hunyuan3D-2.1…",
                        )
                    if infer_parts:
                        card.resume_in_place(
                            {"action": "infer_parts", "material_description": material_description},
                            "Assigning a material to each part…",
                        )
                    if apply_parts:
                        card.resume_in_place(
                            {"action": "apply_parts", "parts": edited_parts,
                             "material_description": material_description},
                            "Rebuilding the per-part material field…",
                        )
                    if infer_vomp:
                        with contextlib.suppress(OSError):
                            controller.preload_vomp_worker(
                                event.workflow.get("compute_env", DEFAULT_COMPUTE_ENV)
                            )
                        card.resume_in_place(
                            {"action": "infer_vomp", "material_description": material_description},
                            "Inferring spatial material properties with VoMP…",
                        )
                    if sent:
                        answer = {
                            "action": "apply_material", "material_mode": material_mode,
                            "material_description": material_description,
                        }
                        if material_mode == "uniform":
                            answer.update({"youngs_modulus": young, "poisson_ratio": poisson,
                                           "density": density or None, "yield_strength": None})
                        card.submit(answer)
                elif kind == "shape_opt_settings":
                    defaults = interrupt["defaults"]
                    strain = interrupt.get("strain_estimate")
                    run_mode = st.selectbox(
                        "Run mode",
                        ("forward", "shape_optimization"),
                        index=0 if defaults["run_mode"] == "forward" else 1,
                        help="Forward runs one FEA solve. Shape optimization repeatedly remeshes and solves.",
                    )
                    # Outside the form, like the run mode, because it decides which
                    # inputs the form shows (a form's widgets only update on submit).
                    gd_lr_auto = bool(defaults.get("gd_lr_auto", False))
                    if run_mode == "shape_optimization":
                        gd_lr_auto = st.checkbox(
                            "Auto learning rate (scale to ε)",
                            gd_lr_auto,
                            key=f"gd_lr_auto_{card.key}",
                            help=(
                                "Off by default. At every update the learning rate is "
                                "set so the largest SDF move is a fraction of the "
                                "diffuse width ε (fraction · ε / max|dJ/ds|); the GD "
                                "learning rate is not used. Steps do not shrink near "
                                "an optimum, so the run ends at its maximum iterations."
                            ),
                        )
                    with st.form(f"shape_opt_{card.key}_{run_mode}"):
                        with st.expander(
                            "Advanced solve settings",
                            key=f"advanced_solve_{card.key}_{run_mode}",
                        ):
                            octree_cols = st.columns(2)
                            remesh_init_depth = octree_cols[0].slider(
                                "Initial octree depth",
                                2,
                                9,
                                int(defaults["remesh_init_octree_depth"]),
                                1,
                                help=(
                                    "Starting grid for the solver octree. The octree is "
                                    "always refined to the final depth around surface-mesh "
                                    "nodes, so the initial depth does not limit preserved "
                                    "surface detail. It controls the starting grid's cost."
                                ),
                            )
                            remesh_depth = octree_cols[1].slider(
                                "Final octree depth",
                                3,
                                10,
                                int(defaults["remesh_octree_depth"]),
                                1,
                                help=(
                                    "Finest depth of the solver octree around surface-mesh "
                                    "nodes. This depth determines how much detail from the "
                                    "input surface the solver can preserve. Each additional "
                                    "depth halves the smallest cell size and increases "
                                    "compute and memory costs. It cannot restore detail "
                                    "missing from the generated surface."
                                ),
                            )
                            # No quadrature degree here: the solver integrates
                            # with 2x2x2 Gauss points and adaptive subcells, and
                            # never read the slider that used to sit here.
                            discretization_cols = st.columns(2)
                            epsilon_factor = discretization_cols[0].slider(
                                "Phase-field width / element size",
                                0.1,
                                4.0,
                                float(defaults["phase_field_epsilon_factor"]),
                                0.1,
                            )
                            st.markdown("**Nonlinear solver**")
                            solver_cols = st.columns(2)
                            relative_tol = solver_cols[0].number_input(
                                "Relative tolerance",
                                min_value=1e-12,
                                value=float(defaults["relative_tol"]),
                                format="%.3e",
                            )
                            absolute_tol = solver_cols[1].number_input(
                                "Absolute tolerance",
                                min_value=1e-14,
                                value=float(defaults["absolute_tol"]),
                                format="%.3e",
                            )
                            max_newton = solver_cols[0].number_input(
                                "Maximum Newton iterations",
                                min_value=1,
                                max_value=500,
                                value=int(defaults["max_newton_iters"]),
                            )
                            load_steps = solver_cols[1].number_input(
                                "Load steps",
                                min_value=1,
                                max_value=100,
                                value=int(defaults["load_steps"]),
                                help=(
                                    "Ramps the load up over this many solves."
                                    if run_mode == "forward" else
                                    "Ramps the load up over the first shape's solve; "
                                    "every later iteration starts from the previous "
                                    "one's displacement at full load."
                                ),
                            )
                            solver_flags = st.columns(2)
                            line_search = solver_flags[0].checkbox(
                                "Newton line search", bool(defaults["line_search_enabled"])
                            )
                            mixed_precision = solver_flags[1].checkbox(
                                "Mixed precision",
                                bool(defaults["use_mixed_precision"]),
                                help=(
                                    "Solve in float64. Needed when the strain is too "
                                    "small for float32 to resolve."
                                ),
                            )
                            unit_diagonal = solver_flags[0].checkbox(
                                "Rescale system to unit diagonal",
                                bool(defaults.get("scale_system_to_unit_diagonal", False)),
                                help=(
                                    "Divides out the row and column scale of the assembled "
                                    "system before the solve, so a boundary penalty far "
                                    "larger than the elastic terms stops dominating the "
                                    "conditioning. Exact, and cheap, but it cannot help "
                                    "when the strain itself is below float32 resolution."
                                ),
                            )
                            stored_solver = defaults.get("linear_solver", "jaxamg")
                            linear_solver = solver_flags[1].selectbox(
                                "Linear solver",
                                LINEAR_SOLVERS,
                                index=LINEAR_SOLVERS.index(
                                    stored_solver if stored_solver in LINEAR_SOLVERS
                                    else "jaxamg"
                                ),
                                format_func=LINEAR_SOLVER_LABELS.get,
                                help=(
                                    "The AMG solve inside each Newton step. JAX-AMG is "
                                    "the default. pyamgx runs the same AMG but keeps its "
                                    "last iterate when AMGX stops short, so Newton's "
                                    "residual decides; try it when JAX-AMG reports a "
                                    "stalled or ill-conditioned linear solve."
                                ),
                            )
                        gd_lr = defaults["gd_lr"]
                        gd_lr_auto_fraction = defaults.get("gd_lr_auto_fraction", 0.5)
                        max_optimize = defaults["max_optimize_iters"]
                        save_every = defaults["output_per_optimize_step"]
                        opt_tol = defaults["optimize_relative_tol"]
                        target_volume = defaults["target_volume_fraction"]
                        volume_as_is = defaults.get("volume_as_is", False)
                        volume_weight = defaults["shape_opt_w_vol"]
                        bc_fix_method = defaults["sdf_bc_fix_method"]
                        reinit = defaults["reinit_sdf"]
                        reinit_every = defaults["reinit_every"]
                        if run_mode == "shape_optimization":
                            st.markdown("**Shape optimization · example preset**")
                            opt_cols = st.columns(2)
                            if gd_lr_auto:
                                gd_lr_auto_fraction = opt_cols[0].number_input(
                                    "Step size (fraction of ε)",
                                    min_value=0.01,
                                    # ShapeOptSettings' bound
                                    max_value=2.0,
                                    value=float(defaults.get("gd_lr_auto_fraction", 0.5)),
                                    help=(
                                        "Every update's largest SDF move, in units of "
                                        "ε; the learning rate is sized to it again at "
                                        "each iteration."
                                    ),
                                )
                            else:
                                gd_lr = opt_cols[0].number_input(
                                    "GD learning rate",
                                    min_value=1e-12,
                                    value=float(defaults["gd_lr"]),
                                    format="%.6g",
                                )
                            max_optimize = opt_cols[0].number_input(
                                "Maximum optimization iterations",
                                min_value=1,
                                max_value=100000,
                                value=int(defaults["max_optimize_iters"]),
                            )
                            save_every = opt_cols[1].number_input(
                                "Save every N iterations (0 disables)",
                                min_value=0,
                                value=int(defaults["output_per_optimize_step"]),
                            )
                            opt_tol = opt_cols[0].number_input(
                                "Optimization relative tolerance",
                                min_value=1e-12,
                                value=float(defaults["optimize_relative_tol"]),
                                format="%.3e",
                            )
                            target_volume = opt_cols[1].number_input(
                                "Target volume fraction",
                                min_value=0.001,
                                # ShapeOptSettings' bound: above 1 the target is a
                                # cap the volume may grow up to
                                max_value=2.0,
                                value=float(defaults["target_volume_fraction"]),
                            )
                            volume_as_is = opt_cols[1].checkbox(
                                "Keep the starting volume",
                                bool(defaults.get("volume_as_is", False)),
                                help=(
                                    "Penalize only growth beyond the volume the "
                                    "shape started with, instead of driving it to "
                                    "the target fraction above -- which is then "
                                    "not read at all."
                                ),
                            )
                            volume_weight = opt_cols[0].number_input(
                                "Volume penalty weight",
                                min_value=0.0,
                                value=float(defaults["shape_opt_w_vol"]),
                            )
                            bc_fix_method = opt_cols[1].selectbox(
                                "BC preservation",
                                BC_FIX_METHODS,
                                index=BC_FIX_METHODS.index(
                                    defaults["sdf_bc_fix_method"]
                                    if defaults["sdf_bc_fix_method"] in BC_FIX_METHODS
                                    else "mask"
                                ),
                            )
                            reinit = st.checkbox(
                                "Reinitialize SDF", bool(defaults["reinit_sdf"])
                            )
                            reinit_every = opt_cols[0].number_input(
                                "Reinitialize every N iterations",
                                min_value=1,
                                value=int(defaults["reinit_every"]),
                            )
                        sent = st.form_submit_button("Run FEA", type="primary")
                    # Outside the form and independent of its widgets, so it renders
                    # with the card and stays put whatever the user selects.  A form
                    # batches its widgets until submit, so anything conditioned on one
                    # of them could only update a step behind.
                    strain_floor = (
                        FLOAT32_RESIDUAL_FACTOR
                        * FLOAT32_EPSILON
                        / float(defaults["relative_tol"])
                        if defaults["relative_tol"]
                        else None
                    )
                    if strain and strain_floor and strain < strain_floor:
                        st.warning(
                            f"Estimated peak strain is {strain:.2e}, below the "
                            f"{strain_floor:.2e} that single precision resolves at a "
                            f"{float(defaults['relative_tol']):.0e} relative tolerance. "
                            "A float32 solve may stall at a fixed residual instead of "
                            "converging; **Mixed precision** runs it in float64.",
                            icon=":material/warning:",
                        )
                    if sent:
                        if remesh_init_depth > remesh_depth:
                            st.error(
                                "Initial octree depth cannot exceed final octree depth."
                            )
                            sent = False
                    if sent:
                        # Noted before the step is handed over, so the monitor
                        # stops showing the previous solve the moment this one
                        # is asked for rather than when it first writes.
                        st.session_state.solve_requested = {
                            **st.session_state.get("solve_requested", {}),
                            event.session_id: time.time(),
                        }
                        answer = {
                            "preset_version": 2,
                            "action": "run",
                            "run_mode": run_mode,
                            "relative_tol": relative_tol,
                            "absolute_tol": absolute_tol,
                            "max_newton_iters": max_newton,
                            "load_steps": load_steps,
                            "remesh_init_octree_depth": remesh_init_depth,
                            "remesh_octree_depth": remesh_depth,
                            "phase_field_epsilon_factor": epsilon_factor,
                            "line_search_enabled": line_search,
                            "use_mixed_precision": mixed_precision,
                            "scale_system_to_unit_diagonal": unit_diagonal,
                            "linear_solver": linear_solver,
                            "gd_lr": gd_lr,
                            "gd_lr_auto": gd_lr_auto,
                            "gd_lr_auto_fraction": gd_lr_auto_fraction,
                            "max_optimize_iters": max_optimize,
                            "output_per_optimize_step": save_every,
                            "optimize_relative_tol": opt_tol,
                            "target_volume_fraction": target_volume,
                            "volume_as_is": volume_as_is,
                            "shape_opt_w_vol": volume_weight,
                            "sdf_bc_fix_method": bc_fix_method,
                            "reinit_sdf": reinit,
                            "reinit_every": reinit_every,
                        }
                        card.submit(answer)
                    if interrupt.get("solve_error"):
                        st.error(interrupt["solve_error"])
                        shown = optimization_shown(interrupt.get("optimization"))
                        if shown:
                            st.caption(shown)
                        # The viewer no longer offers an earlier run's fields as
                        # this case's result; they are still on disk.
                        st.caption(
                            "Results from earlier runs are no longer shown in the "
                            "viewer; they remain under solve/ in the Files panel."
                        )
                    if interrupt.get("solve_log"):
                        st.text_area(
                            "log.txt",
                            value=interrupt["solve_log"],
                            height=220,
                            disabled=True,
                            help=interrupt.get("solve_log_path"),
                        )
                    if interrupt.get("solve_complete"):
                        st.success(
                            "Simulation complete. Results are available in the viewer."
                        )
                elif kind in ("clarification",):
                    with st.form(f"clarification_{card.key}"):
                        value = st.text_input("Response")
                        sent = st.form_submit_button("Send", type="primary")
                    if sent:
                        card.submit(value)
        if event.complete and not event.interrupt:
            with st.chat_message("assistant"):
                st.success("Simulation complete. Results are available in the viewer.")
        if st.session_state.get("job_error"):
            with st.chat_message("assistant"):
                st.error(st.session_state["job_error"], icon=":material/error:")
                # The controller restores the card the rejected answer came
                # from, so sending it again with the fix is what works.
                st.caption("Nothing was applied: the card above is as it was. "
                           "Correct it and send it again.")
        pending = st.session_state.get("pending_job")
        if pending is not None:
            with st.chat_message("assistant"):
                st.markdown(f"_{html.escape(pending['status'])}_")
            # The work is on a worker thread, so this only has to notice when it
            # finishes; the page stays live in the meantime.
            watch_pending_job()

    scroll_target = st.session_state.pop("scroll_conversation", None)
    if scroll_target == "card":
        # No card waiting on an answer (a step still running): the end instead.
        scroll_target = active_card_key or "bottom"
    if scroll_target:
        scroll_conversation(scroll_target, 60)

# How often the live bridge looks for a shape the viewer has not been given.
# A shape optimization saves one every few seconds; the picture would otherwise
# be whatever the page happened to be showing when the run was started.
VIEWER_REFRESH = 2.0


def render_visualization(event: WorkflowEvent | None) -> None:
    """Everything below the Visualization heading: the picker and the viewer.

    Deliberately not a fragment.  The viewer's height comes from a flex chain
    that runs from the panel down to the frame itself (see the rules around
    ``st-key-field_view``); a fragment wraps its contents in a block of its own,
    which is not part of that chain, and the frame collapses to nothing.  A
    running solve reaches the viewer through the live bridge instead.
    """
    st.session_state["live_view"] = None
    if event:
        workflow = event.workflow
        root = controller.session_dir(event.session_id)
        surface = (
            Path(workflow["surface_path"]) if workflow.get("surface_path") else None
        )
        if surface is None or not surface.exists():
            # Re-deciding an earlier step rewinds the workflow past the geometry
            # it produced, so the state stops naming a surface.  The file is
            # still on disk and is still what the viewer should show; every
            # other artifact in this panel is found that way already.
            surface = max(
                (path for path in (root / "geometry").glob("*.glb") if path.is_file()),
                key=lambda path: path.stat().st_mtime,
                default=None,
            )
        if surface is not None:
            # An uploaded OBJ or PLY is shown through a GLB copy of itself.
            surface = viewer_surface(surface, root / "previews")
        bc_surface = root / "boundary_conditions" / "resolved.glb"
        bc_metadata = root / "boundary_conditions" / "resolved.json"
        segments_surface = root / "boundary_conditions" / "segments.glb"
        segments_metadata = root / "boundary_conditions" / "segments.json"
        solve_root = root / "solve"
        displacement_contour = solve_root / "displacement_magnitude.glb"
        result_images = [
            (solve_root / "surface_final.png", "Final analyzed surface"),
        ]

        artifacts = []
        legends: dict[str, list[dict[str, str]]] = {}
        colorbars: dict[str, dict[str, Any]] = {}
        if surface and surface.exists():
            artifacts.append(("Geometry · interactive surface", surface))
        if bc_surface.exists():
            label = "Boundary conditions · selected regions"
            artifacts.append((label, bc_surface))
            if bc_metadata.exists():
                bc_legend = json.loads(bc_metadata.read_text(encoding="utf-8")).get(
                    "legend", []
                )
                # Keep existing sessions consistent with the current BC terminology.
                for item in bc_legend:
                    if item.get("label") == "Dirichlet support":
                        item["label"] = "Dirichlet BC region"
                    elif item.get("label") == "Neumann traction":
                        item["label"] = "Neumann BC region"
                        item["color"] = "#f00000"
                legends[label] = bc_legend
        if segments_surface.exists():
            segment_method = "shape-diameter"
            if segments_metadata.exists():
                segment_method = {
                    "partfield": "PartField", "cad_faces": "CAD-face",
                }.get(json.loads(
                    segments_metadata.read_text(encoding="utf-8")
                ).get("method"), "shape-diameter")
            label = f"Boundary conditions · {segment_method} segments"
            artifacts.append((label, segments_surface))
            if segments_metadata.exists():
                segments = json.loads(
                    segments_metadata.read_text(encoding="utf-8")
                ).get("segments", [])
                legends[label] = [
                    {"label": f"Segment {item['id']}", "color": item["color"]}
                    for item in segments
                ]
        material_root = material_field_root(workflow, root)
        material_labels = {
            "youngs_modulus": "Young's modulus",
            "poisson_ratio": "Poisson ratio",
            "density": "Density",
        }
        material_units = UNIT_SYSTEM_DETAILS.get(
            workflow.get("unit_system", "SI (m-kg-s)"), {}
        )
        for field, title in material_labels.items():
            path = material_root / f"{field}.glb" if material_root else None
            if path is None or not path.exists():
                continue
            label = f"Material field · {title}"
            artifacts.append((label, path))
            metadata = path.with_suffix(path.suffix + ".json")
            if metadata.exists():
                values = json.loads(metadata.read_text(encoding="utf-8"))
                colorbars[label] = {
                    "minimum": values["minimum"],
                    "maximum": values["maximum"],
                    "scale": values.get("scale", "linear"),
                    "unit": (
                        material_units.get("stress_modulus_traction", "")
                        if field == "youngs_modulus"
                        else material_units.get("density", "")
                        if field == "density" else ""
                    ),
                    "title": title,
                }
        # Found before the displacement entry, which no longer waits for the
        # solve-time asset: a shape optimization's surfaces carry displacement
        # from its first saved iteration, and run_forward rebuilds the asset as
        # it goes (and, for a failed run, from the last iteration that solved).
        surface_vtu = newest_surface_vtu(solve_root)
        carried = (
            surface_point_fields(str(surface_vtu), surface_vtu.stat().st_mtime)
            if surface_vtu is not None else ()
        )
        if displacement_contour.exists() or "displacement" in carried:
            label = "FEM results · displacement"
            artifacts.append((
                label,
                displacement_contour if displacement_contour.exists() else surface_vtu,
            ))
            contour_metadata = displacement_contour.with_suffix(".glb.json")
            if contour_metadata.exists():
                values = json.loads(contour_metadata.read_text(encoding="utf-8"))
                minimum, maximum = values["minimum"], values["maximum"]
                unit = UNIT_SYSTEM_DETAILS.get(
                    workflow.get("unit_system", "SI (m-kg-s)"), {}
                ).get("displacement", "")
                colorbars[label] = {
                    "minimum": minimum,
                    "maximum": maximum,
                    "unit": unit,
                }
        # Tensor fields get one entry each; the component is chosen in the
        # viewport and coloured when it is picked, rather than rendering twelve
        # assets per solve that nobody may look at.
        available_tensors = ()
        if surface_vtu is not None:
            available_tensors = tuple(
                f for f in ("strain", "stress") if f in carried
            )
            for name in available_tensors:
                artifacts.append((f"FEM results · {name}", surface_vtu))
            if "displacement" in carried:
                available_tensors += ("displacement",)
        artifacts.extend(
            (f"FEM results · {caption}", path)
            for path, caption in result_images
            if path.exists()
        )

        if not artifacts:
            # Keep the control in place and empty rather than absent, so the
            # panel does not change shape the moment the first artifact lands.
            st.selectbox(
                "Displayed visualization",
                (),
                key=f"visualization-empty-{event.session_id}",
                label_visibility="collapsed",
                placeholder="Nothing to display yet",
                disabled=True,
            )
            render_glb(None)
        else:
            labels = [label for label, _ in artifacts]
            paths = dict(artifacts)
            newest_label = preferred_visualization(
                [(label, path.stat().st_mtime) for label, path in artifacts]
            )
            signature = tuple(
                (label, str(path), path.stat().st_mtime_ns) for label, path in artifacts
            )
            widget_key = f"visualization-{event.session_id}"
            signature_key = f"visualization-signature-{event.session_id}"
            if st.session_state.get(signature_key) != signature:
                st.session_state[widget_key] = newest_label
                st.session_state[signature_key] = signature
            # The label states the obvious above a list of visualizations, and the
            # room is worth more to the tool window below.
            selected = st.selectbox(
                "Displayed visualization",
                labels,
                key=widget_key,
                label_visibility="collapsed",
            )
            selected_path = paths[selected]
            field_entry = next(
                (f for f in available_tensors if selected == f"FEM results · {f}"),
                None,
            )
            # Selector and viewer share one gapless box, so the pair occupies
            # exactly the height the viewer alone gets for every other view --
            # otherwise the block gap between them eats 1rem of viewer.
            view = (
                st.container(border=False, gap=0, key="field_view")
                if field_entry is not None
                else contextlib.nullcontext()
            )
            with view:
                if field_entry is not None:
                    choices = FIELD_COMPONENTS[field_entry]
                    material = workflow.get("material") or {}
                    if field_entry == "stress" and material.get("mode", "uniform") == "uniform":
                        choices += ("safety_factor",)
                    component = (
                        st.segmented_control(
                            "Component",
                            choices,
                            default=choices[0],
                            format_func=lambda name: COMPONENT_LABELS.get(name, name),
                            key=f"field-component-{event.session_id}-{field_entry}",
                            label_visibility="collapsed",
                        )
                        or choices[0]
                    )
                    # How far to exaggerate, and which load step to show.  Both
                    # sit on one row so the pair costs the viewer a single strip
                    # of height rather than two.
                    # Both sliders are drawn inside the viewer.  A widget on
                    # this side costs a rerun, an export and a reload before
                    # anything moves, which is not something anyone would call
                    # dragging; every step is carried into the frame instead.
                    steps = load_step_surfaces(solve_root)
                    scales = deformation_scales(root, workflow)
                    # What a solve's surfaces are depends on what it was: a
                    # forward ramp writes one per load step, an optimization one
                    # per saved iteration.  They are the same files; only the
                    # control that walks them differs, and which it is comes
                    # from the run's own log rather than from the settings now
                    # in the workflow -- re-deciding an earlier card rewinds
                    # those past the run whose surfaces are still on disk.
                    saves_every = optimization_saves(newest_solve_log(root))
                    optimizing = saves_every is not None
                    step_index = max(len(steps) - 1, 0)
                    step_vtu = steps[step_index] if steps else surface_vtu
                    # Strain is dimensionless; stress and displacement carry a unit.
                    units = UNIT_SYSTEM_DETAILS.get(
                        workflow.get("unit_system", "SI (m-kg-s)"), {}
                    )
                    unit = {
                        "strain": "",
                        "stress": units.get("stress_modulus_traction", ""),
                        "displacement": units.get("displacement", ""),
                    }[field_entry]
                    if component == "safety_factor":
                        unit = ""
                        st.caption("Set yield strength and your target factor inside the viewer. The purple marker identifies the lowest-factor surface point. This view assesses ductile yielding.")
                    ramp_frames = None
                    step_label = "Iteration" if optimizing else "Load step"
                    carried: list[int] = []
                    if optimizing and steps:
                        numbers = iteration_numbers(len(steps), saves_every)
                        carried = chosen_iterations(len(steps))
                        try:
                            # Each iteration over its own extremes: the shapes
                            # of an optimization are not one load ramped, and a
                            # reference that moved every time the run saved
                            # would make every asset already built stale.  The
                            # colour bar travels with the frame instead.
                            ramp_frames = [
                                embedded_frame(
                                    Path(component_asset(
                                        str(root),
                                        workflow.get(
                                            "compute_env", DEFAULT_COMPUTE_ENV
                                        ),
                                        str(steps[index]),
                                        steps[index].stat().st_mtime,
                                        field_entry, component, 0.0,
                                        index, max(len(steps), 2),
                                    )),
                                    scales,
                                    name=str(numbers[index]),
                                )
                                for index in carried
                            ]
                        except Exception as exc:
                            ramp_frames, carried = None, []
                            st.warning(
                                f"Could not prepare every iteration: "
                                f"{type(exc).__name__}: {exc}"
                            )
                    if not optimizing and 1 < len(steps) <= MAX_EMBEDDED_STEPS:
                        try:
                            # Every step is coloured against the fully loaded
                            # one.  Left to itself each stretches the colours
                            # over its own largest value, so a ramp that triples
                            # its load looks identical at every step while the
                            # one legend on screen -- the last step's -- claims
                            # to speak for all of them.
                            ramp_frames = [
                                embedded_frame(
                                    Path(component_asset(
                                        str(root),
                                        workflow.get(
                                            "compute_env", DEFAULT_COMPUTE_ENV
                                        ),
                                        str(vtu), vtu.stat().st_mtime,
                                        field_entry, component, 0.0,
                                        index, len(steps), str(steps[-1]),
                                    )),
                                    scales,
                                )
                                for index, vtu in enumerate(steps)
                            ]
                        except Exception as exc:
                            # One step that will not build should not cost the
                            # view; fall back to the single asset below.
                            ramp_frames = None
                            st.warning(
                                f"Could not prepare every load step: "
                                f"{type(exc).__name__}: {exc}"
                            )
                    prebuilt = (
                        contour_shows(displacement_contour, step_vtu)
                        and displacement_contour.with_suffix(".glb.values.bin").exists()
                    )
                    if component == "magnitude" and step_vtu == surface_vtu and prebuilt:
                        # Built at solve time; selected_path already points at
                        # it -- but only while it is newer than the surface it
                        # claims to show.  A run still writing leaves the last
                        # solve's contour on disk, and taking that as this one's
                        # would put a finished shape under a live iteration
                        # number.
                        colorbars[selected] = {
                            **colorbars.get(selected, {}),
                            "title": f"{field_entry.capitalize()} magnitude",
                        }
                    else:
                        try:
                            with st.spinner(f"Colouring {field_entry} {component}…"):
                                selected_path = Path(
                                    component_asset(
                                        str(root),
                                        workflow.get("compute_env", DEFAULT_COMPUTE_ENV),
                                        str(step_vtu),
                                        step_vtu.stat().st_mtime,
                                        field_entry,
                                        component,
                                        0.0,
                                        step_index,
                                        # An optimization's assets are stamped
                                        # with their iteration from the first
                                        # one, rather than only once a second
                                        # exists: otherwise iteration zero is
                                        # written unstamped and then rewritten
                                        # under a stamped name the moment the
                                        # run saves again.
                                        max(len(steps), 2) if optimizing
                                        else len(steps),
                                    )
                                )
                            meta = json.loads(
                                selected_path.with_suffix(
                                    selected_path.suffix + ".json"
                                ).read_text(encoding="utf-8")
                            )
                            colorbars[selected] = {
                                "minimum": 0.0 if component == "safety_factor" else meta["minimum"],
                                "maximum": 5.0 if component == "safety_factor" else meta["maximum"],
                                "unit": unit,
                                "safety_factor": component == "safety_factor",
                                "stress_unit": units.get("stress_modulus_traction", ""),
                                # "von Mises stress", not "Stress von_mises".
                                "title": (
                                    "Yield safety factor" if component == "safety_factor" else
                                    f"von Mises {field_entry}"
                                    if component == "von_mises"
                                    else f"{field_entry.capitalize()} {component}"
                                ),
                            }
                        except Exception as exc:
                            st.error(
                                f"Could not colour {field_entry} {component}: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            selected_path = None
                if selected_path is None:
                    pass
                elif selected_path.suffix.lower() == ".glb":
                    # TripoSG hands back a normalised box roughly two units
                    # across; scale_surface is what gives the shape a size, and
                    # writes it as physical.glb.  Until then the geometry card
                    # is showing something with no length to quote, so its grid
                    # says nothing rather than reading a dimensionless 0.5 as
                    # half a metre.  Every other artifact here is written after
                    # scaling and is in the session's units.
                    unscaled = (
                        surface is not None
                        and selected_path == surface
                        and surface.name != "physical.glb"
                    )
                    render_glb(
                        selected_path,
                        height=360,
                        legend=legends.get(selected),
                        colorbar=colorbars.get(selected),
                        deformation=surface_deformation(selected_path, scales)
                        if field_entry is not None else None,
                        steps=ramp_frames if field_entry is not None else None,
                        length_unit=None if unscaled else UNIT_SYSTEM_DETAILS.get(
                            workflow.get("unit_system") or "", {}
                        ).get("length"),
                        step_label=step_label if field_entry is not None
                        else "Load step",
                        live=field_entry is not None and optimizing
                        and solve_in_flight(),
                        view_key=visualization_key(
                            event.session_id, selected,
                            component if field_entry is not None else None,
                        ),
                    )
                    if field_entry is not None and optimizing and ramp_frames:
                        # What the live bridge needs to build the next shape and
                        # where to put it.  Recorded here because this is the
                        # only place that knows which field is on screen and
                        # which iterations the frame was given.
                        st.session_state["live_view"] = {
                            "root": str(root),
                            "solve_root": str(solve_root),
                            "compute_env": workflow.get(
                                "compute_env", DEFAULT_COMPUTE_ENV
                            ),
                            "field": field_entry,
                            "component": component,
                            "carried": list(carried),
                            # Where the viewer has been brought up to, which is
                            # the newest shape it was just given.
                            "seen": carried[-1],
                            "saves_every": saves_every,
                            "scales": list(scales),
                        }
                else:
                    st.image(selected_path, caption=selected, width="stretch")
    else:
        # Same before a conversation exists as when one has no artifacts yet:
        # the control stays, empty, so the panel does not change shape the
        # moment the first thing lands.
        st.selectbox(
            "Displayed visualization",
            (),
            key="visualization-empty",
            label_visibility="collapsed",
            placeholder="Nothing to display yet",
            disabled=True,
        )
        render_glb(None)


@st.fragment(run_every=VIEWER_REFRESH)
def live_frame_bridge() -> None:
    """Hand the running viewer each shape the optimization writes.

    Pushed into the frame rather than drawn into the page.  Rebuilding the
    viewer would reload every shape already in hand and put the camera back
    where it started, several times a minute, for the whole of a run -- and the
    viewer cannot be redrawn from inside a fragment at all, because it takes its
    height from a flex chain a fragment's own wrapper is not part of.  So this
    carries nothing but a script, and the frame it posts to stays where it is.

    What to send, and what the viewer discards to make room for it, is decided
    by ``next_live_frame`` -- which reads a high-water mark rather than what the
    viewer still holds, because the two part company as soon as it starts
    discarding.
    """
    plan = st.session_state.get("live_view")
    if not plan or not solve_in_flight():
        return
    steps = load_step_surfaces(Path(plan["solve_root"]))
    carried = plan["carried"]
    offer = next_live_frame(plan["seen"], len(steps), len(carried))
    if offer is None:
        return
    index, drop = offer
    surface = steps[index]
    try:
        asset = Path(component_asset(
            plan["root"], plan["compute_env"], str(surface),
            surface.stat().st_mtime, plan["field"], plan["component"],
            0.0, index, max(len(steps), 2),
        ))
    except Exception:
        # The click that asks for this shape reports the failure; a speculative
        # build must not put an error over a running solve.
        return
    frame = embedded_frame(asset, plan["scales"],
                           name=str(index * max(int(plan["saves_every"] or 1), 1)))
    if drop is not None:
        carried.pop(drop)
    carried.append(index)
    st.session_state["live_view"] = {**plan, "carried": carried, "seen": index}
    payload = json.dumps({**frame, "femFrame": True, "drop": drop})
    st.iframe(
        BRIDGE_STYLE
        + f"""
        <script>
          // A unique marker, so Streamlit runs this again for the next shape
          // rather than treating an identical script as already delivered.
          const push = '{time.time_ns()}';
          const frame = window.parent.document.querySelector(
            '[class*="st-key-field_view"] iframe'
          );
          if (frame && frame.contentWindow) {{
            frame.contentWindow.postMessage({payload}, '*');
          }}
        </script>
        """,
        height=1,
    )


with right.container(height=700, border=False, key="visualization_panel"):
    st.markdown(
        '<div class="conversation-header"><div class="conversation-heading">'
        "Visualization</div></div>",
        unsafe_allow_html=True,
    )
    render_visualization(event)


# Invisible, and last among the panels, so it reads the plan the viewer has
# just recorded rather than the one from the rerun before.
with st.container(key="ui_bridge_live_frames"):
    live_frame_bridge()

with right.container(border=False, key="tool_panel"):
    render_tool_window(event)

# Last, so both splits exist to be measured when the handles first place
# themselves.  The frame is only somewhere for the script to run: it is taken
# out of the flow rather than hidden, because it still has to load.
with st.container(key="split_handles"):
    st.iframe(SPLIT_HANDLES, height=1)

# Last of all, so a browser holding this run's number holds everything drawn
# before it; ``redraw_if_lost`` compares the two.
st.session_state["page_drawn"] = str(st.session_state["page_generation"])
st.session_state["page_drawn_at"] = time.time()
with st.container(key="ui_bridge_drawn"):
    st.text_input("Page drawn", key="page_drawn", label_visibility="collapsed", disabled=True)
