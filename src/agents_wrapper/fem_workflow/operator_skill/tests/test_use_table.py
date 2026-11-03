"""The brief's Use table: the use's actions, one row each with its state, size, source, entry
member and holds, which the runner turns into effects and coverage itself.  No API call: the use
anchor (a second reading of the use by a specialist call) is gone.  The objects are neutral
placeholders.  No GPU, no API, no graph run."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import drive  # noqa: E402
import sessionlib as lib  # noqa: E402
from test_frames_and_runner_writes import BRIEF  # noqa: E402

# A placeholder part, no real object: a top plate worked by hand, a side grip, a bottom plate.
CONTACTS = ('- side grip: held over 0.1 m by the hand\n- bottom plate: rests on a support surface\n'
            '{"side grip": {"transmits": "force and moment", "kind": "grip", "over_m": 0.1}, '
            '"bottom plate": {"transmits": "force only", "kind": "rest"}}')
SIZE = ("Real mass: 1 kg\nReal thinnest member: 0.01 m\n"
        "Real section (top plate): 0.01 m deep, I 8.3e-10 m4, A 1e-4 m2\nReal extents: 0.3 x 0.2 x 0.1 m\n"
        'Load path: {"top plate": "series", "side grip": "series", "bottom plate": "series"}\n')
# Each row's lever pinned (lever_m 0.3 m, the largest Real extent) so the numbers below read simply;
# the runner's own defaults are tested apart (test_a_rows_path_and_lever_come_from_its_entry_and_holds).
ROWS = {"A1": {"load": "push", "state": "working", "kind": "applied", "size": 200, "unit": "N", "source": "H5 top",
               "enters": "top plate", "held": ["side grip", "bottom plate"], "lever_m": 0.3},
        "A2": {"load": "side pull", "state": "working", "kind": "applied", "size": 50, "unit": "N",
               "source": "H3 middle", "enters": "top plate", "held": ["side grip", "bottom plate"], "lever_m": 0.3},
        "A3": {"load": "carry", "state": "handling", "kind": "weight", "size": 100, "unit": "N",
               "source": "10 kg x g x 1", "enters": "bottom plate", "held": ["side grip"], "dynamic": True,
               "lever_m": 0.3}}   # a jolt at the grip
TABLE = {"actions": ROWS, "run": "working"}
INSTRUCTION = "a test part"


def table(**changes) -> dict:
    """The Use table with ``changes`` merged in (rows by label: a row dict, or None to drop it)."""
    out = copy.deepcopy(TABLE)
    for key, value in changes.items():
        if key == "rows":
            for label, row in value.items():
                if row is None:
                    out["actions"].pop(label, None)
                else:
                    out["actions"][label] = {**out["actions"].get(label, {}), **row}
        else:
            out[key] = value
    return out


def session_with(tmp_path: Path, use_table: dict | None = None, contacts: str = CONTACTS, size: str = SIZE,
                 name: str = "abc", instruction: str = INSTRUCTION, **_) -> Path:
    """A session whose brief holds the Use table (none: the Use sentence alone) and the stored
    instruction."""
    session = tmp_path / name
    (session / "operator").mkdir(parents=True, exist_ok=True)
    use = "A person pushes and pulls the top plate, the bottom plate on a support surface."
    if use_table is not None:
        use += "\nUse table: " + json.dumps(use_table)
    (session / "operator" / "brief.md").write_text(
        f"## Use\n{use}\n## Contacts\n{contacts}\n## Load\n200 N down on the top plate, H5 top\n"
        f"## Size and material\n{size}")
    drive.store_instruction(session, instruction)
    return session


def gate(session: Path) -> list[str]:
    return drive.use_gate(session / "operator")


# --------------------------------------------------------------- no API call --

def test_the_operator_skill_makes_no_api_call_of_its_own():
    source = "\n".join(p.read_text() for p in SCRIPTS.glob("*.py"))
    for name in ("probe_api", "SpecialistCalls", "infer_use_anchor", "make_use_anchor", "AgentsWrapper",
                 "provider_factory"):
        assert name not in source, name
    import src.agents_wrapper.fem_workflow.specialists as specialists
    assert not hasattr(specialists.SpecialistCalls, "infer_use_anchor")


def test_start_requires_the_instruction_and_stores_it_for_the_checks(tmp_path, monkeypatch):
    parser = drive.parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["start"])
    assert parser.parse_args(["start", "--instruction", "a test part"]).instruction == "a test part"
    with pytest.raises(SystemExit):
        parser.parse_args(["anchor", "abc"])                       # the anchor command is gone
    monkeypatch.delenv("FEM_OPERATOR_INSTRUCTION", raising=False)
    assert drive.start_instruction(tmp_path / "abc", "a test part") is None
    assert drive.stored_instruction(tmp_path / "abc" / "operator") == "a test part"
    assert lib.read_json(tmp_path / "abc" / "operator" / drive.INSTRUCTION)["instruction"] == "a test part"


def test_a_batch_instruction_is_stored_with_the_agents_beside_it(tmp_path, monkeypatch):
    monkeypatch.setenv("FEM_OPERATOR_INSTRUCTION", "a test part")
    differs = drive.start_instruction(tmp_path / "abc", "a test part, painted")
    assert "differs from the user's instruction" in differs
    record = lib.read_json(tmp_path / "abc" / "brief_snapshots" / drive.INSTRUCTION)
    assert record["instruction"] == "a test part" and record["instruction_given"] == "a test part, painted"
    assert drive.start_instruction(tmp_path / "abd", "a test part") is None


# ------------------------------------------------------------------ the table --

def test_a_brief_without_a_use_table_is_refused_naming_its_form(tmp_path):
    [said] = gate(session_with(tmp_path, None))
    assert "the Use section ends with one line Use table: {" in said and '"state"' in said and '"enters"' in said
    assert "working, handling and assembly states" in said
    assert gate(session_with(tmp_path, TABLE)) == []


def test_each_row_carries_its_state_kind_size_source_and_entry(tmp_path):
    said = gate(session_with(tmp_path, table(rows={"A1": {"size": 0, "unit": "kN", "source": ""}})))
    assert any('"size" is a number over 0' in s for s in said)
    assert any('"unit" is "N" or "N m"' in s for s in said)
    assert any('"source" names where the size comes from' in s for s in said)
    said = gate(session_with(tmp_path, table(rows={"A1": {"source": "H5 top", "size": 900}})))
    assert any("900 N is outside H5's range, 100-300 N" in s for s in said)
    # a row x a stated dynamic factor reaches up to three times its top
    assert not [s for s in gate(session_with(tmp_path, table(rows={"A3": {"source": "H9 top", "size": 400}})))
                if "range" in s]
    said = gate(session_with(tmp_path, table(rows={"A1": {"enters": "hinge"}})))
    assert any("\"enters\" 'hinge' is not a member of the Load path" in s for s in said)
    said = gate(session_with(tmp_path, table(rows={"A2": {"state": "idle", "kind": "push"}})))
    assert any('Use table row A2: "state" (working | handling | assembly | storage) and "kind"' in s for s in said)
    # every problem at once
    said = gate(session_with(tmp_path, table(rows={"A1": {"enters": "hinge"}, "A2": {"unit": "kN"}})))
    assert any("'hinge'" in s for s in said) and any('"unit" is "N" or "N m"' in s for s in said)


def test_a_given_size_is_one_the_instruction_states(tmp_path):
    said = gate(session_with(tmp_path, table(rows={"A1": {"source": "given", "size": 200}})))
    assert any('"source": "given", but the user\'s instruction states no 200' in s for s in said)
    assert gate(session_with(tmp_path, table(rows={"A1": {"source": "given", "size": 200}}),
                             instruction="a test part pushed with 200 N")) == []


def test_the_runner_computes_each_rows_effect_and_ignores_the_agents(tmp_path):
    session = session_with(tmp_path, table(rows={"A1": {"effect": {"top plate": 1e9}}}))
    built, problems = drive.use_table(session / "operator")
    assert problems == []
    coverage = drive.use_coverage(session / "operator")
    assert coverage["unit"] == "N m" and coverage["run_effect"] == pytest.approx(75.0)
    # a force times the largest Real extent (0.3 m), at every member from its entry to the hold
    assert built["effects"]["A1"] == {"top plate": 60.0, "side grip": 60.0, "bottom plate": 60.0}
    assert built["effects"]["A3"] == {"side grip": 30.0, "bottom plate": 30.0}
    # the row's own lever replaces the extent; one longer than the object is the object's
    built, _ = drive.use_table(session_with(tmp_path, table(rows={"A2": {"lever_m": 0.1}, "A3": {"lever_m": 5}}))
                               / "operator")
    assert built["effects"]["A2"]["top plate"] == pytest.approx(5.0)
    assert built["effects"]["A3"]["side grip"] == pytest.approx(30.0)
    # a twist is its own size; a row of the surroundings that is no force loads nothing
    twist = {"A2": {"unit": "N m", "size": 4, "source": "H8 top"},
             "+heat": {"load": "heat", "state": "working", "kind": "environment", "size": 0, "source": "derived: none",
                       "enters": "top plate"}}
    built, problems = drive.use_table(session_with(tmp_path, table(rows=twist)) / "operator")
    assert built["effects"]["A2"]["top plate"] == 4 and "+heat" not in built["effects"]


def test_a_minor_row_is_sized_as_another_row(tmp_path):
    cheap = table(rows={"A2": {"source": "as A1", "size": None, "unit": None}})
    assert gate(session_with(tmp_path, cheap)) == []
    coverage = drive.use_coverage(tmp_path / "abc" / "operator")
    assert coverage["cases"]["working"]["top plate"] == pytest.approx(120.0)
    said = gate(session_with(tmp_path, table(rows={"A2": {"source": "as A1", "size": 40}})))
    assert any('sized "as A1", it leaves "size" out or gives A1\'s, 200' in s for s in said)
    for rows in ({"A2": {"source": "as A2", "size": None}}, {"A2": {"source": "as A9", "size": None}},
                 {"A2": {"source": "as A1", "size": None}, "A3": {"source": "as A2", "size": None}}):
        said = gate(session_with(tmp_path, table(rows=rows)))
        assert any('"source": "as <row>" names another row of this table' in s for s in said), rows
    # minor is judged on the runner's own effects: a light force on the same lever is minor
    built, _ = drive.use_table(session_with(tmp_path, table(rows={"A2": {"size": 10, "source": "derived: 10 N"}}))
                               / "operator")
    assert drive.minor_row(built, "A2") and not drive.minor_row(built, "A1")


def test_the_brief_names_the_real_section_member_and_the_real_extents(tmp_path):
    size = SIZE.replace("Real section (top plate):", "Real section:").replace("Real extents: 0.3 x 0.2 x 0.1 m\n", "")
    said = gate(session_with(tmp_path, TABLE, size=size))
    assert any("names its member" in s for s in said) and any("Real extents" in s for s in said)


def test_the_real_section_may_be_a_shape_whose_i_and_a_the_runner_computes():
    rect = drive.real_section_in("Real section (top plate): rectangle 0.04 x 0.01 m")
    assert rect["depth_m"] == 0.01 and rect["I_m4"] == pytest.approx(0.04 * 0.01 ** 3 / 12)
    assert rect["A_m2"] == pytest.approx(4e-4)
    tube = drive.real_section_in("Real section (arm): tube 0.03 x 0.002 m")
    assert tube["I_m4"] == pytest.approx(3.14159265 * (0.03 ** 4 - 0.026 ** 4) / 64, rel=1e-6)
    assert drive.real_section_in("Real section (arm): round 0.02 m")["A_m2"] == pytest.approx(3.14159265e-4, rel=1e-6)
    box = drive.real_section_in("Real section (arm): box 0.05 x 0.1 x 0.004 m")
    assert box["depth_m"] == 0.1 and box["A_m2"] == pytest.approx(0.05 * 0.1 - 0.042 * 0.092)
    assert drive.real_section_in("Real section (arm): 0.1 m deep, I 2e-6 m4, A 1e-3 m2")["I_m4"] == 2e-6
    assert drive.real_section_in("Real section (arm): blob 0.1 m") is None


def test_the_first_run_and_blank_refuse_a_brief_without_a_use_table(tmp_path, monkeypatch):
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    brief = "\n".join(ln for ln in BRIEF.replace("MEMBER", "0.05").splitlines() if not ln.startswith("Use table:"))
    (session / "operator" / "brief.md").write_text(brief)
    monkeypatch.setattr("src.agents_wrapper.fem_workflow.run_workflow.read_compute_env", lambda s: "env")

    def no_plan(*a, **k):
        raise AssertionError("the plan was read: something could be spent")
    monkeypatch.setattr(drive, "load_plan", no_plan)
    drive.cmd_run(argparse.Namespace(output_root=str(tmp_path), session_id="abc", plan="plan.json", patch=None,
                                     two_depths=False, full=False))
    assert printed[-1]["stop"] == "error" and "Use table" in printed[-1]["error"]
    problems = drive.blank_problems(session / "operator" / "brief.md", argparse.Namespace())
    assert any("Use table" in p for p in problems)


# ----------------------------------------------------------------- the states --

def test_every_action_of_a_state_acts_together_and_run_names_a_state(tmp_path):
    rest_only = '- bottom plate: rests on a support surface\n{"bottom plate": {"transmits": "force only", "kind": "rest"}}'
    weighed = {"A1": {**ROWS["A1"], "load": "placed load", "kind": "weight", "source": "4 kg x g x 1", "size": 39,
                      "held": ["bottom plate"]},
               "A2": {**ROWS["A2"], "load": "wind", "kind": "environment", "source": "derived: 0.6 x 20^2 x 0.06",
                      "size": 14.4, "held": ["bottom plate"]}}
    session = session_with(tmp_path, {"actions": weighed, "run": "working"}, contacts=rest_only)
    assert gate(session) == []
    working = drive.use_coverage(session / "operator")["cases"]["working"]
    assert working["top plate"] == pytest.approx((39 + 14.4) * 0.3)        # the placed load and the wind, together
    said = gate(session_with(tmp_path, table(run="work")))
    assert any('the Use table\'s "run" is "work": the state the solve applies, one of' in s for s in said)
    said = gate(session_with(tmp_path, table(run="storage")))
    assert any("\"run\" 'storage' is no state of the Use table's rows" in s for s in said)


def test_every_member_is_covered_on_the_runners_effects_or_declared_uncovered(tmp_path):
    # another state over the run by more than a third at a member: refused, naming both, with numbers
    larger = table(rows={"A3": {"size": 400, "source": "40 kg x g x 1"}})
    [said] = gate(session_with(tmp_path, larger))                  # one message for the state
    assert "state 'handling' exceeds the run state 'working' by more than a third" in said
    assert "120 N m at 'side grip' against 75; 120 N m at 'bottom plate' against 75" in said
    assert '"uncovered"' in said and "Run 'handling' (the Use table's \"run\";" in said and '"enters"' not in said
    # the refusal (recorded) is not cleared by moving the Real section or declaring members it did not name
    operator = tmp_path / "abc" / "operator"
    assert lib.read_json(operator / drive.COVERAGE_REFUSED)["members"] == ["bottom plate", "side grip"]
    cleared = {"run": "working", "member": "bottom plate", "uncovered": {}}
    said = drive.coverage_held(operator, cleared, {"declared_uncovered": {"hinge": "x"}}, {})
    assert "the Real section moved from 'top plate' to 'bottom plate'" in said[0]
    assert "['hinge'], which it did not name, declared uncovered" in said[0] and "Run 'handling'" in said[0]
    # an exit clears it: the refused state run (or a corrected row, the Real section where it was)
    assert drive.coverage_held(operator, {**cleared, "run": "handling"}, {}, {}) == []
    assert not (operator / drive.COVERAGE_REFUSED).exists()
    # a corrected row clears it too, and the record goes: a later move of the Real section is free
    lib.write_json(operator / drive.COVERAGE_REFUSED, {"members": ["side grip"], "cases": ["handling"],
                                                       "member": "top plate", "declared": []})
    assert drive.coverage_held(operator, {**cleared, "member": "top plate"}, {}, {}) == []
    assert not (operator / drive.COVERAGE_REFUSED).exists()
    assert drive.coverage_held(operator, cleared, {}, {}) == []
    # the governing row deleted rather than corrected: the record stays and the refusal holds
    lib.write_json(operator / drive.COVERAGE_REFUSED, {"members": ["side grip"], "cases": ["handling"],
                                                       "member": "top plate", "declared": [], "rows": ["A3"]})
    said = drive.coverage_held(operator, {**cleared, "member": "top plate"}, {"rows": {"A1": {}}}, {})
    assert "row(s) ['A3'] it rested on were removed" in said[0] and (operator / drive.COVERAGE_REFUSED).exists()
    # renamed while it was corrected (the same state, member and holds, a new size): no deletion
    content = {"state": "handling", "enters": "side grip", "held": ["base"], "size": 50.0}
    lib.write_json(operator / drive.COVERAGE_REFUSED, {"members": ["side grip"], "cases": ["handling"],
                                                       "member": "top plate", "declared": [], "rows": ["A3"],
                                                       "row_content": {"A3": content}})
    renamed = {"rows": {"A1": {}, "A4": {**content, "size": 400}}}
    assert drive.coverage_held(operator, {**cleared, "member": "top plate"}, renamed, {}) == []
    # the named members declared uncovered: that exit is taken before the deleted-row check
    lib.write_json(operator / drive.COVERAGE_REFUSED, {"members": ["side grip"], "cases": ["handling"],
                                                       "member": "top plate", "declared": [], "rows": ["A3"]})
    declared = {"rows": {"A1": {}}, "declared_uncovered": {"side grip": "why"}}
    assert drive.coverage_held(operator, {**cleared, "member": "top plate"}, declared, {}) == []
    (operator / drive.COVERAGE_REFUSED).unlink(missing_ok=True)
    # a nearer entry (lever_m) on another state's row is taken with its reason, else set aside and said
    near = {"A3": {"size": 400, "source": "40 kg x g x 1", "lever_m": 0.05}}
    session_with(tmp_path, table(rows=near))
    coverage = drive.use_coverage(tmp_path / "abc" / "operator")
    assert "the default is used" in coverage["lever_notes"][-1]
    assert coverage["cases"]["handling"]["side grip"] == pytest.approx(400 * 0.15)      # the default lever
    (tmp_path / "abc" / "operator" / drive.COVERAGE_REFUSED).unlink(missing_ok=True)
    near["A3"]["lever_why"] = "the load enters beside its hold"
    assert gate(session_with(tmp_path, table(rows=near))) == []
    # declared, with the workflow limit that keeps it out: recorded, printed, a simplified verdict
    limit = {"limit": "dynamic or impact", "why": "the carry is a jolt at the grip",
             "no_approximation": "the jolt has no stated factor"}
    session = session_with(tmp_path, table(rows=larger["actions"], uncovered={"side grip": limit,
                                                                              "bottom plate": limit}))
    assert gate(session) == []
    coverage = lib.read_json(session / "operator" / drive.COVERAGE)
    assert coverage["governing"] == "working" and coverage["run_effect"] == pytest.approx(75.0)
    assert coverage["uncovered"]["side grip"]["declared"] == "dynamic or impact: the carry is a jolt at the grip"
    said = gate(session_with(tmp_path, table(uncovered={"hinge": {"limit": "temperature", "why": "x"}})))
    assert any("\"uncovered\" names ['hinge'], not members of the Load path" in s for s in said)
    said = gate(session_with(tmp_path, table(rows=larger["actions"],
                                             uncovered={"side grip": {"limit": "temperature", "why": "x"},
                                                        "top plate": {"limit": "temperature", "why": "x"}})))
    # the Real section's member may be declared: its governing load cannot be applied, said in the verdict
    assert not any("names the Real section's member" in s for s in said)
    assert any("its dominant row A3 is no action of the surroundings" in s for s in said)


# ------------------------------------------------------------------ the contacts --

def test_the_contacts_are_checked_against_their_own_kinds(tmp_path):
    assert gate(session_with(tmp_path, TABLE)) == []
    said = gate(session_with(tmp_path, table(rows={"A2": {"held": ["palm"]}})))
    assert any("held ['palm'] are not contacts" in s for s in said)
    no_length = CONTACTS.replace(', "over_m": 0.1', "")
    assert any('"over_m"' in s for s in gate(session_with(tmp_path, TABLE, contacts=no_length)))
    clamped_rest = CONTACTS.replace('"bottom plate": {"transmits": "force only"', '"bottom plate": {"transmits": '
                                                                                  '"force and moment"')
    assert any("transmits \"force only\"" in s for s in gate(session_with(tmp_path, TABLE, contacts=clamped_rest)))
    # a grip alone where a grip and a rest are written is the agent's own reading: accepted
    grip_only = ('- side grip: held over 0.1 m\n'
                 '{"side grip": {"transmits": "force and moment", "kind": "grip", "over_m": 0.1}}')
    assert gate(session_with(tmp_path, table(rows={"A1": {"held": ["side grip"]}, "A2": {"held": ["side grip"]},
                                                   "A3": {"enters": "side grip"}}), contacts=grip_only)) == []


def test_the_tipping_exits_keep_the_rest_beside_the_grip():
    tips = {"contacts": ["bottom plate"], "side_force": 30.0, "tipping_force": 2.0}
    said = drive.tipping_exits(tips, {"contacts": {"bottom plate": {}}})
    assert "beside bottom plate, which stays a support" in said and "2 N" in said


# --------------------------------------------------------------------- the report --

def test_the_report_prints_the_use_table_and_the_coverage(tmp_path):
    limit = {"limit": "dynamic or impact", "why": "the carry is a jolt at the grip",
             "no_approximation": "the jolt has no stated factor"}
    session = session_with(tmp_path, table(rows={"A3": {"size": 400, "source": "40 kg x g x 1"}},
                                           uncovered={"side grip": limit, "bottom plate": limit}))
    brief = {"use_table": drive.table_report(session / "operator"),
             "coverage": drive.use_coverage(session / "operator")}
    text = drive.history_section([{"run": "r", "body": {}, "ends": "", "load_N": 1.0}], None, brief)
    assert "Use table A1 (working): push 200 N (H5 top), entering top plate" in text
    assert "at top plate (the Real section's member): the run state 'working' 75 N m" in text
    assert "not covered: at side grip state 'handling' gives 120 N m against the run's 75" in text


# ------------------------------------------------------------- on the regions --

def _rod_session(tmp_path, direction):
    """A 1 m round member along x held on its x = 0 end cap, one push A1 bound to a tip region at
    x = 1 m along ``direction``."""
    import numpy as np
    import trimesh
    session = session_with(tmp_path, table(rows={"A2": None, "A3": {"lever_m": 0.01}}))
    (session / "boundary_conditions").mkdir(parents=True, exist_ok=True)
    (session / "geometry").mkdir(exist_ok=True)
    rod = trimesh.creation.cylinder(radius=0.02, height=1.0, sections=32)
    rod.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    rod.apply_translation([0.5, 0.0, 0.0])
    rod.export(session / "geometry" / "physical.glb")
    np.savez(session / "boundary_conditions" / "resolved.npz", dirichlet_0=rod.face_normals[:, 0] < -0.9)
    lib.write_json(session / "boundary_conditions" / "resolved.json", {"regions": [
        {"name": "end", "key": "dirichlet_0", "category": "dirichlet", "centroid": [0.0, 0.0, 0.0]},
        {"name": "tip", "key": "neumann_0", "category": "neumann", "centroid": [1.0, 0.0, 0.0], "area": 1e-3}]})
    plan = {"loads": {"A1": {"regions": ["tip"]}}, "values": {"neumann_values": {"tip": {"force": direction}}},
            "scale": {"unit_system": "SI (m-kg-s)"}, "material": {"youngs_modulus": 2e11, "density": 7850}}
    return session, plan


def test_on_the_regions_the_lever_is_the_load_regions_distance_to_the_nearest_hold(tmp_path):
    session, plan = _rod_session(tmp_path, [0.0, -1.0, 0.0])
    [(label, lever)] = drive.region_levers(session, plan).items()
    assert label == "A1" and lever["lever_m"] == pytest.approx(1.0, abs=0.03)
    assert lever["arm_m"] == pytest.approx(1.0, abs=0.03) and lever["along_m"] < 0.03
    case = drive.runner_case(session, plan)
    # 200 N across the member 1 m from its hold, not on the 0.3 m Real extent the brief alone gives
    assert case["moment_Nm"] == pytest.approx(200.0, rel=0.03) and case["levers"] == "the regions"
    assert drive.runner_stress(session, plan)["stress_Pa"] == pytest.approx(200 * 0.005 / 8.3e-10, rel=0.03)
    assert case["deflection_m"] == pytest.approx(200 / (3 * 2e11 * 8.3e-10), rel=0.1)
    # along the member: a pull, not a bend
    session, plan = _rod_session(tmp_path / "pull", [1.0, 0.0, 0.0])
    case = drive.runner_case(session, plan)
    assert case["moment_Nm"] < 10 and case["axial_N"] == pytest.approx(200.0)


# ------------------------------------------------------------------ printing --

def test_the_coverage_is_printed_compactly_once_and_then_said_unchanged(tmp_path):
    limit = {"limit": "dynamic or impact", "why": "the carry is a jolt at the grip",
             "no_approximation": "the jolt has no stated factor"}
    session = session_with(tmp_path, table(rows={"A3": {"size": 400, "source": "40 kg x g x 1"}},
                                           uncovered={"side grip": limit, "bottom plate": limit}))
    view = drive.coverage_view(session)
    assert view == {"run": "working", "at top plate": "run 75 N m, largest 'working' 75",
                    "uncovered": {"side grip": "'handling' 120 vs 75, declared dynamic or impact",
                                  "bottom plate": "'handling' 120 vs 75, declared dynamic or impact"}}
    first = drive.compact_stop({"stop": "regions", "coverage": view}, session)
    assert first["coverage"] == view
    again = drive.compact_stop({"stop": "solve", "coverage": view}, session)
    assert again["coverage"] == drive.COVERAGE_UNCHANGED
    # changed: printed whole again, once
    session_with(tmp_path, TABLE)
    changed = drive.coverage_view(session)
    assert drive.compact_stop({"stop": "solve", "coverage": changed}, session)["coverage"] == changed
    assert drive.compact_stop({"stop": "solve", "coverage": changed}, session)["coverage"] == drive.COVERAGE_UNCHANGED
    # the whole record stays on disk
    assert lib.read_json(session / "operator" / drive.COVERAGE)["cases"]["working"]["top plate"] == 75.0


def test_a_new_agent_on_a_resumed_session_is_shown_the_advice_whole_once(tmp_path, monkeypatch):
    session = session_with(tmp_path, TABLE)
    view = drive.coverage_view(session)
    advice = {"stop": "regions", "coverage": view, "note": "a note long enough to be printed once a session, " * 2}
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "agent-1")
    assert drive.compact_stop(dict(advice), session)["note"] == advice["note"]
    again = drive.compact_stop(dict(advice), session)
    assert again["note"] == "as before" and again["coverage"] == drive.COVERAGE_UNCHANGED
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "agent-2")          # resumed by another agent
    fresh = drive.compact_stop(dict(advice), session)
    assert fresh["note"] == advice["note"] and fresh["coverage"] == view
    assert drive.compact_stop(dict(advice), session)["note"] == "as before"
    # the session id is the agent: a subagent's shared process id does not tell it apart
    monkeypatch.setenv("CLAUDE_PID", "4242")
    assert drive.agent_session() == "agent-2"


def test_the_runners_prediction_is_an_upper_bound_a_solve_far_under_it_agrees_with():
    runner = {"max_displacement": 1e-3, "max_stress": 1e8, "source": "the runner's"}
    history = [{"run": "r", "body": {"sha": "b", "size": 1.0}, "ends": "e", "load_N": 1.0, "table_loads": True,
                "prediction": runner, "solved": {"max_displacement": 5e-5, "max_stress": 3e6}}]
    agreed = drive.agreement(history)
    assert agreed["max_stress"]["statement"] == "agrees" and agreed["max_stress"]["bound"] is True
    assert drive.far_off_case(agreed) == {}                    # 0.03 x a conservative bound: no wrong case
    history[0]["solved"]["max_stress"] = 6e8                   # 6 x over it: a wrong case until shown otherwise
    assert list(drive.far_off_case(drive.agreement(history))) == ["max_stress"]
    # the agent's own prediction is held both ways, as before
    history[0]["prediction"] = {"max_displacement": 1e-3, "max_stress": 1e8}
    history[0]["solved"]["max_stress"] = 3e6
    assert "max_stress" in drive.far_off_case(drive.agreement(history))


def test_a_human_actions_row_far_over_the_objects_weight_is_warned(tmp_path):
    light = session_with(tmp_path, TABLE)                       # Real mass 1 kg: 200 N and 50 N of person
    said = drive.load_scale_warning(light)
    assert said.startswith("check the row against the use") and "A1 (H5 top)" in said and "A2" in said
    assert "A3" not in said                                    # a weight row is not a Human actions row
    heavy = session_with(tmp_path, TABLE, size=SIZE.replace("Real mass: 1 kg", "Real mass: 100 kg"), name="heavy")
    assert drive.load_scale_warning(heavy) is None


def test_a_body_once_warning_is_said_at_the_first_stop_of_each_body_only(tmp_path, monkeypatch):
    session = session_with(tmp_path, TABLE)
    body = {"digest": "body1"}
    monkeypatch.setattr(drive, "body_digest", lambda s: body["digest"])
    found = ["check the row against the use: A1", "member likely absent: A1 enters 'part b'. " + drive.ENTRY_FORM,
             "held pieces of plural contacts lie on segments", "another warning"]
    assert drive.compact_stop({"stop": "regions", "warnings": list(found)}, session)["warnings"] == found
    assert drive.compact_stop({"stop": "regions", "warnings": list(found)}, session)["warnings"] == ["another warning"]
    assert drive.compact_stop({"stop": "solve", "warnings": found[:1]}, session)["warnings"] == []
    body["digest"] = "body2"                                     # a new body says them again
    assert drive.compact_stop({"stop": "regions", "warnings": list(found)}, session)["warnings"] == found


def test_the_selector_trees_are_printed_once_each_time_they_change(tmp_path):
    session = session_with(tmp_path, TABLE)
    tree = {"base": {"type": "slab", "params": {"axis": "y", "min_val": 0.0, "max_val": 0.1}}}
    stop = lambda t: {"stop": "regions", "synthesized_selectors": t}  # noqa: E731
    assert drive.compact_stop(stop(tree), session)["synthesized_selectors"] == tree
    assert drive.compact_stop(stop(tree), session)["synthesized_selectors"] == drive.SELECTORS_UNCHANGED
    wider = {"base": {"type": "slab", "params": {"axis": "y", "min_val": 0.0, "max_val": 0.2}}}
    assert drive.compact_stop(stop(wider), session)["synthesized_selectors"] == wider


def test_a_tilted_generated_body_standing_on_rests_is_warned_and_listed(tmp_path, monkeypatch):
    import numpy as np
    import trimesh
    session = session_with(tmp_path, TABLE)
    body = session / "body.glb"
    monkeypatch.setattr(drive, "body_source", lambda s: body)
    monkeypatch.setattr(drive, "body_provenance", lambda s: "triposg")
    monkeypatch.setattr(drive, "body_digest", lambda s: "body1")
    box = trimesh.creation.box(extents=[0.3, 0.1, 0.2])
    box.export(body)
    assert drive.body_tilt(session, {}) < 1.0 and drive.tilt_warning(session, {}) is None
    box.apply_transform(trimesh.transformations.rotation_matrix(np.radians(20.0), [0, 0, 1]))
    box.export(body)
    assert abs(drive.body_tilt(session, {}) - 20.0) < 1.0
    said = drive.tilt_warning(session, {})
    assert "20 degrees off its up axis" in said and "cannot rotate" in said
    assert any("tilted 20.0 degrees" in line for line in drive.runner_simplifications(session, {}, []))
    # a level base under a tilted overhang: only the lowest faces are the base
    level = trimesh.creation.box(extents=[0.3, 0.1, 0.2])
    arm = trimesh.creation.box(extents=[0.3, 0.05, 0.2])
    arm.apply_transform(trimesh.transformations.rotation_matrix(np.radians(30.0), [0, 0, 1]))
    arm.apply_translation([0.0, 0.3, 0.0])
    trimesh.util.concatenate([level, arm]).export(body)
    assert drive.body_tilt(session, {}) < 1.0
    # a built body is the agent's own: not measured
    monkeypatch.setattr(drive, "body_provenance", lambda s: "built")
    assert drive.tilt_warning(session, {}) is None


def test_a_governing_member_under_a_leaf_at_the_deepest_depth_is_said_once_with_the_routes_way(tmp_path, monkeypatch):
    size = SIZE.replace("0.01 m deep, I 8.3e-10 m4, A 1e-4 m2", "rectangle 0.02 x 0.002 m")
    resolution = {"physical_extents": [1.0, 0.5, 0.3]}                 # a leaf of 7.5 mm at depth 8
    built = session_with(tmp_path, TABLE, size=size, name="built")
    monkeypatch.setattr(drive, "body_provenance", lambda s: "built")
    said = drive.governing_unresolved_warning(built, {}, resolution)
    assert "governing member (top plate) is 0.002 m" in said and "equivalent-stiffness section" in said
    # once a body: the compact view leaves it out of the same body's later stops
    monkeypatch.setattr(drive, "body_digest", lambda s: "body1")
    assert drive.compact_stop({"stop": "geometry", "warnings": [said]}, built)["warnings"] == [said]
    assert drive.compact_stop({"stop": "geometry", "warnings": [said]}, built)["warnings"] == []
    made =session_with(tmp_path, TABLE, size=size, name="made")
    monkeypatch.setattr(drive, "body_provenance", lambda s: "triposg")
    monkeypatch.setattr(drive, "route_tried", lambda s, r, c=None: r == "triposg_text")
    plan = {"geometry_routes": ["triposg_text", "drawing", "web_image"]}
    said = drive.governing_unresolved_warning(made, plan, resolution)
    # the real section is the thin one: no other generated route, the web photograph then built
    assert "drawing" not in said and said.index("the web photograph") < said.index("the built body")
    monkeypatch.setattr(drive, "route_tried", lambda s, r, c=None: r in ("triposg_text", "web_image"))
    said = drive.governing_unresolved_warning(made, plan, resolution)
    assert "web photograph" not in said and "the built body at the equivalent-stiffness section" in said
    # a member that resolves says nothing
    fine = session_with(tmp_path, TABLE, size=SIZE.replace("0.01 m deep, I 8.3e-10 m4, A 1e-4 m2",
                                                           "rectangle 0.02 x 0.01 m"), name="fine")
    assert drive.governing_unresolved_warning(fine, plan, resolution) is None


def test_a_lower_bound_claim_far_over_the_runners_estimate_needs_its_cause():
    runner = {"max_displacement": 1e-3, "max_stress": 1e8, "source": "the runner's"}
    history = [{"run": "r", "body": {"sha": "b", "size": 1.0}, "ends": "e", "load_N": 1.0, "table_loads": True,
                "prediction": runner, "solved": {"max_displacement": 2e-2, "max_stress": 3e6}}]
    agreed = drive.agreement(history)
    said = drive.lower_bound_problem("The result disagrees, 20 x, cause not found; a lower bound.", agreed)
    assert said and "20.0 x" in said
    assert drive.lower_bound_problem("... a lower bound, cause found: the load doubled", agreed) is None
    assert drive.lower_bound_problem("The result disagrees, 20 x, cause not found.", agreed) is None
    # the claim is a phrase: a negated one is no claim, and a cause that names nothing is no cause
    assert drive.lower_bound_problem("20 x over it: this is not a lower bound.", agreed) is None
    assert drive.lower_bound_problem("a slower boundary layer", agreed) is None
    assert drive.lower_bound_problem("a lower-bound result, cause found: none", agreed)
    assert drive.lower_bound_problem("a lower bound; cause found: unknown", agreed)
    history[0]["solved"]["max_displacement"] = 5e-3            # 5 x: under the claim's threshold
    assert drive.lower_bound_problem("a lower bound", drive.agreement(history)) is None


# ------------------------------------------------------- paths, levers, sizes --

def _unpinned(**rows):
    out = table(rows=rows)
    for row in out["actions"].values():
        row.pop("lever_m", None)
    return out


def test_a_rows_path_and_lever_come_from_its_entry_and_holds(tmp_path):
    built, problems = drive.use_table(session_with(tmp_path, _unpinned()) / "operator")
    assert problems == []
    # A1 enters the top plate, held at the side grip and the bottom plate: every member on its path,
    # its lever the nearest hold's distance along the path (1 of 2 members x 0.3 m)
    assert built["paths"]["A1"] == ["top plate", "side grip", "bottom plate"]
    assert built["effects"]["A1"]["top plate"] == pytest.approx(200 * 0.15)
    # A3 enters the bottom plate, held at the side grip: the two members between, the same lever
    assert built["paths"]["A3"] == ["side grip", "bottom plate"]
    # a row held on the member it enters is half a member step from its hold, never 0 (naming a hold
    # on the entry member clears nothing); only an explicit lever_m 0 says it sits at its hold
    built, _ = drive.use_table(session_with(tmp_path, _unpinned(A3={"enters": "side grip"})) / "operator")
    assert built["effects"]["A3"] == {"side grip": pytest.approx(100 * 0.3 * 0.5 / 2)}
    at_hold = _unpinned(A3={"enters": "side grip"})
    at_hold["actions"]["A3"].update(lever_m=0, lever_why="it enters at its hold")
    built, _ = drive.use_table(session_with(tmp_path, at_hold) / "operator")
    assert built["effects"]["A3"] == {"side grip": 0.0}
    # a one-member load path: a force row keeps a lever (half the extent), as a twist keeps its N m
    one = SIZE.replace('{"top plate": "series", "side grip": "series", "bottom plate": "series"}',
                       '{"top plate": "series"}')
    held = {"held": ["top plate"]}
    single = CONTACTS.replace('"bottom plate": {', '"top plate": {"transmits": "force only", "kind": "rest"}, '
                              '"spare": {')
    built, _ = drive.use_table(session_with(tmp_path, _unpinned(A1=held, A2=held, A3=held), contacts=single,
                                            size=one) / "operator")
    assert built["effects"]["A1"]["top plate"] == pytest.approx(200 * 0.15)
    # the Load path's list order does not matter
    reversed_size = SIZE.replace('{"top plate": "series", "side grip": "series", "bottom plate": "series"}',
                                 '{"bottom plate": "series", "side grip": "series", "top plate": "series"}')
    turned, _ = drive.use_table(session_with(tmp_path, _unpinned(), size=reversed_size) / "operator")
    assert sorted(turned["paths"]["A1"]) == sorted(["top plate", "side grip", "bottom plate"])
    assert turned["effects"]["A1"]["top plate"] == pytest.approx(30.0)
    # a hold on no member of the path: every member, half the largest extent
    built, _ = drive.use_table(session_with(tmp_path, _unpinned(A3={"held": ["a strap"]})) / "operator")
    assert built["paths"]["A3"] == list(built["members"]) and built["effects"]["A3"]["top plate"] == pytest.approx(15.0)
    # a contact's own "member" names the member it holds
    placed = CONTACTS.replace('"bottom plate": {"transmits": "force only", "kind": "rest"}',
                              '"bottom plate": {"transmits": "force only", "kind": "rest", "member": "side grip"}')
    built, _ = drive.use_table(session_with(tmp_path, _unpinned(A1={"held": ["bottom plate"]}), contacts=placed)
                               / "operator")
    assert built["paths"]["A1"] == ["top plate", "side grip"]


def test_the_real_sections_member_is_declared_under_contact_only_for_a_contact(tmp_path):
    contact = {"limit": "contact between separate bodies", "why": "a separate part presses it",
               "no_approximation": "its force is unknown"}
    over = {"A3": {"size": 400, "source": "40 kg x g x 1", "enters": "top plate", "held": ["side grip"]}}
    said = gate(session_with(tmp_path, table(rows=over, uncovered={"top plate": contact})))
    assert any("it is the Real section's member, and no separate body touches it there" in s for s in said)
    # a load passed through contact by a separate body that is no hold: a contact on the member it enters
    pressed = CONTACTS.replace('"side grip": {', '"pressing part": {"transmits": "force only", "kind": "rest", '
                               '"member": "top plate"}, "side grip": {')
    said = gate(session_with(tmp_path, table(rows=over, uncovered={"top plate": contact}), contacts=pressed))
    assert not any("Real section's member" in s for s in said)
    # held by a rest on that member: a separate body presses there, and the declaration stands
    rest_on_top = CONTACTS.replace('"bottom plate": {"transmits": "force only", "kind": "rest"}',
                                   '"bottom plate": {"transmits": "force only", "kind": "rest", "member": "top plate"}')
    over["A3"]["held"] = ["bottom plate"]
    said = gate(session_with(tmp_path, table(rows=over, uncovered={"top plate": contact}), contacts=rest_on_top))
    assert not any("Real section's member" in s for s in said)


def test_the_runner_sizes_a_row_its_source_fixes(tmp_path):
    rows = {"A1": {"size": None, "unit": None},                                   # H5 top: 300 N
            "A2": {"size": None, "unit": None, "source": "H3 middle"},            # 150 N
            "A3": {"size": None, "unit": None, "source": "10 kg x g x 1.5"}}
    built, problems = drive.use_table(session_with(tmp_path, table(rows=rows)) / "operator")
    assert problems == []
    assert built["sizes"]["A1"] == (300.0, "N") and built["sizes"]["A2"] == (150.0, "N")
    assert built["sizes"]["A3"][0] == pytest.approx(10 * 9.80665 * 1.5)
    # a dynamic Human actions row takes a stated factor, at most the cap
    built, _ = drive.use_table(session_with(tmp_path, table(rows={"A3": {"size": None, "unit": None,
                                                                         "source": "H9 top with factor 2"}}))
                               / "operator")
    assert built["sizes"]["A3"] == (147.0 * 2, "N")
    # a factor on a row that takes none is refused, never dropped
    said = gate(session_with(tmp_path, table(rows={"A1": {"size": None, "unit": None,
                                                          "source": "H5 top with factor 1.5"}})))
    assert any("H5 takes no dynamic factor" in s and 'write "H5 top"' in s for s in said)
    # "as A1" beside A1's runner-filled size: compared with that size
    as_filled = {"A1": {"size": None, "unit": None}, "A2": {"size": 300, "unit": "N", "source": "as A1"}}
    assert not any("as A1" in s for s in gate(session_with(tmp_path, table(rows=as_filled))))
    said = gate(session_with(tmp_path, table(rows={**as_filled, "A2": {"size": 250, "source": "as A1"}})))
    assert any("gives A1's, 300" in s for s in said)
    # a size written beside a weight is checked against it
    said = gate(session_with(tmp_path, table(rows={"A3": {"size": 120, "source": "10 kg x g x 1"}})))
    assert any('"size" 120 is not its source' in s for s in said)
    # a derivation is the runner's arithmetic too: its formula, in numbers and units, gives the size
    assert drive.sized_by_source("derived: 30 kg x g x 0.3 / 1.25", {})[0] == pytest.approx(30 * 9.80665 * 0.3 / 1.25)
    assert drive.derived_value("derived: 2 kPa x 300 cm2") == pytest.approx(60.0)
    assert drive.derived_value("derived: 4 N m") == 4.0 and drive.derived_value("derived: 12") == 12.0
    assert drive.derived_value("derived: the push") is None
    # compound units, powers, parentheses, speeds and a leading minus
    assert drive.derived_value("derived: 1000 kg/m3 x 9.80665 m/s2 x 2 m") == pytest.approx(19613.3)
    assert drive.derived_value("derived: 2 N/mm2 x 100 mm2") == pytest.approx(200.0)
    assert drive.derived_value("derived: 0.5 x 1.2 kg/m3 x (36 km/h)^2 x 2 m2") == pytest.approx(120.0)
    assert drive.derived_value("derived: 0.5 x 2 ** 3") == pytest.approx(4.0)
    assert drive.derived_value("derived: -3 N + 5 N") == pytest.approx(2.0)
    # a number with its units is one term: a unit after a division divides that term
    assert drive.derived_value("derived: 100 N / 2 mm2") == pytest.approx(5e7)
    assert drive.derived_value("derived: 3 kN / (2 cm x 5 mm)") == pytest.approx(3e7)
    assert drive.derived_value("derived: 0.2 MPa / 4") == pytest.approx(5e4)
    assert drive.derived_value("derived: 500 g") is None                     # no grams: g is gravity
    assert drive.sized_by_source("30 kg x g x 0.3 / 1.25", {}) is None        # a weight is the source itself
    built, problems = drive.use_table(session_with(tmp_path, table(rows={"A2": {
        "size": None, "source": "derived: 2 kPa x 300 cm2"}})) / "operator")
    assert problems == [] and built["sizes"]["A2"] == (pytest.approx(60.0), "N")
    said = gate(session_with(tmp_path, table(rows={"A2": {"size": 50, "source": "derived: 2 kPa x 300 cm2"}})))
    assert any('"size" 50 is not its source' in s for s in said)
    said = gate(session_with(tmp_path, table(rows={"A2": {"size": 50, "source": "derived: a light pull"}})))
    assert any("a derivation the runner cannot evaluate" in s for s in said)


def test_the_measured_arm_replaces_a_rows_lever_with_a_note(tmp_path):
    session, plan = _rod_session(tmp_path, [0.0, -1.0, 0.0])
    lib.write_json(session / "operator" / "declared.json", plan)
    brief = session / "operator" / "brief.md"
    brief.write_text(brief.read_text().replace('"lever_m": 0.3', '"lever_m": 0.05', 1))
    # another state's force row has no regions: every state keeps the Load path's levers, never a mix
    coverage = drive.use_coverage(session / "operator")
    assert coverage["levers"] == "the Load path" and coverage["cases"]["working"]["top plate"] == pytest.approx(10.0)
    assert "used once the regions measure every state's rows" in coverage["lever_notes"][0]
    # the regions measure every force row: the measured arm for all
    text = brief.read_text()
    use = json.loads(text.split("Use table: ", 1)[1].split("\n", 1)[0])
    use["actions"].pop("A3")
    brief.write_text(text.replace(text.split("Use table: ", 1)[1].split("\n", 1)[0], json.dumps(use)))
    coverage = drive.use_coverage(session / "operator")
    assert coverage["levers"] == "the regions"
    assert coverage["cases"]["working"]["top plate"] == pytest.approx(200.0, rel=0.05)      # 200 N x the 1 m arm
    assert coverage["lever_notes"] and "lever_m 0.05 m against the regions'" in coverage["lever_notes"][0]


def test_a_run_along_the_member_is_compared_by_stress_not_by_its_zero_moment(tmp_path):
    session, plan = _rod_session(tmp_path, [1.0, 0.0, 0.0])              # a pull along the member
    lib.write_json(session / "operator" / "declared.json", plan)
    brief = session / "operator" / "brief.md"
    text = brief.read_text()
    line = text.split("Use table: ", 1)[1].split("\n", 1)[0]
    use = json.loads(line)
    use["actions"]["A3"] = {"load": "twist", "state": "handling", "kind": "applied", "size": 0.1, "unit": "N m",
                            "source": "derived: 1 N x 0.1 m", "enters": "top plate", "held": ["side grip"]}
    brief.write_text(text.replace(line, json.dumps(use)))
    coverage = drive.use_coverage(session / "operator")
    # 200 N / 1e-4 m2 = 2 MPa against the twist's 0.1 N m x 0.005 m / 8.3e-10 m4 = 0.6 MPa: covered
    assert coverage["unit"] == "MPa" and coverage["direct"] == ["A1"] and coverage["uncovered"] == {}
    assert coverage["run_effect"] == pytest.approx(2.0, rel=0.05)
    assert gate(session) == []


def test_a_varying_pressure_is_shared_among_its_bands_by_the_stated_profile(tmp_path):
    # rising linearly from 0 at the low end to its peak at the high end, along y: the runner shares it
    varies = {"along": [0, 1, 0], "from": 0, "to": 1}
    session = session_with(tmp_path, table(rows={"A2": None, "A1": {"direction": [1, 0, 0], "varies": varies}}))
    assert gate(session) == []
    bands = [{"name": f"band{k}", "category": "neumann", "centroid": [0, k + 0.5, 0], "area": 1.0,
              "bounds": [[0, k, 0], [0, k + 1, 0]]} for k in range(3)]
    plan = {"loads": {"A1": {"regions": ["band0", "band1", "band2"]}}}
    out, said, problem = drive.use_table_values(session, plan, bands)
    forces = [out["values"]["neumann_values"][f"band{k}"]["force"][0] for k in range(3)]
    # intensities 1/6, 3/6, 5/6 of the peak: shares 1/9, 3/9, 5/9 of the row's 200 N
    assert problem is None and forces == pytest.approx([200 / 9, 200 * 3 / 9, 200 * 5 / 9])
    assert "shared by its profile" in said
    # a profile out of its form is refused at G1
    bad = gate(session_with(tmp_path, table(rows={"A1": {"varies": {"along": [0, 1, 0], "from": -1, "to": 1}}})))
    assert any('"varies" is {"along"' in s for s in bad)


def test_a_turned_body_takes_each_rows_direction_in_its_own_frame(tmp_path):
    session = session_with(tmp_path, table(rows={"A2": None, "A1": {"direction": [0, -1, 0]}}))
    # the agent states the body axis that is the use's up; the runner turns rows and gravity alike
    plan = {"body_up": "+z", "loads": {"A1": {"regions": ["top"]}}, "values": {"acceleration": [0, -9.80665, 0]}}
    turn = drive.use_to_body(session, plan)
    assert turn == ((0, 1), (2, 1), (1, -1)) and drive.turned_direction([0, -1, 0], turn) == [0.0, 0.0, -1.0]
    assert drive.body_vector(plan) == [0.0, 0.0, -9.80665]
    regions = [{"name": "top", "category": "neumann", "centroid": [0, 0, 0]}]
    out, said, problem = drive.use_table_values(session, plan, regions)
    assert problem is None and out["values"]["neumann_values"]["top"]["force"] == [0.0, 0.0, -200.0]
    assert "row directions and the acceleration turned into the body's frame" in said
    # the plan keeps the use frame's acceleration: the turn is applied where it meets the body
    assert out["values"]["acceleration"] == [0, -9.80665, 0]
    # a body as the brief stands is not turned
    assert drive.use_to_body(session, {}) is None and drive.use_to_body(session, {"body_up": "+y"}) is None


def test_a_tipping_or_reaction_refusal_takes_the_contact_it_names_without_a_new_body(tmp_path, monkeypatch):
    rest_only = '- bottom plate: rests\n{"bottom plate": {"transmits": "force only", "kind": "rest"}}'
    rows = {"A1": {"held": ["bottom plate"]}, "A2": {"held": ["bottom plate"]}}
    session = session_with(tmp_path, table(rows=rows), contacts=rest_only)
    operator = session / "operator"
    body = tmp_path / "body.glb"
    body.write_bytes(b"x" * 10)
    monkeypatch.setattr(drive, "body_source", lambda s: body)
    assert sorted(drive.brief_contacts_v2(operator)) == ["bottom plate"]           # frozen with the body
    # a grip added beside the rest: refused as a change after the body ...
    session_with(tmp_path, table(rows={"A1": {"held": ["bottom plate", "side grip"]}, "A2": {"held": ["bottom plate"]}}))
    assert drive.contacts_frozen_problems(operator)
    # ... until a tipping refusal asks for one: then taken on this body and frozen again
    drive.allow_contact_additions(operator, drive.ADD_FOR_TIPPING)
    assert sorted(drive.brief_contacts_v2(operator)) == ["bottom plate", "side grip"]
    assert drive.contacts_frozen_problems(operator) == []
    record = lib.read_json(session / "brief_snapshots" / drive.CONTACTS_AT_BODY)
    assert not drive._additions(record) and "side grip" in record["edited"]
    # one addition per refusal: a second contact added after it is a change after the body
    two = ('- bottom plate: rests\n{"bottom plate": {"transmits": "force only", "kind": "rest"}, '
           '"side grip": {"transmits": "force and moment", "kind": "grip", "over_m": 0.1}, '
           '"hand": {"transmits": "force and moment", "kind": "grip", "over_m": 0.1}, '
           '"strap": {"transmits": "force only", "kind": "rest"}}')
    held = {"held": ["bottom plate", "side grip", "hand", "strap"]}
    session_with(tmp_path, table(rows={"A1": held, "A2": {"held": ["bottom plate"]}}), contacts=two)
    assert drive.contacts_frozen_problems(operator)
    drive.allow_contact_additions(operator, drive.ADD_FOR_TIPPING)
    assert sorted(drive.brief_contacts_v2(operator)) == ["bottom plate", "hand", "side grip"]
    assert drive.contacts_frozen_problems(operator)                       # the strap: not this refusal's kind
    # a reaction-cap refusal takes no rest or hang that only spreads the reaction
    drive.allow_contact_additions(operator, drive.ADD_FOR_REACTION["other"], "reaction")
    assert "strap" not in drive.brief_contacts_v2(operator)
    # the permission expires when its refusal clears another way, or at a solve
    drive.revoke_contact_additions(operator, "tipping")
    record_now = lambda: lib.read_json(session / "brief_snapshots" / drive.CONTACTS_AT_BODY)  # noqa: E731
    assert drive._additions(record_now()) == {"reaction": sorted(drive.ADD_FOR_REACTION["other"])}
    # a second standing refusal keeps the first's kinds: the union is allowed
    drive.allow_contact_additions(operator, drive.ADD_FOR_SLIDING, "sliding")
    assert set(drive._additions(record_now())) == {"reaction", "sliding"}
    drive.revoke_contact_additions(operator, "sliding")
    drive.revoke_contact_additions(operator, "reaction")
    assert not drive._additions(record_now())
    drive.allow_contact_additions(operator, drive.ADD_FOR_TIPPING)
    drive.revoke_contact_additions(operator)
    assert not drive._additions(lib.read_json(session / "brief_snapshots" / drive.CONTACTS_AT_BODY))
    # a fastening taken for tipping takes a moment over a length; a spot is no exit
    assert not drive.addable({"kind": "fastening", "transmits": "force only"}, drive.ADD_FOR_TIPPING)
    assert drive.addable({"kind": "fastening", "transmits": "force and moment", "over_m": 0.1}, drive.ADD_FOR_TIPPING)
    session_with(tmp_path, table(rows={"A1": {"held": ["bottom plate", "side grip"]}, "A2": {"held": ["bottom plate"]}}))
    extra = drive.thin_declaration(session, {}, summary={})
    line = drive.history_section([{"run": "r", "body": {}, "ends": "", "load_N": 1.0}], None, extra)
    assert "contacts taken after the body was made, at a gate's request: side grip" in line
    # its own line: a part declared off the load path is still reported beside it
    line = drive.history_section([{"run": "r", "body": {}, "ends": "", "load_N": 1.0}], None,
                                 {"thin_off_load_path": "a thin part", "contacts_edited_after_body": ["side grip"]})
    assert "declared off the load path: a thin part" in line and "at a gate's request: side grip" in line
