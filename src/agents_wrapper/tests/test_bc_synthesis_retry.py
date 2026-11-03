"""A region synthesis that returns no support for a request that holds the body is asked again.

In a held-out run (2026-10-10) four previews of one body failed with "at least one Dirichlet
BC is required" although every request opened with "Fix ...": the held face was named by the
direction it faces, and the model made it a load.  The repair call only sees the answer, so it
returned the same; the question is asked once more with the holding clauses named.
"""

import json

import pytest

from src.agents_wrapper import GenerationResponse, TokenUsage
from src.agents_wrapper.fem_workflow import specialists as specialists_module
from src.agents_wrapper.fem_workflow.specialists import SpecialistCalls, hold_clauses

SELECTOR = {"type": "surface_segment", "params": {"segment_ids": [0]}, "children": []}
ALL_LOADS = {"dirichlet": [], "neumann": [
    {"name": "held_face", "selector": SELECTOR, "traction": [0, 0, 1]},
    {"name": "tip", "selector": SELECTOR, "traction": None}]}
FIXED = {"dirichlet": [{"name": "held_face", "selector": SELECTOR, "value": 0.0, "components": [0, 1, 2]}],
         "neumann": [{"name": "tip", "selector": SELECTOR, "traction": None}]}
REQUEST = ("Fix all components of the middle part: its face toward +z, between 0.4 m and 0.6 m. "
           "Apply a traction toward +z on the tip.")


class ScriptedAgent:
    model = "fake-model"

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def generate_structured(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.answers.pop(0), GenerationResponse("{}", TokenUsage(10, 4, 14))


def _calls(tmp_path, agent):
    calls = object.__new__(SpecialistCalls)
    calls.session_dir = tmp_path
    calls.cache_dir = tmp_path / "api_cache"
    calls.cache_dir.mkdir()
    calls.provider = "groq"
    calls.agent = agent
    calls._seen = lambda purpose, image, legend=None: {}
    return calls


def test_the_holding_clauses_of_a_request_are_read_by_their_first_word():
    assert hold_clauses(REQUEST) == ["Fix all components of the middle part: its face toward +z, "
                                     "between 0.4 m and 0.6 m."]
    assert hold_clauses("Clamp the base.\nPush down on the top") == ["Clamp the base."]
    assert hold_clauses("Apply a traction on the top; the load is supported by the hand.") == []
    # a sentence that goes on to a load is named by its hold part only
    assert hold_clauses("Fix the base and push 100 N down on the top.") == ["Fix the base"]
    assert hold_clauses("Clamp the left end, then apply a traction in -y on the right end.") == ["Clamp the left end"]
    assert hold_clauses("Hold the rim, and pull the hook up. Fix the foot.") == ["Hold the rim", "Fix the foot."]
    # "and" between two held parts is no load
    assert hold_clauses("Fix the base and the back face.") == ["Fix the base and the back face."]
    # a region named by a load word is held, not a load: the verb takes an object or a direction
    for held in ("Fix the base and pull ring.", "Clamp the frame, loading plate and pressing face.",
                 "Hold the handle and push rod faces toward +z.", "Fix the base, load cell and press plate."):
        assert hold_clauses(held) == [held]
    assert hold_clauses("Fix the base and pull the ring up.") == ["Fix the base"]
    assert hold_clauses("Fix the base, then load it with 50 N.") == ["Fix the base"]
    assert hold_clauses("Fix the base and presses down on the lid.") == ["Fix the base"]


def test_a_proposal_with_no_support_is_asked_again_with_the_holds_named(tmp_path):
    # first answer all loads, the repair returns the same, the corrected question gets a support
    agent = ScriptedAgent([ALL_LOADS, ALL_LOADS, FIXED])
    proposal, _ = _calls(tmp_path, agent).synthesize_bcs(REQUEST, {"segments": []}, None)
    assert [item.name for item in proposal.dirichlet] == ["held_face"]
    assert len(agent.prompts) == 3
    asked = json.loads(agent.prompts[2])
    assert asked["request"] == REQUEST and "held_face" not in asked["correction"]
    assert "Fix all components of the middle part" in asked["correction"]
    # both failures and the answer are in the session's transcript
    log = [json.loads(line) for line in (tmp_path / "api_log.jsonl").read_text().splitlines()]
    assert [bool(row.get("error")) for row in log] == [True, False]


def test_a_request_that_holds_nothing_is_not_asked_again(tmp_path):
    from pydantic import ValidationError
    agent = ScriptedAgent([ALL_LOADS, ALL_LOADS])
    with pytest.raises(ValidationError, match=specialists_module.NO_DIRICHLET):
        _calls(tmp_path, agent).synthesize_bcs("Apply a traction toward +z on the tip.", {"segments": []}, None)
    assert len(agent.prompts) == 2              # the answer and its repair, no third call


def test_the_selector_prompt_says_a_hold_is_a_support_whatever_its_face_points():
    from pathlib import Path
    text = (Path(specialists_module.__file__).parent / "skill" / "references"
            / "boundary_conditions_api.md").read_text()
    assert "A fix, hold or clamp clause is\nDirichlet even when it names where its face points" in text


# A held-out run (2026-10-10): requests of 6 to 14 regions all came back cut at the 768-token
# budget, and each preview failed as "not a single JSON value".
BANDS = ("Fix all components of the base underside. "
         + " ".join(f"Apply a traction in +z on the wall faces between y = {k / 10:.1f} and {(k + 1) / 10:.1f} m."
                    for k in range(8)))


class BudgetAgent:
    """Answers within its budget, or is cut off at it (finish_reason "length")."""
    model = "fake-model"

    def __init__(self, needs):
        self.needs = needs
        self.budgets = []
        self.prompts = []

    def generate_structured(self, prompt, **kwargs):
        from src.agents_wrapper.errors import APIResponseError
        budget = kwargs["max_output_tokens"]
        self.budgets.append(budget)
        self.prompts.append(prompt)
        if budget < self.needs:
            cut = GenerationResponse('{"dirichlet":[{"na', TokenUsage(10, budget, 10 + budget), finish_reason="length")
            raise APIResponseError("the API did not return a single JSON value", raw_text=cut.text, response=cut)
        return FIXED, GenerationResponse("{}", TokenUsage(10, 400, 410), finish_reason="stop")


def test_the_regions_a_request_names_are_counted():
    from src.agents_wrapper.fem_workflow.specialists import requested_regions
    assert requested_regions(BANDS) == 9
    assert requested_regions("Fix the base and push 100 N down on the top.") == 2
    assert requested_regions("Fix the bottom face at y = 0.03 m; apply a traction on the top.") == 2
    assert requested_regions("") == 1


def test_the_synthesis_budget_scales_with_the_regions_and_has_a_ceiling():
    from src.agents_wrapper.fem_workflow.specialists import MAX_OUTPUT_TOKENS, bc_synthesis_budget
    few, many = bc_synthesis_budget(REQUEST), bc_synthesis_budget(BANDS)
    assert few >= 768 and many > few
    assert many >= 200 * 9                       # room for every tree of the nine regions
    assert bc_synthesis_budget(BANDS * 10) == MAX_OUTPUT_TOKENS


# The fixture's model may think, so every budget carries the stage's reasoning headroom.
HEADROOM = specialists_module.REASONING_HEADROOM_BY_PURPOSE["bc_synthesis"]


def test_a_long_region_request_gets_the_room_it_needs(tmp_path):
    agent = BudgetAgent(needs=HEADROOM + 2000)   # 2000 answer tokens: more than the old fixed 768
    proposal, _ = _calls(tmp_path, agent).synthesize_bcs(BANDS, {"segments": []}, None)
    assert [item.name for item in proposal.dirichlet] == ["held_face"]
    assert agent.budgets == [HEADROOM + specialists_module.bc_synthesis_budget(BANDS)]


def test_a_cut_reply_is_asked_again_with_more_room_never_repaired(tmp_path):
    first = HEADROOM + specialists_module.bc_synthesis_budget(REQUEST)
    agent = BudgetAgent(needs=first + 1)
    _calls(tmp_path, agent).synthesize_bcs(REQUEST, {"segments": []}, None)
    assert agent.budgets[0] == first and agent.budgets[1] > first and len(agent.budgets) == 2
    assert all("repair_json" not in prompt for prompt in agent.prompts)


def test_a_reply_cut_off_again_says_the_request_is_too_long(tmp_path):
    from src.agents_wrapper.errors import APIResponseError
    agent = BudgetAgent(needs=10 ** 6)
    with pytest.raises(APIResponseError, match="the region request is too long for one reply; merge regions "
                                               "or describe fewer"):
        _calls(tmp_path, agent).synthesize_bcs(BANDS, {"segments": []}, None)
    assert len(agent.budgets) == 2 and agent.budgets[-1] - HEADROOM <= specialists_module.MAX_OUTPUT_TOKENS
    log = [json.loads(line) for line in (tmp_path / "api_log.jsonl").read_text().splitlines()]
    assert "too long for one reply" in log[-1]["error"]
