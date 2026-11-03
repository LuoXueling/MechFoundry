"""Minimal, cached API calls used at reasoning-heavy workflow stages."""

from __future__ import annotations

import colorsys
import hashlib
import json
import time
from pathlib import Path
from typing import Any, TypeVar

from PIL import Image
from pydantic import BaseModel, ValidationError

from src.agents_wrapper import (
    APIResponseError, AgentsWrapper, DeepSeekAPI, GeminiAPI, GroqAPI,
)

from .models import (
    APIReceipt, BCProposal, BCPromptRefinement, BCValueInference,
    CADDescription, GeometryRefinement, ImageDescription, MaterialInference, PartMaterials,
    PhysicalScaleInference,
    UnloadedBCValueInference,
)

# Evidence images are read only for identity and orientation, so they are
# downscaled per purpose rather than sent at the source resolution.
# A reasoning model spends output tokens thinking before it emits the JSON, and that
# spend counts against max_output_tokens.  Without headroom the budget is exhausted
# before any answer appears: the reply comes back empty with finish_reason "length",
# and the provider rejects the empty generation.
#
# Granted by role once, on the assumption that only the text stage reasons.  That was
# wrong as soon as a reasoning model could read pictures too -- deepseek-flash spends
# its whole budget thinking about the image and returns nothing.  So it now follows
# the model, and is granted by default: a reasoning model left without it fails,
# while a model that does not reason only loses some output-per-minute quota to a
# ceiling it never reaches.  List one here when that quota is worth protecting.
REASONING_HEADROOM = 1024
NON_REASONING_MODELS = frozenset({"qwen/qwen3.8-27b"})
# Providers whose API documents a dial for how much a model thinks.
REASONING_OPTIONAL_PROVIDERS = frozenset({"deepseek"})
# Measured per stage on deepseek-flash, not chosen on taste.  Thinking is billed
# as output, and output costs 4x a cache-missed input token and 200x a cached
# one, so it is the expensive half of a call and has to earn its place.  Mostly
# it did not.  Physical scale and material returned character-identical answers
# for 3-4x the output tokens.  Geometry refinement spent 6x to say the same
# thing at greater length.  Region synthesis cost ~5350 tokens against ~142,
# burned the whole budget on three of four first attempts and returned nothing,
# and named the base by segment in two runs where "none" managed three.  The
# image stages returned no answer at all at any level above "none": a 1296- and
# then a 2592-token budget went entirely on reasoning, both attempts, every
# time.  Only the supports-and-loads rewrite ever came back better for thinking
# -- "apply a downward load of 80 kg on the seat" rather than "apply 80 kg load
# on the seat" -- and one direction word did not carry the rest.
#
# So nothing thinks, until there is a case where thinking demonstrably produces
# a better answer.  Naming a purpose in the map buys it back for that stage
# alone; a stage that wants more room for it than the default belongs in
# REASONING_HEADROOM_BY_PURPOSE too.
DEFAULT_REASONING_EFFORT = "none"
REASONING_EFFORT: dict[str, str] = {}
# Room for the thinking, on top of the stage's own answer.  Selector trees are
# the longest answer in the workflow and thinking about them ran to 1569 tokens
# in one measured run, so the default that suits a one-line rewrite would cut
# that off mid-thought and return nothing.
REASONING_HEADROOM_BY_PURPOSE = {"bc_synthesis": 2048}

# Every image-bearing stage runs in two steps: the vision model turns the picture into
# words, then the text model reasons over those words with everything else. Each stage
# briefs the vision model on what the image is and what to report, because "describe
# this" yields prose about lighting and mood rather than the one fact the stage needs.
VISION_BRIEFS = {
    # One reading of the geometry itself, shared by every stage that follows it.
    # The stages used to look at whatever the user supplied -- a photograph, a
    # scribble -- but that is evidence about the request, not about the mesh
    # that was built from it, and a generator does not always build what was
    # asked for.  Asking the same question of the same render means the first
    # stage to want the answer pays for the call and the rest hit the cache.
    "geometry_overview": (
        "six orthographic views of the geometry now being analysed, each panel "
        "titled by its nearest world-axis viewing direction (+x, -x, +y, -y, +z, -z). "
        "World +y is the object's up direction. The +/-x and +/-z side views are "
        "elevated 20 degrees toward +y and keep +y toward screen-up; the exact "
        "+/-y axial views instead use +z as screen-up",
        "what the object is, the parts it is made of and how they connect, its "
        "rough proportions and symmetry, which axis it stands up along, which part "
        "forms its structural bulk as opposed to padding or covering, and any "
        "recognisable feature that hints at its real size",
        80,
    ),
    "bc_synthesis": (
        "a six-view montage of one surface, flat-shaded by surface segment "
        "with a numbered colour legend; each panel is titled by its nearest world-axis "
        "viewing direction (+x, -x, +y, -y, +z, -z). World +y is the object's up direction. "
        "The +/-x and +/-z views are elevated 20 degrees toward +y with +y toward "
        "screen-up; the exact +/-y views use +z as screen-up. The object's front "
        "faces +z, so the +z panel is its front view and the -z panel its back. "
        "`legend` gives every segment id the colour it is painted, where its middle "
        "sits in the object's own box, and which way its surface points",
        "one clause for every segment id in `legend`, naming what that segment is or "
        "what it covers -- a named part such as seat, backrest, armrest, leg, base or "
        "rim, or the surface it spans. `legend` gives each id its colour and the "
        "measured position of its middle. Use both to find the region, and let the "
        "position overrule what a view seems to show: a segment sitting low in the "
        "object is a foot or a leg, not an armrest, however much the shape beside it "
        "looks like one. Then report the part and nothing else -- `seat`, not `seat, "
        "flat`; `backrest`, not `upright backrest at the rear`. Add a distinguishing "
        "word only where two ids would otherwise carry the same name, and only the "
        "word that separates them: left or right, upper or lower, front or back, "
        "inner or outer. Take that word from how their measured positions in "
        "`legend` differ, so that no two ids end up reading the same. Never report an "
        "axis, a coordinate, a measurement or a pose. "
        "Cover every id, and say so when a segment is too small or hidden to identify",
        120,
    ),
}

PREVIEW_SIZE = {
    # Six panels share the width, so the preview's long edge divides by three
    # before anything is read: at 512 each view is ~170px and the object inside
    # it perhaps 100px.  That was once enough for the montage and is not any
    # more -- the segmentation it carries went from a handful of shape-diameter
    # blocks to PartField's parts, and telling five of those apart is a finer
    # judgment than telling the old ones apart.  Measured on one sofa: at 512
    # the armrests came back as the backrest and the backrest as a front face;
    # at 1024, with nothing else changed, both were named correctly.  It costs
    # 281 input tokens on a DeepSeek call.  The plain render stays at 768,
    # where a blocky armchair stopped reading as "a single L-shaped bracket".
    "bc_synthesis": 1024, "geometry_overview": 768,
    # Two stacked six-view renders, so each panel gets half the height again.
    "part_materials": 1280,
}

# What each picture is told about its subject beforehand.  `requested` is what
# the user asked for and is explicitly not evidence; `object` is the reading the
# plain render already produced, and is.  The overview is the stage that decides
# what the object is, so it hears only the request.  The montage hears both: the
# request says what the parts would be called if the generator got it right, and
# the reading is there to overrule it when the generator did not -- told the
# request alone, it named a seat and a backrest on a C-shaped block with
# neither, and those names go on to place supports and loads.
CONTEXT_KEYS = {
    "geometry_overview": ("requested",),
    "bc_synthesis": ("object", "requested"),
}

NO_DIRICHLET = "at least one Dirichlet BC is required"
# A clause that holds the body opens with one of these verbs ("Fix all components
# of ...", "Clamp the ...").  Read only to decide whether a proposal with no held
# region contradicts its own request; the regions themselves stay the model's.
HOLD_VERBS = ("fix", "fixed", "hold", "held", "clamp", "clamped", "pin", "pinned",
              "anchor", "anchored", "restrain", "restrained", "support", "supported",
              "fasten", "fastened", "bolt", "bolted", "weld", "welded")


def hold_clauses(description: str) -> list[str]:
    """The clauses of a region request that hold the body: each sentence (split at
    a full stop, semicolon or line break followed by a space) whose first word is a
    holding verb.  A sentence that goes on to a load ("Fix the base and push 100 N down
    on the top.") is named up to that load: its hold part only, so the load is not
    asked back as a support."""
    import re
    clauses = re.split(r"(?<=[.;])\s+|\n+", description or "")
    found = []
    for clause in clauses:
        words = re.findall(r"[A-Za-z]+", clause)
        if words and words[0].lower() in HOLD_VERBS:
            hold = re.split(LOAD_CONTINUES, clause, maxsplit=1, flags=re.I)[0]
            found.append(hold.strip() if hold != clause else clause.strip())
    return found


# Where a holding sentence turns to a load: "," or "and" before a load verb, a whole word in its
# verb form followed by what a verb takes (an object, a number, a direction): the same words open
# region names ("pull ring", "loading plate", "pressing face", "push rod"), which are no load.
LOAD_CONTINUES = (r"\s*(?:,\s*(?:and\s+)?|\s+and\s+)(?:then\s+)?(?=(?:push|pushes|pull|pulls|press|presses|apply|"
                  r"applies|load|loads|twist|twists)\s+(?:(?:the|a|an|it|its|them|their|each|both|this|that|on|down|"
                  r"up|in|into|toward|towards|along|against|with|outward|inward|away)\b|[+-]?\d))")


T = TypeVar("T", bound=BaseModel)


# The montage paints segments from matplotlib's tab20 (see workflow_tool.py):
# ten hues, each in a strong and a pale form.  Naming those twenty outright is
# what keeps every pair distinct -- derived names put two of them on the same
# words, and a legend where two ids share a colour is worse than no legend.
TAB20_NAMES = {
    "1f77b4": "blue", "aec7e8": "light blue",
    "ff7f0e": "orange", "ffbb78": "light orange",
    "2ca02c": "green", "98df8a": "light green",
    "d62728": "red", "ff9896": "light red",
    "9467bd": "purple", "c5b0d5": "light purple",
    "8c564b": "brown", "c49c94": "light brown",
    "e377c2": "pink", "f7b6d2": "light pink",
    "7f7f7f": "grey", "c7c7c7": "light grey",
    "bcbd22": "olive", "dbdb8d": "light olive",
    "17becf": "cyan", "9edae5": "light cyan",
}


def color_name(value: str) -> str:
    """Plain-words name for a hex colour, e.g. "#aec7e8" -> "light blue".

    The palette in use is named exactly; anything else is derived from hue and
    lightness, so a change of palette still yields a usable name.
    """
    text = value.strip().lstrip("#").lower()
    if text in TAB20_NAMES:
        return TAB20_NAMES[text]
    if len(text) != 6:
        return ""
    try:
        red, green, blue = (int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return ""
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    degrees = hue * 360.0
    if saturation < 0.12:
        for limit, name in ((0.15, "black"), (0.35, "dark grey"), (0.65, "grey"),
                            (0.9, "light grey")):
            if lightness < limit:
                return name
        return "white"
    name = "red"
    for limit, candidate in (
        (15, "red"), (45, "orange"), (68, "yellow"), (80, "olive"),
        (160, "green"), (200, "cyan"), (255, "blue"), (290, "purple"),
        (335, "pink"),
    ):
        if degrees < limit:
            name = candidate
            break
    # Brown is what people call a muted orange-red; by hue alone it comes out as
    # red or orange and stops matching what is on screen.
    if 5 <= degrees < 45 and saturation < 0.6:
        name = "brown"
    if lightness > 0.66:
        return f"light {name}"
    if lightness < 0.32:
        return f"dark {name}"
    return name


# Height gets five steps and its own words.  The render states that +y is up,
# so "near the top" is the fact the stage actually needs, while "high y" makes
# it translate an axis label first -- and thirds put a backrest and a seat in
# the same bucket, which is the confusion that has outlasted every other fix.
# Depth is worded too, on the convention at FRONT_AXIS; width alone keeps the
# axis form, because left and right have no measurement to settle whose they
# are.
_HEIGHTS = ("at the bottom", "low", "at mid height", "high", "at the top")

# Depth, in the same words the facings use, so one end of z reads the same way
# wherever the stage meets it.  Three steps rather than height's five: the
# confusion that wanted five of them was a foot against the panel above it,
# which is a difference in height.
_DEPTHS = ("at the back", "mid depth", "at the front")


# Where the generator puts the front.  TripoSG builds its output facing +z --
# an image-conditioned generation places the view it was given toward the
# camera -- so the front view is the one the +z panel shows, on every session
# checked.  This is a convention about the generator, not a fact about
# furniture: a surface that arrived some other way may not honour it, and the
# only thing that goes wrong then is a segment named for the wrong end of z.
# It is asserted because the alternative was worse.  Left to work it out, the
# stage named the back face of a sofa the "front panel" while naming the
# segment facing the other way the backrest -- and neither a rule telling it
# the two cannot both be the front, nor the measured relation between them in
# the legend, moved it.  Naming the ends outright does.  Text2CAD parts carry no
# such convention; for them this only fixes which end the names call "front".
FRONT_AXIS = "+z"

# Which way a segment's facets actually point, in the words the height steps are
# already written in.  A flat patch across the bottom of a box and a seat are
# the same address in thirds -- "mid x, at the bottom, mid depth" fits both -- and
# the stage called one of them a seat.  Where it points is the measurement that
# separates them, and it is already computed per segment.  x keeps the axis
# form: left and right depend on whether they are the viewer's or the object's,
# and that is a choice with no measurement behind it.
_FACING_WORDS = {
    "+y": "faces upward", "-y": "faces downward",
    "+z": "faces the front", "-z": "faces the back",
}

# How much of a segment's facets must share a direction (within 45 degrees)
# before it is worth stating, and how much each of two opposite directions must
# hold before the segment is called two-sided.  The shares count facets, not
# area, so a finely meshed region weighs more than its size.  A shape with no dominant facing says nothing.
_FACING_SHARE = 0.5
_FACING_PAIR_SHARE = 0.25


def _dominant_facing(fractions: Any) -> str:
    """The way this segment points, when most of its facets agree on one."""
    try:
        shares = {str(key): float(value) for key, value in fractions.items()}
    except (AttributeError, TypeError, ValueError):
        return ""
    if not shares:
        return ""
    direction, share = max(shares.items(), key=lambda item: item[1])
    if share >= _FACING_SHARE:
        return _FACING_WORDS.get(direction, f"faces the {direction} side")
    for axis in "xyz":
        low, high = shares.get(f"-{axis}", 0.0), shares.get(f"+{axis}", 0.0)
        if min(low, high) >= _FACING_PAIR_SHARE:
            return f"faces both {axis} sides"
    return ""


def _bounded_position(centroid: Any, bounds: Any) -> str:
    """Where a centroid sits in the object's own box, measured not guessed.

    Judging this from a render is precisely what a vision model gets wrong --
    it read a backrest as a base and put both armrests on the same side --
    while the segmentation already knows it exactly, so it is supplied as an
    anchor for finding the region.
    """
    try:
        low, high = (list(edge) for edge in bounds)
        point = list(centroid)
    except (TypeError, ValueError):
        return ""
    if len(low) != 3 or len(high) != 3 or len(point) != 3:
        return ""
    words = []
    for index, axis in enumerate("xyz"):
        try:
            span = float(high[index]) - float(low[index])
            fraction = (float(point[index]) - float(low[index])) / span
        except (TypeError, ValueError, ZeroDivisionError):
            return ""
        if axis == "y":
            step = min(int(max(fraction, 0.0) * len(_HEIGHTS)), len(_HEIGHTS) - 1)
            words.append(_HEIGHTS[step])
        elif axis == "z":
            words.append(
                _DEPTHS[0 if fraction < 0.34 else 2 if fraction > 0.66 else 1]
            )
        else:
            words.append(
                f"{'low' if fraction < 0.34 else 'high' if fraction > 0.66 else 'mid'} {axis}"
            )
    return ", ".join(words)


PROVIDERS = {"deepseek": DeepSeekAPI, "groq": GroqAPI, "gemini": GeminiAPI}


def provider_factory(name: str):
    """The adapter for a provider name, or a message naming the ones there are."""
    try:
        return PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"unsupported provider: {name}; choose from {', '.join(PROVIDERS)}"
        ) from None


def density_unit(modulus_unit: str) -> str:
    """The density unit that goes with a modulus unit's unit system."""
    return (
        "tonne/mm³" if "MPa" in modulus_unit else
        "lbf·s²/in⁴" if "psi" in modulus_unit else "kg/m³"
    )


# Replies that failed in a held-out run (2026-10-10) were each one bracket over or short; at
# temperature 0 the repair request, told only "not a single JSON value", sent the same broken
# object back, and every resend of the step failed the same way.  A mend must be the only one:
# a bracket missing after an object's last entry can also be "mended" at the end of the reply,
# which moves the keys after it into that object (a review, 2026-10-10: gravity became a load
# region), so every place before the parser's stop is tried, and two readings are no mend.
TRUNCATED = ("length", "max_tokens")

# A region synthesis returns one selector tree per region, so its reply grows with the regions the
# request names.  In a held-out run (2026-10-10) every request of 6 to 14 regions came back cut at
# about 2.4 k characters, the whole 768-token budget, and seven previews failed as "not a single JSON
# value" (about 220 s): a compact tree of an "and" of a segment and a slab runs 120-170 tokens.  The
# budget therefore scales with the regions asked for, to a ceiling no answer of sensible size
# reaches (output is billed as produced, so the ceiling itself costs nothing).
BC_SYNTHESIS_BASE_TOKENS = 1024
BC_SYNTHESIS_TOKENS_PER_REGION = 200
MAX_OUTPUT_TOKENS = 4096
# What a reply cut off at the budget means to whoever reads the error, by stage.
TRUNCATION_ADVICE = {
    "bc_synthesis": "the region request is too long for one reply; merge regions or describe fewer",
}
# The verbs that open a region clause of a request ("Fix ...", "Apply a traction ..."): one region each.
REGION_VERBS = HOLD_VERBS + ("apply", "applies", "push", "pushes", "pull", "pulls", "press", "presses",
                             "load", "loads", "twist", "twists")


def requested_regions(description: str) -> int:
    """How many regions a request names: each sentence (split as hold_clauses splits) that opens
    with a holding or loading verb, and each load a holding sentence goes on to ("Fix the base and
    push down on the top")."""
    import re
    count = 0
    for clause in re.split(r"(?<=[.;])\s+|\n+", description or ""):
        words = re.findall(r"[A-Za-z]+", clause)
        if words and words[0].lower() in REGION_VERBS:
            count += 1 + len(re.findall(LOAD_CONTINUES, clause, flags=re.I))
    return max(count, 1)


def bc_synthesis_budget(description: str) -> int:
    """The answer budget of a region synthesis, scaled with the regions requested."""
    return min(MAX_OUTPUT_TOKENS,
               BC_SYNTHESIS_BASE_TOKENS + BC_SYNTHESIS_TOKENS_PER_REGION * requested_regions(description))


def _unfenced(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    return text


def _parse_error(exc: Exception, raw: str | None) -> str:
    """The error the repair request carries: the parser's own message, where it stopped and
    the text there, so the model can find the slip (a bare "not a single JSON value" was
    answered with the same object)."""
    error = str(exc).splitlines()[0][:240]
    if isinstance(exc, APIResponseError) and raw:
        text = _unfenced(raw)
        try:
            json.loads(text)
        except json.JSONDecodeError as parse:
            error += (f": {parse.msg} at character {parse.pos}, after "
                      f"{text[max(0, parse.pos - 40):parse.pos]!r}")
        except ValueError:
            pass
    return error


def _truncated(response: Any, budget: int | None = None) -> bool:
    """Whether the provider stopped at the token budget: its finish reason says so, or the reply
    used the whole budget.  A cut-off reply is never mended: what follows the cut is unknown."""
    reason = str(getattr(response, "finish_reason", None) or "").lower()
    if reason in TRUNCATED or reason.endswith("max_tokens"):
        return True
    usage = getattr(response, "usage", None)
    used = getattr(usage, "output_tokens", None)
    return bool(budget and used and used >= budget)


def _one_bracket_mend(raw: str, schema: type[BaseModel]) -> dict | None:
    """A reply one bracket away from valid JSON, mended, when exactly one reading exists: a
    ``]`` or ``}`` dropped, or added before a ``,``, ``]``, ``}`` or the end, anywhere up to where
    the parser stopped, and the result accepted by the stage's schema as any reply must.  Two
    different readings (a bracket closing an object early or late) are no mend: the reply goes
    to the repair request."""
    text = _unfenced(raw)
    try:
        json.loads(text)
        return None
    except json.JSONDecodeError as parse:
        stop = parse.pos
    except ValueError:
        return None
    seen: set[str] = set()
    readings: dict[str, dict] = {}
    for at in range(min(stop, len(text)), -1, -1):
        candidates = [text[:at] + text[at + 1:]] if at < len(text) and text[at] in "]}" else []
        if at == len(text) or text[at] in ",]}":
            candidates += [text[:at] + bracket + text[at:] for bracket in "]}"]
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                value = json.loads(candidate)
                schema.model_validate(value)
            except (ValueError, ValidationError):
                continue
            if isinstance(value, dict):
                readings[json.dumps(value, sort_keys=True)] = value
                if len(readings) > 1:
                    return None
    return next(iter(readings.values()), None)


class SpecialistCalls:
    """Run schema-constrained calls with a filesystem content cache."""

    def __init__(
        self, session_dir: Path, provider: str, model: str | None = None,
        vision_model: str | None = None, vision_provider: str | None = None,
        requested: str = "", overview: str = "",
    ):
        self.session_dir = session_dir
        self.cache_dir = session_dir / "api_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Reading a montage and reasoning over text are different jobs, so each
        # role gets its own provider as well as its own model: the strongest
        # text model on one service need not be the one that reads pictures.
        vision_provider = vision_provider or provider
        self.agent = AgentsWrapper(provider_factory(provider)(), model=model)
        shares_text_agent = vision_provider == provider and (
            not vision_model or vision_model == model
        )
        self.vision_agent = (
            self.agent
            if shares_text_agent
            else AgentsWrapper(
                provider_factory(vision_provider)(), model=vision_model
            )
        )
        self.provider = provider
        self.vision_provider = vision_provider
        # What the geometry was asked to be, and what reading it actually got.
        # Which of the two a picture is shown depends on what the picture is
        # for; see CONTEXT_KEY.
        self.requested = requested
        self.overview = overview
        self.receipts: list[APIReceipt] = []

    def _agent_for(self, image):
        return self.vision_agent if image is not None else self.agent

    def _provider_for(self, agent) -> str:
        """Which service served a call -- the two roles need not share one."""
        return self.provider if agent is self.agent else self.vision_provider

    def describe_image(
        self, purpose: str, image: Path, legend: dict[str, str] | None = None
    ) -> str:
        """Turn the image into the words the text stage needs, or "" if absent.

        ``legend`` names what the picture's colours mean.  Reading a swatch out
        of a rendered key is exactly the kind of thing a vision model gets
        subtly wrong, and every later stage is keyed on the segment ids, so the
        mapping is supplied rather than inferred.
        """
        subject, wanted, limit = VISION_BRIEFS[purpose]
        payload = {
            "task": "describe_image",
            "subject": subject,
            "describe": wanted,
            "word_limit": limit,
        }
        available = {"requested": self.requested, "object": self.overview}
        for key in CONTEXT_KEYS.get(purpose, ()):
            if available[key]:
                payload[key] = available[key]
        if legend:
            payload["legend"] = legend
        prompt = json.dumps(payload, separators=(",", ":"))
        described, receipt = self._cached_call(
            "image_description", prompt, ImageDescription,
            self._instructions("image_description.md"), image, 32 + 2 * limit,
            preview_purpose=purpose,
        )
        self.receipts.append(receipt)
        return described.description

    def _seen(
        self, purpose: str, image: Path | None, legend: dict[str, str] | None = None
    ) -> dict[str, str]:
        """The vision stage's words, ready to fold into a text-only payload."""
        if image is None:
            return {}
        return {"image_description": self.describe_image(purpose, image, legend)}

    @staticmethod
    def _segment_legend(metadata: dict[str, Any]) -> dict[str, str]:
        """Segment id -> the colour it is drawn in, named rather than coded.

        A hex triplet is not something that can be matched against a rendered
        fill; "dark blue" is.  The name is what the description has to key on,
        so it is stated outright instead of being left to be worked out.
        """
        segments = metadata.get("segments") or []
        bounds = metadata.get("bounds") or []
        legend = {}
        for segment in segments:
            if not isinstance(segment, dict) or "id" not in segment:
                continue
            name = color_name(str(segment.get("color") or ""))
            if not name:
                continue
            seat = _bounded_position(segment.get("centroid"), bounds)
            # Size separates what position cannot.  Thirds of a box put a foot
            # and the side panel above it at the same address, and the stage,
            # given nothing to tell them apart with, invented a difference; a
            # 0.7% patch and a 38% mass are plainly not the same kind of thing.
            try:
                share = float(segment.get("area_fraction"))
            except (TypeError, ValueError):
                share = None
            # A CAD face knows what kind of surface it is, which is what tells
            # a bolt hole's bore from the flat face it passes through.
            cad_surface = segment.get("cad_surface")
            measured = [part for part in (
                f"{cad_surface} face" if cad_surface else "",
                f"{share * 100:.1f}% of the surface" if share else "", seat,
                _dominant_facing(segment.get("facing_fraction_45deg")),
            ) if part]

            legend[str(segment["id"])] = ", ".join([name, *measured])
        return legend

    def refine_geometry(
        self, description: str, overview: str = ""
    ) -> tuple[GeometryRefinement, APIReceipt]:
        """Edit the description into a generation prompt.

        This stage used to describe the user's own photo or scribble, which
        bought nothing: the reference is handed to the generator directly as
        conditioning, so putting a second-hand account of it into the prompt
        only paid for a vision call to say what the generator could already
        see.  What does help, on a revision, is knowing what came out last
        time -- and by then the six-view render has usually been described
        already for scaling or material, so that reading is reused where it
        exists and skipped where it does not.  No picture is sent either way.
        """
        prompt = json.dumps(
            {"task": "refine_geometry", "description": description,
             **({"image_description": overview} if overview else {})},
            separators=(",", ":"),
        )
        system = self._instructions("geometry.md")
        return self._cached_call(
            "geometry_refinement", prompt, GeometryRefinement, system, None, 192
        )

    def refine_cad_geometry(self, description: str) -> tuple[GeometryRefinement, APIReceipt]:
        """Rewrite the description as the sketch-and-extrude construction Text2CAD reads.

        Kept apart from ``refine_geometry``: that prompt is tuned for TripoSG's
        short visual phrases, while Text2CAD knows no object names -- "a hex
        nut" comes back a box, "a hexagonal prism with a circular hole through
        its center" a nut -- so this one spells out profiles, extrusions and holes.
        """
        prompt = json.dumps(
            {"task": "refine_cad_geometry", "description": description},
            separators=(",", ":"),
        )
        return self._cached_call(
            "cad_geometry_refinement", prompt, GeometryRefinement,
            self._instructions("cad_geometry.md"), None, 192,
        )

    def refine_cadcoder_geometry(self, description: str) -> tuple[CADDescription, APIReceipt]:
        """Write the expert-level construction description CAD-Coder transcribes.

        CAD-Coder was tuned on step-by-step descriptions with explicit sketch
        coordinates and follows them faithfully, but cannot work out what a
        named object looks like -- "a hex nut" comes back a box.  So this stage
        decides the geometry and spells out every loop, radius and extrusion.
        """
        prompt = json.dumps(
            {"task": "refine_cadcoder_geometry", "description": description},
            separators=(",", ":"),
        )
        return self._cached_call(
            "cadcoder_geometry_refinement", prompt, CADDescription,
            # A part with many pieces (a bookshelf, a table) runs long; cutting
            # the answer off mid-string leaves invalid JSON.
            self._instructions("cadcoder_geometry.md"), None, 4000,
        )

    def infer_physical_scale(
        self, description: str, request: str, length_unit: str,
        overview: str = "",
    ) -> tuple[PhysicalScaleInference, APIReceipt]:
        payload: dict[str, Any] = {
            "task": "infer_physical_scale",
            "description": description,
            "length_unit": length_unit,
            **({"image_description": overview} if overview else {}),
        }
        if request:
            payload["request"] = request
        return self._cached_call(
            "physical_scale_inference", json.dumps(payload, separators=(",", ":")),
            PhysicalScaleInference, self._instructions("physical_scale.md"), None, 64,
        )

    def refine_bc_prompt(
        self, geometry: str, description: str, segments: str = ""
    ) -> tuple[BCPromptRefinement, APIReceipt]:
        """Edit the supports-and-loads text into something selectable.

        ``segments`` is what the surface's numbered segments were found to be.
        The stage after this one can only select regions that exist, so a region
        phrased in the segmentation's own words is one it can bind; phrased in
        the user's, it may have nothing to attach to.
        """
        prompt = json.dumps(
            {"task": "refine_boundary_prompt", "geometry": geometry,
             "description": description,
             **({"segments": segments} if segments else {})},
            separators=(",", ":"),
        )
        return self._cached_call(
            "bc_prompt_refinement", prompt, BCPromptRefinement,
            self._instructions("boundary_condition_prompt.md"), None, 600,
        )

    def infer_material(
        self, description: str, request: str, modulus_unit: str,
        overview: str = "",
    ) -> tuple[MaterialInference, APIReceipt]:
        """Infer isotropic parameters from the same evidence as physical scaling."""
        payload: dict[str, Any] = {
            "task": "infer_material",
            "description": description,
            "modulus_unit": modulus_unit,
            "density_unit": density_unit(modulus_unit),
            **({"image_description": overview} if overview else {}),
        }
        if request:
            payload["request"] = request
        return self._cached_call(
            "material_interpretation", json.dumps(payload, separators=(",", ":")),
            MaterialInference, self._instructions("material.md", "material_values.md"), None, 192,
        )

    def assign_part_materials(
        self, montage: Path, metadata: dict[str, Any], modulus_unit: str,
        request: str = "", segments: str = "",
    ) -> tuple[PartMaterials, APIReceipt]:
        """Isotropic parameters for every part in a colour-over-parts montage.

        The uniform stage's rules and table (``material_values.md``) apply per
        part.  One vision call: a material is read off colour and texture,
        which a description in words would lose.  ``segments`` is the segment
        stage's reading of what each part is -- the seat, the legs -- the same
        words the boundary conditions were chosen by.
        """
        payload: dict[str, Any] = {
            "task": "assign_part_materials",
            "parts": self._segment_legend(metadata),
            "modulus_unit": modulus_unit,
            "density_unit": density_unit(modulus_unit),
        }
        for key, value in (("requested", self.requested), ("object", self.overview),
                           ("segments", segments), ("request", request)):
            if value:
                payload[key] = value
        result, receipt = self._cached_call(
            "part_materials", json.dumps(payload, separators=(",", ":")),
            PartMaterials,
            self._instructions("part_material.md", "material_values.md",
                               shared_preamble=False),
            montage, 32 + 40 * len(payload["parts"]),
        )
        self.receipts.append(receipt)
        return result, receipt

    def synthesize_bcs(
        self, description: str, metadata: dict[str, Any], montage: Path
    ) -> tuple[BCProposal, APIReceipt]:
        payload = {
            "task": "select_boundary_regions",
            "request": description,
            "geometry": self._compact_geometry(metadata),
            **self._seen(
                "bc_synthesis", montage, self._segment_legend(metadata)
            ),
        }
        system = self._instructions("boundary_conditions_api.md")
        # Selector trees are the longest output in the workflow, one per region
        # (bc_synthesis_budget); overrunning the budget cuts the JSON off, or
        # makes the provider reject an empty generation as a bad request.
        try:
            return self._cached_call(
                "bc_synthesis", json.dumps(payload, separators=(",", ":")),
                BCProposal, system, None, bc_synthesis_budget(description),
            )
        except ValidationError as exc:
            holds = hold_clauses(description)
            if not holds or NO_DIRICHLET not in str(exc):
                raise
            # A request whose clauses begin "Fix ..." came back with every region
            # a load: in a held-out run (2026-10-10) four previews of one body
            # failed so, the held face being described by the direction it faces
            # ("the face toward the user (+z)"), which reads like a traction's.
            # The repair call cannot mend it (it only sees the answer), so the
            # question is asked once more with the holding clauses named.
            payload["correction"] = (
                "An earlier answer to this request put no region in dirichlet. "
                "These clauses hold the body, so each is a dirichlet item, whatever "
                "direction words describe the face it holds: "
                + " | ".join(holds)
                + ". Only a clause that applies a load is a neumann item."
            )
            return self._cached_call(
                "bc_synthesis", json.dumps(payload, separators=(",", ":")),
                BCProposal, system, None, bc_synthesis_budget(description),
            )

    def infer_bc_values(
        self, previous_description: str, value_description: str,
        unit_system: str, units: dict[str, str],
        region_statistics: list[dict[str, Any]],
        loaded: bool = True,
    ) -> tuple[BCValueInference, APIReceipt]:
        """Infer values from the description alone.

        The current values are deliberately not sent: with them in the prompt the
        model tended to echo them back, which made the call a silent no-op.  The
        controller back-fills any region the response omits.

        ``loaded`` says whether there is a Neumann region to value at all.  An
        empty answer is normally a refusal to answer and is rejected; where the
        proposal holds no loaded region it is the only correct one.
        """
        prompt = json.dumps({
            "task": "infer_boundary_condition_values",
            "previous_description": previous_description,
            "value_description": value_description,
            "unit_system": unit_system,
            "units": {
                "displacement": units["displacement"],
                "force": units["force"],
                "traction": units["stress_modulus_traction"],
                "body_force": units["body_force"],
                "acceleration": units["acceleration"],
                "region_area": f"{units['length']}^2",
            },
            "region_statistics": region_statistics,
        }, separators=(",", ":"))
        return self._cached_call(
            "bc_value_inference", prompt,
            BCValueInference if loaded else UnloadedBCValueInference,
            self._instructions("boundary_condition_values.md"), None, 384,
        )

    @staticmethod
    def _compact_geometry(metadata: dict[str, Any]) -> dict[str, Any]:
        """Retain decision-relevant geometry evidence without repeated diagnostics."""

        def rounded(value):
            if isinstance(value, float):
                return round(value, 5)
            if isinstance(value, list):
                return [rounded(item) for item in value]
            return value

        segments = metadata.get("segments", [])
        compact = {
            "bounds": rounded(metadata.get("bounds")),
            "segmentation": {
                key: metadata.get(key)
                for key in (
                    "method", "num_clusters", "num_rays", "smoothing", "cone_angle_deg"
                )
                if metadata.get(key) is not None
            },
            "segmentation_informative": len(segments) > 1,
        }
        if len(segments) > 1:
            compact["segments"] = [
                {
                    key: rounded(segment.get(key))
                    for key in (
                        "id", "area_fraction", "bounds", "centroid",
                        "facing_fraction_45deg",
                    )
                }
                for segment in segments
            ]
        return compact

    # Long enough to read a prompt in full in the tool window, short enough that
    # a session's transcript stays a text file rather than a database.
    TRANSCRIPT_LIMIT = 8000

    def _record_call(self, receipt: APIReceipt, prompt: str, response: Any,
                     image: Path | None, error: str | None = None) -> None:
        """Append one call to the session transcript the tool window reads.

        The response cache holds what was returned but not what was asked, and a
        cache hit leaves no trace at all, so neither answers "what did we send?".
        This does, and it never raises: a failure to log must not fail the call.
        A failed call is recorded too, with ``error`` and whatever came back,
        since what the model proposed is exactly what a failure leaves unknown.
        """
        def clip(text: str) -> str:
            return text if len(text) <= self.TRANSCRIPT_LIMIT else (
                text[:self.TRANSCRIPT_LIMIT] + f"… [{len(text)} chars]"
            )

        entry = {
            "timestamp": time.time(),
            "purpose": receipt.purpose,
            "provider": receipt.provider,
            "model": receipt.model,
            "cached": receipt.cached,
            "total_tokens": receipt.total_tokens,
            "prompt": clip(prompt),
            "response": clip(response if isinstance(response, str)
                             else json.dumps(response, indent=1, default=str)),
            # The image itself is megabytes of base64; its name is what identifies it.
            "image": image.name if image else None,
            **({"error": error} if error else {}),
        }
        try:
            with (self.session_dir / "api_log.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            pass

    def _cached_call(
        self,
        purpose: str,
        prompt: str,
        schema: type[T],
        system: str,
        image: Path | None,
        max_tokens: int,
        preview_purpose: str | None = None,
    ) -> tuple[T, APIReceipt]:
        preview = self._preview(
            image, max_size=PREVIEW_SIZE.get(preview_purpose or purpose, 768)
        ) if image else None
        agent = self._agent_for(preview)
        provider = self._provider_for(agent)
        # Every stage here asks for a short, validated JSON object, so where a
        # provider exposes the dial it is set per stage rather than left at its
        # default; see the table.  "medium" is not a level the service has -- it
        # maps that back onto high -- so the modest setting is "low".
        reasoning = (
            REASONING_EFFORT.get(purpose, DEFAULT_REASONING_EFFORT)
            if provider in REASONING_OPTIONAL_PROVIDERS
            else None
        )
        thinks = (
            reasoning not in (None, "none")
            or (reasoning is None and agent.model not in NON_REASONING_MODELS)
        )
        headroom = REASONING_HEADROOM_BY_PURPOSE.get(purpose, REASONING_HEADROOM)
        budget = max_tokens + (headroom if thinks else 0)

        def ask(text: str, system_prompt: str, max_output_tokens: int):
            """One attempt, always carrying whatever image the stage was given."""
            return agent.generate_structured(
                text,
                image=preview,
                image_mime_type="image/webp" if preview else None,
                system_prompt=system_prompt,
                max_output_tokens=max_output_tokens,
                temperature=0.0,
                reasoning_effort=reasoning,
            )
        digest = hashlib.sha256()
        digest.update(provider.encode())
        digest.update(agent.model.encode())
        digest.update(system.encode())
        digest.update(prompt.encode())
        if preview:
            digest.update(preview.read_bytes())
        key = digest.hexdigest()
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                value = schema.model_validate(payload["value"])
                receipt = APIReceipt.model_validate(payload["receipt"])
                cached_receipt = receipt.model_copy(update={"cached": True})
                self._record_call(cached_receipt, prompt,
                                  value.model_dump(mode="json"), preview)
                return value, cached_receipt
            except (OSError, KeyError, json.JSONDecodeError, ValidationError):
                cache_path.unlink(missing_ok=True)

        returned: Any = None  # the last thing the model sent back, for the log
        mended = None         # a reply mended here is used once, never cached as the model's answer
        try:
            try:
                value, response = ask(prompt, system, budget)
                returned = value
                parsed = schema.model_validate(value)
            except (APIResponseError, ValidationError) as exc:
                raw = exc.raw_text if isinstance(exc, APIResponseError) else json.dumps(value, separators=(",", ":"))
                returned = raw
                error = _parse_error(exc, raw)
                cut = isinstance(exc, APIResponseError) and _truncated(exc.response, budget)
                mended = (_one_bracket_mend(raw, schema)
                          if isinstance(exc, APIResponseError) and raw and not cut else None)
                if cut:
                    # Cut off at the budget: a repair request, held to the same budget and asked to
                    # write the whole object again, has no more room to finish it.  The question is
                    # asked once more with twice the room, up to the ceiling; cut again, the error
                    # says so plainly.
                    wider = min(2 * budget, MAX_OUTPUT_TOKENS + (budget - max_tokens))
                    try:
                        if wider <= budget:
                            raise exc
                        value, response = ask(prompt, system, wider)
                    except APIResponseError as again:
                        if again is not exc and not _truncated(again.response, wider):
                            raise
                        returned = again.raw_text or returned
                        advice = TRUNCATION_ADVICE.get(purpose, "ask for a shorter answer")
                        raise APIResponseError(
                            f"the reply was cut off at the {max(wider, budget)}-token budget: {advice}",
                            raw_text=again.raw_text, response=again.response) from again
                elif mended is not None:
                    # One bracket short or over: mended here, validated, and nothing re-asked.
                    value, response = mended, exc.response or type("R", (), {"usage": None})()
                elif raw and raw.strip():
                    value, response = ask(
                        json.dumps(
                            {"task": "repair_json", "invalid_response": raw, "error": error},
                            separators=(",", ":"),
                        ),
                        system + "\nCorrect the supplied response. Return only the corrected JSON object.",
                        budget,
                    )
                else:
                    # Nothing came back to correct.  Asking a vision model to repair
                    # an empty string used to send the prompt on without the image,
                    # and it answered, honestly and uselessly, that it could see no
                    # image -- which then got stored as the description.  Re-ask the
                    # real question instead, with room for an answer that overran.
                    value, response = ask(prompt, system, budget * 2)
                returned = value
                parsed = schema.model_validate(value)
        except Exception as exc:
            # Not cached -- a fresh call may well succeed, and the card that made
            # this one offers it -- but logged, or what the model proposed is
            # lost with the error that rejected it.
            if isinstance(exc, ValidationError):
                detail = "; ".join(
                    f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}"
                    for item in exc.errors()[:3])
            else:
                detail = (str(exc).splitlines() or [""])[0]
            self._record_call(
                APIReceipt(purpose=purpose, provider=provider, model=agent.model, cache_key=key),
                prompt, "" if returned is None else returned, preview,
                error=f"{type(exc).__name__}: {detail[:400]}",
            )
            raise
        usage = response.usage
        receipt = APIReceipt(
            purpose=purpose,
            provider=self._provider_for(agent),
            model=agent.model,
            cache_key=key,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
        )
        self._record_call(receipt, prompt, parsed.model_dump(mode="json"), preview)
        if mended is not None:
            return parsed, receipt
        cache_path.write_text(
            json.dumps(
                {"value": parsed.model_dump(mode="json"), "receipt": receipt.model_dump(mode="json")},
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        return parsed, receipt

    def _preview(self, path: Path, max_size: int = 768) -> Path:
        """Create a derived, bounded image while preserving the original asset."""

        preview_dir = self.session_dir / "previews"
        preview_dir.mkdir(exist_ok=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        target = preview_dir / f"{digest}-{max_size}.webp"
        if not target.exists():
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((max_size, max_size))
                image.save(target, "WEBP", quality=80, method=6)
        return target

    @staticmethod
    def _instructions(*references: str, shared_preamble: bool = True) -> str:
        """The stage's own reference, on the workflow preamble unless refused.

        SKILL.md orients a text specialist: the stage list, the selector
        vocabulary, and how to weigh an `image_description` written by someone
        else.  The vision stage is that someone else.  None of it applies to
        looking at a picture, its one relevant rule -- return bare JSON -- is
        already in image_description.md, and it is ~530 tokens on the most
        expensive call in the workflow, so that call does without it.
        """
        skill = Path(__file__).resolve().parent / "skill"
        body = "\n".join(
            (skill / "references" / reference).read_text(encoding="utf-8")
            for reference in references
        )
        if not shared_preamble:
            return body
        return (skill / "SKILL.md").read_text(encoding="utf-8") + "\n" + body
