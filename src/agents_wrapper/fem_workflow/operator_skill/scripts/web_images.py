#!/usr/bin/env python
"""Licensed photographs of an object from keyless open-collection APIs, for the image route.

When the generated routes (TripoSG text, an SDXL drawing, Hunyuan3D from it) give no
recognisable object, a real photograph of one often does: the image generators follow a picture
closely.  It is the last route before a body built by hand (the user's rule, 2026-10-11).  This
module is the only code that reaches the web for it, and the runner (``drive.py``) is its only
caller: the agent writes query words in its plan, never a URL.

* **Licences.**  Only CC0, public domain (dedication or the Public Domain Mark) and CC BY pass,
  each read from the source's own per-item licence field, and every downloaded image carries
  its attribution (title, author, source page, licence and its URL, the query).  CC BY-SA is
  left out on purpose: whether a mesh and the simulation fields made from a photograph adapt it
  is unsettled, and ShareAlike would then bind them; NC and ND are left out as well, and so is a
  rights advisory such as "no known restrictions", which is no licence (the user's decision;
  it removed the Library of Congress, whose items pass only through that advisory).
* **Sources, in order:** Wikimedia Commons, the Met (v1.1 search; v1 returns 410 since
  2026-10-01), Rijksmuseum, Wellcome Collection, the Art Institute of Chicago, the Cleveland
  Museum of Art, Openverse, the Minneapolis Institute of Art.  All keyless (the user declined a
  Smithsonian key).  Commons comes first for its breadth (it mirrors many museums' CC0 studio
  photos); the museums give plain studio backgrounds; Openverse (snapshots) is the last resort
  before Minneapolis.
* **Etiquette.**  One informative User-Agent naming the project's page (Wikimedia's policy asks
  for contact information); the Art Institute's Cloudflare refuses an e-mail in it, so its
  contact goes in ``AIC-User-Agent``.  Requests are serial with a pause per source, and the
  daily and per-minute caps (Openverse 200 a day and 20 a minute anonymous, the Art Institute
  60 a minute) are kept in a ledger every session shares, under a file lock.

The sources, their licence fields and limits were probed on 2026-10-08.

    python web_images.py process --phrase WORDS IMAGE FOLDER [IMAGE FOLDER ...]   # compute env, GPU

The search has no command line: only the runner calls it (``main``).
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterator

SOURCES = ("commons", "met", "rijksmuseum", "wellcome", "aic", "cleveland", "openverse", "mia")
PROJECT = "https://github.com/LuoXueling/implicit_optimize"
USER_AGENT = f"implicit-optimize-research/0.1 ({PROJECT}) python-urllib/3"
AIC_HEADERS = {"User-Agent": USER_AGENT, "AIC-User-Agent": f"implicit-optimize ({PROJECT})"}
# Seconds between two requests to one source, and the caps a source states.  Commons asks for
# serial requests ("be considerate"); the Art Institute for no more than one a second; Openverse
# allows 20 a minute and 200 a day anonymously.
PAUSE_S = {"commons": 1.0, "met": 0.1, "rijksmuseum": 1.0, "wellcome": 0.5, "aic": 1.0,
           "cleveland": 0.5, "openverse": 3.0, "mia": 1.0}
PER_DAY = {"openverse": 200}
PER_MINUTE = {"aic": 60, "openverse": 20}
MAX_IMAGE_BYTES = 25 * 2 ** 20
# A request waits at most REQUEST_S, and one search at most SEARCH_S in all: 8 sources at the 40 s
# timeout before could hold the run over 5 minutes on a dead network (review of 2026-10-11).
# Past the cap the sources not yet asked are recorded unreachable.
REQUEST_S = 10.0
SEARCH_S = 60.0
_deadline = [float("inf")]      # the running search's end, which every request's timeout respects
WANT = 6             # candidates downloaded and processed for one search
SHOW = 3             # at most this many survivors of the pre-screen shown at the stop
PER_SOURCE = 3       # from any one source, so one archive's style does not fill the stop
CC0 = "https://creativecommons.org/publicdomain/zero/1.0/"
PDM = "https://creativecommons.org/publicdomain/mark/1.0/"
RIJKS_PASS = ("creativecommons.org/publicdomain/mark/1.0", "creativecommons.org/publicdomain/zero/1.0",
              "creativecommons.org/licenses/by/4.0")
WELLCOME_PASS = ("cc-0", "pdm", "cc-by")
OPENVERSE_PASS = ("cc0", "pdm", "by")


class SourceLimit(RuntimeError):
    """A source's daily cap is reached: it is skipped, not waited for."""


# ------------------------------------------------------------------ licences --

def commons_licence(meta: dict[str, Any]) -> tuple[str, str] | None:
    """(licence, licence URL) when a Commons file's licence passes, else None.  The value is
    ``extmetadata.License``: ``cc0``, ``pd`` and ``cc-by-<version>`` pass; ``cc-by-sa-*``,
    GFDL and an empty value do not, nor does any ``Restrictions`` (trademark, personality)."""
    value = str((meta.get("License") or {}).get("value") or "").strip().lower()
    if str((meta.get("Restrictions") or {}).get("value") or "").strip():
        return None
    if value not in ("cc0", "pd") and not re.fullmatch(r"cc-by-\d(\.\d)?", value):
        return None
    name = str((meta.get("LicenseShortName") or {}).get("value") or value)
    url = str((meta.get("LicenseUrl") or {}).get("value") or {"cc0": CC0, "pd": PDM}.get(value, ""))
    return name, url


def _text(value: Any) -> str:
    """Plain text of an HTML fragment (Commons' Artist field), whitespace collapsed."""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", str(value or ""))).split())


def _candidate(source: str, item_id: Any, title: Any, author: Any, page: Any, licence: str,
               licence_url: str, image_url: str, query: str) -> dict[str, Any]:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(item_id)).strip("-")[:40] or "item"
    return {"id": f"{source}-{slug}", "source": source, "title": _text(title)[:200],
            "author": _text(author)[:200] or "unknown", "source_url": str(page or ""),
            "licence": licence, "licence_url": licence_url, "image_url": image_url, "query": query}


# ------------------------------------------------------------------- parsers --
# Each takes a source's response as it comes and returns the candidates whose licence passes,
# in the source's order.  They read no network, so the tests feed them recorded responses.

def parse_commons(data: dict[str, Any], query: str) -> list[dict[str, Any]]:
    pages = sorted(((data.get("query") or {}).get("pages") or {}).values(), key=lambda p: p.get("index", 0))
    out = []
    for page in pages:
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        licence = commons_licence(meta)
        if not licence or str(info.get("mime") or "") not in ("image/jpeg", "image/png"):
            continue
        # The original, or a 2048-wide rendition of a larger one (standard widths only).
        image = info.get("thumburl") if (info.get("width") or 0) > 2048 and info.get("thumburl") else info.get("url")
        title = str(page.get("title") or "").removeprefix("File:")
        out.append(_candidate("commons", page.get("pageid"), title, (meta.get("Artist") or {}).get("value"),
                              info.get("descriptionurl"), licence[0], licence[1], image, query))
    return out


def parse_met_object(data: dict[str, Any], query: str) -> dict[str, Any] | None:
    """One Met object record: CC0 when ``isPublicDomain`` (the search's own filter is not
    honoured), with its primary image."""
    if not data.get("isPublicDomain") or not data.get("primaryImage"):
        return None
    return _candidate("met", data.get("objectID"), data.get("title"),
                      data.get("artistDisplayName") or data.get("culture"), data.get("objectURL"),
                      "CC0 1.0", CC0, data["primaryImage"], query)


def parse_rijks_record(xml: str, query: str) -> dict[str, Any] | None:
    """One Rijksmuseum EDM record (OAI GetRecord): ``edm:rights`` must be PDM, CC0 or CC BY 4.0."""
    rights = re.search(r'<edm:rights rdf:resource="([^"]+)"', xml)
    image = re.search(r'<edm:isShownBy rdf:resource="([^"]+)"', xml)
    about = re.search(r'<edm:ProvidedCHO rdf:about="([^"]+)"', xml)
    if not rights or not image or not any(p in rights.group(1) for p in RIJKS_PASS):
        return None
    titles = re.findall(r'<dc:title(?: xml:lang="(\w+)")?>([^<]*)<', xml)
    title = next((t for lang, t in titles if lang == "en"), titles[0][1] if titles else "")
    creator = re.search(r"<dc:creator[^>]*>([^<]*)<", xml)
    licence = ("Public Domain Mark 1.0" if "mark/1.0" in rights.group(1) else
               "CC0 1.0" if "zero/1.0" in rights.group(1) else "CC BY 4.0")
    page = about.group(1) if about else ""
    return _candidate("rijksmuseum", page.rsplit("/", 1)[-1], title, creator.group(1) if creator else "",
                      page, licence, rights.group(1), image.group(1), query)


def parse_wellcome(data: dict[str, Any], query: str) -> list[dict[str, Any]]:
    out = []
    for item in data.get("results") or []:
        location = (item.get("locations") or [{}])[0]
        lic = location.get("license") or {}
        if lic.get("id") not in WELLCOME_PASS or not str(location.get("url") or "").endswith("/info.json"):
            continue
        source = item.get("source") or {}
        people = [((c.get("agent") or {}).get("label")) for c in source.get("contributors") or []]
        out.append(_candidate("wellcome", item.get("id"), source.get("title"),
                              "; ".join(p for p in people if p) or location.get("credit") or "Wellcome Collection",
                              f"https://wellcomecollection.org/works/{source.get('id')}", lic.get("label") or lic["id"],
                              lic.get("url") or "", location["url"][:-len("info.json")] + "full/full/0/default.jpg",
                              query))
    return out


def parse_aic(data: dict[str, Any], query: str) -> list[dict[str, Any]]:
    out = []
    for item in data.get("data") or []:
        if not item.get("is_public_domain") or not item.get("image_id"):
            continue
        out.append(_candidate("aic", item.get("id"), item.get("title"), item.get("artist_display"),
                              f"https://www.artic.edu/artworks/{item.get('id')}", "CC0 1.0", CC0,
                              f"https://www.artic.edu/iiif/2/{item['image_id']}/full/1686,/0/default.jpg", query))
    return out


def parse_cleveland(data: dict[str, Any], query: str) -> list[dict[str, Any]]:
    out = []
    for item in data.get("data") or []:
        images = item.get("images") or {}
        image = (images.get("print") or images.get("web") or {}).get("url")
        if item.get("share_license_status") != "CC0" or not image:
            continue
        people = [c.get("description") for c in item.get("creators") or []]
        out.append(_candidate("cleveland", item.get("id"), item.get("title"),
                              "; ".join(p for p in people if p) or item.get("culture"), item.get("url"),
                              "CC0 1.0", CC0, image, query))
    return out


def parse_openverse(data: dict[str, Any], query: str) -> list[dict[str, Any]]:
    """Openverse does not verify licences itself: the item's own page is recorded as the source,
    where the licence is to be confirmed before a figure uses the picture."""
    out = []
    for item in data.get("results") or []:
        licence = str(item.get("license") or "").lower()
        if licence not in OPENVERSE_PASS or not item.get("url"):
            continue
        name = {"cc0": "CC0", "pdm": "Public Domain Mark", "by": "CC BY"}[licence]
        version = f" {item['license_version']}" if item.get("license_version") else ""
        out.append({**_candidate("openverse", item.get("id"), item.get("title"), item.get("creator"),
                                 item.get("foreign_landing_url"), name + version, item.get("license_url") or "",
                                 item["url"], query),
                    "licence_checked_at_source": False})
    return out


def parse_mia(data: dict[str, Any], query: str) -> list[dict[str, Any]]:
    out = []
    for hit in (data.get("hits") or {}).get("hits") or []:
        item = hit.get("_source") or {}
        if item.get("rights_type") != "Public Domain" or str(item.get("public_access")) != "1" \
                or not item.get("image_width"):
            continue
        out.append(_candidate("mia", item.get("id"), item.get("title"), item.get("artist"),
                              f"https://collections.artsmia.org/art/{item.get('id')}", "Public Domain", PDM,
                              f"https://1.api.artsmia.org/800/{item.get('id')}.jpg", query))
    return out


# ------------------------------------------------------------------- network --

@contextlib.contextmanager
def _ledger(path: Path) -> Iterator[dict[str, Any]]:
    """The request ledger every session shares, read and written under one lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except ValueError:
            state = {}
        today = date.today().isoformat()
        if state.get("day") != today:
            state = {"day": today, "count": {}, "last": state.get("last", {}), "recent": state.get("recent", {})}
        yield state
        path.write_text(json.dumps(state), encoding="utf-8")


def polite_wait(ledger: Path, source: str, api: bool = True, now: Callable[[], float] = time.time,
                sleep: Callable[[float], None] = time.sleep) -> None:
    """Wait out the source's pause, and keep its caps: a minute cap is waited for, a day cap
    refuses.  ``api`` False is an image download: paused, not counted."""
    with _ledger(ledger) as state:
        if api and state["count"].get(source, 0) >= PER_DAY.get(source, 10 ** 9):
            raise SourceLimit(f"{source}: {PER_DAY[source]} requests today, its daily cap")
        recent = [t for t in state["recent"].get(source, []) if now() - t < 60.0]
        if api and len(recent) >= PER_MINUTE.get(source, 10 ** 9):
            sleep(max(60.0 - (now() - recent[0]), 0.0) + 0.5)
        gap = now() - state["last"].get(source, 0.0)
        if gap < PAUSE_S[source]:
            sleep(PAUSE_S[source] - gap)
        stamp = now()
        state["last"][source] = stamp
        if api:
            state["count"][source] = state["count"].get(source, 0) + 1
            state["recent"][source] = [t for t in recent if stamp - t < 60.0] + [stamp]


class SearchCap(TimeoutError):
    """The search's overall cap (SEARCH_S) is reached."""


def fetch(url: str, source: str, ledger: Path, api: bool = True, limit: int = 8 * 2 ** 20) -> bytes:
    polite_wait(ledger, source, api)
    left = _deadline[0] - time.time()
    if left <= 0.5:
        raise SearchCap(f"the search's {SEARCH_S:.0f} s cap was reached")
    headers = AIC_HEADERS if source == "aic" else {"User-Agent": USER_AGENT}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                timeout=min(REQUEST_S, left)) as response:
        body = response.read(limit + 1)
    if len(body) > limit:
        raise ValueError(f"over {limit // 2 ** 20} MB")
    return body


def _json(url: str, source: str, ledger: Path) -> Any:
    return json.loads(fetch(url, source, ledger).decode("utf-8"))


def _q(query: str) -> str:
    return urllib.parse.quote(query)


def _words(query: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", query.lower()) if len(w) > 2]


def search_source(source: str, query: str, ledger: Path, want: int = PER_SOURCE) -> list[dict[str, Any]]:
    """Up to ``want`` passing candidates from one source, in its own order (a few requests)."""
    q = _q(query)
    if source == "commons":
        url = ("https://commons.wikimedia.org/w/api.php?action=query&format=json&maxlag=5&generator=search"
               f"&gsrnamespace=6&gsrlimit=20&gsrsearch=filetype:bitmap%20{q}&prop=imageinfo"
               "&iiprop=url|size|mime|extmetadata&iiurlwidth=2048&iiextmetadatafilter=License|"
               "LicenseShortName|LicenseUrl|Artist|Restrictions")
        return parse_commons(_json(url, source, ledger), query)[:want]
    if source == "met":
        ids = (_json(f"https://collectionapi.metmuseum.org/public/collection/v1.1/search?q={q}"
                     "&hasImages=true&limit=12&offset=0", source, ledger) or {}).get("objectIDs") or []
        found, words = [], _words(query)
        for object_id in ids[:8]:
            record = _json(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}",
                           source, ledger)
            item = parse_met_object(record, query)
            # The hits include prints that only mention the word: the object's own name must hold one.
            name = f"{record.get('objectName', '')} {record.get('title', '')}".lower()
            if item and (not words or any(w in name for w in words)):
                found.append(item)
            if len(found) >= want:
                break
        return found
    if source == "rijksmuseum":
        page = _json(f"https://data.rijksmuseum.nl/search/collection?description={q}&imageAvailable=true",
                     source, ledger)
        found = []
        for entry in (page.get("orderedItems") or [])[:6]:
            xml = fetch("https://data.rijksmuseum.nl/oai?verb=GetRecord&metadataPrefix=edm&identifier="
                        + _q(entry["id"]), source, ledger).decode("utf-8", "replace")
            item = parse_rijks_record(xml, query)
            if item:
                found.append(item)
            if len(found) >= want:
                break
        return found
    if source == "wellcome":
        return parse_wellcome(_json(f"https://api.wellcomecollection.org/catalogue/v2/images?query={q}"
                                    "&locations.license=cc-0,pdm,cc-by&pageSize=20&include=source.contributors",
                                    source, ledger), query)[:want]
    if source == "aic":
        return parse_aic(_json(f"https://api.artic.edu/api/v1/artworks/search?q={q}"
                               "&query[term][is_public_domain]=true&fields=id,title,artist_display,image_id,"
                               "is_public_domain&limit=20", source, ledger), query)[:want]
    if source == "cleveland":
        return parse_cleveland(_json(f"https://openaccess-api.clevelandart.org/api/artworks/?q={q}"
                                     "&cc0=1&has_image=1&limit=20", source, ledger), query)[:want]
    if source == "openverse":
        return parse_openverse(_json(f"https://api.openverse.org/v1/images/?q={q}&license=cc0,pdm,by"
                                     "&category=photograph&page_size=20", source, ledger), query)[:want]
    if source == "mia":
        # In the URL path: a '/' in the words must not change the path.
        return parse_mia(_json(f"https://search.artsmia.org/{urllib.parse.quote(query, safe='')}"
                               "%20public_access:1?size=20", source, ledger), query)[:want]
    raise ValueError(f"unknown source {source!r}")


def relevant(candidate: dict[str, Any], query: str) -> bool:
    """Every query word (a plural's stem) is in the item's title.  The museums match a word anywhere
    in a record: a query for a modern product filled the stop with prints titled after something
    else (measured on one query, two runs: 3 of 4 candidates each), and kept the sources that have
    the product (Openverse) from being reached."""
    title = candidate.get("title", "").lower()
    return all(re.sub(r"(es|s)$", "", w) in title for w in _words(query))


# Words a caption uses for the people in a photograph: such a photograph shows the object in use,
# often small, and carries personality rights, so it is ranked after the others (never dropped).
PEOPLE_WORDS = ("man", "men", "woman", "women", "people", "person", "boy", "girl", "child", "children",
                "crew", "sailor", "sailors", "soldier", "soldiers", "worker", "workers", "officer",
                "technician", "lieutenant", "colonel", "sergeant", "seaman", "portrait", "family")


def rank(candidate: dict[str, Any]) -> tuple[int, int]:
    """Photographs without people in the caption first, then the shortest titles: a photograph of
    the object itself is titled after it, a scene that shows it in passing has a long caption."""
    words = re.findall(r"[a-z]+", candidate.get("title", "").lower())
    return (int(any(w in PEOPLE_WORDS for w in words)), len(candidate.get("title", "").split()))


def search(query: str, ledger: Path, want: int = WANT, per_source: int = PER_SOURCE,
           sources: tuple[str, ...] = SOURCES,
           one: Callable[..., list[dict[str, Any]]] = search_source,
           cap_s: float = SEARCH_S, now: Callable[[], float] = time.time) -> dict[str, Any]:
    """Query the sources in order until ``want`` relevant candidates pass; a source that fails or
    is at its cap is logged and skipped (a network failure is not an error).  Past ``cap_s`` in
    all, the sources not yet asked are logged unreachable."""
    found: list[dict[str, Any]] = []
    log = []
    _deadline[0] = now() + cap_s
    for source in sources:
        if len(found) >= want:
            break
        if now() >= _deadline[0]:
            log.append({"source": source, "error": f"unreachable: the search's {cap_s:.0f} s cap was reached"})
            continue
        try:
            items = sorted((i for i in one(source, query, ledger, 3 * per_source) if relevant(i, query)),
                           key=rank)
            items = items[:min(per_source, want - len(found))]
            log.append({"source": source, "passed": len(items)})
            found += [i for i in items if i["id"] not in {f["id"] for f in found}]
        except SourceLimit as exc:
            log.append({"source": source, "skipped": str(exc)})
        except Exception as exc:
            log.append({"source": source, "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
    _deadline[0] = float("inf")         # a download after the search keeps REQUEST_S alone
    return {"query": query, "candidates": found[:want], "sources": log}


def download(candidate: dict[str, Any], folder: Path, ledger: Path,
             get: Callable[..., bytes] = fetch) -> dict[str, Any]:
    """The candidate's image into ``folder`` beside its ``attribution.json``."""
    body = get(candidate["image_url"], candidate["source"], ledger, api=False, limit=MAX_IMAGE_BYTES)
    kind = "png" if body[:8] == b"\x89PNG\r\n\x1a\n" else "jpg" if body[:2] == b"\xff\xd8" else None
    if kind is None:
        raise ValueError("not a JPEG or PNG")
    folder.mkdir(parents=True, exist_ok=True)
    image = folder / f"original.{kind}"
    image.write_bytes(body)
    record = {**{k: candidate[k] for k in ("id", "title", "author", "source_url", "licence", "licence_url",
                                            "query", "source", "image_url")},
              "licence_checked_at_source": candidate.get("licence_checked_at_source", True),
              "retrieved": date.today().isoformat(), "sha256": hashlib.sha256(body).hexdigest(),
              "credit": credit_line(candidate)}
    (folder / "attribution.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return {**record, "original": image}


def credit_line(record: dict[str, Any]) -> str:
    """The figure credit: title, author, licence, source page."""
    unchecked = record.get("licence_checked_at_source") is False
    return (f"\"{record.get('title') or 'untitled'}\" by {record.get('author') or 'unknown'}, "
            f"{record.get('licence')} ({record.get('licence_url')})"
            f"{' as listed by Openverse, not checked at the source page' if unchecked else ''}, "
            f"{record.get('source_url')}")


# ---------------------------------------------------------------- processing --
# Each downloaded photograph is made into what a drawing is before the agent sees it: the named
# object found and cropped (Grounding DINO), cut out from its background (RMBG-1.4, exactly as a
# drawing is: hunyuan3d_wrapper/draw_image.py), and pre-screened, so that a print, a scene or a
# failed cut-out never reaches the stop.  Grounding DINO runs in the solver env (transformers 5)
# as in mechfoundry-hunyuan3d (4.46), probed 2026-10-11; it is optional and loaded offline only.

DETECTOR = "IDEA-Research/grounding-dino-base"
BOX_THRESHOLD, TEXT_THRESHOLD = 0.25, 0.2
PAD = 0.08                 # of the box's width and height, on each side
SCREEN_SIDE = 128          # the mask is screened at this size
KEPT_SHARE = (0.05, 0.90)  # the cut-out's share of the crop: under, no object kept; over, no background removed
LARGE_PIECE = 0.2          # a piece of the mask this share of the largest or more is a large one
THUMB_SIDE = 320           # the thumbnail the stop shows


def detector_cached() -> bool:
    """Whether Grounding DINO's weights are in the Hugging Face cache: without them the whole
    photograph is cut out (no crop), and nothing is downloaded."""
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME") or
                                                       Path.home() / ".cache" / "huggingface") / "hub")
    folder = hub / ("models--" + DETECTOR.replace("/", "--")) / "snapshots"
    return any(folder.glob("*/model.safetensors")) or any(folder.glob("*/pytorch_model.bin"))


def pieces(mask) -> list[int]:
    """The areas of the 4-connected pieces of a boolean mask, largest first (no scipy here)."""
    import numpy as np
    seen = np.zeros(mask.shape, dtype=bool)
    areas = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        seen[y0, x0], stack, area = True, [(y0, x0)], 0
        while stack:
            y, x = stack.pop()
            area += 1
            for v, u in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1] and mask[v, u] and not seen[v, u]:
                    seen[v, u] = True
                    stack.append((v, u))
        areas.append(area)
    return sorted(areas, reverse=True)


def screen(image) -> dict[str, Any]:
    """The pre-screen of one cut-out (an RGBA PIL image): the share of it kept, its large pieces,
    its colour, and the reasons it is dropped (none: it may be shown)."""
    import numpy as np
    from PIL import Image
    image = image.convert("RGBA")
    rgba = np.asarray(image)
    solid = rgba[..., 3] > 127
    share = float(solid.mean()) if solid.size else 0.0
    small = image.getchannel("A")
    small.thumbnail((SCREEN_SIDE, SCREEN_SIDE), Image.NEAREST)
    areas = pieces(np.asarray(small) > 127)
    large = sum(a >= LARGE_PIECE * areas[0] for a in areas) if areas else 0
    rgb = rgba[..., :3][solid].astype(int)
    chroma = float(np.median(rgb.max(axis=1) - rgb.min(axis=1))) if len(rgb) else 0.0
    reasons = []
    if share < KEPT_SHARE[0]:
        reasons.append(f"the cut-out keeps {share:.0%} of the image: no object")
    if share > KEPT_SHARE[1]:
        reasons.append(f"the cut-out keeps {share:.0%} of the image: the background was not removed")
    if large > 1:
        reasons.append(f"{large} large pieces: several objects, or one cut apart")
    # No greyscale reason: measured on 16 real candidates (2026-10-11) it dropped a bare steel
    # object on a grey backdrop and a black-and-white photograph that made a good body as often
    # as it dropped a print; the colour is recorded and a print is judged at the stop instead.
    return {"kept_share": round(share, 3), "large_pieces": int(large), "chroma": round(chroma, 1),
            "reasons": reasons}


def padded(box: list[float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    """``box`` grown by PAD of its width and height on each side, inside the image."""
    x0, y0, x1, y1 = box
    pw, ph = PAD * (x1 - x0), PAD * (y1 - y0)
    return (int(max(0, x0 - pw)), int(max(0, y0 - ph)), int(min(size[0], x1 + pw)), int(min(size[1], y1 + ph)))


def _detect(model, processor, image, text: str, device: str) -> tuple[list[float] | None, float | None]:
    """The top-scoring box of ``text`` in ``image`` and its score, or (None, None)."""
    import torch
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    post = processor.post_process_grounded_object_detection
    sizes = [image.size[::-1]]
    try:            # transformers 5 names the box threshold `threshold`, 4.x `box_threshold`
        found = post(outputs, inputs.input_ids, threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
                     target_sizes=sizes)[0]
    except TypeError:
        found = post(outputs, inputs.input_ids, box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
                     target_sizes=sizes)[0]
    if not len(found["boxes"]):
        return None, None
    k = int(torch.argmax(found["scores"]))
    return [float(v) for v in found["boxes"][k].tolist()], float(found["scores"][k])


def thumbnail(cut, target: Path) -> None:
    """The cut-out on white, at most THUMB_SIDE a side."""
    from PIL import Image
    image = cut.convert("RGBA")
    flat = Image.new("RGB", image.size, "white")
    flat.paste(image, mask=image.getchannel("A"))
    flat.thumbnail((THUMB_SIDE, THUMB_SIDE))
    flat.save(target)


def process_files(phrase: str, pairs: list[tuple[Path, Path]]) -> dict[str, Any]:
    """Each original detected, cropped, cut out and screened into its folder (cut.png, thumb.png,
    screen.json); run in the compute env, on a GPU.  The detector's text is the query's noun phrase
    and a full stop; with no box (or no detector) the whole photograph is cut out."""
    repo = Path(__file__).resolve().parents[5]
    sys.path.insert(0, os.fspath(repo))
    os.environ["HF_HUB_OFFLINE"] = "1"         # the detector is used only from the cache, never fetched
    import torch
    from PIL import Image
    from src.hunyuan3d_wrapper.draw_image import cut_out, load_rmbg
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device: RMBG-1.4 cuts out on a GPU only")
    net = load_rmbg()
    if net is None:
        raise RuntimeError("RMBG-1.4 is not available (not downloaded, and offline): no cut-out")
    device = "cuda"
    detector = processor = None
    note = None
    if detector_cached():
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            processor = AutoProcessor.from_pretrained(DETECTOR)
            detector = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR).to(device).eval()
        except Exception as exc:       # optional: the whole photograph is cut out
            note = f"no crop ({type(exc).__name__}: {str(exc)[:160]})"
    else:
        note = f"no crop: {DETECTOR} is not cached"
    text = phrase.strip().lower().rstrip(".") + "."
    out = []
    for source, folder in pairs:
        folder.mkdir(parents=True, exist_ok=True)
        try:
            # one photograph that cannot be read (corrupt, over Pillow's pixel limit) or processed is
            # dropped with its reason; the others are still shown
            image = Image.open(source).convert("RGB")
            image.thumbnail((1536, 1536))
            box = score = None
            if detector is not None:
                box, score = _detect(detector, processor, image, text, device)
            cut = cut_out(net, image.crop(padded(box, image.size)) if box else image)
            cut.save(folder / "cut.png")
            thumbnail(cut, folder / "thumb.png")
            record = {"box": [round(v, 1) for v in box] if box else None,
                      "score": round(score, 3) if score is not None else None, **screen(cut)}
        except Exception as exc:
            record = {"reasons": [f"processing failed: {type(exc).__name__}: {str(exc)[:120]}"]}
        (folder / "screen.json").write_text(json.dumps(record), encoding="utf-8")
        out.append({"folder": os.fspath(folder), **record})
    return {"processed": out, "detector": DETECTOR if detector is not None else None,
            **({"note": note} if note else {})}


def main(argv: list[str] | None = None) -> None:
    """The command line does only the processing, which the runner runs in the compute env.
    Searching and downloading have no command line on purpose: the runner calls them in-process,
    behind the web_image stop, its attempt counter and the route order, so no command can fetch
    outside them."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("process", help="IMAGE FOLDER pairs: detect, crop, cut out, screen (compute env)")
    p.add_argument("--phrase", required=True, help="the query's noun phrase: the detector's text")
    p.add_argument("files", nargs="+")
    args = parser.parse_args(argv)
    if len(args.files) % 2:
        parser.error("process takes IMAGE FOLDER pairs")
    pairs = [(Path(a), Path(b)) for a, b in zip(args.files[::2], args.files[1::2])]
    print(json.dumps(process_files(args.phrase, pairs)))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
