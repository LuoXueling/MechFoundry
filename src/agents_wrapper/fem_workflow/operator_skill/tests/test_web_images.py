"""The web-image module's licence filters, ledger and attribution, on responses recorded from each
source (trimmed to the fields read): no network.

    python -m pytest src/agents_wrapper/fem_workflow/operator_skill/tests/test_web_images.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import web_images as web  # noqa: E402


def _meta(licence, short="", url="", restrictions="", artist="<a href='x'>A. Person</a>"):
    return {"License": {"value": licence}, "LicenseShortName": {"value": short},
            "LicenseUrl": {"value": url}, "Restrictions": {"value": restrictions}, "Artist": {"value": artist}}


COMMONS = {"query": {"pages": {
    "3": {"pageid": 3, "index": 3, "title": "File:Item pd.jpg", "imageinfo": [{
        "url": "https://upload.example/pd.jpg", "width": 3300, "thumburl": "https://upload.example/pd-2048.jpg",
        "mime": "image/jpeg", "descriptionurl": "https://commons.example/File:Item_pd.jpg",
        "extmetadata": _meta("pd", "Public domain")}]},
    "1": {"pageid": 1, "index": 1, "title": "File:Item zero.png", "imageinfo": [{
        "url": "https://upload.example/zero.png", "width": 800, "mime": "image/png",
        "descriptionurl": "https://commons.example/File:Item_zero.png",
        "extmetadata": _meta("cc0", "CC0", "http://creativecommons.org/publicdomain/zero/1.0/deed.en")}]},
    "2": {"pageid": 2, "index": 2, "title": "File:Item sa.jpg", "imageinfo": [{
        "url": "https://upload.example/sa.jpg", "width": 800, "mime": "image/jpeg",
        "extmetadata": _meta("cc-by-sa-4.0", "CC BY-SA 4.0")}]},
    "4": {"pageid": 4, "index": 4, "title": "File:Item by.jpg", "imageinfo": [{
        "url": "https://upload.example/by.jpg", "width": 800, "mime": "image/jpeg",
        "extmetadata": _meta("cc-by-4.0", "CC BY 4.0", "https://creativecommons.org/licenses/by/4.0")}]},
    "5": {"pageid": 5, "index": 5, "title": "File:Item brand.jpg", "imageinfo": [{
        "url": "https://upload.example/tm.jpg", "width": 800, "mime": "image/jpeg",
        "extmetadata": _meta("cc0", restrictions="trademarked")}]},
    "6": {"pageid": 6, "index": 6, "title": "File:Item kogl.jpg", "imageinfo": [{
        "url": "https://upload.example/kogl.jpg", "width": 800, "mime": "image/jpeg",
        "extmetadata": {"LicenseUrl": {"value": "http://www.kogl.or.kr/info/licenseType1.do"}}}]},
    "7": {"pageid": 7, "index": 7, "title": "File:Item.svg", "imageinfo": [{
        "url": "https://upload.example/item.svg", "width": 800, "mime": "image/svg+xml",
        "extmetadata": _meta("cc0")}]}}}}

RIJKS_EDM = """<ore:Aggregation rdf:about="https://id.rijksmuseum.nl/200107292#aggregation">
<edm:isShownBy rdf:resource="https://iiif.micr.io/aqYhe/full/max/0/default.jpg"/>
<edm:rights rdf:resource="http://creativecommons.org/publicdomain/mark/1.0/"/></ore:Aggregation>
<edm:ProvidedCHO rdf:about="https://id.rijksmuseum.nl/200107292">
<dc:creator>anonymous</dc:creator>
<dc:title xml:lang="en">Item with flowering plants</dc:title>
<dc:title xml:lang="nl">Ding met bloemen</dc:title></edm:ProvidedCHO>"""


def test_commons_passes_cc0_pd_and_cc_by_only_in_its_own_order():
    found = web.parse_commons(COMMONS, "item")
    assert [c["id"] for c in found] == ["commons-1", "commons-3", "commons-4"]
    zero, pd, by = found
    assert zero["image_url"] == "https://upload.example/zero.png"            # the original, small enough
    assert pd["image_url"] == "https://upload.example/pd-2048.jpg"           # a large original's rendition
    assert pd["licence_url"] == web.PDM and by["licence"] == "CC BY 4.0"
    assert zero["author"] == "A. Person" and zero["title"] == "Item zero.png" and zero["query"] == "item"


def test_met_takes_the_object_record_flag_not_the_search_filter():
    record = {"objectID": 8356, "isPublicDomain": True, "primaryImage": "https://images.example/8356.jpg",
              "title": "Item", "artistDisplayName": "", "culture": "American",
              "objectURL": "https://www.metmuseum.org/art/collection/search/8356"}
    item = web.parse_met_object(record, "item")
    assert item["licence"] == "CC0 1.0" and item["author"] == "American" and item["id"] == "met-8356"
    assert web.parse_met_object({**record, "isPublicDomain": False}, "item") is None
    assert web.parse_met_object({**record, "primaryImage": ""}, "item") is None


def test_rijksmuseum_reads_the_edm_rights_and_the_english_title():
    item = web.parse_rijks_record(RIJKS_EDM, "item")
    assert item["licence"] == "Public Domain Mark 1.0" and item["title"] == "Item with flowering plants"
    assert item["image_url"].endswith("/full/max/0/default.jpg") and item["id"] == "rijksmuseum-200107292"
    rights = "http://rightsstatements.org/vocab/InC/1.0/"
    assert web.parse_rijks_record(RIJKS_EDM.replace("http://creativecommons.org/publicdomain/mark/1.0/", rights),
                                  "item") is None


def test_wellcome_aic_cleveland_openverse_loc_and_mia_filter_on_their_own_fields():
    wellcome = {"results": [
        {"id": "a", "locations": [{"url": "https://iiif.example/a/info.json",
                                   "license": {"id": "cc-by", "label": "CC BY", "url": "https://cc/by"}}],
         "source": {"id": "w1", "title": "An item", "contributors": []}},
        {"id": "b", "locations": [{"url": "https://iiif.example/b/info.json", "license": {"id": "cc-by-nc"}}],
         "source": {"id": "w2", "title": "x"}}]}
    found = web.parse_wellcome(wellcome, "item")
    assert [c["id"] for c in found] == ["wellcome-a"]
    assert found[0]["image_url"] == "https://iiif.example/a/full/full/0/default.jpg"
    assert found[0]["author"] == "Wellcome Collection"
    aic = {"data": [{"id": 1, "title": "Item", "image_id": "abc", "is_public_domain": True},
                    {"id": 2, "title": "Item", "image_id": "def", "is_public_domain": False},
                    {"id": 3, "title": "Item", "image_id": None, "is_public_domain": True}]}
    assert [c["image_url"] for c in web.parse_aic(aic, "item")] == [
        "https://www.artic.edu/iiif/2/abc/full/1686,/0/default.jpg"]
    cleveland = {"data": [{"id": 9, "title": "Item", "share_license_status": "CC0",
                           "images": {"print": {"url": "https://cma.example/9_print.jpg"}}},
                          {"id": 10, "title": "Item", "share_license_status": "Copyrighted",
                           "images": {"print": {"url": "https://cma.example/10.jpg"}}}]}
    assert [c["id"] for c in web.parse_cleveland(cleveland, "item")] == ["cleveland-9"]
    openverse = {"results": [
        {"id": "o1", "license": "by", "license_version": "2.0", "url": "https://flickr.example/1_b.jpg",
         "foreign_landing_url": "https://flickr.example/photos/1", "license_url": "https://cc/by/2.0", "creator": "C"},
        {"id": "o2", "license": "by-sa", "url": "https://flickr.example/2_b.jpg"},
        {"id": "o3", "license": "by-nc", "url": "https://flickr.example/3_b.jpg"},
        {"id": "o4", "license": "cc0", "url": "https://flickr.example/4_b.jpg"}]}
    found = web.parse_openverse(openverse, "item")
    assert [c["id"] for c in found] == ["openverse-o1", "openverse-o4"] and found[0]["licence"] == "CC BY 2.0"
    assert found[0]["licence_checked_at_source"] is False      # Openverse does not verify its licences
    mia ={"hits": {"hits": [{"_source": {"id": 14955, "title": "Item", "rights_type": "Public Domain",
                                          "public_access": 1, "image_width": 4468}},
                             {"_source": {"id": 2, "title": "Item", "rights_type": "Rights Reserved",
                                          "public_access": 1, "image_width": 100}}]}}
    assert [c["image_url"] for c in web.parse_mia(mia, "item")] == ["https://1.api.artsmia.org/800/14955.jpg"]


def test_search_goes_in_order_takes_at_most_two_a_source_and_skips_a_failing_one(tmp_path):
    asked = []

    def one(source, query, ledger, want):
        asked.append((source, want))
        if source == "met":
            raise OSError("network unreachable")
        if source == "rijksmuseum":
            raise web.SourceLimit("rijksmuseum: cap")
        return ([web._candidate(source, "x", "A print of something else", "a", "p", "CC0", web.CC0, "u", query)]
                + [web._candidate(source, k, f"Items {k}", "a", "p", "CC0", web.CC0, "u", query)
                   for k in range(want)])

    result = web.search("item", tmp_path / "ledger.json", one=one)
    assert [s for s, _ in asked] == ["commons", "met", "rijksmuseum", "wellcome"]
    assert [c["source"] for c in result["candidates"]] == ["commons"] * 3 + ["wellcome"] * 3
    assert all(c["title"].startswith("Items") for c in result["candidates"])   # the print is not shown
    short = web.search("item", tmp_path / "ledger.json", sources=("commons",), one=lambda s, q, l, w: [
        web._candidate(s, 1, "Sailors carrying an item across the deck, 1966", "a", "p", "CC0", web.CC0, "u", q),
        web._candidate(s, 2, "Item", "a", "p", "CC0", web.CC0, "u", q)])
    assert [c["id"] for c in short["candidates"]] == ["commons-2", "commons-1"]   # the object's own photo first
    assert web.relevant({"title": "An improvised red item on a wheeled frame"}, "red items")
    assert not web.relevant({"title": "A red thing R01.jpg"}, "red items")
    assert "network unreachable" in result["sources"][1]["error"] and "skipped" in result["sources"][2]
    assert web.SOURCES == ("commons", "met", "rijksmuseum", "wellcome", "aic", "cleveland", "openverse", "mia")
    assert set(web.PAUSE_S) == set(web.SOURCES) and not hasattr(web, "parse_loc")


def test_a_request_waits_ten_seconds_and_a_search_sixty_in_all(tmp_path, monkeypatch):
    clock = [0.0]

    def one(source, query, ledger, want):
        clock[0] += 25.0                       # each source answers nothing after 25 s
        raise OSError("timed out")

    result = web.search("item", tmp_path / "ledger.json", one=one, now=lambda: clock[0])
    assert [s["source"] for s in result["sources"]] == list(web.SOURCES)
    assert sum("timed out" in s["error"] for s in result["sources"]) == 3       # 0, 25, 50 s: asked
    assert all("unreachable" in s["error"] for s in result["sources"][3:])
    assert web.REQUEST_S == 10.0 and web._deadline[0] == float("inf")
    timeouts = []

    class Answer:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n):
            return b"{}"

    monkeypatch.setattr(web, "polite_wait", lambda *a, **k: None)
    monkeypatch.setattr(web.urllib.request, "urlopen", lambda request, timeout: timeouts.append(timeout) or Answer())
    web.fetch("https://example.invalid/x", "commons", tmp_path / "ledger.json")
    assert timeouts == [10.0]
    web._deadline[0] = web.time.time() - 1
    with pytest.raises(web.SearchCap):
        web.fetch("https://example.invalid/x", "commons", tmp_path / "ledger.json")
    web._deadline[0] = float("inf")


def test_a_no_known_restrictions_advisory_passes_no_source():
    """A rights advisory is no licence (the user's decision): every source refuses it."""
    nkr = "No known restrictions on publication"
    commons = {"query": {"pages": {"1": {"pageid": 1, "index": 1, "title": "File:Item.jpg", "imageinfo": [{
        "url": "https://upload.example/i.jpg", "width": 800, "mime": "image/jpeg",
        "extmetadata": _meta("no known restrictions", nkr)}]}}}}
    assert web.parse_commons(commons, "item") == []
    assert web.parse_met_object({"objectID": 1, "isPublicDomain": False, "rightsAndReproduction": nkr,
                                 "primaryImage": "https://images.example/1.jpg"}, "item") is None
    assert web.parse_rijks_record(RIJKS_EDM.replace("http://creativecommons.org/publicdomain/mark/1.0/",
                                                    "http://rightsstatements.org/vocab/NKC/1.0/"), "item") is None
    assert web.parse_wellcome({"results": [{"id": "a", "locations": [{"url": "https://iiif.example/a/info.json",
                                "license": {"id": "nkr", "label": nkr}}], "source": {"id": "w"}}]}, "item") == []
    assert web.parse_aic({"data": [{"id": 1, "image_id": "abc", "is_public_domain": False}]}, "item") == []
    assert web.parse_cleveland({"data": [{"id": 9, "share_license_status": nkr,
                                          "images": {"print": {"url": "https://cma.example/9.jpg"}}}]}, "item") == []
    assert web.parse_openverse({"results": [{"id": "o", "license": "nkr", "url": "https://x/o.jpg"}]}, "item") == []
    assert web.parse_mia({"hits": {"hits": [{"_source": {"id": 3, "rights_type": nkr, "public_access": 1,
                                                         "image_width": 9}}]}}, "item") == []


def test_photographs_with_people_rank_after_the_others_and_are_kept():
    found = web.search("item", Path("unused"), sources=("commons",), one=lambda s, q, l, w: [
        web._candidate(s, 1, "Item", "a", "p", "CC0", web.CC0, "u", q),
        web._candidate(s, 2, "Two workers lifting an item", "a", "p", "CC0", web.CC0, "u", q),
        web._candidate(s, 3, "A long caption about an item on a shelf", "a", "p", "CC0", web.CC0, "u", q)])
    assert [c["id"] for c in found["candidates"]] == ["commons-1", "commons-3", "commons-2"]


def test_the_search_has_no_command_line_and_the_path_query_is_quoted_whole(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        web.main(["search", "--query", "item", "--out", str(tmp_path)])
    urls = []
    monkeypatch.setattr(web, "_json", lambda url, source, ledger: urls.append(url) or {})
    web.search_source("mia", "a/b item", tmp_path / "ledger.json")
    assert urls[0].startswith("https://search.artsmia.org/a%2Fb%20item%20public_access:1")


def test_the_ledger_keeps_each_source_s_pause_and_caps(tmp_path):
    clock, slept = [1000.0], []

    def now():
        return clock[0]

    def sleep(seconds):
        slept.append(round(seconds, 2))
        clock[0] += seconds

    ledger = tmp_path / "ledger.json"
    web.polite_wait(ledger, "openverse", now=now, sleep=sleep)
    web.polite_wait(ledger, "openverse", now=now, sleep=sleep)
    assert slept == [3.0]                                     # Openverse: 20 a minute
    for _ in range(18):
        web.polite_wait(ledger, "openverse", now=now, sleep=sleep)
    slept.clear()
    web.polite_wait(ledger, "openverse", now=now, sleep=sleep)   # the 21st within a minute waits it out
    assert slept and slept[0] > 3.0
    state = json.loads(ledger.read_text())
    state["count"]["openverse"] = 200
    ledger.write_text(json.dumps(state))
    with pytest.raises(web.SourceLimit, match="daily cap"):
        web.polite_wait(ledger, "openverse", now=now, sleep=sleep)
    web.polite_wait(ledger, "openverse", api=False, now=now, sleep=sleep)   # an image download is not counted


def test_a_download_writes_the_image_and_its_attribution(tmp_path):
    candidate = web.parse_commons(COMMONS, "item")[2]

    def get(url, source, ledger, api, limit):
        assert url == candidate["image_url"] and source == "commons" and api is False
        return b"\xff\xd8\xff\xe0 a jpeg"

    record = web.download(candidate, tmp_path / candidate["id"], tmp_path / "ledger.json", get=get)
    saved = json.loads((tmp_path / candidate["id"] / "attribution.json").read_text())
    assert record["original"].name == "original.jpg" and record["original"].exists()
    for key in ("title", "author", "licence", "licence_url", "query", "sha256", "retrieved"):
        assert saved[key]
    assert saved["credit"].startswith('"Item by.jpg" by A. Person, CC BY 4.0 (https://creativecommons.org/licenses/by/4.0)')
    assert saved["licence_checked_at_source"] is True
    with pytest.raises(ValueError, match="JPEG or PNG"):
        web.download(candidate, tmp_path / "x", tmp_path / "ledger.json", get=lambda *a, **k: b"<html>")
    listed = web.parse_openverse({"results": [{"id": "o", "license": "by", "url": "https://x/o.jpg",
                                               "foreign_landing_url": "https://x/p"}]}, "item")[0]
    web.download(listed, tmp_path / "o", tmp_path / "ledger.json", get=lambda *a, **k: b"\xff\xd8 jpeg")
    saved = json.loads((tmp_path / "o" / "attribution.json").read_text())
    assert saved["licence_checked_at_source"] is False and "as listed by Openverse" in saved["credit"]


def test_the_user_agent_names_the_project_and_never_an_address():
    assert "github.com" in web.USER_AGENT and "@" not in web.USER_AGENT
    assert "@" not in web.AIC_HEADERS["User-Agent"] and "AIC-User-Agent" in web.AIC_HEADERS


def _cut(mask, colour=(200, 60, 40)):
    """An RGBA cut-out: ``colour`` where ``mask`` holds, transparent elsewhere."""
    import numpy as np
    from PIL import Image
    rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)
    rgba[mask] = (*colour, 255)
    return Image.fromarray(rgba, "RGBA")


def test_the_pre_screen_drops_no_object_no_cut_out_several_pieces_and_grey():
    import numpy as np
    one = np.zeros((200, 200), dtype=bool)
    one[40:160, 60:140] = True
    assert web.screen(_cut(one))["reasons"] == []                      # one coloured object, cut out
    tiny = np.zeros((200, 200), dtype=bool)
    tiny[10:20, 10:20] = True
    assert "no object" in web.screen(_cut(tiny))["reasons"][0]
    assert "background was not removed" in web.screen(_cut(np.ones((200, 200), dtype=bool)))["reasons"][0]
    two = np.zeros((200, 200), dtype=bool)
    two[40:160, 10:80] = True
    two[40:160, 120:190] = True
    assert web.screen(_cut(two))["large_pieces"] == 2
    speck = one.copy()
    speck[5:8, 5:8] = True                                               # a speck is no second piece
    assert web.screen(_cut(speck))["large_pieces"] == 1
    grey = web.screen(_cut(one, (120, 122, 118)))           # a grey object is kept: judged at the stop
    assert grey["reasons"] == [] and grey["chroma"] < 10


def test_the_crop_is_the_box_padded_eight_percent_inside_the_photograph(tmp_path, monkeypatch):
    assert web.padded([100, 50, 200, 150], (1000, 1000)) == (92, 42, 208, 158)
    assert web.padded([0, 0, 100, 100], (100, 100)) == (0, 0, 100, 100)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert not web.detector_cached()                                    # optional: no crop without it
    snapshot = tmp_path / "models--IDEA-Research--grounding-dino-base" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"x")
    assert web.detector_cached()
    with pytest.raises(SystemExit):
        web.main(["process", "--phrase", "item", "only-one-file"])


def test_one_unreadable_photograph_is_dropped_with_its_reason_and_no_cuda_fails_the_whole(tmp_path, monkeypatch):
    """No GPU: torch and the cut-out are stand-ins, so the per-image guard is what is tested."""
    import types
    from PIL import Image
    cuda = [True]
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda[0])))
    monkeypatch.setitem(sys.modules, "src.hunyuan3d_wrapper.draw_image", types.SimpleNamespace(
        load_rmbg=lambda: object(), cut_out=lambda net, image: image.convert("RGBA")))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))          # no detector: no crop
    good, bad = tmp_path / "good.png", tmp_path / "bad.jpg"
    Image.new("RGB", (64, 64), "red").save(good)
    bad.write_bytes(b"\xff\xd8 not really a jpeg")
    out = web.process_files("item", [(bad, tmp_path / "a"), (good, tmp_path / "b")])
    assert "processing failed" in out["processed"][0]["reasons"][0]
    assert (tmp_path / "b" / "cut.png").exists() and "processing failed" not in str(out["processed"][1])
    assert "processing failed" in json.loads((tmp_path / "a" / "screen.json").read_text())["reasons"][0]
    cuda[0] = False
    with pytest.raises(RuntimeError, match="no CUDA device"):
        web.process_files("item", [(good, tmp_path / "c")])
