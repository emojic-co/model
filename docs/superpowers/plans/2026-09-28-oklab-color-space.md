# OKLAB Color Space Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the model's raw-sRGB 3×[-1,1]-per-color representation with an OKLAB-based one, so the color GAN's energy-distance loss operates in a perceptually uniform space, and propagate OKLAB as the working color space through the web app and Android app (both still render final pixels as sRGB/hex, since neither platform can practically render native OKLab at the pixel level — Android's `ColorSpace.Named.OKLAB` needs API 34+ and this app's minSdk is 26).

**Architecture:** `COLOR_DIM` stays 9 (3 colors × 3 channels, each channel in [-1,1]) — no model architecture change. Only the *meaning* of those 9 floats changes: each channel-triple now encodes `[L-unit, a-unit, b-unit]` where `L = (unit0+1)/2 ∈ [0,1]`, `a = unit1*0.4 ∈ [-0.4,0.4]`, `b = unit2*0.4 ∈ [-0.4,0.4]`. Dataset loading (`model/data.py`) converts sRGB hex → OKLAB → clamps a/b to ±0.4 → maps to unit range. Every other consumer (GAN, critic, ONNX export, web, Android) works in this unit-encoded OKLAB space; sRGB only reappears at the very last mile (a pixel needs an actual RGB value to display), converted via `oklabToSrgb`. Per an explicit scope decision below, the web app renders using native CSS `oklab()` strings (no lossy sRGB round-trip for on-screen color), while Android converts to sRGB hex for `Color.parseColor`/Compose (no native OKLab rendering path available at minSdk 26).

**Tech Stack:** PyTorch (model), TypeScript/React + Vitest (web), Kotlin + JUnit (Android).

**Spec:** `todo.txt` lines 1–15 (the OKLAB discussion note), refined through conversation. No separate spec doc exists; the Global Constraints below are the spec, transcribed from that discussion plus the scope decision the user made when this plan was requested (web renders native `oklab()` CSS; Android stays hex/`Color`).

## Global Constraints

- Model color representation: 3×[-1,1] per color, unchanged shape (`COLOR_DIM = 9`).
- Decode (model-unit → real OKLAB): `L = (c[0] + 1) / 2`, `a = c[1] * 0.4`, `b = c[2] * 0.4`.
- Encode (dataset hex → model-unit): convert sRGB → OKLAB, clamp `a`/`b` to `[-0.4, 0.4]`, then map to `[-1, 1]` via the inverse of the decode formula.
- The only place raw RGB is the working representation is the raw dataset (hex strings in `data/*.jsonl`). Everywhere else — GAN, critic, ONNX output, web app, Android app — works in OKLAB (unit-encoded or real), converting to sRGB only at the pixel-rendering boundary.
- Web renders colors with native CSS `oklab()` strings, not hex, wherever a model-derived or OKLAB-math-derived color reaches a style/canvas property.
- Android renders colors as sRGB hex / `Color.parseColor`, unchanged from today, because Compose has no practical native OKLab rendering path at minSdk 26.
- This migration invalidates every existing `.pt` checkpoint (the color GAN's output space changes meaning even though its tensor shape doesn't) — a full retrain is required before the report/goals numbers are meaningful again.

## Review Focus

- **Saturated dataset colors whose true OKLAB a/b exceed ±0.4** — should clamp silently during dataset loading, not raise or silently corrupt the tensor. (Real sRGB gamut colors top out well under 0.4 in practice, but the clamp is the safety net the user explicitly asked for, and it must be proven to behave, not just assumed unreachable.)
- **Model output at the exact tanh boundary (±1.0)** — decode must map `-1 → L=0,a=-0.4,b=-0.4` and `1 → L=1,a=0.4,b=0.4` without going out of the documented ranges, on both the export side (ONNX) and every decode implementation (Python, JS, Kotlin).
- **Stale `meta.json` without `color_ab_range`** (an old cached PWA asset, or a dev who forgot to re-export) — decode functions must fall back to the documented default (0.4) instead of crashing on `undefined`/`null`.
- **The transient window between changing decode logic and retraining** — once `export_onnx.py` stops applying the old `(x+1)*127.5` RGB transform, any *already-exported* `model.onnx` (trained under the old RGB semantics) will decode to nonsense colors until Task 8's retrain runs. This is expected, not a regression — verification in Tasks 5–7 must rely on the unit tests (which pin the codec math directly), not on "the live card looks right," until Task 8 is done.
- **CSS `oklab()` actually being applied, not silently dropped** — an invalid CSS color value is ignored by the browser (previous value/default persists) rather than erroring, so a typo in `toCssOklab`'s output format could look like "nothing happened" instead of a visible failure. Task 6's manual verification step must visually confirm gradients/text actually take on the predicted color, in an actual browser, not just check for console errors.

---

## Task 1: OKLAB codec in `model/data.py`

**Files:**
- Modify: `model/data.py:42-66` (replace `hex2rgb`/`colors2tensor`/`rnd_color_tensor` region)
- Modify: `model/model.py:169-177` (`Critic.forward` docstring)
- Test: `model/test_data.py` (new file)

**Interfaces:**
- Produces: `AB_RANGE: float` (module constant, `0.4`), `hex_to_color_unit(h: str) -> tuple[float, float, float]`, `color_unit_to_hex(unit: tuple[float, float, float]) -> str`, `color_tensor_to_hexes(vec9: torch.Tensor) -> list[str]`, `colors2tensor(colors: list[str]) -> torch.Tensor` (unchanged signature, new internals), `COLOR_DIM = 9` (unchanged).

- [ ] **Step 1: Write the failing tests**

```python
# model/test_data.py
import math

import torch

from model.data import (
    AB_RANGE,
    COLOR_DIM,
    color_tensor_to_hexes,
    color_unit_to_hex,
    colors2tensor,
    hex_to_color_unit,
)


def test_white_and_black_are_exact_endpoints():
    l, a, b = hex_to_color_unit("#ffffff")
    assert math.isclose(l, 1.0, abs_tol=1e-6)
    assert math.isclose(a, 0.0, abs_tol=1e-6)
    assert math.isclose(b, 0.0, abs_tol=1e-6)

    l, a, b = hex_to_color_unit("#000000")
    assert math.isclose(l, -1.0, abs_tol=1e-6)
    assert math.isclose(a, 0.0, abs_tol=1e-6)
    assert math.isclose(b, 0.0, abs_tol=1e-6)


def test_encode_decode_round_trips_for_representative_colors():
    for hexcolor in ["#ffffff", "#000000", "#ff0000", "#00ff00", "#0000ff",
                      "#808080", "#3a7bd5", "#c94f4f"]:
        unit = hex_to_color_unit(hexcolor)
        assert all(-1.0 - 1e-9 <= v <= 1.0 + 1e-9 for v in unit)
        back = color_unit_to_hex(unit)
        assert back == hexcolor


def test_ab_clamp_keeps_unit_channels_in_range():
    # Simulate an out-of-gamut a/b (can't happen from real sRGB hex, but the
    # clamp must still hold if it ever did).
    over_range_ab = AB_RANGE * 10
    clamped = max(-AB_RANGE, min(AB_RANGE, over_range_ab))
    assert clamped == AB_RANGE
    unit = clamped / AB_RANGE
    assert -1.0 <= unit <= 1.0


def test_colors2tensor_shape_and_range():
    t = colors2tensor(["#ffffff", "#000000", "#3a7bd5"])
    assert t.shape == (COLOR_DIM,)
    assert torch.all(t >= -1.0 - 1e-6) and torch.all(t <= 1.0 + 1e-6)


def test_color_tensor_to_hexes_matches_colors2tensor_round_trip():
    hexes = ["#a8e2f4", "#78c9f4", "#282e36"]
    t = colors2tensor(hexes)
    assert color_tensor_to_hexes(t) == hexes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest model/test_data.py -v`
Expected: FAIL — `ImportError: cannot import name 'AB_RANGE' from 'model.data'` (none of the new names exist yet).

- [ ] **Step 3: Implement the codec**

Replace `model/data.py:42-66` (the `COLOR_DIM = 9` block through `rnd_color_tensor`) with:

```python
COLOR_DIM = 9
AB_RANGE = 0.4


def hex2rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (
        int(h[0:2], 16),
        int(h[2:4], 16),
        int(h[4:6], 16))


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    v = 12.92 * c if c <= 0.0031308 else 1.055 * max(c, 0.0) ** (1 / 2.4) - 0.055
    return v * 255


def srgb_to_oklab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    r = _srgb_to_linear(rgb[0] / 255)
    g = _srgb_to_linear(rgb[1] / 255)
    b = _srgb_to_linear(rgb[2] / 255)
    l_ = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m_ = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s_ = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * l_ + 0.793617785 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.428592205 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.808675766 * s_,
    )


def oklab_to_srgb(lab: tuple[float, float, float]) -> tuple[float, float, float]:
    L, a, b = lab
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.291485548 * b) ** 3
    return (
        _linear_to_srgb(4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_),
        _linear_to_srgb(-1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_),
        _linear_to_srgb(-0.0041960863 * l_ - 0.7034186147 * m_ + 1.707614701 * s_),
    )


def hex_to_color_unit(h: str) -> tuple[float, float, float]:
    """sRGB hex -> model-space unit triple: rgb2oklab, clamp a/b to
    +/-AB_RANGE, map to [-1, 1] (L = 2*L-1, a/b = v/AB_RANGE)."""
    L, a, b = srgb_to_oklab(hex2rgb(h))
    a = max(-AB_RANGE, min(AB_RANGE, a))
    b = max(-AB_RANGE, min(AB_RANGE, b))
    return (2 * L - 1, a / AB_RANGE, b / AB_RANGE)


def color_unit_to_hex(unit: tuple[float, float, float]) -> str:
    """Model-space unit triple -> sRGB hex (see Critic.forward / ColorGen)."""
    n0, n1, n2 = unit
    L = (n0 + 1) / 2
    a = n1 * AB_RANGE
    b = n2 * AB_RANGE
    r, g, b_ = oklab_to_srgb((L, a, b))

    def clamp(v: float) -> int:
        return max(0, min(255, round(v)))

    return f"#{clamp(r):02x}{clamp(g):02x}{clamp(b_):02x}"


def colors2tensor(colors: list[str]) -> torch.Tensor:
    vals = [v for h in colors for v in hex_to_color_unit(h)]
    return torch.tensor(vals, dtype=torch.float32)


def color_tensor_to_hexes(vec9: torch.Tensor) -> list[str]:
    assert vec9.shape == (9,), "Input tensor must be of shape (9,)"
    flat = vec9.tolist()
    return [color_unit_to_hex(tuple(flat[i:i + 3])) for i in range(0, 9, 3)]


def sample_colors_tensor(colors: list[list[str]]) -> torch.Tensor:
    if not colors:
        return torch.zeros(COLOR_DIM)
    i = int(torch.randint(len(colors), (1,)).item())
    return colors2tensor(colors[i])


def rnd_color_tensor(n: int, device=None) -> torch.Tensor:
    """Uniform random points in the model's unit color cube -- used only as
    negative/adversarial samples for the critic, not as real colors."""
    return torch.rand((n, COLOR_DIM), device=device) * 2.0 - 1.0
```

Update `model/model.py:169-171` (the `Critic.forward` docstring):

```python
    def forward(self, cond: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        """`colors` is OKLAB-unit-encoded in [-1, 1] (see `ColorGen.forward` /
        `model.data.colors2tensor` / `model.data.AB_RANGE`)."""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest model/test_data.py -v`
Expected: PASS (all 5 tests).

- [ ] **Step 5: Commit**

```bash
git add model/data.py model/model.py model/test_data.py
git commit -m "feat(model): switch color codec from raw RGB to OKLAB-unit"
```

---

## Task 2: Export raw OKLAB-unit output from ONNX, publish `color_ab_range`

**Files:**
- Modify: `model/export_onnx.py:69-77` (`ExportWrapper.forward`), `:104-124` (`export_web`)
- Test: `model/test_export_onnx.py` (new file)

**Interfaces:**
- Consumes: `model.data.AB_RANGE` (from Task 1).
- Produces: ONNX `color` output now raw tanh range `[-1, 1]` (previously `(x+1)*127.5`); `meta.json` gains `"color_ab_range": 0.4`.

- [ ] **Step 1: Write the failing test**

```python
# model/test_export_onnx.py
import torch

from model.data import AB_RANGE
from model.export_onnx import ExportWrapper
from model.model import ColorGen, EmojiHead, StyleHead, TextEncoder


def test_export_wrapper_color_output_is_raw_unit_range():
    wrapper = ExportWrapper(
        TextEncoder(), StyleHead(), EmojiHead(), ColorGen()
    ).eval()
    x = torch.zeros(1, 42, dtype=torch.long)
    with torch.no_grad():
        _, _, color = wrapper(x)
    assert color.shape == (5, 9)
    assert torch.all(color >= -1.0 - 1e-6) and torch.all(color <= 1.0 + 1e-6)
    # would be ~[0, 255] under the old RGB export transform -- pin that it's not
    assert float(color.abs().max()) <= 1.0 + 1e-6


def test_export_web_meta_has_color_ab_range(tmp_path, monkeypatch):
    import model.export_onnx as export_onnx

    monkeypatch.setattr(export_onnx, "WEB_PUBLIC", tmp_path / "web")
    monkeypatch.setattr(export_onnx, "ANDROID_ASSETS_DIR", tmp_path / "android")
    wrapper = ExportWrapper(
        TextEncoder(), StyleHead(), EmojiHead(), ColorGen()
    ).eval()
    export_onnx.export_web(wrapper)

    import json
    meta = json.loads((tmp_path / "web" / "meta.json").read_text())
    assert meta["color_ab_range"] == AB_RANGE
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest model/test_export_onnx.py -v`
Expected: FAIL — the color-range assertion fails under the current `(x+1)*127.5` transform (values go up to ~255, not 1.0), and `color_ab_range` is absent from `meta`.

- [ ] **Step 3: Implement**

In `model/export_onnx.py`, change `ExportWrapper.forward` (around line 69-77):

```python
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.enc(x)
        style_logits = self.style(emb)
        emoji_logits = self.emoji(emb)
        cond = emb.expand(COLOR_SAMPLES, -1)
        color = self.gen(cond)
        return style_logits, emoji_logits, color
```

Add the import and the meta field (around line 12-21 and 104-117):

```python
from model.data import CHARS, PAD_IDX, AB_RANGE
```

```python
    meta = {
        "chars": CHARS,
        "pad_idx": PAD_IDX,
        "max_text_len": MAX_TEXT_LEN,
        "emojis": EMOJIS,
        "styles": STYLES,
        "color_ab_range": AB_RANGE,
        "exported_at": datetime.now(UTC).isoformat(timespec="minutes"),
        "model_meta": getattr(wrapper.enc, "_pt_meta", None),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest model/test_export_onnx.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/export_onnx.py model/test_export_onnx.py
git commit -m "feat(model): export raw OKLAB-unit color output, publish color_ab_range"
```

---

## Task 3: `model/pred.py` uses the shared codec

**Files:**
- Modify: `model/pred.py:1-31` (drop the local `rgb_to_hex`, use `color_tensor_to_hexes`), `:92` (call site)
- Test: `model/test_pred.py` (new file)

**Interfaces:**
- Consumes: `model.data.color_tensor_to_hexes` (Task 1).

- [ ] **Step 1: Write the failing test**

```python
# model/test_pred.py
import torch

from model.data import COLOR_DIM
from model.pred import predict, read_rows


def test_read_rows_normalizes_and_drops_empty():
    rows = read_rows(['{"text": "  Hello   WORLD  "}', '{"text": ""}'])
    assert rows == [{"text": "hello world"}]


def test_predict_emits_three_valid_hex_colors(tmp_path, monkeypatch):
    import model.pred as pred_mod
    from model.model import ColorGen, EmojiHead, StyleHead, TextEncoder
    from model.runmeta import save_pt
    from files import PtFile

    pt_dir = tmp_path
    save_pt(TextEncoder().state_dict(), PtFile.ENC.in_dir(pt_dir), stage="test")
    save_pt(ColorGen().state_dict(), PtFile.GEN.in_dir(pt_dir), stage="test")
    save_pt(StyleHead().state_dict(), PtFile.STYLE.in_dir(pt_dir), stage="test")
    save_pt(EmojiHead().state_dict(), PtFile.EMOJI.in_dir(pt_dir), stage="test")

    records = predict([{"text": "hello world"}], pt_dir)
    assert len(records) == 1
    r = records[0]
    assert len(r["bg"]) == 2
    for h in [*r["bg"], r["fg"]]:
        assert h.startswith("#") and len(h) == 7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest model/test_pred.py -v`
Expected: FAIL only if `rgb_to_hex`'s old `(rgb + 1.0) * 127.5` math and the new codec disagree loudly enough to break hex parsing — more importantly this pins current behavior before the refactor so a mistake in Step 3 is caught. (If it already passes, that's fine — proceed to Step 3 and re-run after to confirm no regression.)

- [ ] **Step 3: Implement**

In `model/pred.py`, delete `rgb_to_hex` (lines 23-31) and its use; import and call the shared helper instead:

```python
from model.data import EMOJIS, STYLES, color_tensor_to_hexes, normalize, text_to_tensor
```

```python
            hexes = color_tensor_to_hexes(gen(emb).squeeze(0))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest model/test_pred.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/pred.py model/test_pred.py
git commit -m "refactor(model): pred.py reuses the shared OKLAB codec instead of its own rgb_to_hex"
```

---

## Task 4: `tools/report.py` uses the shared codec, fixes L/chroma math

**Files:**
- Modify: `tools/report.py:39-46` (imports), `:69-70` (threshold constants), `:1413-1450` (`_hex_to_rgb`/`_rgb_to_hex`/`_l_chroma`/`_pure_distance`)
- Test: `tools/test_report.py` (extend)

**Interfaces:**
- Consumes: `model.data.hex_to_color_unit`, `color_unit_to_hex`, `AB_RANGE` (Task 1).

- [ ] **Step 1: Write the failing test**

Add to `tools/test_report.py`:

```python
def test_l_chroma_decodes_real_oklab_values():
    from tools.report import _l_chroma

    # unit (1,0,0) three times -> real OKLAB L=1, a=b=0 -> chroma 0
    l, chroma = _l_chroma([1, 0, 0, 1, 0, 0, 1, 0, 0])
    assert abs(l - 1.0) < 1e-6
    assert abs(chroma - 0.0) < 1e-6

    # unit (-1, 1, 0) -> L=0, a=AB_RANGE, b=0 -> chroma = AB_RANGE
    l, chroma = _l_chroma([-1, 1, 0, -1, 1, 0, -1, 1, 0])
    assert abs(l - 0.0) < 1e-6
    assert abs(chroma - 0.4) < 1e-6


def test_pure_distance_uses_oklab_unit_space():
    from tools.report import _pure_distance

    # a palette that's exactly pure red in both bg slots should have 0
    # distance from the "red" pure reference.
    from model.data import hex_to_color_unit

    red_unit = hex_to_color_unit("#ff0000")
    pred9 = [*red_unit, *red_unit, 0, 0, 0]
    assert _pure_distance(pred9, "red") < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tools/test_report.py -k "l_chroma or pure_distance" -v`
Expected: FAIL — `_l_chroma` currently returns naive RGB-channel mean/variance, not real OKLAB L/chroma, so the first test's exact values won't match; `_pure_distance` currently measures distance in the old RGB-unit space via the local `_hex_to_rgb`.

- [ ] **Step 3: Implement**

In `tools/report.py`, update the import block (around line 39-46):

```python
from model.data import (
    AB_RANGE,
    EVAL_PATH,
    TRAIN_PATH,
    color_unit_to_hex,
    colors2tensor,
    hex_to_color_unit,
    read,
    read_color_keywords,
    text_to_tensor,
)
```

Rename the threshold constant (around line 69) for clarity now that it's OKLAB-space, not RGB-space:

```python
CARD_PURE_THRESHOLD_OKLAB = 0.251
CARD_PURE_THRESHOLD_L = 0.6
```

(Update the two other references to `CARD_PURE_THRESHOLD_RGB` — the `"pure_threshold_rgb"` dict key at line ~925 and the `_pure_threshold` function below — to `CARD_PURE_THRESHOLD_OKLAB` / `"pure_threshold_oklab"`.)

Replace `_hex_to_rgb`/`_rgb_to_hex`/`_l_chroma`/`_pure_distance` (around line 1413-1450):

```python
def _l_chroma(rgb9) -> tuple[float, float]:
    flat = torch.tensor(rgb9, dtype=torch.float32).reshape(3, 3)
    ls, chromas = [], []
    for row in flat.tolist():
        n0, n1, n2 = row
        L = (n0 + 1) / 2
        a = n1 * AB_RANGE
        b = n2 * AB_RANGE
        ls.append(L)
        chromas.append((a ** 2 + b ** 2) ** 0.5)
    return sum(ls) / len(ls), sum(chromas) / len(chromas)


PURE_HEX = {
    "red": "#ff0000",
    "green": "#00ff00",
    "blue": "#0000ff",
    "dark": "#000000",
    "bright": "#ffffff",
}


def _pure_threshold(color: str) -> float:
    if color in ("dark", "bright"):
        return CARD_PURE_THRESHOLD_L
    return CARD_PURE_THRESHOLD_OKLAB


def _pure_distance(pred9, color: str) -> float:
    p = torch.tensor(pred9, dtype=torch.float32).reshape(3, 3)
    bg = p[:2].mean(dim=0)
    pure = torch.tensor(hex_to_color_unit(PURE_HEX[color]), dtype=torch.float32)
    if color in ("dark", "bright"):
        return (bg.mean() - pure.mean()).abs().item()
    return (bg - pure).norm().item()
```

Update the remaining call sites that used the deleted `_hex_to_rgb`/`_rgb_to_hex` (lines ~874-876, 891-893, 987-988, 1047-1049, 1102-1107) to call `hex_to_color_unit`/`color_unit_to_hex` instead — same argument shapes, direct rename:

```python
        gold9 = (
            list(hex_to_color_unit(r["bg"][0]))
            + list(hex_to_color_unit(r["bg"][1]))
            + list(hex_to_color_unit(r["fg"]))
        )
```

```python
                "bg1": color_unit_to_hex(tuple(flat[0:3])),
                "bg2": color_unit_to_hex(tuple(flat[3:6])),
                "text_color": color_unit_to_hex(tuple(flat[6:9])),
```

(apply the same `color_unit_to_hex(tuple(flat[a:b]))` substitution at each of the other `_rgb_to_hex(flat[...])` call sites listed above).

And the goals-target dict key (around line 925):

```python
        "pure_threshold_oklab": CARD_PURE_THRESHOLD_OKLAB,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tools/test_report.py -v`
Expected: PASS (including the pre-existing suite — check none of it hardcoded `pure_threshold_rgb` as a dict key or asserted old `_l_chroma` values).

- [ ] **Step 5: Commit**

```bash
git add tools/report.py tools/test_report.py
git commit -m "refactor(report): use shared OKLAB codec, fix L/chroma to real OKLAB values"
```

---

## Task 5: `web/src/model.js` — OKLAB-native codec and color math

**Files:**
- Modify: `web/src/model.js` (full color section, lines 19-143)
- Modify: `web/src/model.test.js` (color-related `describe` blocks)

**Interfaces:**
- Produces: `AB_RANGE`, `BLACK`, `WHITE`, `hexToOklab(hex) -> [L,a,b]`, `toCssOklab([L,a,b]) -> string`, `decodeColors(color9, abRange?) -> {bg1,bg2,text_color}` (each an `[L,a,b]` triple), `decodeColorList(flat, abRange?) -> palette[]`, `contrastRatio(oklabA, oklabB) -> number`, `fixContrast(palette, minContrast?) -> palette`, `mixColors(oklabA, oklabB, t?) -> [L,a,b]`, `patternTint(bg1, bg2) -> [L,a,b]`. `srgbToOklab`/`oklabToSrgb` stay exported (still used internally and by tests). `decodeColors`/`fixContrast`/`mixColors`/`patternTint` no longer take or return hex strings.

- [ ] **Step 1: Write the failing tests**

Replace the `decodeColors`, `decodeColorList`, `contrastRatio`, `fixContrast` describe blocks in `web/src/model.test.js`, and add new ones, per the following (this replaces lines 1-16 imports and 48-155 of the existing file; `normalize`/`encode`/`argmax`/`softmax`/`sigmoid` blocks at the top and bottom are unchanged and stay as-is):

```js
import { describe, it, expect } from 'vitest'
import {
  normalize,
  encode,
  argmax,
  softmax,
  sigmoid,
  decodeColors,
  decodeColorList,
  srgbToOklab,
  oklabToSrgb,
  hexToOklab,
  toCssOklab,
  contrastRatio,
  fixContrast,
  mixColors,
  patternTint,
  CONTRAST_MIN,
  BLACK,
  WHITE,
} from './model'

// ...(normalize / encode describe blocks unchanged)...

describe('decodeColors', () => {
  it('maps unit -1/1 to black/white OKLab triples', () => {
    const decoded = decodeColors([1, 0, 0, -1, 0, 0, 1, 0, 0])
    expect(decoded.bg1).toEqual([1, 0, 0])
    expect(decoded.bg2).toEqual([0, 0, 0])
    expect(decoded.text_color).toEqual([1, 0, 0])
  })
  it('maps unit 0 to the neutral midpoint L=0.5, a=0, b=0', () => {
    const decoded = decodeColors([0, 0, 0, 0, 0, 0, 0, 0, 0])
    expect(decoded.bg1).toEqual([0.5, 0, 0])
  })
  it('scales a/b by the given ab_range', () => {
    const decoded = decodeColors([0, 1, -1, 0, 0, 0, 0, 0, 0], 0.3)
    expect(decoded.bg1[0]).toBeCloseTo(0.5, 10)
    expect(decoded.bg1[1]).toBeCloseTo(0.3, 10)
    expect(decoded.bg1[2]).toBeCloseTo(-0.3, 10)
  })
})

describe('decodeColorList', () => {
  it('chunks a flat 45-value buffer into 5 palettes', () => {
    const list = decodeColorList(new Float32Array(45).fill(0))
    expect(list).toHaveLength(5)
    for (const p of list) {
      expect(p.bg1).toEqual([0.5, 0, 0])
      expect(p.bg2).toEqual([0.5, 0, 0])
      expect(p.text_color).toEqual([0.5, 0, 0])
    }
  })
  it('decodes each chunk independently', () => {
    const flat = [...Array(9).fill(-1), ...Array(9).fill(1)]
    expect(decodeColorList(flat)).toEqual([
      { bg1: [0, 0, 0], bg2: [0, 0, 0], text_color: [0, 0, 0] },
      { bg1: [1, 0, 0], bg2: [1, 0, 0], text_color: [1, 0, 0] },
    ])
  })
})

describe('oklab', () => {
  it('round-trips sRGB through OKLab', () => {
    for (const rgb of [
      [255, 255, 255],
      [0, 0, 0],
      [128, 64, 200],
      [20, 180, 90],
    ]) {
      const back = oklabToSrgb(srgbToOklab(rgb)).map(Math.round)
      expect(back).toEqual(rgb)
    }
  })
  it('white is L≈1 with near-zero a/b', () => {
    const [L, a, b] = srgbToOklab([255, 255, 255])
    expect(L).toBeCloseTo(1, 3)
    expect(a).toBeCloseTo(0, 3)
    expect(b).toBeCloseTo(0, 3)
  })
})

describe('hexToOklab / toCssOklab', () => {
  it('converts white and black hex to their exact OKLab triples', () => {
    expect(hexToOklab('#ffffff')[0]).toBeCloseTo(1, 5)
    expect(hexToOklab('#000000')).toEqual([0, 0, 0])
  })
  it('formats an OKLab triple as a CSS oklab() string', () => {
    expect(toCssOklab([0.6, -0.05, 0.12])).toBe('oklab(60.00% -0.0500 0.1200)')
  })
})

describe('contrastRatio', () => {
  it('is 21 for black on white and 1 for a colour on itself', () => {
    expect(contrastRatio(BLACK, WHITE)).toBeCloseTo(21, 4)
    const c = hexToOklab('#3a7bd5')
    expect(contrastRatio(c, c)).toBeCloseTo(1, 5)
  })
  it('is symmetric', () => {
    const a = hexToOklab('#123456')
    const b = hexToOklab('#abcdef')
    expect(contrastRatio(a, b)).toBeCloseTo(contrastRatio(b, a), 10)
  })
})

describe('fixContrast', () => {
  it('leaves a readable palette untouched (same object)', () => {
    const p = {
      bg1: hexToOklab('#a8e2f4'),
      bg2: hexToOklab('#78c9f4'),
      text_color: hexToOklab('#282e36'),
    }
    expect(fixContrast(p)).toBe(p)
  })

  it('repairs a low-contrast palette so both stops clear the threshold', () => {
    const p = {
      bg1: hexToOklab('#2b2b2b'),
      bg2: hexToOklab('#3a3a3a'),
      text_color: hexToOklab('#444444'),
    }
    const fixed = fixContrast(p)
    expect(fixed.text_color).not.toBe(p.text_color)
    expect(fixed.bg1).toBe(p.bg1)
    expect(fixed.bg2).toBe(p.bg2)
    expect(contrastRatio(fixed.text_color, fixed.bg1)).toBeGreaterThanOrEqual(CONTRAST_MIN)
    expect(contrastRatio(fixed.text_color, fixed.bg2)).toBeGreaterThanOrEqual(CONTRAST_MIN)
  })

  it('nudges lightness while roughly preserving hue', () => {
    const p = {
      bg1: hexToOklab('#c94f4f'),
      bg2: hexToOklab('#d46b4b'),
      text_color: hexToOklab('#b84a3a'),
    }
    const fixed = fixContrast(p)
    const [, a0, b0] = p.text_color
    const [, a1, b1] = fixed.text_color
    expect(Math.sign(a1)).toBe(Math.sign(a0))
    expect(Math.sign(b1)).toBe(Math.sign(b0))
  })

  it('respects a custom threshold', () => {
    const p = {
      bg1: hexToOklab('#ffffff'),
      bg2: hexToOklab('#f4f4f4'),
      text_color: hexToOklab('#8a8a8a'),
    }
    const fixed = fixContrast(p, 4.5)
    expect(minOf(fixed)).toBeGreaterThanOrEqual(4.5)
  })
})

function minOf({ bg1, bg2, text_color }) {
  return Math.min(contrastRatio(text_color, bg1), contrastRatio(text_color, bg2))
}

describe('mixColors', () => {
  it('at t=0 returns the first color, at t=1 the second', () => {
    const a = hexToOklab('#ff0000')
    const b = hexToOklab('#0000ff')
    expect(mixColors(a, b, 0)).toEqual(a)
    expect(mixColors(a, b, 1)).toEqual(b)
  })
  it('at t=0.5 is the midpoint on each channel', () => {
    expect(mixColors([0, 0, 0], [1, 0.4, -0.4], 0.5)).toEqual([0.5, 0.2, -0.2])
  })
})

describe('patternTint', () => {
  it('floors lightness at 0.94 while keeping hue', () => {
    const tint = patternTint([0.5, 0.1, -0.1], [0.5, 0.1, -0.1])
    expect(tint).toEqual([0.94, 0.1, -0.1])
  })
  it('keeps lightness above 0.94 if the mix is already lighter', () => {
    const tint = patternTint([0.98, 0.05, 0], [0.98, 0.05, 0])
    expect(tint[0]).toBeCloseTo(0.98, 10)
  })
})

// ...(argmax/softmax/sigmoid describe blocks unchanged)...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd web && npm test -- model.test.js`
Expected: FAIL — `decodeColors`/`fixContrast`/etc. still operate on hex strings under the old contract; `hexToOklab`, `toCssOklab`, `mixColors`, `patternTint`, `BLACK`, `WHITE` aren't exported yet.

- [ ] **Step 3: Implement**

Replace `web/src/model.js` lines 19-143 (from `export function decodeColors` through the end of `patternTint`) with:

```js
export const AB_RANGE = 0.4

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
}

function linearToSrgb(c) {
  const v = c <= 0.0031308 ? 12.92 * c : 1.055 * Math.max(c, 0) ** (1 / 2.4) - 0.055
  return v * 255
}

export function srgbToOklab([r0, g0, b0]) {
  const r = srgbToLinear(r0 / 255)
  const g = srgbToLinear(g0 / 255)
  const b = srgbToLinear(b0 / 255)
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ]
}

export function oklabToSrgb([L, a, b]) {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3
  return [
    linearToSrgb(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
    linearToSrgb(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
    linearToSrgb(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s),
  ]
}

export function hexToOklab(hex) {
  return srgbToOklab(hexToRgb(hex))
}

export function toCssOklab([L, a, b]) {
  return `oklab(${(L * 100).toFixed(2)}% ${a.toFixed(4)} ${b.toFixed(4)})`
}

function decodeColor([n0, n1, n2], abRange) {
  return [(n0 + 1) / 2, n1 * abRange, n2 * abRange]
}

export function decodeColors(color, abRange = AB_RANGE) {
  return {
    bg1: decodeColor([color[0], color[1], color[2]], abRange),
    bg2: decodeColor([color[3], color[4], color[5]], abRange),
    text_color: decodeColor([color[6], color[7], color[8]], abRange),
  }
}

export function decodeColorList(flat, abRange = AB_RANGE) {
  const arr = Array.from(flat)
  const out = []
  for (let i = 0; i + 9 <= arr.length; i += 9) out.push(decodeColors(arr.slice(i, i + 9), abRange))
  return out
}

function relLuminance([r, g, b]) {
  return (
    0.2126 * srgbToLinear(r / 255) +
    0.7152 * srgbToLinear(g / 255) +
    0.0722 * srgbToLinear(b / 255)
  )
}

export function contrastRatio(oklabA, oklabB) {
  const la = relLuminance(oklabToSrgb(oklabA))
  const lb = relLuminance(oklabToSrgb(oklabB))
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05)
}

export const CONTRAST_MIN = 3
export const BLACK = [0, 0, 0]
export const WHITE = [1, 0, 0]

function minMargin(fg, bg1, bg2) {
  return Math.min(contrastRatio(fg, bg1), contrastRatio(fg, bg2))
}

export function fixContrast(palette, minContrast = CONTRAST_MIN) {
  const { bg1, bg2, text_color } = palette
  const ok = (fg) => minMargin(fg, bg1, bg2) >= minContrast
  if (ok(text_color)) return palette

  const [L0, a, b] = text_color
  const STEP = 0.02
  let best = null
  let bestCost = Infinity
  for (const dir of [-1, 1]) {
    for (let L = L0 + dir * STEP; L >= 0 && L <= 1; L += dir * STEP) {
      const cand = [L, a, b]
      if (ok(cand)) {
        if (Math.abs(L - L0) < bestCost) {
          best = cand
          bestCost = Math.abs(L - L0)
        }
        break
      }
    }
  }
  if (!best) {
    best = minMargin(BLACK, bg1, bg2) >= minMargin(WHITE, bg1, bg2) ? BLACK : WHITE
  }
  return { bg1, bg2, text_color: best }
}

export function mixColors(a, b, t = 0.5) {
  const [L1, a1, b1] = a
  const [L2, a2, b2] = b
  return [L1 + (L2 - L1) * t, a1 + (a2 - a1) * t, b1 + (b2 - b1) * t]
}

export function patternTint(bg1, bg2) {
  const [L, a, b] = mixColors(bg1, bg2)
  return [Math.max(0.94, L), a, b]
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd web && npm test -- model.test.js`
Expected: PASS (all describe blocks).

- [ ] **Step 5: Commit**

```bash
git add web/src/model.js web/src/model.test.js
git commit -m "feat(web): switch model.js color codec/math from hex/RGB to OKLAB triples + CSS oklab()"
```

---

## Task 6: Wire OKLAB triples through the web app's render paths

**Files:**
- Modify: `web/src/hooks/useOnnx.js:53`
- Modify: `web/src/feelings.js:3`
- Modify: `web/src/StylePreview.jsx:4-7,11,18,27-28,69`
- Modify: `web/src/components/Card.jsx:4-8,50,54,57,94`
- Modify: `web/src/components/ColorBar.jsx:1,12,17` (add import)
- Modify: `web/src/hooks/useCardImage.js:4,96-97,101,109,156-159`

**Interfaces:**
- Consumes: everything Task 5 exports from `web/src/model.js`.

- [ ] **Step 1: `useOnnx.js` — pass the meta-supplied ab_range**

In `web/src/hooks/useOnnx.js:53`, change:

```js
      palettes: decodeColorList(out.color.data),
```

to:

```js
      palettes: decodeColorList(out.color.data, m.color_ab_range),
```

- [ ] **Step 2: `feelings.js` — DEFAULT_COLORS becomes an OKLAB-triple palette**

In `web/src/feelings.js`, add the import and change line 3:

```js
import { hexToOklab } from './model'
```

```js
export const DEFAULT_COLORS = {
  bg1: hexToOklab('#a8e2f4'),
  bg2: hexToOklab('#78c9f4'),
  text_color: hexToOklab('#282e36'),
}
```

- [ ] **Step 3: `StylePreview.jsx` — convert loaded hex samples, render via `toCssOklab`**

Change the import and `watermarkInk` (lines 4-7):

```js
import { contrastRatio, fixContrast, patternTint, hexToOklab, toCssOklab, BLACK, WHITE } from './model'

function watermarkInk(bg) {
  return contrastRatio(BLACK, bg) >= contrastRatio(WHITE, bg) ? '#000000' : '#ffffff'
}
```

Change `PreviewCard` (lines 10-19 and the `style` blocks at 26-30, 37):

```js
function PreviewCard({ name, lang, entry, sample, globalSettings, patterns }) {
  const colors = fixContrast({
    bg1: hexToOklab(sample.colors.bg1),
    bg2: hexToOklab(sample.colors.bg2),
    text_color: hexToOklab(sample.colors.text_color),
  })
  const displayText = sample.text.trim() ? sample.text : "What's on your mind?"
  const textRef = useFitText(displayText, {
    min: globalSettings.textMinRatio * 100,
    max: globalSettings.textMaxRatio * 100,
    key: name,
  })
  const tint = toCssOklab(patternTint(colors.bg1, colors.bg2))
```

```js
        style={{
          background: `linear-gradient(135deg, ${toCssOklab(colors.bg1)}, ${toCssOklab(colors.bg2)})`,
          color: toCssOklab(colors.text_color),
          fontFamily: `"${entry.font}", sans-serif`,
        }}
```

Line 69's `watermarkInk(colors.bg2)` call needs no change — `colors.bg2` is now already an OKLab triple, matching `watermarkInk`'s new expected input.

- [ ] **Step 4: `Card.jsx` — same pattern for the live card**

Change lines 4-8:

```js
import { contrastRatio, patternTint, toCssOklab, BLACK, WHITE } from '../model'

function watermarkInk(bg) {
  return contrastRatio(BLACK, bg) >= contrastRatio(WHITE, bg) ? '#000000' : '#ffffff'
}
```

Change lines 50-57:

```js
          const layers = patternLayers(shown.feeling, toCssOklab(patternTint(colors.bg1, colors.bg2)))
          return {
            backgroundImage: [
              ...layers.map((l) => l.image),
              `linear-gradient(135deg, ${toCssOklab(colors.bg1)}, ${toCssOklab(colors.bg2)})`,
            ].join(', '),
            backgroundSize: [...patternSizeCss(layers), 'auto'].join(', '),
            color: toCssOklab(colors.text_color),
```

Line 94's `watermarkInk(colors.bg2)` needs no change, same reasoning as Step 3.

- [ ] **Step 5: `ColorBar.jsx` — wrap gradient/text colors**

Add the import and wrap `c.bg1`/`c.bg2`/`c.text_color`:

```js
import { toCssOklab } from '../model'

export function ColorBar({ palettes, active, onPick, ready = true, count = 5 }) {
  const items = ready && palettes.length ? palettes.slice(0, count) : null
  return (
    <div className="color-bar-container">
      <div className="color-bar">
        {items
          ? items.map((c, i) => (
            <button
              key={i}
              type="button"
              className={i === active ? 'active' : undefined}
              style={{ backgroundImage: `linear-gradient(135deg, ${toCssOklab(c.bg1)}, ${toCssOklab(c.bg2)})` }}
              aria-label={`color ${i + 1}`}
              aria-pressed={i === active}
              onClick={() => onPick(i)}
            >
              <span aria-hidden="true" style={{ color: toCssOklab(c.text_color) }}>
                Aa
              </span>
            </button>
          ))
```

(rest of the file unchanged.)

- [ ] **Step 6: `useCardImage.js` — Canvas fillStyle via `toCssOklab`**

Change line 4:

```js
import { contrastRatio, patternTint, toCssOklab, BLACK, WHITE } from '../model'
```

Change lines 96-97, 101, 109, 156-159:

```js
  grad.addColorStop(0, toCssOklab(colors.bg1))
  grad.addColorStop(1, toCssOklab(colors.bg2))
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, S, S)

  const layers = patternLayers(feeling, toCssOklab(patternTint(colors.bg1, colors.bg2)))
```

```js
  ctx.fillStyle = toCssOklab(colors.text_color)
```

```js
  ctx.fillStyle =
    contrastRatio(BLACK, colors.bg2) >= contrastRatio(WHITE, colors.bg2)
      ? '#000000'
      : '#ffffff'
```

- [ ] **Step 7: Run the web test suite**

Run: `cd web && npm test`
Expected: PASS across the whole suite (not just `model.test.js` — check for any other test file that constructs a palette object and would now need `hexToOklab`, e.g. component snapshot/render tests referencing `bg1`/`bg2`/`text_color`).

- [ ] **Step 8: Manual verification (dev server)**

Run: `cd web && npm run dev`, open the app, type a sentence (≥3 chars). Because `web/public/model.onnx` hasn't been retrained yet (Task 8), the *actual hues* will look arbitrary/wrong — that's expected per Review Focus. What must be verified visually here is narrower: that the card, color bar, and style-preview page (`/style-preview.html`) all render **some** solid gradient/text color (not black, not transparent, not "the CSS silently failed to parse"), and that toggling "fix low-contrast palettes" still visibly changes the text color. Then click "copy" to exercise `useCardImage.js`'s canvas path and confirm the copied image also has a visible (not blank/black) background — this is the `oklab()`-in-Canvas compatibility check called out in Review Focus.

- [ ] **Step 9: Commit**

```bash
git add web/src/hooks/useOnnx.js web/src/feelings.js web/src/StylePreview.jsx \
  web/src/components/Card.jsx web/src/components/ColorBar.jsx web/src/hooks/useCardImage.js
git commit -m "feat(web): render model/design colors as OKLAB triples via native CSS oklab()"
```

---

## Task 7: Android — OKLAB-unit decode in `ModelIo.kt`

**Files:**
- Modify: `android/app/src/main/java/ing/emojify/model/Meta.kt:16-24`
- Modify: `android/app/src/main/java/ing/emojify/model/ModelIo.kt:24-47`
- Modify: `android/app/src/main/java/ing/emojify/model/OnnxPredictor.kt:39`
- Modify: `android/app/src/test/java/ing/emojify/model/ModelIoTest.kt:30-38`

**Interfaces:**
- Produces: `Meta.color_ab_range: Double` (default `0.4`), `decodeColors(c: FloatArray, abRange: Double = 0.4): Palette`, `decodeColorList(flat: FloatArray, abRange: Double = 0.4): List<Palette>`.
- `Palette`/`contrastRatio`/`fixContrast`/`mixColors`/`patternTint` stay hex-based and unchanged (Android renders sRGB, per Global Constraints).

- [ ] **Step 1: Write the failing test**

Replace the `decodeColorList chunks flat floats into 9-value palettes` test in `ModelIoTest.kt:30-38`:

```kotlin
    @Test
    fun `decodeColorList interprets floats as OKLAB-unit, not raw RGB bytes`() {
        // unit (1,0,0) -> white; unit (-1,0,0) -> black; unit (1,0,0) again -> white
        val flat = floatArrayOf(1f, 0f, 0f, -1f, 0f, 0f, 1f, 0f, 0f)
        val palettes = decodeColorList(flat)
        assertEquals(1, palettes.size)
        assertEquals("#ffffff", palettes[0].bg1)
        assertEquals("#000000", palettes[0].bg2)
        assertEquals("#ffffff", palettes[0].textColor)
    }

    @Test
    fun `decodeColorList respects a custom ab_range`() {
        val flat = floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
        val palettes = decodeColorList(flat, abRange = 0.3)
        // unit 0 always maps to L=0.5, a=0, b=0 regardless of ab_range
        assertEquals("#787878", palettes[0].bg1)
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd android && ./gradlew :app:testDebugUnitTest --tests "ing.emojify.model.ModelIoTest"`
Expected: FAIL — `decodeColorList` today treats the floats as already-0-255 RGB, so `1f, 0f, 0f` decodes to a near-black swatch (`#010000`), not white.

- [ ] **Step 3: Implement**

In `android/app/src/main/java/ing/emojify/model/ModelIo.kt`, replace `decodeColors`/`decodeColorList` (lines 24-47):

```kotlin
data class Palette(val bg1: String, val bg2: String, val textColor: String)

private const val DEFAULT_AB_RANGE = 0.4

private fun clampByte(v: Double): Int = max(0.0, min(255.0, Math.round(v).toDouble())).toInt()

private fun toHex(r: Double, g: Double, b: Double): String {
    fun h(v: Double) = clampByte(v).toString(16).padStart(2, '0')
    return "#${h(r)}${h(g)}${h(b)}"
}

private fun unitToOklab(n0: Double, n1: Double, n2: Double, abRange: Double): Triple<Double, Double, Double> =
    Triple((n0 + 1.0) / 2.0, n1 * abRange, n2 * abRange)

private fun decodeSwatch(c: FloatArray, offset: Int, abRange: Double): String {
    val (l, a, b) = unitToOklab(c[offset].toDouble(), c[offset + 1].toDouble(), c[offset + 2].toDouble(), abRange)
    val (r, g, bch) = oklabToSrgb(Triple(l, a, b))
    return toHex(r, g, bch)
}

fun decodeColors(c: FloatArray, abRange: Double = DEFAULT_AB_RANGE): Palette = Palette(
    bg1 = decodeSwatch(c, 0, abRange),
    bg2 = decodeSwatch(c, 3, abRange),
    textColor = decodeSwatch(c, 6, abRange),
)

fun decodeColorList(flat: FloatArray, abRange: Double = DEFAULT_AB_RANGE): List<Palette> {
    val out = mutableListOf<Palette>()
    var i = 0
    while (i + 9 <= flat.size) {
        out.add(decodeColors(flat.copyOfRange(i, i + 9), abRange))
        i += 9
    }
    return out
}
```

(`oklabToSrgb`, `srgbToOklab`, `hexToRgb`, `rgbToHex`, `contrastRatio`, `fixContrast`, `mixColors`, `patternTint` below stay exactly as they are — they're unaffected since they're still hex-in/hex-out.)

In `Meta.kt`, add the field (line 16-24):

```kotlin
@Serializable
data class Meta(
    val chars: String,
    val pad_idx: Int,
    val max_text_len: Int,
    val emojis: List<String>,
    val styles: List<String>,
    val exported_at: String? = null,
    val model_meta: ModelMetaInfo? = null,
    val color_ab_range: Double = 0.4,
)
```

In `OnnxPredictor.kt`, thread it through (line 39):

```kotlin
                return Prediction(emojiLogits, styleLogits, decodeColorList(colorFlat, meta.color_ab_range), ms)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd android && ./gradlew :app:testDebugUnitTest --tests "ing.emojify.model.ModelIoTest"`
Expected: PASS (all tests in the file, including the untouched `contrastRatio`/`fixContrast` ones).

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/java/ing/emojify/model/Meta.kt \
  android/app/src/main/java/ing/emojify/model/ModelIo.kt \
  android/app/src/main/java/ing/emojify/model/OnnxPredictor.kt \
  android/app/src/test/java/ing/emojify/model/ModelIoTest.kt
git commit -m "feat(android): decode model color output as OKLAB-unit instead of raw RGB bytes"
```

---

## Task 8: Retrain, recalibrate, and close out the docs

**Files:**
- Modify: `goals.yml` (`color generator: energy distance:` block)
- Modify: `tools/report.py` (`CARD_PURE_THRESHOLD_OKLAB`, `CARD_PURE_THRESHOLD_L` values, if the new report indicates they're off)
- Modify: `CLAUDE.md` (the `model/color.py` line in the Project section)

This task is a runbook, not TDD — there is no way to know the right calibration numbers before seeing real output from a model trained under the new color codec. Every earlier task is verifiable in isolation; this one closes the loop.

- [ ] **Step 1: Confirm a clean tree and regenerate derived data**

```bash
git status  # must be clean -- train.py's require_clean_tree() will abort otherwise
bun run regen
```

- [ ] **Step 2: Full retrain**

```bash
uv run train --local
```

This retrains the encoder + heads + GAN from scratch (Tasks 1-4 changed what the color tensors mean, invalidating every prior checkpoint) and auto-runs `tools/report.py` at the end, which also re-runs `model/export_onnx.py` (Task 2's change), refreshing `web/public/model.onnx` + `meta.json` and the Android assets.

- [ ] **Step 3: Read the new report, recalibrate thresholds**

Open the freshly written `report/<newest>/report.html`. Look at:
- The "color generator" energy-distance numbers per keyword (`coffee`, `chocolate`, `lemon`, `dark`, `bright`, `green`, `blue`, `gold`, `" red "`) — compare against `goals.yml`'s `color generator: energy distance:` targets (currently all `0.1`, tuned for the old RGB-unit space). Update each target in `goals.yml` to a value that's achievable-but-meaningful given the new OKLAB-space numbers actually observed (the same judgment call the `0.1` values originally represented, just re-anchored to the new units).
- The "cards" section's `pure_accuracy`/`pure_mean_distance` per color (`red`, `green`, `blue`, `dark`, `bright`). If the hit rate looks saturated (near 0% or 100% for most colors), adjust `CARD_PURE_THRESHOLD_OKLAB` and/or `CARD_PURE_THRESHOLD_L` in `tools/report.py` (currently `0.251` / `0.6`, also tuned for the old space) to a value that produces a meaningfully discriminating pass/fail split, then re-run `uv run report --pt pt/` to confirm the new thresholds render sensibly.

- [ ] **Step 4: Update the CLAUDE.md line describing `model/color.py`**

In `CLAUDE.md`, under "Architecture (read the code, not this file)":

```
- `model/color.py` (`energy_distance`, used by the GAN loss, computed directly on RGB)
```

becomes:

```
- `model/color.py` (`energy_distance`, used by the GAN loss, computed directly on OKLAB-encoded colors — see `model/data.py`'s `hex_to_color_unit`/`AB_RANGE`)
```

- [ ] **Step 5: Commit**

```bash
git add goals.yml tools/report.py CLAUDE.md pt/ web/public/ android/app/src/main/assets/ report/ runs/
git commit -m "chore: retrain under OKLAB color codec, recalibrate report thresholds"
```

(Confirm with `git status` beforehand exactly which generated paths are tracked vs. gitignored in this repo — `CLAUDE.md` notes `.pt` is gitignored and `web/public/` should be committed; add only what `git status` actually shows as trackable.)

---

## Execution Notes

- Tasks 1-4 (Python/model) have no dependency on Tasks 5-7 (web/Android) and vice versa; they can run in either order or in parallel. Task 8 depends on all of 1-7 being done, since it retrains against the final codec and both apps' decode logic.
- Nothing before Task 8 should be judged by "does the app show the right color" — only by its own unit tests. The live app will show incorrect colors from the moment Task 2 lands until Task 8's retrain completes; this is expected (see Review Focus).
