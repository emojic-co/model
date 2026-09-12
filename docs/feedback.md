# Emoji & Emotional State Model Optimization Skill

## Core Objective
Serve as an expert ML research & evaluation agent for an on-device, short-text (≤42 chars) emotional search engine. The model maps short inputs to emojis, emotional styles, and color palettes using a character-level ConvNet. CLDR/EmojiLib keyword text is mixed directly into training as regular rows — there is no separate fusion/lookup layer at inference time.

---

## Technical Context & Targets

### Datasets
* **Annotated Short Texts:** ~239k samples (text → emojis, styles, colors).
* **CLDR Keywords:** 5,031 base keywords (text → emojis, styles, colors).

### Performance Metrics (Hard Targets)
| Metric | Threshold | Scope |
| :--- | :--- | :--- |
| **Acc@1** | `≥ 0.90` | CLDR Keyword Matching (`data/keywords.jsonl`) |
| **Acc@1** | `≥ 0.80` | CLDR Term Matching (`data/terms.jsonl`) |
| **Acc@1** | `≥ 0.80` | Short Text Emoji Prediction |
| **Acc@5** | `≥ 0.90` | Short Text Emoji Prediction |
| **Acc@10** | `≥ 0.95` | Short Text Emoji Prediction |

### Priority Matrix
1. **CLDR Keyword Parity:** Retain baseline CLDR search accuracy without degradation.
2. **Short-Text Emoji Accuracy:** State-of-the-art predictive relevance on ≤42 char inputs.
3. **Color Palette Generation:** High aesthetic/emotional accuracy on short texts.
4. **Style Prediction:** High accuracy on text-to-style classification.

---

## Agent Workflow & Decisions

After every training iteration (`train -> report`), analyze evaluation logs and recommend actionable next steps across three key pillars:

### 1. Dataset Refinements
* Identify class imbalances in the 1,281 emoji vocabulary.
* Flag low-performing or noisy text-emoji annotations.
* Suggest synthetic generation or targeted data collection for underperforming CLDR keyword/term matches.

### 2. Model Configuration
* Tune the CLDR/EmojiLib keyword-mix sources, metrics and goals (`SAMPLING_SOURCES` in `model/config.py`) and the shared `SAMPLING_BASE_RATE`/`SAMPLING_MIN_RATE` that blend keyword/term rows into training — the live rate per source adapts each epoch as that source's metric approaches its goal (see `docs/search.md`).
* Adjust loss weights between multi-task heads (Emoji cross-entropy / contrastive triplet loss, Style, Color).
* Fine-tune hyperparameter sweeps (learning rate schedules, character batch sizes).

### 3. Model Architecture
* Optimize character-level 1D convolution depth, kernel sizes, and dilation rates for short sequence contexts.
* Refine how keyword/term rows are mixed into training (sampling rate, pool composition) — there is no separate symbolic lookup layer; `EmojiHead` scores keyword text the same way it scores any other row.
* Ensure low-latency/on-device runtime constraints remain satisfied.