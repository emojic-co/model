# Emoji & Emotional State Model Optimization Skill

## Core Objective
Serve as an expert ML research & evaluation agent for an on-device, short-text (≤42 chars) emotional search engine. The model maps short inputs to emojis (1,281 target vocabulary), emotional styles, and color palettes using a character-level ConvNet + keyword fusion architecture.

---

## Technical Context & Targets

### Datasets
* **Annotated Short Texts:** ~239k samples (text → emojis, styles, colors).
* **CLDR Keywords:** 5,031 base keywords (text → emojis, styles, colors).

### Performance Metrics (Hard Targets)
| Metric | Threshold | Scope |
| :--- | :--- | :--- |
| **Acc@1** | `≥ 0.95` | CLDR Exact Keyword Matching |
| **Acc@1** | `≥ 0.90` | CLDR Fuzzy Keyword Matching |
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
* Suggest synthetic generation or targeted data collection for underperforming CLDR fuzzy matches.

### 2. Model Configuration
* Tune fusion head weights (balancing deterministic CLDR keyword lookup vs. neural embedding predictions).
* Adjust loss weights between multi-task heads (Emoji cross-entropy / contrastive triplet loss, Style, Color).
* Fine-tune hyperparameter sweeps (learning rate schedules, character batch sizes).

### 3. Model Architecture
* Optimize character-level 1D convolution depth, kernel sizes, and dilation rates for short sequence contexts.
* Refine the fusion layer architecture connecting symbolic keyword features with deep feature representations.
* Ensure low-latency/on-device runtime constraints remain satisfied.