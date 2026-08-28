// Standalone in-browser inference. The model (model.onnx) and everything the
// Python side owns (char vocab, MAX_TEXT_LEN, emoji/feeling label sets) come
// from meta.json, written by train.py's export_web -- nothing here is hardcoded
// from the Python side. The feeling color palette is separate data (palette.json,
// not touched by Python) and is fetched alongside it.

const input = document.getElementById("input");
const card = document.getElementById("card");
const emojiEl = document.getElementById("emoji");
const typedEl = document.getElementById("typed");
const feelingEl = document.getElementById("feeling");
const copyBtn = document.getElementById("copy");
const toastEl = document.getElementById("toast");

// One Google-fonts webfont per feeling (loaded in index.html). The mood of the
// typeface is meant to echo the mood of the feeling.
const FEELING_FONTS = {
  Happy: '"Fredoka", system-ui, sans-serif',
  Excited: '"Bangers", system-ui, cursive',
  Calm: '"Quicksand", system-ui, sans-serif',
  Sad: '"Playfair Display", Georgia, serif',
  Angry: '"Anton", system-ui, sans-serif',
  Anxious: '"Shantell Sans", system-ui, cursive',
  Neutral: '"Inter", system-ui, sans-serif',
};

const argmax = (arr) => {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
};

let META;
let PALETTE;
let CHAR2IDX;
let session;
let seq = 0;
// Latest prediction, mirrored so the "copy" button can redraw it on a canvas.
let current = null;

// Mirror data.py's normalize(): collapse whitespace, lowercase, clamp any run
// of 3+ identical chars down to 2, then drop anything not in the model vocab.
// Must stay byte-identical to data.py or browser inference sees a different
// input distribution than training.
function normalize(text) {
  const t = text
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase()
    .replace(/(.)\1{2,}/g, "$1$1");
  let out = "";
  for (const c of t) if (CHAR2IDX.has(c)) out += c;
  return out;
}

// Mirror main.py's encode(): char indices, right-padded to max_text_len.
function encode(text) {
  const norm = normalize(text).slice(0, META.max_text_len);
  const ids = new Array(META.max_text_len).fill(META.pad_idx);
  for (let i = 0; i < norm.length; i++) ids[i] = CHAR2IDX.get(norm[i]);
  return BigInt64Array.from(ids, BigInt);
}

async function update() {
  const text = input.value;
  typedEl.textContent = text;
  if (!session) return;

  const mine = ++seq;
  const tensor = new ort.Tensor("int64", encode(text), [1, META.max_text_len]);
  const out = await session.run({ input: tensor });
  if (mine !== seq) return; // a newer keystroke already won

  const emoji = META.emojis[argmax(out.emoji_logits.data)];
  const feeling = META.feelings[argmax(out.feeling_logits.data)];
  const pal = PALETTE[feeling] ?? PALETTE.Neutral;

  emojiEl.textContent = emoji;
  feelingEl.textContent = feeling;
  card.style.background = `linear-gradient(135deg, ${pal.bg1}, ${pal.bg2})`;
  card.style.color = pal.text_color;
  card.style.fontFamily = FEELING_FONTS[feeling] ?? FEELING_FONTS.Neutral;

  current = { text, emoji, feeling, pal };
}

// The primary family name (the quoted token) out of a FEELING_FONTS stack,
// e.g. '"Playfair Display", Georgia, serif' -> 'Playfair Display'.
const fontName = (stack) => stack.match(/"([^"]+)"/)?.[1] ?? null;

// Wait for the feeling's webfont so canvas text is not drawn in a fallback
// face before the real one loads. Best-effort: a load failure just falls back.
async function ensureFont(stack) {
  const name = fontName(stack);
  if (!name || !document.fonts) return;
  try {
    await Promise.all([
      document.fonts.load(`600 24px "${name}"`),
      document.fonts.load(`400 13px "${name}"`),
    ]);
  } catch {
    /* fallback face is acceptable */
  }
}

// Greedy word wrap against a pixel width, capped at `maxLines`.
function wrapLines(ctx, text, maxWidth, maxLines) {
  const words = text.split(/\s+/).filter(Boolean);
  const lines = [];
  let line = "";
  for (const w of words) {
    const next = line ? `${line} ${w}` : w;
    if (line && ctx.measureText(next).width > maxWidth) {
      lines.push(line);
      line = w;
    } else {
      line = next;
    }
  }
  if (line) lines.push(line);
  return lines.slice(0, maxLines);
}

// Redraw the current card onto a 512x512 canvas and hand back a PNG blob.
// Hand-drawn (gradient + emoji + text) rather than a DOM snapshot so it needs
// no html2canvas-style dependency.
async function cardToBlob() {
  const S = 512;
  const { text, emoji, feeling, pal } = current;
  const stack = FEELING_FONTS[feeling] ?? FEELING_FONTS.Neutral;
  await ensureFont(stack);

  const canvas = document.createElement("canvas");
  canvas.width = S;
  canvas.height = S;
  const ctx = canvas.getContext("2d");

  // Rounded square (20/600 of the card, scaled), transparent outside.
  const r = Math.round((20 / 600) * S);
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(0, 0, S, S, r);
  else ctx.rect(0, 0, S, S);
  ctx.clip();

  // linear-gradient(135deg, ...): top-left -> bottom-right on a square.
  const grad = ctx.createLinearGradient(0, 0, S, S);
  grad.addColorStop(0, pal.bg1);
  grad.addColorStop(1, pal.bg2);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, S, S);

  ctx.fillStyle = pal.text_color;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  // Emoji: a dedicated emoji stack so the glyph renders regardless of `stack`.
  ctx.font = `120px "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif`;
  ctx.fillText(emoji, S / 2, S * 0.36);

  // Typed text, wrapped.
  ctx.font = `600 24px ${stack}`;
  const lines = wrapLines(ctx, text, S - 96, 4);
  let ty = S * 0.6;
  for (const line of lines) {
    ctx.fillText(line, S / 2, ty);
    ty += 32;
  }

  // Feeling label.
  ctx.font = `600 13px ${stack}`;
  if ("letterSpacing" in ctx) ctx.letterSpacing = "3.5px";
  ctx.globalAlpha = 0.85;
  ctx.fillText(feeling.toUpperCase(), S / 2, S * 0.84);
  ctx.globalAlpha = 1;

  return new Promise((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("toBlob failed"))), "image/png");
  });
}

let toastTimer;
function toast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove("show"), 1600);
}

async function copyCard() {
  if (!current) {
    toast("nothing to copy yet");
    return;
  }
  try {
    const blob = await cardToBlob();
    await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    toast("copied to clipboard ✓");
  } catch (err) {
    console.error(err);
    toast("copy failed");
  }
}

// Bottom-of-page reference row: one small square per feeling, each rendered
// with that feeling's own font family and gradient background.
function buildFeelingRow() {
  const row = document.getElementById("feelings");
  row.replaceChildren();
  for (const feeling of META.feelings) {
    const pal = PALETTE[feeling] ?? PALETTE.Neutral;
    const sq = document.createElement("div");
    sq.className = "swatch";
    sq.textContent = feeling;
    sq.style.background = `linear-gradient(135deg, ${pal.bg1}, ${pal.bg2})`;
    sq.style.color = pal.text_color;
    sq.style.fontFamily = FEELING_FONTS[feeling] ?? FEELING_FONTS.Neutral;
    row.appendChild(sq);
  }
}

(async () => {
  [META, PALETTE] = await Promise.all([
    fetch("./meta.json").then((r) => r.json()),
    fetch("./palette.json").then((r) => r.json()),
  ]);
  CHAR2IDX = new Map([...META.chars].map((c, i) => [c, i]));
  buildFeelingRow();

  ort.env.wasm.numThreads = 1; // GitHub Pages sends no COOP/COEP headers
  // Absolute URL so ORT resolves it against the page, not against ort.wasm.min.js
  // (which already lives in vendor/ and would give vendor/vendor/).
  ort.env.wasm.wasmPaths = new URL("./vendor/", document.baseURI).href;
  session = await ort.InferenceSession.create("./model.onnx");

  input.addEventListener("input", update);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      copyCard();
    }
  });
  copyBtn.addEventListener("click", copyCard);
  update();
})();
