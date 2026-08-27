// Standalone in-browser inference. The model (model.onnx) and everything
// main.py owns (char vocab, MAX_TEXT_LEN, emoji/feeling label sets, the feeling
// color palette) come from meta.json, written by build_web.py -- nothing here
// is hardcoded from the Python side.

const input = document.getElementById("input");
const card = document.getElementById("card");
const emojiEl = document.getElementById("emoji");
const typedEl = document.getElementById("typed");
const feelingEl = document.getElementById("feeling");

// Oklab [L, a, b] -> CSS oklab() color string.
const oklab = ([L, a, b]) => `oklab(${L.toFixed(4)} ${a.toFixed(4)} ${b.toFixed(4)})`;

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
let CHAR2IDX;
let session;
let seq = 0;

// Mirror main.py's normalize(): collapse whitespace, lowercase, drop anything
// not in the model vocab.
function normalize(text) {
  const t = text.replace(/\s+/g, " ").trim().toLowerCase();
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
  const pal = META.feeling_palette[feeling];

  emojiEl.textContent = emoji;
  feelingEl.textContent = feeling;
  card.style.background = `linear-gradient(135deg, ${oklab(pal.bg1)}, ${oklab(pal.bg2)})`;
  card.style.color = oklab(pal.text_color);
  card.style.fontFamily = FEELING_FONTS[feeling] ?? FEELING_FONTS.Neutral;
}

// Bottom-of-page reference row: one small square per feeling, each rendered
// with that feeling's own font family and gradient background.
function buildFeelingRow() {
  const row = document.getElementById("feelings");
  row.replaceChildren();
  for (const feeling of META.feelings) {
    const pal = META.feeling_palette[feeling];
    const sq = document.createElement("div");
    sq.className = "swatch";
    sq.textContent = feeling;
    sq.style.background = `linear-gradient(135deg, ${oklab(pal.bg1)}, ${oklab(pal.bg2)})`;
    sq.style.color = oklab(pal.text_color);
    sq.style.fontFamily = FEELING_FONTS[feeling] ?? FEELING_FONTS.Neutral;
    row.appendChild(sq);
  }
}

(async () => {
  META = await (await fetch("./meta.json")).json();
  CHAR2IDX = new Map([...META.chars].map((c, i) => [c, i]));
  buildFeelingRow();

  ort.env.wasm.numThreads = 1; // GitHub Pages sends no COOP/COEP headers
  ort.env.wasm.wasmPaths = "./vendor/";
  session = await ort.InferenceSession.create("./model.onnx");

  input.addEventListener("input", update);
  update();
})();
