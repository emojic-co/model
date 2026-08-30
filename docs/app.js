const input = document.getElementById("input");
const counterEl = document.getElementById("counter");
const counterMaxEl = document.getElementById("counter-max");
const card = document.getElementById("card");
const emojiEl = document.getElementById("emoji");
const typedEl = document.getElementById("typed");
const feelingEl = document.getElementById("feeling");
const copyBtn = document.getElementById("copy");
const toastEl = document.getElementById("toast");
const dbgFeelingsEl = document.getElementById("dbg-feelings");
const dbgEmojisEl = document.getElementById("dbg-emojis");

const FEELING_FONTS = {
  Happy: '"Fredoka", system-ui, sans-serif',
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

const softmax = (arr) => {
  let m = -Infinity;
  for (const x of arr) if (x > m) m = x;
  const exps = Array.from(arr, (x) => Math.exp(x - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
};

let META;
let PALETTE;
let CHAR2IDX;
let session;
let seq = 0;
let current = null;

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

function encode(text) {
  const norm = normalize(text).slice(0, META.max_text_len);
  const ids = new Array(META.max_text_len).fill(META.pad_idx);
  for (let i = 0; i < norm.length; i++) ids[i] = CHAR2IDX.get(norm[i]);
  return BigInt64Array.from(ids, BigInt);
}

async function update() {
  const text = input.value;
  typedEl.textContent = text;
  counterEl.firstChild.textContent = String(text.length);
  counterEl.classList.toggle("full", text.length >= input.maxLength);
  if (!session) return;

  const mine = ++seq;
  const tensor = new ort.Tensor("int64", encode(text), [1, META.max_text_len]);
  const out = await session.run({ input: tensor });
  if (mine !== seq) return;

  const emoji = META.emojis[argmax(out.emoji_logits.data)];
  const feeling = META.feelings[argmax(out.feeling_logits.data)];
  const pal = PALETTE[feeling] ?? PALETTE.Neutral;

  renderDebug(out.feeling_logits.data, out.emoji_logits.data);

  emojiEl.textContent = emoji;
  feelingEl.textContent = feeling;
  card.style.background = `linear-gradient(135deg, ${pal.bg1}, ${pal.bg2})`;
  card.style.color = pal.text_color;
  card.style.fontFamily = FEELING_FONTS[feeling] ?? FEELING_FONTS.Neutral;

  current = { text, emoji, feeling, pal };
}

function renderDebug(feelingLogits, emojiLogits) {
  const fp = softmax(feelingLogits);
  const feelings = META.feelings
    .map((name, i) => ({ label: name, p: fp[i] }))
    .sort((a, b) => b.p - a.p);
  dbgFeelingsEl.replaceChildren(...feelings.map(probRow));

  const ep = softmax(emojiLogits);
  const emojis = META.emojis
    .map((ch, i) => ({ label: ch, p: ep[i] }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 10);
  dbgEmojisEl.replaceChildren(...emojis.map(probRow));
}

function probRow({ label, p }) {
  const li = document.createElement("li");
  const name = document.createElement("span");
  name.className = "dbg-label";
  name.textContent = label;
  const track = document.createElement("span");
  track.className = "dbg-track";
  const fill = document.createElement("span");
  fill.className = "dbg-fill";
  fill.style.width = `${(p * 100).toFixed(1)}%`;
  track.appendChild(fill);
  const pct = document.createElement("span");
  pct.className = "dbg-pct";
  pct.textContent = `${(p * 100).toFixed(1)}%`;
  li.append(name, track, pct);
  return li;
}

const fontName = (stack) => stack.match(/"([^"]+)"/)?.[1] ?? null;

async function ensureFont(stack) {
  const name = fontName(stack);
  if (!name || !document.fonts) return;
  try {
    await Promise.all([
      document.fonts.load(`600 24px "${name}"`),
      document.fonts.load(`400 13px "${name}"`),
    ]);
  } catch {}
}

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

async function cardToBlob() {
  const S = 512;
  const { text, emoji, feeling, pal } = current;
  const stack = FEELING_FONTS[feeling] ?? FEELING_FONTS.Neutral;
  await ensureFont(stack);

  const canvas = document.createElement("canvas");
  canvas.width = S;
  canvas.height = S;
  const ctx = canvas.getContext("2d");

  const r = Math.round((20 / 600) * S);
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(0, 0, S, S, r);
  else ctx.rect(0, 0, S, S);
  ctx.clip();

  const grad = ctx.createLinearGradient(0, 0, S, S);
  grad.addColorStop(0, pal.bg1);
  grad.addColorStop(1, pal.bg2);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, S, S);

  ctx.fillStyle = pal.text_color;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  ctx.font = `120px "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif`;
  ctx.fillText(emoji, S / 2, S * 0.36);

  ctx.font = `600 24px ${stack}`;
  const lines = wrapLines(ctx, text, S - 96, 4);
  let ty = S * 0.6;
  for (const line of lines) {
    ctx.fillText(line, S / 2, ty);
    ty += 32;
  }

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
  let CONFIG;
  [META, CONFIG, PALETTE] = await Promise.all([
    fetch("./meta.json").then((r) => r.json()),
    fetch("./config.json").then((r) => r.json()),
    fetch("./palette.json").then((r) => r.json()),
  ]);
  CHAR2IDX = new Map([...META.chars].map((c, i) => [c, i]));
  input.maxLength = CONFIG.max_text_len;
  counterMaxEl.textContent = `/${CONFIG.max_text_len}`;
  if (META.exported_at) {
    const d = new Date(META.exported_at);
    document.getElementById("model-date").textContent = Number.isNaN(d.getTime())
      ? META.exported_at
      : d.toLocaleString(undefined, {
          year: "numeric",
          month: "short",
          day: "numeric",
          hour: "numeric",
          minute: "2-digit",
          timeZoneName: "short",
        });
  }
  buildFeelingRow();

  ort.env.wasm.numThreads = 1;
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
