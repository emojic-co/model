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

let seq = 0;

async function update() {
  const text = input.value;
  typedEl.textContent = text;

  const mine = ++seq;
  let res;
  try {
    res = await fetch(`/predict?text=${encodeURIComponent(text)}`);
  } catch {
    return;
  }
  if (mine !== seq) return; // a newer keystroke already won

  const r = await res.json();
  emojiEl.textContent = r.emoji;
  feelingEl.textContent = r.feeling;
  card.style.background = `linear-gradient(135deg, ${oklab(r.bg1)}, ${oklab(r.bg2)})`;
  card.style.color = oklab(r.text_color);
  card.style.fontFamily = FEELING_FONTS[r.feeling] ?? FEELING_FONTS.Neutral;
}

input.addEventListener("input", update);
update();

// Bottom-of-page reference row: one small square per feeling, each rendered
// with that feeling's own font family and gradient background.
async function buildFeelingRow() {
  const row = document.getElementById("feelings");
  let feelings;
  try {
    feelings = await (await fetch("/feelings")).json();
  } catch {
    return;
  }
  row.replaceChildren();
  for (const f of feelings) {
    const sq = document.createElement("div");
    sq.className = "swatch";
    sq.textContent = f.feeling;
    sq.style.background = `linear-gradient(135deg, ${oklab(f.bg1)}, ${oklab(f.bg2)})`;
    sq.style.color = oklab(f.text_color);
    sq.style.fontFamily = FEELING_FONTS[f.feeling] ?? FEELING_FONTS.Neutral;
    row.appendChild(sq);
  }
}
buildFeelingRow();
