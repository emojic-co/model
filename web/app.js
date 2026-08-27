const input = document.getElementById("input");
const card = document.getElementById("card");
const emojiEl = document.getElementById("emoji");
const typedEl = document.getElementById("typed");
const feelingEl = document.getElementById("feeling");

// Oklab [L, a, b] -> CSS oklab() color string.
const oklab = ([L, a, b]) => `oklab(${L.toFixed(4)} ${a.toFixed(4)} ${b.toFixed(4)})`;

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
}

input.addEventListener("input", update);
update();
