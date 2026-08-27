"""Regenerate data.jsonl from scratch.

Emits 60 emojis x 7 feelings x 5 short texts = 2100 samples, one JSON object
per line with keys: emoji, feeling, text. Rows are grouped emoji-major then
feeling-minor. Deterministic — running it twice produces the same file.

Each text is a short (roughly 2-6 word) phrase that pairs a word tied to the
emoji's picture (EMOJI_WORDS) with a cue for the row's feeling (FEELING_MOODS),
e.g. 🍕 + Angry -> "cold crust, fed up".

Run:  uv run gen_data.py
"""

import json

from main import EMOJIS, feeling

# 6 short, lowercase, punctuation-free phrases evoking each emoji's picture.
# Only the first 5 are used per feeling; the 6th is spare for de-duping.
EMOJI_WORDS = {
    "😀": [
        "big grin",
        "wide smile",
        "beaming face",
        "cheerful hello",
        "bright smile",
        "happy face",
    ],
    "😂": [
        "belly laugh",
        "that joke",
        "tears of laughter",
        "giggle fit",
        "the punchline",
        "cracking up",
    ],
    "🥹": [
        "happy tears",
        "tight throat",
        "welling up",
        "tender moment",
        "a heartfelt hug",
        "holding back tears",
    ],
    "😍": [
        "heart eyes",
        "total crush",
        "swooning",
        "adoring this",
        "smitten",
        "love struck",
    ],
    "🤔": [
        "deep thought",
        "a chin scratch",
        "hmm moment",
        "puzzling it over",
        "weighing options",
        "pondering hard",
    ],
    "🥳": [
        "party hat",
        "confetti toss",
        "big celebration",
        "the bash",
        "party time",
        "streamers",
    ],
    "😎": [
        "cool shades",
        "sunglasses on",
        "effortless cool",
        "the swagger",
        "too cool",
        "slick look",
    ],
    "😭": [
        "the big cry",
        "many tears",
        "teary eyes",
        "a crying spell",
        "wet cheeks",
        "sobbing hard",
    ],
    "💀": [
        "dead tired",
        "just bones",
        "skull vibes",
        "totally wiped",
        "worn out",
        "grave mood",
    ],
    "🔥": [
        "roaring flames",
        "blazing fire",
        "burning hot",
        "wildfire",
        "red hot",
        "up in flames",
    ],
    "❤️": [
        "full heart",
        "warm love",
        "a heartbeat",
        "true love",
        "big heart",
        "close hug",
    ],
    "💯": [
        "full marks",
        "hundred percent",
        "nailed it",
        "perfect score",
        "all in",
        "no notes",
    ],
    "✨": [
        "little sparkles",
        "glitter",
        "shimmer",
        "magic dust",
        "shiny glow",
        "a twinkle",
    ],
    "👍": [
        "thumbs up",
        "all good",
        "approved",
        "sounds good",
        "green light",
        "nice work",
    ],
    "👏": [
        "big applause",
        "clap clap",
        "standing ovation",
        "well done",
        "applause",
        "bravo",
    ],
    "🙌": [
        "hands up",
        "praise hands",
        "we did it",
        "raise the roof",
        "hooray",
        "high fives",
    ],
    "🙏": [
        "please please",
        "thank you",
        "prayer hands",
        "a grateful bow",
        "fingers crossed",
        "hoping hard",
    ],
    "💪": [
        "strong arms",
        "flex the muscle",
        "pure power",
        "the grind",
        "lifting heavy",
        "staying strong",
    ],
    "🧠": [
        "big brain",
        "brain power",
        "a smart idea",
        "deep thinking",
        "mind working",
        "brainstorm",
    ],
    "👀": [
        "wide eyes",
        "side eye",
        "watching close",
        "peeking",
        "cant look away",
        "eyes on it",
    ],
    "🐶": [
        "wagging tail",
        "puppy eyes",
        "good dog",
        "fetch time",
        "wet nose",
        "the dog walk",
    ],
    "🐱": [
        "purring cat",
        "soft paws",
        "kitten nap",
        "the cat stare",
        "whiskers",
        "a quiet meow",
    ],
    "🦁": [
        "the lion roar",
        "the mane",
        "king of pride",
        "fierce roar",
        "big cat",
        "prowling lion",
    ],
    "🦉": [
        "night owl",
        "wise owl",
        "soft hooting",
        "big owl eyes",
        "silent wings",
        "late owl hours",
    ],
    "🐙": [
        "eight arms",
        "octopus reach",
        "the tentacles",
        "an ink cloud",
        "deep sea",
        "grabbing everything",
    ],
    "🌲": [
        "pine forest",
        "tall trees",
        "woodland trail",
        "evergreen",
        "forest air",
        "under the pines",
    ],
    "🌺": [
        "bright bloom",
        "flower petals",
        "full blossom",
        "fresh bouquet",
        "the flowerbed",
        "petals open",
    ],
    "🌈": [
        "full rainbow",
        "seven colors",
        "sky arc",
        "rainbow light",
        "color band",
        "bright arc",
    ],
    "☀️": [
        "warm sun",
        "bright sunshine",
        "clear sky",
        "a sunbeam",
        "midday sun",
        "golden light",
    ],
    "⭐": [
        "bright star",
        "night stars",
        "make a wish",
        "gold star",
        "starlight",
        "shining star",
    ],
    "🍕": [
        "hot slice",
        "melty cheese",
        "pepperoni",
        "pizza night",
        "the crust",
        "the pizza box",
    ],
    "🌮": [
        "street taco",
        "taco truck",
        "spicy salsa",
        "taco night",
        "soft shell",
        "guac and taco",
    ],
    "🍣": [
        "sushi roll",
        "fresh nigiri",
        "soy and wasabi",
        "sushi platter",
        "raw fish",
        "the sushi bar",
    ],
    "☕": [
        "hot coffee",
        "first cup",
        "espresso shot",
        "coffee steam",
        "morning brew",
        "a cafe latte",
    ],
    "🍺": [
        "cold beer",
        "frothy pint",
        "the pub",
        "beers with friends",
        "pint glass",
        "happy hour",
    ],
    "⚽": [
        "the goal",
        "kickoff",
        "penalty kick",
        "match day",
        "soccer pitch",
        "final whistle",
    ],
    "🎉": [
        "party popper",
        "confetti burst",
        "big celebration",
        "a surprise",
        "streamers",
        "party time",
    ],
    "🚀": [
        "rocket launch",
        "blast off",
        "liftoff",
        "to the moon",
        "the countdown",
        "shooting up",
    ],
    "✈️": [
        "the flight",
        "boarding now",
        "takeoff",
        "window seat",
        "jet plane",
        "wheels up",
    ],
    "🎸": [
        "electric guitar",
        "power chord",
        "the riff",
        "band practice",
        "strumming",
        "guitar solo",
    ],
    "💡": [
        "bright idea",
        "lightbulb moment",
        "the spark",
        "a new idea",
        "it clicked",
        "bulb on",
    ],
    "💎": [
        "shiny diamond",
        "rare gem",
        "precious stone",
        "sparkling jewel",
        "the diamond",
        "flawless gem",
    ],
    "📱": [
        "the phone",
        "buzzing screen",
        "a new text",
        "endless scrolling",
        "phone in hand",
        "a notification",
    ],
    "🎁": [
        "wrapped gift",
        "the present",
        "ribbon and bow",
        "a surprise box",
        "gift for you",
        "unwrapping",
    ],
    "🔒": [
        "locked tight",
        "the padlock",
        "secure lock",
        "keys and lock",
        "locked door",
        "sealed shut",
    ],
    "🌍": [
        "the whole world",
        "planet earth",
        "around the globe",
        "the blue planet",
        "world map",
        "far away lands",
    ],
    "🏆": [
        "gold trophy",
        "first place",
        "the big win",
        "champions cup",
        "top prize",
        "the trophy shelf",
    ],
    "🎨": [
        "paint palette",
        "fresh canvas",
        "brush strokes",
        "mixing colors",
        "the painting",
        "art studio",
    ],
    "🔮": [
        "crystal ball",
        "the future",
        "fortune telling",
        "misty glass",
        "what comes next",
        "reading signs",
    ],
    "📍": [
        "drop a pin",
        "the location",
        "map marker",
        "right here",
        "pinned spot",
        "you are here",
    ],
    "💼": [
        "the briefcase",
        "big meeting",
        "the work day",
        "the office",
        "a client deal",
        "nine to five",
    ],
    "🩺": [
        "the checkup",
        "doctor visit",
        "the stethoscope",
        "test results",
        "the clinic",
        "heartbeat check",
    ],
    "💻": [
        "the laptop",
        "typing code",
        "the build",
        "screen glow",
        "late night coding",
        "the deadline",
    ],
    "⏰": [
        "the alarm",
        "ticking clock",
        "time is up",
        "early wake up",
        "clock hands",
        "running late",
    ],
    "🚗": [
        "the drive",
        "open road",
        "car engine",
        "a road trip",
        "traffic jam",
        "behind the wheel",
    ],
    "🌾": [
        "golden field",
        "wheat stalks",
        "harvest time",
        "the farm",
        "a country road",
        "grain fields",
    ],
    "⛈️": [
        "the storm",
        "thunder crack",
        "heavy rain",
        "lightning flash",
        "dark clouds",
        "storm rolling in",
    ],
    "🧩": [
        "puzzle piece",
        "the missing bit",
        "it fits",
        "the jigsaw",
        "solving it",
        "the last piece",
    ],
    "👑": [
        "gold crown",
        "the throne",
        "royal crown",
        "a king today",
        "crown jewels",
        "wear the crown",
    ],
    "🕊️": [
        "white dove",
        "peace dove",
        "calm wings",
        "an olive branch",
        "the dove flies",
        "quiet peace",
    ],
}

# 8 short cues per feeling; combined with an emoji word to set the mood.
FEELING_MOODS = {
    "Happy": [
        "pure joy",
        "so happy",
        "love it",
        "big smile",
        "feeling great",
        "heart full",
        "all good",
        "what a day",
    ],
    "Excited": [
        "cannot wait",
        "so hyped",
        "lets go",
        "buzzing",
        "pumped up",
        "here we go",
        "so excited",
        "hype is real",
    ],
    "Calm": [
        "so calm",
        "at peace",
        "deep breath",
        "nice and slow",
        "all quiet",
        "no rush",
        "settled now",
        "easy does it",
    ],
    "Sad": [
        "feeling low",
        "so sad",
        "miss it",
        "heavy heart",
        "tears close",
        "a hard day",
        "all gone now",
        "quiet ache",
    ],
    "Angry": [
        "so mad",
        "furious",
        "fed up",
        "had enough",
        "not again",
        "hate this",
        "raging",
        "done with it",
    ],
    "Anxious": [
        "so nervous",
        "worried sick",
        "on edge",
        "what if",
        "cant relax",
        "stomach in knots",
        "dreading it",
        "so shaky",
    ],
    "Neutral": [
        "nothing new",
        "just noting",
        "as usual",
        "same as ever",
        "no big deal",
        "a plain day",
        "it happened",
        "nothing special",
    ],
}


def build_rows() -> list[dict]:
    rows: list[dict] = []
    for emoji in EMOJIS:
        words = EMOJI_WORDS[emoji]
        for feel in feeling:
            moods = FEELING_MOODS[feel]
            # Stable per-pair rotation over the mood cues.
            offset = sum(ord(c) for c in emoji + feel) % len(moods)
            seen: set[str] = set()
            for i in range(5):
                word = words[i]
                step = 0
                while True:
                    mood = moods[(i + offset + step) % len(moods)]
                    text = f"{word}, {mood}" if i % 2 == 0 else f"{mood}, {word}"
                    if text not in seen:  # distinct words already keep these apart
                        break
                    step += 1
                seen.add(text)
                rows.append({"emoji": emoji, "feeling": feel, "text": text})
    return rows


if __name__ == "__main__":
    rows = build_rows()
    with open("data.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to data.jsonl")
