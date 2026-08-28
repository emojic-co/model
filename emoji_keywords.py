"""Hand-curated keyword lists for the model test suite (see test_model.py).

For each of the ~50 most frequent emojis in data.jsonl, a short list of words
and phrases that a person would strongly associate with that emoji. The test
suite feeds each keyword to the model and checks whether the paired emoji is
predicted. The lists are intentionally about the *meaning* of the emoji, not
copied from any training text.
"""

# emoji -> keywords strongly associated with it
EMOJI_KEYWORDS: dict[str, list[str]] = {
    "😤": ["hmph", "so done with this", "fed up", "over it", "steaming mad"],
    "😰": ["cold sweat", "so nervous", "panicking", "worried sick", "dreading this"],
    "😬": ["awkward", "yikes", "cringe", "so tense", "eek"],
    "🎉": ["party", "congrats", "we did it", "let's celebrate", "hooray"],
    "😌": ["relieved", "at peace", "content", "finally relaxed", "so calm now"],
    "😠": ["angry", "annoyed", "mad", "irritated", "grr"],
    "😟": ["worried", "concerned", "uneasy", "nervous about this", "anxious"],
    "😊": ["so happy", "smiling", "feeling good", "glad", "warm and happy"],
    "😔": ["sad", "feeling down", "disappointed", "heavy heart", "feeling low"],
    "😞": ["let down", "bummed", "gutted", "so disappointed", "dejected"],
    "🤩": ["amazing", "wow", "so excited", "incredible", "starstruck"],
    "😄": ["haha", "great day", "laughing", "cheerful", "so glad"],
    "💔": ["heartbroken", "broke up", "broken heart", "devastated", "miss you so much"],
    "😡": ["furious", "raging", "livid", "pissed off", "so angry"],
    "😢": ["crying", "tearing up", "want to cry", "in tears", "so sad"],
    "🤔": ["hmm", "thinking", "not sure", "let me think", "curious"],
    "😅": ["phew", "close call", "nervous laugh", "awkward haha", "barely made it"],
    "😩": ["so tired", "exhausted", "can't anymore", "drained", "weary"],
    "😣": ["struggling", "so hard", "ugh", "frustrated", "hang in there"],
    "🤬": ["cursing", "enraged", "absolutely livid", "so furious", "seeing red"],
    "☕": ["coffee", "espresso", "morning brew", "latte", "need caffeine"],
    "🔥": ["on fire", "lit", "crushing it", "so hot", "fire"],
    "🙂": ["okay", "it's fine", "sure", "alright", "no worries"],
    "✨": ["sparkle", "magical", "shiny", "glowing", "something special"],
    "😱": ["omg", "terrified", "so scared", "shocked", "screaming"],
    "🙄": ["eye roll", "whatever", "seriously", "ugh again", "rolling my eyes"],
    "😨": ["scared", "afraid", "frightened", "so anxious", "fear"],
    "🙌": ["yes finally", "praise hands", "hands up", "so grateful", "hallelujah"],
    "🌧️": ["rain", "rainy day", "pouring outside", "grey skies", "drizzle"],
    "⏳": ["waiting", "running out of time", "hourglass", "time is up", "so slow"],
    "🥳": ["birthday", "party time", "celebrating", "woohoo", "let's party"],
    "😴": ["sleepy", "going to bed", "need sleep", "yawn", "so drowsy"],
    "😭": ["sobbing", "bawling", "crying so hard", "ugly crying", "can't stop crying"],
    "😂": ["lol", "so funny", "hilarious", "dying laughing", "rofl"],
    "🚀": ["launch", "rocket", "blast off", "to the moon", "shipping it"],
    "😒": ["unamused", "not impressed", "meh", "really though", "so annoyed"],
    "🌿": ["plants", "greenery", "fresh leaves", "garden", "nature walk"],
    "👻": ["ghost", "boo", "spooky", "haunted", "halloween"],
    "😵‍💫": ["overwhelmed", "dizzy", "head spinning", "too much", "can't focus"],
    "🕊️": ["peace", "serenity", "dove", "tranquil", "calm and peaceful"],
    "😐": ["neutral", "no comment", "blank stare", "indifferent", "meh"],
    "🧘": ["meditating", "yoga", "breathe", "mindfulness", "zen"],
    "🤫": ["shh", "secret", "keep it quiet", "don't tell anyone", "hush"],
    "🌮": ["taco", "tacos", "taco night", "burrito", "mexican food"],
    "🎶": ["music", "song", "melody", "listening to music", "tunes"],
    "😕": ["confused", "unsure", "puzzled", "not sure about this", "hmm no"],
    "⚡": ["lightning", "electric energy", "thunderbolt", "power surge", "a jolt"],
    "💪": ["strong", "you got this", "workout", "powerful", "stay strong"],
    "🥰": ["love you", "adore you", "so much love", "in love", "my heart is full"],
    "🌙": ["moon", "goodnight", "night sky", "moonlight", "late night"],
}
