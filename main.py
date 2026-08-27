from torch import nn

feeling = [
    'Calm',
    'Cute',
    'Happy',
    'Loud',
    'Excited',
]

vocab = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?;:()[]{}<>@#$%^&*'
emojis = "😀😂😍🥰😎🤔😅😭🥳🙃👍👎👏🙏💪🔥💯❤️✨⭐🎉🚀🍕🍔🍟🍦🍩🍺🍷☕🏀⚽🎮🎲🎸🎨✈️🚗🚲🌴🌈☀️🌙⭐🐶🐱🦁🐼🦊🍎🍌🥑🌶️🍿🍻🥂🏆🎯🎶🎤💡🔑📌⚡💥🎉👑💍💎💖💔💤🤖👽💀👻💩🎃🔮🚀🚢⛵🚗🚲🚨🏆⚽🏀🏈⚾🎾🎱🎮🎯🎲🎨🎤🎶🎷🎸🎹🎺🥁📱💻🎥📷📸🔍💡🔦🕯️💰💎⚖️🛒🎁🎈🎉🎊✉️📦📌📍🔑🔒🔓❤️‍🔥💖"


class Model(nn.Module):
    def __init__(self, embed_dim=8, hidden_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.emoji = nn.Linear(hidden_dim, len(emojis))
        self.feeling = nn.Linear(hidden_dim, len(feeling))
        # RGB values for gradient background start color
        self.bg1 = nn.Linear(hidden_dim, 3)
        # RGB values for gradient background end color
        self.bg2 = nn.Linear(hidden_dim, 3)

    def forward(self, x, state=None):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        out, (h_n, c_n) = self.lstm(x, state)

        # Take the hidden state of the final time step
        last_step = out[:, -1, :]  # (batch_size, hidden_dim)
        emoji_logits = self.emoji(last_step)  # (batch_size, len(emojis))
        feeling_logits = self.feeling(last_step)  # (batch_size, len(feeling))
        bg1 = self.bg1(last_step)  # (batch_size, 3)
        bg2 = self.bg2(last_step)  # (batch_size, 3)

        return emoji_logits, feeling_logits, bg1, bg2, (h_n, c_n)


if __name__ == "__main__":
    ...
