import torch
import torch.nn as nn

# ----------------------------
# Discriminator (same as training)
# ----------------------------
class Discriminator(nn.Module):
    def __init__(self, vocab_size, hidden=128, nclass=6, max_len=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.adv = nn.Linear(hidden, 1)
        self.cls = nn.Linear(hidden, nclass)

    def forward(self, x):
        emb = self.embed(x)
        _, (h, _) = self.lstm(emb)
        h = h[-1]
        validity = torch.sigmoid(self.adv(h))
        cls_logits = self.cls(h)
        return validity, cls_logits

# ----------------------------
# Load model and predict
# ----------------------------
def test_model(model_path, user_input):
    ckpt = torch.load(model_path, map_location="cpu")
    vocab, mtypes = ckpt["vocab"], ckpt["mtypes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = Discriminator(len(vocab), hidden=128, nclass=len(mtypes)).to(device)
    D.load_state_dict(ckpt["D"])
    D.eval()

    # Encode input
    max_len = 256
    arr = [vocab.get(c, 3) for c in user_input][:max_len]
    arr = arr + [0] * (max_len - len(arr))
    seq = torch.tensor([arr], device=device)

    with torch.no_grad():
        _, cls_logits = D(seq)
        pred = cls_logits.argmax(-1).item()
        print("🔍 Predicted Malware Type:", mtypes[pred])

if __name__ == "__main__":
    # Example usage
    test_model("gan_classifier.pt", "obfuscated_function_example_here")
