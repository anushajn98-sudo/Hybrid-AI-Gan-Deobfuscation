import json, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# ----------------------------
# Dataset
# ----------------------------
class CodeDataset(Dataset):
    def __init__(self, path, vocab=None, max_len=256):
        self.samples = [json.loads(line) for line in open(path, encoding="utf-8")]
        self.max_len = max_len

        if vocab is None:
            chars = set()
            for s in self.samples:
                chars.update(list(s["obfuscated"]))
            self.vocab = {c: i+4 for i, c in enumerate(sorted(chars))}
            self.vocab["<pad>"] = 0
            self.vocab["<sos>"] = 1
            self.vocab["<eos>"] = 2
            self.vocab["<unk>"] = 3
        else:
            self.vocab = vocab
        self.ivocab = {i: c for c, i in self.vocab.items()}

        self.mtypes = ["benign", "ransomware", "spyware", "trojan", "keylogger", "data-stealer"]
        self.mmap = {t: i for i, t in enumerate(self.mtypes)}

    def __len__(self): return len(self.samples)

    def encode(self, s):
        arr = [self.vocab.get(c, 3) for c in s][:self.max_len]
        return arr

    def pad(self, arr):
        return arr + [0] * (self.max_len - len(arr))

    def __getitem__(self, idx):
        s = self.samples[idx]
        obf = self.pad(self.encode(s["obfuscated"]))
        label = self.mmap[s["malware_type"]] if s["label"] == "malware" else 0
        return torch.tensor(obf), torch.tensor(label)

# ----------------------------
# GAN Models
# ----------------------------
class Generator(nn.Module):
    def __init__(self, vocab_size, hidden=128, max_len=256):
        super().__init__()
        self.latent_dim = hidden
        self.max_len = max_len
        self.fc = nn.Linear(hidden, max_len * hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, z):
        x = self.fc(z).view(-1, self.max_len, self.latent_dim)
        x, _ = self.lstm(x)
        logits = self.out(x)
        return logits.argmax(-1)  # generated sequence

class Discriminator(nn.Module):
    def __init__(self, vocab_size, hidden=128, nclass=6, max_len=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.adv = nn.Linear(hidden, 1)       # real/fake
        self.cls = nn.Linear(hidden, nclass)  # malware type

    def forward(self, x):
        emb = self.embed(x)
        _, (h, _) = self.lstm(emb)
        h = h[-1]
        validity = torch.sigmoid(self.adv(h))
        cls_logits = self.cls(h)
        return validity, cls_logits

# ----------------------------
# Training Function
# ----------------------------
def train_gan(trainfile, savepath, epochs=20):
    ds = CodeDataset(trainfile)
    train_size = int(0.9 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    vocab_size = len(ds.vocab)
    max_len = ds.max_len
    nclass = len(ds.mtypes)

    G = Generator(vocab_size, hidden=128, max_len=max_len).to(device)
    D = Discriminator(vocab_size, hidden=128, nclass=nclass, max_len=max_len).to(device)

    opt_G = optim.Adam(G.parameters(), lr=1e-3)
    opt_D = optim.Adam(D.parameters(), lr=1e-3)

    adversarial_loss = nn.BCELoss()
    classification_loss = nn.CrossEntropyLoss()

    val_accs = []
    train_losses = []

    for epoch in range(epochs):
        D.train(); G.train()
        tot_loss, n = 0, 0

        for real_seq, label in train_dl:
            real_seq, label = real_seq.to(device), label.to(device)
            batch_size = real_seq.size(0)

            # Train Generator
            opt_G.zero_grad()
            z = torch.randn(batch_size, 128, device=device)
            fake_seq = G(z)
            validity, _ = D(fake_seq.detach())
            g_loss = adversarial_loss(validity, torch.ones_like(validity))
            g_loss.backward()
            opt_G.step()

            # Train Discriminator
            opt_D.zero_grad()
            real_validity, cls_logits = D(real_seq)
            d_real_loss = adversarial_loss(real_validity, torch.ones_like(real_validity))
            d_cls_loss = classification_loss(cls_logits, label)
            fake_validity, _ = D(fake_seq.detach())
            d_fake_loss = adversarial_loss(fake_validity, torch.zeros_like(fake_validity))
            d_loss = d_real_loss + d_fake_loss + d_cls_loss
            d_loss.backward()
            opt_D.step()

            tot_loss += d_loss.item()
            n += 1

        train_losses.append(tot_loss/n)

        # Validation accuracy
        D.eval()
        acc_sum, m = 0, 0
        with torch.no_grad():
            for real_seq, label in val_dl:
                real_seq, label = real_seq.to(device), label.to(device)
                _, cls_logits = D(real_seq)
                acc_sum += (cls_logits.argmax(-1) == label).float().mean().item()
                m += 1
        val_accs.append(acc_sum/m)

        print(f"Epoch {epoch+1}/{epochs} | D Loss {train_losses[-1]:.4f} | Val Acc {val_accs[-1]:.4f}")

    # Save discriminator
    torch.save({"D": D.state_dict(), "vocab": ds.vocab, "mtypes": ds.mtypes}, savepath)

    # Plot metrics
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_accs, label="Val Accuracy")
    plt.legend()
    plt.savefig("gan_training_metrics.png")
    print("✅ Training complete. Discriminator (classifier) saved as", savepath)

if __name__ == "__main__":
    train_gan("python_deobf_dataset.jsonl", "gan_classifier.pt", epochs=20)
