#!/usr/bin/python3
"""Render a digit dataset from the fonts installed on this machine, train a 256-32-10 network with numpy
on the CPU, and write model.bin for nn_client (weights, biases and a held-out test set)."""
import random, struct, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

S, HID, N_TRAIN, N_TEST = 16, 32, 16000, 1024
random.seed(0); np.random.seed(0)

paths = sorted(set(l.split(':')[0] for l in subprocess.run(['fc-list', ':', 'file'], capture_output=True, text=True).stdout.splitlines()
                   if l.split(':')[0].lower().endswith(('.ttf', '.otf'))))
def glyph(font, ch):
    im = Image.new('L', (64, 64), 0); ImageDraw.Draw(im).text((32, 32), ch, fill=255, font=font, anchor='mm'); return im
fonts = []
for p in paths:
    try:
        f = ImageFont.truetype(p, 40)
        g = [np.asarray(glyph(f, str(d)), dtype=np.float32) for d in range(10)]
        if min(x.sum() for x in g) < 255 * 40: continue                      # missing or empty digit glyphs
        if min(np.abs(g[a] - g[b]).sum() for a in range(10) for b in range(a)) < 255 * 30: continue  # tofu boxes / symbol fonts
        fonts.append(p)
    except Exception:
        pass
random.shuffle(fonts)
n_hold = max(4, len(fonts) // 5)
test_fonts, train_fonts = fonts[:n_hold], fonts[n_hold:]
print(f'{len(fonts)} usable fonts: {len(train_fonts)} for training, {len(test_fonts)} held out for testing')

def sample(font_paths):
    d = random.randrange(10)
    f = ImageFont.truetype(random.choice(font_paths), random.randint(30, 44))
    im = glyph(f, str(d)).rotate(random.uniform(-12, 12), resample=Image.BILINEAR)
    box = im.getbbox()
    im = im.crop(box)
    side = random.randint(11, 14)
    w, h = im.size; sc = side / max(w, h)
    im = im.resize((max(1, round(w * sc)), max(1, round(h * sc))), Image.BILINEAR)
    out = Image.new('L', (S, S), 0)
    out.paste(im, (random.randint(0, S - im.size[0]), random.randint(0, S - im.size[1])))
    x = np.asarray(out, dtype=np.float32) / 255.0
    x = np.clip(x * random.uniform(0.7, 1.0) + np.random.normal(0, 0.05, x.shape), 0, 1)
    return x.reshape(-1).astype(np.float32), d

def make(n, fp):
    X = np.zeros((n, S * S), np.float32); y = np.zeros(n, np.int64)
    for i in range(n): X[i], y[i] = sample(fp)
    return X, y
Xtr, ytr = make(N_TRAIN, train_fonts); Xte, yte = make(N_TEST, test_fonts)

W1 = (np.random.randn(HID, S * S) * np.sqrt(2 / (S * S))).astype(np.float32); b1 = np.zeros(HID, np.float32)
W2 = (np.random.randn(10, HID) * np.sqrt(2 / HID)).astype(np.float32); b2 = np.zeros(10, np.float32)
params = [W1, b1, W2, b2]; m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
def forward(X): H = np.maximum(X @ W1.T + b1, 0); return H, H @ W2.T + b2
step = 0
for epoch in range(40):
    perm = np.random.permutation(N_TRAIN)
    for k in range(0, N_TRAIN, 64):
        idx = perm[k:k + 64]; X = Xtr[idx]; H, L = forward(X)
        P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True); P[np.arange(len(idx)), ytr[idx]] -= 1; P /= len(idx)
        gW2 = P.T @ H; gb2 = P.sum(0); dH = (P @ W2) * (H > 0); gW1 = dH.T @ X; gb1 = dH.sum(0)
        step += 1
        for p, g, mm, vv in zip(params, [gW1, gb1, gW2, gb2], m, v):
            g = g + 1e-4 * p; mm[:] = 0.9 * mm + 0.1 * g; vv[:] = 0.999 * vv + 0.001 * g * g
            p -= 2e-3 * (mm / (1 - 0.9 ** step)) / (np.sqrt(vv / (1 - 0.999 ** step)) + 1e-8)
    if epoch % 10 == 9: print(f'epoch {epoch + 1}: train acc {(forward(Xtr)[1].argmax(1) == ytr).mean():.4f}, test acc {(forward(Xte)[1].argmax(1) == yte).mean():.4f}')
print('max |W1| %.2f, max |W2| %.2f' % (abs(W1).max(), abs(W2).max()))
with open('model.bin', 'wb') as f:
    f.write(struct.pack('<4i', S * S, HID, 10, N_TEST))
    for a in (W1, b1, W2, b2, Xte): f.write(a.astype('<f4').tobytes())
    f.write(yte.astype(np.uint8).tobytes())
Image.fromarray((np.hstack([Xte[i].reshape(S, S) for i in range(16)]) * 255).astype(np.uint8)).resize((16 * 64, 64), Image.NEAREST).save('samples.png')
print('wrote model.bin and samples.png; first 16 test labels:', ''.join(str(int(d)) for d in yte[:16]))
