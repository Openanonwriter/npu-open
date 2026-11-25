#!/usr/bin/python3 -I
"""Train a fully connected MNIST classifier with numpy on the CPU and write a model file for npu_client.
usage: train_mnist.py MNIST_DIR OUT.bin [hidden sizes...]   (default hidden: 128)"""
import gzip, struct, sys
import numpy as np

d, out = sys.argv[1], sys.argv[2]
hidden = [int(x) for x in sys.argv[3:]] or [128]
def images(name):
    with gzip.open(f'{d}/{name}') as f:
        magic, n, r, c = struct.unpack('>4i', f.read(16)); assert magic == 2051
        return np.frombuffer(f.read(n * r * c), np.uint8).reshape(n, r * c).astype(np.float32) / 255.0
def labels(name):
    with gzip.open(f'{d}/{name}') as f:
        magic, n = struct.unpack('>2i', f.read(8)); assert magic == 2049
        return np.frombuffer(f.read(n), np.uint8).astype(np.int64)
Xtr, ytr = images('train-images-idx3-ubyte.gz'), labels('train-labels-idx1-ubyte.gz')
Xte, yte = images('t10k-images-idx3-ubyte.gz'), labels('t10k-labels-idx1-ubyte.gz')
sizes = [Xtr.shape[1]] + hidden + [10]
np.random.seed(0)
W = [(np.random.randn(sizes[i + 1], sizes[i]) * np.sqrt(2 / sizes[i])).astype(np.float32) for i in range(len(sizes) - 1)]
B = [np.zeros(sizes[i + 1], np.float32) for i in range(len(sizes) - 1)]
params = [p for wb in zip(W, B) for p in wb]; m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
def forward(X):
    acts = [X]
    for i in range(len(W)):
        z = acts[-1] @ W[i].T + B[i]
        acts.append(np.maximum(z, 0) if i + 1 < len(W) else z)
    return acts
step = 0
for epoch in range(8):
    perm = np.random.permutation(len(Xtr))
    for k in range(0, len(Xtr), 64):
        idx = perm[k:k + 64]; acts = forward(Xtr[idx])
        L = acts[-1]; P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True); P[np.arange(len(idx)), ytr[idx]] -= 1; P /= len(idx)
        grads = []; delta = P
        for i in reversed(range(len(W))):
            grads[:0] = [delta.T @ acts[i], delta.sum(0)]
            if i: delta = (delta @ W[i]) * (acts[i] > 0)
        step += 1
        for p, g, mm, vv in zip(params, grads, m, v):
            mm[:] = 0.9 * mm + 0.1 * g; vv[:] = 0.999 * vv + 0.001 * g * g
            p -= 1e-3 * (mm / (1 - 0.9 ** step)) / (np.sqrt(vv / (1 - 0.999 ** step)) + 1e-8)
    print(f'epoch {epoch + 1}: test accuracy {(forward(Xte)[-1].argmax(1) == yte).mean():.4f}', flush=True)
print('layer sizes', sizes, 'largest weight %.2f' % max(abs(w).max() for w in W),
      'largest activation %.1f' % max(abs(a).max() for a in forward(Xte)[1:]))
with open(out, 'wb') as f:
    f.write(b'NPU2' + struct.pack('<i', len(W)) + struct.pack(f'<{len(sizes)}i', *sizes) + struct.pack('<i', len(Xte)))
    for w, b in zip(W, B): f.write(w.astype('<f4').tobytes()); f.write(b.astype('<f4').tobytes())
    f.write(Xte.astype('<f4').tobytes()); f.write(yte.astype(np.uint8).tobytes())
print('wrote', out)
