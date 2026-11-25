#!/usr/bin/python3 -I
"""Flow-based anomaly scoring on the open NPU.

  ids.py features CAPTURE OUT.npz     packets -> one 32-feature record per flow per 5-second window
  ids.py train BENIGN.npz ATTACK.npz  train an autoencoder on benign flows (numpy, CPU); write model.bin, meta.json
  ids.py eval OUTPUTS.bin             score the accelerator's reconstructions and report detection quality
  ids.py report FLOWS.npz MODEL OUTPUTS.bin  list the most anomalous flows of a capture scored with the trained model
  ids.py export FLOWS.npz OUT.f32     raw feature rows for the Rust demonstrations (rust/src/bin/usecases)

Only packet headers are used: addresses, ports, sizes, times and TCP flags. Addresses never enter the features."""
import json, math, os, struct, subprocess, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW = 5.0
PORTS = [22, 53, 80, 123, 443, 5353]
NAMES = (['log_pkts_out', 'log_pkts_in', 'log_bytes_out', 'log_bytes_in', 'log_duration', 'mean_size_out', 'mean_size_in',
          'std_size_out', 'std_size_in', 'log_mean_gap', 'log_std_gap', 'bytes_out_share', 'pkts_out_share',
          'syn_share', 'synack_share', 'fin_share', 'rst_share', 'psh_share', 'is_tcp', 'is_udp', 'port_wellknown', 'port_registered']
         + [f'port_{p}' for p in PORTS]
         + ['log_src_flows', 'log_src_ports', 'log_src_hosts', 'src_unanswered_share'])
assert len(NAMES) == 32

def read_packets(path):
    fields = ['frame.time_epoch', 'ip.src', 'ipv6.src', 'ip.dst', 'ipv6.dst', 'ip.proto', 'ipv6.nxt',
              'tcp.srcport', 'tcp.dstport', 'udp.srcport', 'udp.dstport', 'frame.len', 'tcp.flags']
    cmd = ['tshark', '-r', path, '-T', 'fields', '-E', 'separator=\t', '-E', 'occurrence=f'] + [x for f in fields for x in ('-e', f)]
    for line in subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.splitlines():
        c = line.split('\t')
        if len(c) < 13: continue
        src, dst = c[1] or c[2], c[3] or c[4]
        proto = c[5] or c[6]
        if not src or not dst or proto not in ('6', '17'): continue
        sp, dp = (c[7], c[8]) if proto == '6' else (c[9], c[10])
        if not sp or not dp: continue
        yield float(c[0]), src, dst, int(proto), int(sp), int(dp), int(c[11]), int(c[12], 16) if c[12] else 0

def features(path):
    flows = {}
    for t, src, dst, proto, sp, dp, size, flags in read_packets(path):
        w = int(t // WINDOW)
        a, b = (src, sp), (dst, dp)
        key = (w, proto) + ((a, b) if a <= b else (b, a))
        f = flows.get(key)
        if f is None:   # the first packet seen in the window defines who is "out" (the initiator)
            f = flows[key] = dict(w=w, proto=proto, src=src, dst=dst, sp=sp, dp=dp, t=[], so=[], si=[], fl=defaultdict(int))
        out = (src, sp) == (f['src'], f['sp'])
        f['t'].append(t); (f['so'] if out else f['si']).append(size)
        if proto == 6:
            for name, bit, only in (('syn', 0x02, 0x12), ('synack', 0x12, 0x12), ('fin', 0x01, 0x01), ('rst', 0x04, 0x04), ('psh', 0x08, 0x08)):
                if flags & only == bit: f['fl'][name] += 1
    per_src = defaultdict(list)
    for f in flows.values(): per_src[(f['w'], f['src'])].append(f)
    X, meta = [], []
    lg = lambda v: math.log1p(v)
    for f in flows.values():
        so, si, t = np.array(f['so'] or [0.0]), np.array(f['si'] or [0.0]), np.sort(np.array(f['t']))
        n_out, n_in = len(f['so']), len(f['si']); n = n_out + n_in
        gaps = np.diff(t) if len(t) > 1 else np.array([0.0])
        peers = per_src[(f['w'], f['src'])]
        port = min(f['sp'], f['dp'])
        row = [lg(n_out), lg(n_in), lg(so.sum()), lg(si.sum()), lg(t[-1] - t[0]), so.mean() / 1500, si.mean() / 1500,
               so.std() / 1500, si.std() / 1500, lg(gaps.mean() * 1000), lg(gaps.std() * 1000),
               so.sum() / max(so.sum() + si.sum(), 1), n_out / n,
               f['fl']['syn'] / n, f['fl']['synack'] / n, f['fl']['fin'] / n, f['fl']['rst'] / n, f['fl']['psh'] / n,
               float(f['proto'] == 6), float(f['proto'] == 17), float(port < 1024), float(1024 <= port < 32768)]
        row += [float(port == p) for p in PORTS]
        row += [lg(len(peers)), lg(len({p['dp'] for p in peers})), lg(len({p['dst'] for p in peers})),
                sum(1 for p in peers if not p['si']) / len(peers)]
        X.append(row)
        meta.append(f"{f['src']}:{f['sp']} -> {f['dst']}:{f['dp']} {'tcp' if f['proto'] == 6 else 'udp'} window {f['w']} pkts {n_out}/{n_in}")
    return np.array(X, np.float32).reshape(-1, 32), meta

def write_model(path, W, B, X, y):
    sizes = [W[0].shape[1]] + [w.shape[0] for w in W]
    with open(path, 'wb') as f:
        f.write(b'NPU2' + struct.pack('<i', len(W)) + struct.pack(f'<{len(sizes)}i', *sizes) + struct.pack('<i', len(X)))
        for w, b in zip(W, B): f.write(w.astype('<f4').tobytes()); f.write(b.astype('<f4').tobytes())
        f.write(X.astype('<f4').tobytes()); f.write(np.asarray(y, np.uint8).tobytes())

def forward(W, B, X):
    acts = [X]
    for i in range(len(W)):
        z = acts[-1] @ W[i].T + B[i]
        acts.append(np.maximum(z, 0) if i + 1 < len(W) else z)
    return acts

def auc(scores, labels):
    order = np.argsort(scores); ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1; n1, n0 = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def cmd_features(cap, out):
    X, meta = features(cap)
    np.savez(out, X=X, meta=np.array(meta))
    print(f'{cap}: {len(X)} flow records with {X.shape[1]} features -> {out}')

def cmd_train(benign, attack):
    Xb, Xa = np.load(benign)['X'], np.load(attack)['X']
    rng = np.random.default_rng(0); idx = rng.permutation(len(Xb))
    n_tr, n_va = int(0.6 * len(Xb)), int(0.2 * len(Xb))
    tr, va, te = Xb[idx[:n_tr]], Xb[idx[n_tr:n_tr + n_va]], Xb[idx[n_tr + n_va:]]
    mu, sd = tr.mean(0), tr.std(0) + 1e-3
    norm = lambda X: np.clip((X - mu) / sd, -8, 8).astype(np.float32)
    tr, va, te, at = norm(tr), norm(va), norm(te), norm(Xa)
    sizes = [32, 32, 8, 32, 32]
    W = [(rng.standard_normal((sizes[i + 1], sizes[i])) * np.sqrt(2 / sizes[i])).astype(np.float32) for i in range(4)]
    B = [np.zeros(sizes[i + 1], np.float32) for i in range(4)]
    params = [p for wb in zip(W, B) for p in wb]; m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    step = 0
    for epoch in range(400):
        perm = rng.permutation(len(tr))
        for k in range(0, len(tr), 32):
            x = tr[perm[k:k + 32]]; x_in = x + rng.normal(0, 0.1, x.shape).astype(np.float32)   # denoising
            acts = forward(W, B, x_in); delta = 2 * (acts[-1] - x) / x.size; grads = []
            for i in reversed(range(4)):
                grads[:0] = [delta.T @ acts[i], delta.sum(0)]
                if i: delta = (delta @ W[i]) * (acts[i] > 0)
            step += 1
            for p, g, mm, vv in zip(params, grads, m, v):
                g = g + 1e-5 * p; mm[:] = 0.9 * mm + 0.1 * g; vv[:] = 0.999 * vv + 0.001 * g * g
                p -= 2e-3 * (mm / (1 - 0.9 ** step)) / (np.sqrt(vv / (1 - 0.999 ** step)) + 1e-8)
    score = lambda X: ((forward(W, B, X)[-1] - X) ** 2).mean(1)
    thr = float(np.quantile(score(va), 0.99))
    Xe = np.vstack([te, at]); ye = np.r_[np.zeros(len(te)), np.ones(len(at))].astype(np.uint8)
    write_model(f'{HERE}/model.bin', W, B, Xe, ye)
    np.savez(f'{HERE}/model.npz', mu=mu, sd=sd, **{f'W{i}': W[i] for i in range(4)}, **{f'B{i}': B[i] for i in range(4)})
    json.dump(dict(threshold=thr, n_benign_test=int(len(te)), n_attack=int(len(at)), n_train=int(len(tr)), n_val=int(len(va))), open(f'{HERE}/meta.json', 'w'))
    print(f'trained on {len(tr)} benign records; validation {len(va)}, benign test {len(te)}, attack {len(at)}')
    print(f'reconstruction error: train {score(tr).mean():.4f}, benign test {score(te).mean():.4f}, attack {score(at).mean():.4f}; threshold (99th percentile of validation) {thr:.4f}')

def read_outputs(path):
    with open(path, 'rb') as f:
        n, no = struct.unpack('<2i', f.read(8))
        npu = np.frombuffer(f.read(4 * n * no), '<f4').reshape(n, no); cpu = np.frombuffer(f.read(4 * n * no), '<f4').reshape(n, no)
    return npu, cpu

def model_inputs(path):
    with open(path, 'rb') as f:
        assert f.read(4) == b'NPU2'; nl, = struct.unpack('<i', f.read(4)); sz = struct.unpack(f'<{nl + 1}i', f.read(4 * (nl + 1))); nt, = struct.unpack('<i', f.read(4))
        for l in range(nl): f.read(4 * sz[l] * sz[l + 1] + 4 * sz[l + 1])
        X = np.frombuffer(f.read(4 * nt * sz[0]), '<f4').reshape(nt, sz[0]); y = np.frombuffer(f.read(nt), np.uint8)
    return X, y

def cmd_eval(outputs):
    X, y = model_inputs(f'{HERE}/model.bin'); meta = json.load(open(f'{HERE}/meta.json')); thr = meta['threshold']
    npu, cpu = read_outputs(outputs)
    s_npu, s_cpu = ((npu - X) ** 2).mean(1), ((cpu - X) ** 2).mean(1)
    for name, s in (('accelerator', s_npu), ('CPU', s_cpu)):
        fp = (s[y == 0] > thr).mean(); tp = (s[y == 1] > thr).mean()
        print(f'{name:11s}: AUC {auc(s, y):.4f}  detected {100 * tp:.1f}% of {int((y == 1).sum())} attack records  false alarms {100 * fp:.1f}% of {int((y == 0).sum())} benign records')
    print(f'same alarm decision on {int(((s_npu > thr) == (s_cpu > thr)).sum())} of {len(y)} records; largest score difference {np.abs(s_npu - s_cpu).max():.4f} (threshold {thr:.4f})')

def cmd_export(flows, out):
    """Write a capture's raw flow features as u32 rows, u32 width, float32 data (for the Rust demonstrations)."""
    X = np.load(flows)['X']
    with open(out, 'wb') as f:
        f.write(struct.pack('<2I', *X.shape)); f.write(X.astype('<f4').tobytes())
    print(f'{len(X)} flow records -> {out}')

def cmd_prepare(flows, out):
    """Normalise a capture's flow records with the trained model's statistics and write a model file to score them."""
    d = np.load(flows); mdl = np.load(f'{HERE}/model.npz')
    X = np.clip((d['X'] - mdl['mu']) / mdl['sd'], -8, 8).astype(np.float32)
    write_model(out, [mdl[f'W{i}'] for i in range(4)], [mdl[f'B{i}'] for i in range(4)], X, np.zeros(len(X)))
    print(f'{len(X)} flow records ready in {out}')

def cmd_report(flows, model, outputs):
    d = np.load(flows); X, _ = model_inputs(model); npu, _ = read_outputs(outputs)
    thr = json.load(open(f'{HERE}/meta.json'))['threshold']; s = ((npu - X) ** 2).mean(1)
    print(f'{len(s)} flow records scored on the accelerator; {int((s > thr).sum())} above the alarm threshold {thr:.3f}')
    for i in np.argsort(-s)[:12]:
        worst = np.argsort(-((npu[i] - X[i]) ** 2))[:2]
        print(f"  {s[i]:8.2f} {'ALARM' if s[i] > thr else '     '} {d['meta'][i]}   (most unusual: {', '.join(NAMES[j] for j in worst)})")

if __name__ == '__main__':
    cmds = dict(features=cmd_features, train=cmd_train, eval=cmd_eval, prepare=cmd_prepare, report=cmd_report, export=cmd_export)
    if len(sys.argv) < 2 or sys.argv[1] not in cmds: sys.exit(__doc__)
    cmds[sys.argv[1]](*sys.argv[2:])
