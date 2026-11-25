#!/usr/bin/python3 -I
"""Convert downloaded real datasets into the row files the Rust demonstrations read
(u32 rows, u32 width, float32 data; labels as one byte per row in a .u8 file next to it).

  prep_real.py nslkdd DIR     KDDTrain+.txt / KDDTest+.txt -> nsl-train.f32/.u8, nsl-test.f32/.u8 (96 features)
  prep_real.py mnist DIR OUT [train|t10k] [limit]  images -> OUT (rows x 784) with labels
  prep_real.py ping FILE OUT  `ping -D` output -> OUT (n x 1, round-trip milliseconds; lost replies repeat the last value)
  prep_real.py wifi FILE OUT  repeated `iw station dump` output -> OUT (n x 6: signal, signal avg, tx bitrate, rx bitrate, tx retries/s, tx packets/s)
"""
import gzip, re, struct, sys
import numpy as np

def save(path, X, y=None):
    X = np.asarray(X, np.float32)
    with open(path, 'wb') as f:
        f.write(struct.pack('<2I', *X.shape)); f.write(X.astype('<f4').tobytes())
    if y is not None: open(path[:-4] + '.u8', 'wb').write(np.asarray(y, np.uint8).tobytes())
    print(f'{path}: {X.shape[0]} rows x {X.shape[1]}')

def nslkdd(d):
    rows = {k: [l.rstrip('\n').split(',') for l in open(f'{d}/KDD{k}+.txt')] for k in ('Train', 'Test')}
    tr = rows['Train']
    protos = sorted({r[1] for r in tr}); flags = sorted({r[3] for r in tr})
    svc_count = {}
    for r in tr: svc_count[r[2]] = svc_count.get(r[2], 0) + 1
    services = [s for s, _ in sorted(svc_count.items(), key=lambda kv: -kv[1])[:20]]
    logcols = {0, 4, 5, 22, 23, 31, 32}          # duration, src_bytes, dst_bytes, count, srv_count, dst_host_count, dst_host_srv_count
    # attack families of NSL-KDD (public documentation of the dataset): 1 DoS, 2 probe, 3 R2L, 4 U2R
    fam = {**{a: 1 for a in 'back land neptune pod smurf teardrop apache2 udpstorm processtable mailbomb worm'.split()},
           **{a: 2 for a in 'satan ipsweep nmap portsweep mscan saint'.split()},
           **{a: 3 for a in 'guess_passwd ftp_write imap phf multihop warezmaster warezclient spy xlock xsnoop snmpguess snmpgetattack httptunnel sendmail named'.split()},
           **{a: 4 for a in 'buffer_overflow loadmodule rootkit perl sqlattack xterm ps'.split()}}
    def enc(r):
        num = [float(r[i]) for i in range(41) if i not in (1, 2, 3)]
        idx = [i for i in range(41) if i not in (1, 2, 3)]
        num = [np.log1p(v) if i in logcols else v for i, v in zip(idx, num)]
        v = num + [float(r[1] == p) for p in protos] + [float(r[3] == f) for f in flags] + [float(r[2] == s) for s in services]
        return v + [0.0] * (96 - len(v))
    for k in ('Train', 'Test'):
        X = np.array([enc(r) for r in rows[k]], np.float32)
        y = [0 if r[41] == 'normal' else fam.get(r[41], 5) for r in rows[k]]
        save(f'{d}/nsl-{k.lower()}.f32', X, y)
    print('feature count before padding:', 38 + len(protos) + len(flags) + len(services))

def mnist(d, out, which='t10k', limit='0'):
    with gzip.open(f'{d}/{which}-images-idx3-ubyte.gz') as f:
        _, n, r, c = struct.unpack('>4i', f.read(16)); n = int(limit) or n; X = np.frombuffer(f.read(n * r * c), np.uint8).reshape(n, r * c) / 255.0
    with gzip.open(f'{d}/{which}-labels-idx1-ubyte.gz') as f:
        f.read(8); y = np.frombuffer(f.read(n), np.uint8)
    save(out, X, y)

def ping(path, out):
    last, series, expect = 0.0, [], None
    for l in open(path, errors='replace'):
        m = re.search(r'icmp_seq=(\d+).*time=([\d.]+) ms', l)
        if not m: continue
        seq, t = int(m.group(1)), float(m.group(2))
        if expect is not None:
            series += [last] * max(0, seq - expect)     # lost replies
        series.append(t); last = t; expect = seq + 1
    save(out, np.array(series, np.float32).reshape(-1, 1))

def wifi(path, out):
    recs, cur, t = [], {}, None
    for l in open(path, errors='replace'):
        l = l.strip()
        if re.fullmatch(r'\d+\.\d+', l):
            if t is not None and cur: recs.append((t, dict(cur)))
            t, cur = float(l), {}
        else:
            m = re.match(r'(signal|signal avg|tx bitrate|rx bitrate|tx retries|tx packets):\s+(-?[\d.]+)', l)
            if m: cur[m.group(1)] = float(m.group(2))
    rows = []
    for (t0, a), (t1, b) in zip(recs, recs[1:]):
        if all(k in b for k in ('signal', 'signal avg', 'tx bitrate', 'rx bitrate', 'tx retries', 'tx packets')) and t1 > t0:
            rows.append([b['signal'], b['signal avg'], b['tx bitrate'], b['rx bitrate'],
                         (b['tx retries'] - a.get('tx retries', b['tx retries'])) / (t1 - t0), (b['tx packets'] - a.get('tx packets', b['tx packets'])) / (t1 - t0)])
    save(out, np.array(rows, np.float32).reshape(-1, 6))

if __name__ == '__main__':
    cmds = dict(nslkdd=nslkdd, mnist=mnist, ping=ping, wifi=wifi)
    if len(sys.argv) < 2 or sys.argv[1] not in cmds: sys.exit(__doc__)
    cmds[sys.argv[1]](*sys.argv[2:])
