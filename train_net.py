"""
train_net.py -- train vitals_net (see vitals_net.py) on MCD-rPPG features. PyTorch, CPU is enough.

    python build_mcd_dataset.py --out mcd_features            # per-frame features + PPG for ~540 people
    python train_net.py --data mcd_features --epochs 40       # -> vitals_net_weights.npz

Training uses only the 'train' people, and model selection only the 'validation'
people (splits.json). The 'test' people are never read here.

What the network learns from each example: a window of 6-15 s of the colour
traces the app measures, and the finger PPG over the same window. Targets:
  * heart rate: the PPG's spectral peak, as a probability spread over nearby
    1-BPM bins (a Gaussian, sigma 1.5 BPM);
  * waveform: the band-passed PPG itself (negative Pearson correlation, a
    small time shift allowed because the finger and the face pulse at
    slightly different moments).
Augmentation, so it does not just memorise this dataset:
  * time-stretch the window (input and PPG together): heart rates from ~45 to
    ~170 BPM even though most people here were between 60 and 120;
  * camera noise, coloured light flicker that also shows on the wall (it has
    to learn that what the wall does is not a pulse), and dropped patches.
Validation runs every window through vitals_net.prepare, the same path the
app uses, and compares with the current hand-made pipeline (POS) on the same
windows.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import vitals_net as vn
from build_mcd_dataset import load_features
from evaluate_video import read_ground_truth, reference_bpm
from vitals_dsp import FS, HR_HIGH, HR_LOW, analyze_heart_rate, bandpass, pick_peak, power_spectrum, snr_db

torch.set_num_threads(max(1, os.cpu_count() - 1))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_recording(folder):
    """Features + PPG on an even 30 Hz grid. Missing samples are interpolated; `valid` remembers where."""
    f = load_features(folder)
    t_gt, ppg, _ = read_ground_truth(os.path.join(folder, "ground_truth.txt"))
    t = f["t"]
    tu = np.arange(t[0], min(t[-1], t_gt[-1]), 1.0 / FS)
    rgb = np.full((vn.N_TRACES, len(tu), 3), np.nan, np.float32)
    valid = np.zeros((vn.N_TRACES, len(tu)), bool)
    for i, name in enumerate(vn.TRACES):
        raw = f["background"] if name == "background" else f["patches"].get(name)
        if raw is None:
            continue
        ok = ~np.isnan(raw).any(axis=1)
        if name != "background":
            ok &= f["face"]
        if ok.sum() < 100:
            continue
        for c in range(3):
            rgb[i, :, c] = np.interp(tu, t[ok], raw[ok, c])
        valid[i] = np.interp(tu, t, ok.astype(float)) > 0.5
    return {"name": os.path.basename(folder), "rgb": rgb, "valid": valid,
            "ppg": np.interp(tu, t_gt, ppg).astype(np.float32)}


def ppg_label(seg):
    """PPG segment -> (heart rate in BPM, signal quality in dB)."""
    b = bandpass(seg - seg.mean(), HR_LOW, HR_HIGH)
    f, p = power_spectrum(b)
    f0 = pick_peak(f, p, HR_LOW, HR_HIGH)
    return 60.0 * f0, snr_db(f, p, f0, HR_LOW, HR_HIGH), b


def make_example(rec, L, rng, augment=True):
    """One training window of L samples, or None if the reference is unusable."""
    n = rec["ppg"].shape[0]
    s = rng.uniform(0.75, 1.35) if augment else 1.0          # time stretch: heart rate x s
    span = int(round(L * s))
    if span + 2 >= n:
        return None
    start = int(rng.integers(0, n - span - 1))
    pos = start + np.arange(L) * (span / L)
    i0 = np.floor(pos).astype(int)
    frac = pos - i0

    def seg(a):                                              # fractional resampling along time
        f = frac if a.ndim == 1 else frac[:, None]
        return a[i0] * (1 - f) + a[i0 + 1] * f

    bpm, quality, wave = ppg_label(seg(rec["ppg"].astype(float)))
    if quality < 0.0 or not 42.0 <= bpm <= 178.0:
        return None
    tt = np.arange(L) / FS
    flicker = None
    if augment and rng.random() < 0.5:                       # coloured light flicker, wall included
        f = rng.uniform(0.7, 3.5)
        amp = rng.uniform(0.0, 0.004)
        color = rng.uniform(-0.5, 1.0, 3)
        flicker = 1.0 + amp * np.sin(2 * np.pi * f * tt + rng.uniform(0, 6.3))[:, None] * color
    noise = rng.uniform(0.0, 0.8) if augment else 0.0
    rgb = []
    for i in range(vn.N_TRACES):
        if rec["valid"][i, start:start + span + 1].mean() < 0.8 or np.isnan(rec["rgb"][i, 0, 0]):
            rgb.append(None)
            continue
        if augment and 0 < i < vn.N_TRACES - 1 and rng.random() < 0.1:   # drop a patch now and then
            rgb.append(None)
            continue
        x = seg(rec["rgb"][i].astype(float))
        if flicker is not None:
            x = x * flicker
        if noise:
            x = x + rng.normal(0, noise, x.shape)
        rgb.append(x)
    if rgb[0] is None:
        return None
    return vn.window_input(rgb), bpm, (wave / (wave.std() + 1e-9)).astype(np.float32)


class Windows(torch.utils.data.IterableDataset):
    """Endless batches; every batch has one random window length (6-15 s)."""

    def __init__(self, recs, batch, seed):
        self.recs, self.batch, self.seed = recs, batch, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0) + int(time.time()))
        while True:
            L = int(rng.integers(int(vn.MIN_WINDOW_SEC * FS), int(15 * FS) + 1))
            xs, bpms, waves = [], [], []
            while len(xs) < self.batch:
                ex = make_example(self.recs[rng.integers(len(self.recs))], L, rng)
                if ex is not None:
                    xs.append(ex[0])
                    bpms.append(ex[1])
                    waves.append(ex[2])
            yield torch.from_numpy(np.stack(xs)), torch.tensor(bpms, dtype=torch.float32), torch.from_numpy(np.stack(waves))


# --------------------------------------------------------------------------
# Network (mirrors vitals_net.PulseNet)
# --------------------------------------------------------------------------

class Net(nn.Module):
    def __init__(self, hidden=32, dilations=(1, 2, 4, 8, 16), spec_hidden=16):
        super().__init__()
        self.dilations = list(dilations)
        self.mix = nn.Conv1d(vn.N_IN, hidden, 1)
        self.mix_bn = nn.BatchNorm1d(hidden)
        self.c1 = nn.ModuleList([nn.Conv1d(hidden, hidden, 5, dilation=d, padding=2 * d) for d in dilations])
        self.bn = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in dilations])
        self.c2 = nn.ModuleList([nn.Conv1d(hidden, hidden, 1) for _ in dilations])
        self.wave = nn.Conv1d(hidden, 1, 1)
        self.spec1 = nn.Conv1d(hidden, spec_hidden, 1)
        self.spec2 = nn.Conv1d(spec_hidden, 1, 5, padding=2)
        self.register_buffer("bins", torch.tensor(vn._BINS, dtype=torch.float32))

    def forward(self, x):
        h = F.relu(self.mix_bn(self.mix(x)))
        for c1, bn, c2 in zip(self.c1, self.bn, self.c2):
            h = h + c2(F.relu(bn(c1(h))))
        wave = self.wave(h)[:, 0]
        L = h.shape[-1]
        win = torch.hann_window(L, periodic=False, dtype=h.dtype)
        power = torch.fft.rfft(h * win, n=vn.NFFT).abs() ** 2
        binned = power @ self.bins
        s = torch.log1p(binned / (binned.mean(-1, keepdim=True) + 1e-9))
        logits = self.spec2(F.relu(self.spec1(s)))[:, 0]
        return wave, logits


def export(net, path, meta):
    """Fold BatchNorm into the convolutions and save the numpy weights."""
    net = net.eval()

    def fold(conv, bn):
        g = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        return (conv.weight * g[:, None, None]).detach().numpy(), ((conv.bias - bn.running_mean) * g + bn.bias).detach().numpy()

    w = {}
    w["mix.w"], w["mix.b"] = fold(net.mix, net.mix_bn)
    for i, (c1, bn, c2) in enumerate(zip(net.c1, net.bn, net.c2)):
        w[f"block{i}.w"], w[f"block{i}.b"] = fold(c1, bn)
        w[f"block{i}.w2"], w[f"block{i}.b2"] = c2.weight.detach().numpy(), c2.bias.detach().numpy()
    for name, layer in (("wave", net.wave), ("spec.w1", net.spec1), ("spec.w2", net.spec2)):
        key_w, key_b = (f"{name}.w", f"{name}.b") if name == "wave" else (name, name.replace("w", "b"))
        w[key_w], w[key_b] = layer.weight.detach().numpy(), layer.bias.detach().numpy()
    w = {k: v.astype(np.float32) for k, v in w.items()}
    np.savez_compressed(path, dilations=np.array(net.dilations), meta=np.array(json.dumps(meta)), **w)


def losses(net, x, bpm, wave_t, sigma=1.5):
    wave, logits = net(x)
    bins = torch.tensor(vn.BPM_BINS, dtype=torch.float32)
    target = torch.exp(-0.5 * ((bins[None] - bpm[:, None]) / sigma) ** 2)
    target = target / target.sum(-1, keepdim=True)
    ce = -(target * F.log_softmax(logits, -1)).sum(-1).mean()
    w = (wave - wave.mean(-1, keepdim=True)) / (wave.std(-1, keepdim=True) + 1e-6)
    best = None
    for shift in range(-6, 7, 2):                              # +-0.2 s
        c = (torch.roll(w, shift, -1) * wave_t).mean(-1)
        best = c if best is None else torch.maximum(best, c)
    return ce, (1 - best).mean()


def logits_to_bpm(logits):
    p = torch.softmax(logits, -1)
    i = p.argmax(-1).clamp(1, len(vn.BPM_BINS) - 2)
    lp = torch.log(p + 1e-12)
    a, b, c = lp.gather(1, (i - 1)[:, None])[:, 0], lp.gather(1, i[:, None])[:, 0], lp.gather(1, (i + 1)[:, None])[:, 0]
    denom = a - 2 * b + c
    off = torch.where(denom < 0, 0.5 * (a - c) / denom, torch.zeros_like(denom))
    bpm = torch.tensor(vn.BPM_BINS, dtype=torch.float32)[i] + off
    conf = (p * ((torch.tensor(vn.BPM_BINS)[None] - bpm[:, None]).abs() <= 5)).sum(-1)
    return bpm, conf


# --------------------------------------------------------------------------
# Validation: the app's own input path, compared with the current pipeline
# --------------------------------------------------------------------------

def validation_windows(folders, window_sec, step=3.0):
    """[(recording, end, prepared input, reference, POS estimate)] for every window."""
    out = []
    for folder in folders:
        f = load_features(folder)
        t_gt, ppg, _ = read_ground_truth(os.path.join(folder, "ground_truth.txt"))
        t = f["t"]
        for end in np.arange(15.0, t[-1], step):
            sel = (t > end - window_sec) & (t <= end) & f["face"]
            ref = reference_bpm(t_gt, ppg, end)
            if ref is None or sel.sum() < window_sec * 10:
                continue
            tr = {n: v[sel] for n, v in f["patches"].items()}
            x = vn.prepare(t[sel], tr, f["background"][sel])
            pos = analyze_heart_rate(t[sel], tr, f["background"][sel], patch_map=False)
            if x is not None and pos is not None:
                out.append((os.path.basename(folder), end, x, ref, pos["bpm"]))
    return out


def evaluate(net, windows):
    net.eval()
    est, conf = [], []
    with torch.no_grad():
        by_len = {}
        for k, w in enumerate(windows):
            by_len.setdefault(w[2].shape[1], []).append(k)
        res = {}
        for ks in by_len.values():
            for j in range(0, len(ks), 256):
                part = ks[j:j + 256]
                _, logits = net(torch.from_numpy(np.stack([windows[k][2] for k in part])))
                b, c = logits_to_bpm(logits)
                for k, bb, cc in zip(part, b.tolist(), c.tolist()):
                    res[k] = (bb, cc)
    est = np.array([res[k][0] for k in range(len(windows))])
    conf = np.array([res[k][1] for k in range(len(windows))])
    ref = np.array([w[3] for w in windows])
    pos = np.array([w[4] for w in windows])
    e, ep = np.abs(est - ref), np.abs(pos - ref)
    return {"net_MAE": float(e.mean()), "net_within5": float((e <= 5).mean()),
            "pos_MAE": float(ep.mean()), "pos_within5": float((ep <= 5).mean()),
            "conf_auc_proxy": float(np.corrcoef(conf, (e <= 5).astype(float))[0, 1]),
            "est": est.tolist(), "conf": conf.tolist(), "ref": ref.tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="mcd_features")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=250, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wave-weight", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=0, help="use at most this many training recordings (smoke test)")
    ap.add_argument("--train-split", default="train", help="split to train on ('validation' only for a smoke test)")
    ap.add_argument("--out", default="vitals_net_weights.npz")
    args = ap.parse_args()

    with open(os.path.join(args.data, "splits.json")) as f:
        splits = json.load(f)

    def folders(split):
        return [os.path.join(args.data, f"{p}_{s}") for p in splits[split] for s in ("before", "after")
                if os.path.exists(os.path.join(args.data, f"{p}_{s}", "features.npz"))]

    train_f, val_f = folders(args.train_split), folders("validation")
    if args.max_train:
        train_f = train_f[:args.max_train]
    print(f"train recordings {len(train_f)}, validation recordings {len(val_f)}", flush=True)
    t0 = time.time()
    recs = [load_recording(f) for f in train_f]
    val15 = validation_windows(val_f, 15.0)
    val8 = validation_windows(val_f, 8.0)
    print(f"data ready in {time.time() - t0:.0f} s: {len(val15)} + {len(val8)} validation windows", flush=True)

    net = Net(hidden=args.hidden)
    print(f"parameters: {sum(p.numel() for p in net.parameters())}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * args.steps)
    loader = torch.utils.data.DataLoader(Windows(recs, args.batch, seed=0), batch_size=None,
                                         num_workers=args.workers, persistent_workers=args.workers > 0)
    best, history = None, []
    it = iter(loader)
    for epoch in range(args.epochs):
        net.train()
        tl, te = [], time.time()
        for _ in range(args.steps):
            x, bpm, wave_t = next(it)
            ce, pear = losses(net, x, bpm, wave_t)
            loss = ce + args.wave_weight * pear
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            tl.append((ce.item(), pear.item()))
        m15, m8 = evaluate(net, val15), evaluate(net, val8)
        tl = np.mean(tl, axis=0)
        row = {"epoch": epoch + 1, "ce": float(tl[0]), "pearson_loss": float(tl[1]),
               **{f"v15_{k}": v for k, v in m15.items() if not isinstance(v, list)},
               **{f"v8_{k}": v for k, v in m8.items() if not isinstance(v, list)}}
        history.append(row)
        print(f"epoch {epoch + 1:2d} ({time.time() - te:.0f} s) loss ce {tl[0]:.3f} wave {tl[1]:.3f} | "
              f"15 s windows: net MAE {m15['net_MAE']:.2f} within5 {m15['net_within5']:.0%} "
              f"(POS {m15['pos_MAE']:.2f} / {m15['pos_within5']:.0%}) | 8 s: net {m8['net_MAE']:.2f} "
              f"{m8['net_within5']:.0%} (POS {m8['pos_MAE']:.2f} / {m8['pos_within5']:.0%}) | "
              f"conf~correct r {m15['conf_auc_proxy']:.2f}",
              flush=True)
        score = m15["net_MAE"] + m8["net_MAE"]
        if best is None or score < best:
            best = score
            meta = {"epoch": epoch + 1, "args": vars(args), "validation": row,
                    "train_recordings": len(train_f), "validation_recordings": len(val_f)}
            export(net, args.out, meta)
            torch.save(net.state_dict(), args.out.replace(".npz", ".pt"))
    with open(args.out.replace(".npz", "_history.json"), "w") as f:
        json.dump(history, f, indent=1)
    print(f"best validation score {best:.2f}; saved {args.out}")


if __name__ == "__main__":
    main()
