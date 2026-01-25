import numpy as np
from copy import deepcopy
import time
import gc
import warnings
from scipy.stats import spearmanr, rankdata
import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================================================
# Chatterjee correlation (your style, with safety for n<2)
# =========================================================
def chatterjee_correlation(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if n < 2:
        return np.float64(0.0)

    sort_idx = np.argsort(x)
    y_sorted = y[sort_idx]
    r = rankdata(y_sorted, method="ordinal")

    diff_sum = np.sum(np.abs(np.diff(r)))
    denom = (n**2 - 1)
    if denom <= 0:
        return np.float64(0.0)
    xi = 1 - (3 * diff_sum) / denom
    if not np.isfinite(xi):
        return np.float64(0.0)
    return np.float64(xi)

# =========================================================
# helpers
# =========================================================
def _as_vec32(y, L):
    # Force 1D float32 length L
    if isinstance(y, np.ndarray):
        if y.dtype != np.float32:
            y = y.astype(np.float32, copy=False)
        y = y.ravel()
        if y.size != L:
            y = np.resize(y, L).astype(np.float32, copy=False)
        return y
    y = np.asarray(y, dtype=np.float32).ravel()
    if y.size != L:
        y = np.resize(y, L).astype(np.float32, copy=False)
    return y

def _fast_abs_pearson(a, b, eps=1e-12):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = a.size
    if n < 2:
        return 0.0
    am = a.mean()
    bm = b.mean()
    da = a - am
    db = b - bm
    num = float(np.dot(da, db))
    den = float(np.sqrt(np.dot(da, da) * np.dot(db, db)) + eps)
    if den <= 0 or (not np.isfinite(num)) or (not np.isfinite(den)):
        return 0.0
    c = num / den
    if not np.isfinite(c):
        return 0.0
    return abs(c)

def safe_corr(a, b):
    """
    You asked not to change the loss "meaning".
    This keeps your intended form:
      |Pearson| * |Spearman| * max(xi(a,b),0) * max(xi(b,a),0)
    (Your original xi.max(0) was a bug; this is the intended clamp.)
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or b.size < 2:
        return 0.0

    pear = _fast_abs_pearson(a, b)
    sp = spearmanr(a, b).correlation
    if not np.isfinite(sp):
        sp = 0.0
    sp = abs(float(sp))

    xi1 = float(chatterjee_correlation(a, b))
    xi2 = float(chatterjee_correlation(b, a))
    if not np.isfinite(xi1): xi1 = 0.0
    if not np.isfinite(xi2): xi2 = 0.0
    xi1 = xi1 if xi1 > 0.0 else 0.0
    xi2 = xi2 if xi2 > 0.0 else 0.0

    return float(pear * sp * xi1 * xi2)

# =========================================================
# 1D helper TT/TT2 (your original)
# =========================================================
def TT(x):
    x = np.asarray(x)
    if x.size < 3:
        return x.copy()
    return np.concatenate((np.ones(1, x.dtype),
                           (x[:-2] + x[1:-1]*2 + x[2:]) * 0.25,
                           np.ones(1, x.dtype)))

def TT2(x):
    x = np.asarray(x)
    if x.size < 3:
        return x.copy()
    return np.concatenate((np.ones(1, x.dtype),
                           ((x[:-2] - x[1:-1])**2 + (x[1:-1] - x[2:])**2) * 0.5,
                           np.ones(1, x.dtype)))

# =========================================================
# Polynomial "attention" (your original, kept)
# =========================================================
def _attn_poly_fast(k, v, q, deg: int):
    k = np.nan_to_num(k, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float64).ravel()
    v = np.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float64).ravel()
    q = np.nan_to_num(q, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float64).ravel()

    scale = np.max(np.abs(k)) + 1e-12
    kn = k / scale
    qn = q / scale

    size = deg + 1
    max_pow = 2 * deg

    with np.errstate(over='ignore', invalid='ignore'):
        P = kn[:, None] ** np.arange(max_pow + 1, dtype=np.float64)

    if not np.all(np.isfinite(P)):
        return np.zeros_like(q, dtype=np.float64)

    A = np.empty((size, size), dtype=np.float64)
    for i in range(size):
        A[i] = P[:, i:i+size].sum(axis=0)

    b = (v[:, None] * P[:, :size]).sum(axis=0)

    ridge = 1e-8 * np.trace(A) / size if size > 0 else 1e-10
    A = A + np.eye(size, dtype=np.float64) * max(ridge, 1e-10)

    try:
        coef = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        try:
            coef = np.linalg.lstsq(A, b, rcond=1e-15)[0]
        except np.linalg.LinAlgError:
            return np.zeros_like(q, dtype=np.float64)

    Q = qn[:, None] ** np.arange(size, dtype=np.float64)
    return (Q @ coef).astype(np.float64)

def attn_poly3_fast(k, v, q):  return _attn_poly_fast(np.nan_to_num(k), np.nan_to_num(v), np.nan_to_num(q), deg=3)
def attn_poly5_fast(k, v, q):  return _attn_poly_fast(np.nan_to_num(k), np.nan_to_num(v), np.nan_to_num(q), deg=5)
def attn_poly11_fast(k, v, q): return _attn_poly_fast(np.nan_to_num(k), np.nan_to_num(v), np.nan_to_num(q), deg=11)

gg   = lambda x: np.abs(x)**(1/3)  * np.sign(x)
ggg  = lambda x: np.abs(x)**0.2    * np.sign(x)
gggg = lambda x: np.abs(x)**(1/11) * np.sign(x)

# =========================================================
# function sets (YOUR LISTS)
# NOTE: these must accept 1D arrays (L,)
# =========================================================
funcs_1 = [
    lambda x: x + 1,
    lambda x: x * 2,
    lambda x: x ** 2,
    lambda x: -x,
    lambda x: 1 / (x ** 2 + 1e-12),
    lambda x: np.sort(x),
    lambda x: np.argsort(x).astype(np.float32),
    lambda x: np.argsort(np.argsort(x)).astype(np.float32),
    lambda x: np.fft.fft(x).real / x.shape[0],
    lambda x: np.fft.fft(x).imag / x.shape[0],
    lambda x: TT(x),
    lambda x: TT2(x),
    lambda x: TT(TT(x)),
    lambda x: TT(TT(TT(TT(x)))),
    lambda x: x * 0.5,
    lambda x: np.flip(x),
    lambda x: np.concatenate((np.zeros(1, x.dtype), x[:-1])),
    lambda x: np.concatenate((x[1:], np.zeros(1, x.dtype))),
    lambda x: np.concatenate((np.zeros(2, x.dtype), x[:-2])),
    lambda x: np.concatenate((x[2:], np.zeros(2, x.dtype))),
    lambda x: np.concatenate((np.zeros(4, x.dtype), x[:-4])),
    lambda x: np.concatenate((x[4:], np.zeros(4, x.dtype))),
    lambda x: np.mean(x, dtype=np.float64) + x * 0,
    lambda x: (np.log(np.std(x) + 1e-12)).astype(np.float64) + x * 0,
    lambda x: x * 0.1,
    lambda x: (x - np.mean(x)) / (np.std(x) + 1e-12),
    lambda x: x * 10,
    lambda x: x ** 3,
    lambda x: np.sin(x * np.pi),
    lambda x: x - 1,
    lambda x: x * 0.9,
    lambda x: x * 1.1,
    lambda x: np.sign(x) * np.abs(x) ** (1/3),
    lambda x: np.tanh(x),
    lambda x: x * -0.01,
    lambda x: x * 0.01,
    lambda x: np.fft.ifft(np.abs(np.fft.fft(x + 0j)) ** 2).real,
    lambda x: np.cumsum(x) / (np.arange(x.size, dtype=np.float64)+1.0),
    lambda x: np.cumsum(x),
    lambda x: np.cumprod(x / np.sqrt(np.mean(x ** 2) + 1e-12)),
    lambda x: np.abs(x),
    lambda x: np.maximum(x, 0),
    lambda x: x * len(x),
    lambda x: x / len(x),
    lambda x: np.take(TT(np.take(x, np.argsort(x))), np.argsort(np.argsort(x))),
    lambda x: np.abs(x)**(1/3) * np.sign(x),
]

funcs_2 = [
    lambda x, y: x + y,
    lambda x, y: x - y,
    lambda x, y: x * y,
    lambda x, y: x / (y ** 2 + 1e-12),
    lambda x, y: x / (np.abs(y) + 1e-12),
    lambda x, y: (x - y) ** 2,
    lambda x, y: (x + y) / 2,
    lambda x, y: np.sqrt(x ** 2 + y ** 2),
    lambda x, y: np.fft.ifft(np.fft.fft(x) * np.fft.fft(y) / x.shape[0]).real,
    lambda x, y: np.fft.ifft(np.fft.fft(x) ** 2 / (np.fft.fft(y) + 1e-12)).real,
    lambda x, y: np.take(x, np.argsort(y)),
    lambda x, y: np.take(TT(np.take(x, np.argsort(y))), np.argsort(np.argsort(y))),
    lambda x, y: np.maximum(x, y),
    lambda x, y: np.minimum(x, y),
    lambda x, y: np.sin(x * np.pi * y),
    lambda x, y: attn_poly3_fast(x, y, y),
    lambda x, y: gg(attn_poly3_fast(x**3, y**3, y**3)),
    lambda x, y: attn_poly5_fast(x, y, y),
    lambda x, y: ggg(attn_poly5_fast(x**5, y**5, y**5)),
    lambda x, y: attn_poly11_fast(x, y, y),
    lambda x, y: gggg(attn_poly11_fast(x**11, y**11, y**11)),
]

funcs_3 = [
    lambda x, y, z: np.sqrt((x - y) ** 2 + (y - z) ** 2 + (z - x) ** 2),
    lambda x, y, z: (x + y + z) / 3,
    lambda x, y, z: np.sign(x * y * z) * np.abs(x * y * z) ** (1/3),
    lambda x, y, z: x + y - z,
    lambda x, y, z: np.sqrt(x ** 2 + y ** 2 + z ** 2),
    lambda x, y, z: np.fft.ifft(np.fft.fft(x) * np.fft.fft(y) / (np.fft.fft(z) + 1e-12)).real,
    lambda x, y, z: np.fft.ifft(np.fft.fft(x) * np.fft.fft(np.tanh(y)) / (np.fft.fft(np.tanh(z)) + 1e-12)).real,
    lambda x, y, z: np.fft.ifft(np.fft.fft(x) * np.fft.fft(np.tanh(y)+1) / (np.fft.fft(np.tanh(z)+1) + 1e-12)).real,
    lambda x, y, z: attn_poly3_fast(x**3, y**3, z),
    lambda x, y, z: gg(attn_poly3_fast(x, y**3, z)),
    lambda x, y, z: attn_poly5_fast(x, y, z),
    lambda x, y, z: ggg(attn_poly5_fast(x**5, y**5, z**5)),
    lambda x, y, z: attn_poly5_fast(x**5, y**5, z),
    lambda x, y, z: ggg(attn_poly5_fast(x, y**5, z)),
    lambda x, y, z: attn_poly11_fast(x, y, z),
    lambda x, y, z: gggg(attn_poly11_fast(x**11, y**11, z**11)),
    lambda x, y, z: attn_poly11_fast(x**11, y**11, z),
    lambda x, y, z: gggg(attn_poly11_fast(x, y**11, z)),
    lambda x, y, z: np.take(np.take(x, np.argsort(y)), np.argsort(np.argsort(z))),
    lambda x, y, z: np.take(TT(np.take(x, np.argsort(y))), np.argsort(np.argsort(z))),
    lambda x, y, z: np.take(TT(TT(np.take(x, np.argsort(y)))), np.argsort(np.argsort(z))),
    lambda x, y, z: np.take(TT2(np.take(x, np.argsort(y))), np.argsort(np.argsort(z))),
    lambda x, y, z: np.take(TT2(TT2(np.take(x, np.argsort(y)))), np.argsort(np.argsort(z))),
]

i0t = funcs_1
i1t = funcs_2
i2t = funcs_3
len_i0 = len(i0t)
len_i1 = len(i1t)
len_i2 = len(i2t)

# =========================================================
# build function-time distribution T (your intent)
# =========================================================
def build_T_distribution_1d(
    i0t, i1t, i2t,
    L=64,
    repeats=40,
    warmup=3,
    power=0.7,
    seed=0,
    eps=1e-12,
):
    rng = np.random.default_rng(seed)

    def _bench_unary(f):
        x = rng.normal(0, 1, size=L).astype(np.float32)
        for _ in range(warmup):
            y = f(x)
            x = _as_vec32(y, L)
        t0 = time.perf_counter()
        for _ in range(repeats):
            x = rng.normal(0, 1, size=L).astype(np.float32)
            y = f(x)
            _ = _as_vec32(y, L)
        t1 = time.perf_counter()
        return (t1 - t0) / repeats

    def _bench_binary(f):
        x = rng.normal(0, 1, size=L).astype(np.float32)
        y = rng.normal(0, 1, size=L).astype(np.float32)
        for _ in range(warmup):
            z = f(x, y)
            _ = _as_vec32(z, L)
        t0 = time.perf_counter()
        for _ in range(repeats):
            x = rng.normal(0, 1, size=L).astype(np.float32)
            y = rng.normal(0, 1, size=L).astype(np.float32)
            z = f(x, y)
            _ = _as_vec32(z, L)
        t1 = time.perf_counter()
        return (t1 - t0) / repeats

    def _bench_ternary(f):
        x = rng.normal(0, 1, size=L).astype(np.float32)
        y = rng.normal(0, 1, size=L).astype(np.float32)
        z = rng.normal(0, 1, size=L).astype(np.float32)
        for _ in range(warmup):
            w = f(x, y, z)
            _ = _as_vec32(w, L)
        t0 = time.perf_counter()
        for _ in range(repeats):
            x = rng.normal(0, 1, size=L).astype(np.float32)
            y = rng.normal(0, 1, size=L).astype(np.float32)
            z = rng.normal(0, 1, size=L).astype(np.float32)
            w = f(x, y, z)
            _ = _as_vec32(w, L)
        t1 = time.perf_counter()
        return (t1 - t0) / repeats

    times = []
    for f in i0t:
        try: dt = _bench_unary(f)
        except Exception: dt = 1e9
        times.append(dt)
    for f in i1t:
        try: dt = _bench_binary(f)
        except Exception: dt = 1e9
        times.append(dt)
    for f in i2t:
        try: dt = _bench_ternary(f)
        except Exception: dt = 1e9
        times.append(dt)

    times = np.asarray(times, dtype=np.float64)
    w = 1.0 / (np.maximum(times, eps) ** power)
    if (not np.isfinite(w).all()) or w.sum() <= 0:
        w = np.ones_like(w)
    T = (w / w.sum()).astype(np.float64)
    return T, times

T, times = build_T_distribution_1d(i0t, i1t, i2t, L=64, repeats=20, warmup=2)
print("T built. funcs:", len(T))

# =========================================================
# dataset generator (your original)
# =========================================================
A = {"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"+":10,"-":11,"*":12,"/":13,"=":14}

def generatetekito():
    S  = int(np.random.randint(1, 2 ** np.random.randint(4, 32)))
    S2 = int(np.random.randint(1, 2 ** np.random.randint(4, 32)))
    S3 = int(np.random.randint(1, 2 ** np.random.randint(4, 32)))
    enz = np.random.randint(0, 13)
    if enz == 0:  return f"{S}+{S2}={S+S2}"
    if enz == 1:  return f"{S}-{S2}={S-S2}"
    if enz == 2:  return f"{S}*{S2}={S*S2}"
    if enz == 3:  return f"{S}/{S2}={S//S2}"
    if enz == 4:  return f"{S}+{S2}+{S3}={S+S2+S3}"
    if enz == 5:  return f"{S}+{S2}-{S3}={S+S2-S3}"
    if enz == 6:  return f"{S}+{S2}*{S3}={S+S2*S3}"
    if enz == 7:  return f"{S}*{S2}+{S3}={S*S2+S3}"
    if enz == 8:  return f"{S}*{S2}-{S3}={S*S2-S3}"
    if enz == 9:  return f"{S}-{S2}*{S3}={S-S2*S3}"
    if enz == 10: return f"{S}*{S2}/{S3}={int(S*S2/S3)}"
    if enz == 11: return f"{S}-{S2}/{S3}={S - S2//S3}"
    return f"{S}/{S2}*{S3}={int(S/S2*S3)}"

def gendata(g, tt):
    at = [A[_] for _ in g]
    s = ((len(at) - tt) / len(at))
    perm = np.random.choice(len(at), len(at), replace=False)
    for i in range(tt):
        t = np.random.randint(0, 15)
        while t == at[perm[i]]:
            t = np.random.randint(0, 15)
        at[perm[i]] = t

    gt = np.zeros((15, len(at)), dtype=np.float32)
    for j in range(len(at)):
        gt[at[j], j] = 1.0
    return gt, np.float32(s)

# =========================================================
# greedy engine (G3 removed)
# =========================================================
def build_used_mask(G1, G2, MODELLEN, num_inputs, out_idx):
    used = np.zeros(MODELLEN, dtype=np.bool_)
    stack = [out_idx]
    while stack:
        n = stack.pop()
        if n < 0 or n >= MODELLEN:
            continue
        if used[n]:
            continue
        used[n] = True
        if n < num_inputs:
            continue
        fid = int(G2[n])
        if fid < len_i0:
            a = int(abs(G1[n, 0])); stack.append(a)
        elif fid < len_i0 + len_i1:
            a = int(abs(G1[n, 0])); b = int(abs(G1[n, 1]))
            stack.append(a); stack.append(b)
        else:
            a = int(abs(G1[n, 0])); b = int(abs(G1[n, 1])); c = int(abs(G1[n, 2]))
            stack.append(a); stack.append(b); stack.append(c)
    used[:num_inputs] = True
    return used

def build_parents(G1, G2, MODELLEN, num_inputs, used):
    deg = np.zeros(MODELLEN, dtype=np.int32)
    for p in range(num_inputs, MODELLEN):
        if not used[p]:
            continue
        fid = int(G2[p])
        if fid < len_i0:
            a = int(abs(G1[p,0])); deg[a] += 1
        elif fid < len_i0 + len_i1:
            a = int(abs(G1[p,0])); b = int(abs(G1[p,1]))
            deg[a] += 1; deg[b] += 1
        else:
            a = int(abs(G1[p,0])); b = int(abs(G1[p,1])); c = int(abs(G1[p,2]))
            deg[a] += 1; deg[b] += 1; deg[c] += 1

    offsets = np.zeros(MODELLEN + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(deg)
    buf = np.empty(offsets[-1], dtype=np.int32)
    cur = offsets[:-1].copy()

    for p in range(num_inputs, MODELLEN):
        if not used[p]:
            continue
        fid = int(G2[p])
        if fid < len_i0:
            a = int(abs(G1[p,0]))
            buf[cur[a]] = p; cur[a] += 1
        elif fid < len_i0 + len_i1:
            a = int(abs(G1[p,0])); b = int(abs(G1[p,1]))
            buf[cur[a]] = p; cur[a] += 1
            buf[cur[b]] = p; cur[b] += 1
        else:
            a = int(abs(G1[p,0])); b = int(abs(G1[p,1])); c = int(abs(G1[p,2]))
            buf[cur[a]] = p; cur[a] += 1
            buf[cur[b]] = p; cur[b] += 1
            buf[cur[c]] = p; cur[c] += 1

    return offsets, buf

def affected_cone(n, used, parent_offsets, parent_buf):
    if not used[n]:
        return np.array([], dtype=np.int32)
    q = [n]
    seen = np.zeros_like(used)
    seen[n] = True
    while q:
        x = q.pop()
        s = parent_offsets[x]; e = parent_offsets[x+1]
        for i in range(s, e):
            p = int(parent_buf[i])
            if not used[p] or seen[p]:
                continue
            seen[p] = True
            q.append(p)
    idx = np.flatnonzero(seen)
    idx.sort()  # child < parent (DAG) so index order works
    return idx.astype(np.int32)

def pack_batch_list(batch, num_inputs, L_eval=64):
    targets = np.asarray([s for (_, s) in batch], dtype=np.float64)
    Xlist = []
    for (gt, _) in batch:
        x = np.asarray(gt, dtype=np.float32)
        x = np.resize(x, (num_inputs, L_eval)).astype(np.float32, copy=False)
        Xlist.append(x)  # (num_inputs, L)
    return Xlist, targets

def forward_base_list(G1, G2, Xlist, used, num_inputs):
    MODELLEN = G1.shape[0]
    base_out_list = []
    for x_inputs in Xlist:
        L = x_inputs.shape[1]
        out = [None] * MODELLEN
        for i in range(num_inputs):
            out[i] = x_inputs[i]
        for n in range(num_inputs, MODELLEN):
            if not used[n]:
                continue
            fid = int(G2[n])
            try:
                if fid < len_i0:
                    a = int(abs(G1[n,0]))
                    y = i0t[fid](out[a])
                elif fid < len_i0 + len_i1:
                    a = int(abs(G1[n,0])); b = int(abs(G1[n,1]))
                    y = i1t[fid - len_i0](out[a], out[b])
                else:
                    a = int(abs(G1[n,0])); b = int(abs(G1[n,1])); c = int(abs(G1[n,2]))
                    y = i2t[fid - (len_i0 + len_i1)](out[a], out[b], out[c])
                out[n] = _as_vec32(y, L)
            except Exception:
                out[n] = np.zeros(L, dtype=np.float32)
        base_out_list.append(out)
    return base_out_list

def build_aff_metadata(aff, G1, G2, num_inputs, MODELLEN):
    """
    aff: int32 sorted ascending
    returns:
      nodes: (K,)
      pos: (MODELLEN,) int32, node->k or -1
      fid0,a0,b0,c0: (K,) int32 base graph info
      pa,pb,pc: (K,) int32, child position in aff or -1
    """
    aff = np.asarray(aff, dtype=np.int32)
    K = aff.size
    nodes = aff

    pos = np.full(MODELLEN, -1, dtype=np.int32)
    for k in range(K):
        pos[int(nodes[k])] = k

    fid0 = np.empty(K, dtype=np.int32)
    a0   = np.empty(K, dtype=np.int32)
    b0   = np.empty(K, dtype=np.int32)
    c0   = np.empty(K, dtype=np.int32)

    pa = np.empty(K, dtype=np.int32)
    pb = np.empty(K, dtype=np.int32)
    pc = np.empty(K, dtype=np.int32)

    for k in range(K):
        node = int(nodes[k])
        if node < num_inputs:
            fid0[k] = -1
            a0[k] = b0[k] = c0[k] = -1
            pa[k] = pb[k] = pc[k] = -1
            continue

        fid = int(G2[node])
        fid0[k] = fid

        a = int(abs(G1[node, 0]))
        b = int(abs(G1[node, 1]))
        c = int(abs(G1[node, 2]))
        a0[k] = a
        b0[k] = b
        c0[k] = c

        pa[k] = pos[a] if (0 <= a < MODELLEN) else -1
        pb[k] = pos[b] if (0 <= b < MODELLEN) else -1
        pc[k] = pos[c] if (0 <= c < MODELLEN) else -1

    return nodes, pos, fid0, a0, b0, c0, pa, pb, pc


def eval_candidate_logits_fast(
    n, cand_fid, cand_a, cand_b, cand_c,
    nodes, pos, fid0, a0, b0, c0, pa, pb, pc,
    base_out_list, num_inputs, out_idx, L,
    scratch2d,
):
    """
    candidates evaluated on ALL samples:
      returns logits (B,) float64 (mean(out_idx vector))
    scratch2d: (K,L) float32 reused for every sample
    """
    B = len(base_out_list)
    K = nodes.size

    out_pos = pos[out_idx]  # -1 if out not in aff (then unaffected)
    logits = np.empty(B, dtype=np.float64)

    # per sample
    for si in range(B):
        base = base_out_list[si]

        # fill scratch for this sample
        for k in range(K):
            node = int(nodes[k])

            if node < num_inputs:
                # input node: directly from base
                scratch2d[k, :] = base[node]
                continue

            # choose op/children (override only at node==n)
            if node == n:
                fid = cand_fid
                a = cand_a; b = cand_b; c = cand_c
                # child positions for override
                pA = pos[a] if 0 <= a < pos.size else -1
                pB = pos[b] if 0 <= b < pos.size else -1
                pC = pos[c] if 0 <= c < pos.size else -1
            else:
                fid = int(fid0[k])
                a = int(a0[k]); b = int(b0[k]); c = int(c0[k])
                pA = int(pa[k]); pB = int(pb[k]); pC = int(pc[k])

            # fetch children (scratch if in aff and already computed, else base)
            try:
                if fid < len_i0:
                    xa = scratch2d[pA, :] if pA >= 0 else base[a]
                    y = i0t[fid](xa)
                elif fid < len_i0 + len_i1:
                    xa = scratch2d[pA, :] if pA >= 0 else base[a]
                    xb = scratch2d[pB, :] if pB >= 0 else base[b]
                    y = i1t[fid - len_i0](xa, xb)
                else:
                    xa = scratch2d[pA, :] if pA >= 0 else base[a]
                    xb = scratch2d[pB, :] if pB >= 0 else base[b]
                    xc = scratch2d[pC, :] if pC >= 0 else base[c]
                    y = i2t[fid - (len_i0 + len_i1)](xa, xb, xc)

                yy = _as_vec32(y, L)
                scratch2d[k, :] = yy
            except Exception:
                scratch2d[k, :] = 0.0

        # pick output
        if out_pos >= 0:
            logits[si] = float(np.mean(scratch2d[out_pos, :], dtype=np.float64))
        else:
            # out_idx not affected -> base output
            logits[si] = float(np.mean(base[out_idx], dtype=np.float64))

    return logits

def optimize_one_node_exhaustive_safe_corr(
    n, G1, G2, base_out_list, used, aff,
    Xlist, targets,
    num_inputs, out_idx,
    MODELLEN_for_radius=None,
):
    MODELLEN = G1.shape[0]
    if MODELLEN_for_radius is None:
        MODELLEN_for_radius = MODELLEN

    if aff.size == 0 or n < num_inputs:
        return False, None

    B = len(Xlist)
    L = Xlist[0].shape[1]

    # base score (current)
    base_logits = np.asarray([float(np.mean(base_out_list[i][out_idx])) for i in range(B)], dtype=np.float64)
    base_score = safe_corr(base_logits, targets)

    # radii: your rule (MODELLEN-based)
    r2 = max(1, int(np.sqrt(max(1, MODELLEN_for_radius)) / 2))
    r3 = max(1, int(np.cbrt(max(1, MODELLEN_for_radius)) / 2))

    oa = int(abs(G1[n, 0])); ob = int(abs(G1[n, 1])); oc = int(abs(G1[n, 2]))

    def neigh(center, r, upper_exclusive):
        lo = max(0, center - r)
        hi = min(upper_exclusive, center + r + 1)
        if lo >= hi:
            return np.array([max(0, min(center, upper_exclusive - 1))], dtype=np.int32)
        return np.arange(lo, hi, dtype=np.int32)

    # unary: all previous nodes
    cand_u = np.arange(0, n, dtype=np.int32)

    # binary/ternary candidate sets: (near original) ∪ (near n)
    if n > 0:
        near_n2 = neigh(n - 1, r2, n)
        near_n3 = neigh(n - 1, r3, n)

        cand_a2 = np.unique(np.concatenate([neigh(min(oa, n-1), r2, n), near_n2])).astype(np.int32)
        cand_b2 = np.unique(np.concatenate([neigh(min(ob, n-1), r2, n), near_n2])).astype(np.int32)

        cand_a3 = np.unique(np.concatenate([neigh(min(oa, n-1), r2, n), near_n2])).astype(np.int32)
        cand_b3 = np.unique(np.concatenate([neigh(min(ob, n-1), r2, n), near_n2])).astype(np.int32)
        cand_c3 = np.unique(np.concatenate([neigh(min(oc, n-1), r3, n), near_n3])).astype(np.int32)
    else:
        cand_a2 = cand_b2 = cand_a3 = cand_b3 = cand_c3 = np.array([], dtype=np.int32)

    # aff metadata once
    nodes, pos, fid0, a0, b0, c0, pa, pb, pc = build_aff_metadata(
        aff, G1, G2, num_inputs, MODELLEN
    )

    # scratch buffer reused (K,L)
    K = nodes.size
    scratch2d = np.empty((K, L), dtype=np.float32)

    best_score = base_score
    best = (int(G2[n]), int(G1[n,0]), int(G1[n,1]), int(G1[n,2]))

    # --- unary exhaustive: funcs x all nodes ---
    for fid in tqdm.tqdm(range(len_i0)):
        for a in cand_u:
            a = int(a)
            if a >= n:
                continue
            logits = eval_candidate_logits_fast(
                n, int(fid), a, 0, 0,
                nodes, pos, fid0, a0, b0, c0, pa, pb, pc,
                base_out_list, num_inputs, out_idx, L,
                scratch2d,
            )
            sc = safe_corr(logits, targets)
            if sc > best_score + 1e-12:
                best_score = sc
                best = (int(fid), a, 0, 0)

    # --- binary exhaustive: funcs x (cand_a2 x cand_b2) ---
    for bi in tqdm.tqdm(range(len_i1)):
        fid = len_i0 + bi
        for a in cand_a2:
            a = int(a)
            if a >= n:
                continue
            for b in cand_b2:
                b = int(b)
                if b >= n:
                    continue
                logits = eval_candidate_logits_fast(
                    n, int(fid), a, b, 0,
                    nodes, pos, fid0, a0, b0, c0, pa, pb, pc,
                    base_out_list, num_inputs, out_idx, L,
                    scratch2d,
                )
                sc = safe_corr(logits, targets)
                if sc > best_score + 1e-12:
                    best_score = sc
                    best = (int(fid), a, b, 0)

    # --- ternary exhaustive: funcs x (cand_a3 x cand_b3 x cand_c3) ---
    for ci in tqdm.tqdm(range(len_i2)):
        fid = len_i0 + len_i1 + ci
        for a in cand_a3:
            a = int(a)
            if a >= n:
                continue
            for b in cand_b3:
                b = int(b)
                if b >= n:
                    continue
                for c in cand_c3:
                    c = int(c)
                    if c >= n:
                        continue
                    logits = eval_candidate_logits_fast(
                        n, int(fid), a, b, c,
                        nodes, pos, fid0, a0, b0, c0, pa, pb, pc,
                        base_out_list, num_inputs, out_idx, L,
                        scratch2d,
                    )
                    sc = safe_corr(logits, targets)
                    if sc > best_score + 1e-12:
                        best_score = sc
                        best = (int(fid), a, b, c)

    # apply best
    if best_score > base_score + 1e-12:
        G2[n] = best[0]
        G1[n,0] = best[1]
        G1[n,1] = best[2]
        G1[n,2] = best[3]
        return True, best_score

    return False, base_score

def run_greedy_fast(
    MODELLEN=8192,
    iters=10000,
    samples=1024,
    dataset=131072,
    seed=0,
    num_inputs=15,
    L_eval=64,
):
    rng = np.random.default_rng(seed)
    out_idx = MODELLEN - 1

    # dataset build
    traindats = []
    for _ in tqdm.tqdm(range(dataset), desc="building dataset"):
        gp = generatetekito()
        gt, score = gendata(gp, np.random.randint(1, len(gp)))
        traindats.append((gt, score))

    # init graph (DAG)
    G1 = np.zeros((MODELLEN, 3), dtype=np.int64)
    for n in range(num_inputs, MODELLEN):
        G1[n,0] = rng.integers(0, n)
        G1[n,1] = rng.integers(0, n)
        G1[n,2] = rng.integers(0, n)

    total_funcs = len_i0 + len_i1 + len_i2
    G2 = rng.choice(total_funcs, size=(MODELLEN,), p=T).astype(np.int64)

    history = []
    batches = dataset // samples

    for step in range(iters):
        bi = step % batches
        batch = traindats[bi*samples:(bi+1)*samples]
        Xlist, targets = pack_batch_list(batch, num_inputs, L_eval=L_eval)

        used = build_used_mask(G1, G2, MODELLEN, num_inputs, out_idx)
        parent_offsets, parent_buf = build_parents(G1, G2, MODELLEN, num_inputs, used)

        base_out_list = forward_base_list(G1, G2, Xlist, used, num_inputs)
        base_logits = np.asarray([np.mean(base_out_list[i][out_idx]) for i in range(len(Xlist))], dtype=np.float64)
        base_score = safe_corr(base_logits, targets)

        #min_touch = max(num_inputs, MODELLEN - greedy_window)

        stack = [out_idx]
        visited = np.zeros(MODELLEN, dtype=np.bool_)
        best_score = base_score
        improved_any = False

        while stack:
            n = int(stack.pop())
            if n < num_inputs or visited[n]:
                continue
            visited[n] = True

            aff = affected_cone(n, used, parent_offsets, parent_buf)
            improved, new_score = optimize_one_node_exhaustive_safe_corr(
                n, G1, G2, base_out_list, used, aff,
                Xlist, targets,
                num_inputs, out_idx,
            )

            if improved:
                improved_any = True
                best_score = new_score

                # rebuild caches after improvement (simple & consistent)
                used = build_used_mask(G1, G2, MODELLEN, num_inputs, out_idx)
                parent_offsets, parent_buf = build_parents(G1, G2, MODELLEN, num_inputs, used)
                base_out_list = forward_base_list(G1, G2, Xlist, used, num_inputs)

            # push upstream children according to current function arity
            fid = int(G2[n])
            if fid < len_i0:
                a = int(abs(G1[n,0]))
                if a >= num_inputs: stack.append(a)
            elif fid < len_i0 + len_i1:
                a = int(abs(G1[n,0])); b = int(abs(G1[n,1]))
                if a >= num_inputs: stack.append(a)
                if b >= num_inputs: stack.append(b)
            else:
                a = int(abs(G1[n,0])); b = int(abs(G1[n,1])); c = int(abs(G1[n,2]))
                if a >= num_inputs: stack.append(a)
                if b >= num_inputs: stack.append(b)
                if c >= num_inputs: stack.append(c)
            print(f"{step} corr={best_score:.6f} improved={improved_any}")

        history.append(best_score)
        if step % 100 == 0:
            np.savez("greedy_fast_full.npz", G1=G1, G2=G2, history=np.asarray(history, dtype=np.float64))

        gc.collect()

    return G1, G2, history

# =========================================================
# main
# =========================================================
if __name__ == "__main__":
    G1, G2, hist = run_greedy_fast(
        MODELLEN=1024,
        iters=10000,
        samples=512,
        dataset=131072,
        seed=0,
        num_inputs=15,
        L_eval=64,
    )
    print("done. best corr:", float(np.max(hist)) if len(hist) else 0.0)
