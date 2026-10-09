"""A small word-level language model + the 'green list' watermark of Kirchenbauer et al. (ICML 2023).
At each step, the previous word and a secret key split the vocabulary into a 'green' half and a 'red' half;
green words get a bonus (delta) before sampling. Detection counts green words and computes a z-score."""
import re, glob, collections, hashlib, math
import numpy as np
KEY = b"delini-secret-key"; GAMMA = 0.5
def words(path):
    t = open(path, encoding="utf-8", errors="ignore").read()
    s, e = t.find("*** START"), t.find("*** END")
    t = t[t.find("\n", s) + 1 if s >= 0 else 0: e if e > 0 else None]
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[.,;!?]", t.lower())
TRAIN = ["1342", "98", "2701", "1400", "158", "1260", "145", "514"]; HUMAN = ["84", "345"]
def green(prev, w):
    h = hashlib.blake2b(KEY + prev.encode() + b"|" + w.encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") % 10000 < GAMMA * 10000
class LM:
    def __init__(self):
        toks = [w for b in TRAIN for w in words(f"corpus/{b}.txt")]
        self.uni = collections.Counter(toks); self.bi = collections.defaultdict(collections.Counter); self.tri = collections.defaultdict(collections.Counter)
        for a, b in zip(toks, toks[1:]): self.bi[a][b] += 1
        for a, b, c in zip(toks, toks[1:], toks[2:]): self.tri[(a, b)][c] += 1
        self.n_tokens = len(toks); self.vocab = list(self.uni); self.cache = {}; self.mcache = {}
        uw = np.array([self.uni[w] for w in self.vocab], float); self.uni_p = uw / uw.sum()
    def dist(self, p2, p1):
        k = (p2, p1)
        if k not in self.cache:
            t, b = self.tri.get(k, {}), self.bi.get(p1, {})
            cands = list(set(t) | set(b)) or ["."]
            tt, bt = sum(t.values()) or 1, sum(b.values()) or 1
            p = np.array([0.6 * t.get(c, 0) / tt + 0.4 * b.get(c, 0) / bt for c in cands]) + 1e-12
            self.cache[k] = (cands, np.log(p / p.sum()))
        return self.cache[k]
    def generate(self, n, rng, delta=0.0, key_word=None, start=None):
        out = list(start or rng.choice(self.vocab, 2, p=self.uni_p)); lp_sum = 0.0
        for _ in range(n):
            cands, lp = self.dist(out[-2], out[-1])
            prev = key_word if key_word is not None else out[-1]
            if delta:
                mk = (out[-2], out[-1], prev)
                if mk not in self.mcache: self.mcache[mk] = np.array([green(prev, c) for c in cands], float)
                bonus = delta * self.mcache[mk]
            else: bonus = 0.0
            z = lp + bonus; z = np.exp(z - z.max()); z /= z.sum()
            i = rng.choice(len(cands), p=z); out.append(cands[i]); lp_sum += lp[i]
        return out[2:], -lp_sum / n          # tokens, base-model perplexity (log scale per token)
def zscore(toks, prev0=None):
    pairs = list(zip(toks, toks[1:])); T = len(pairs)
    if T == 0: return 0.0
    g = sum(green(a, b) for a, b in pairs)
    return (g - GAMMA * T) / math.sqrt(T * GAMMA * (1 - GAMMA))
