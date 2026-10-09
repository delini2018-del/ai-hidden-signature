import json, time, numpy as np
from wmlab import LM, zscore, words, HUMAN
t0 = time.time(); lm = LM(); print("trained on", lm.n_tokens, "tokens, vocab", len(lm.vocab), round(time.time() - t0), "s", flush=True)
rng = np.random.default_rng(0); N, L = 200, 200; R = {"n_texts": N, "length": L, "train_tokens": lm.n_tokens, "vocab": len(lm.vocab)}
import pickle, os
gens = pickle.load(open('gens.pkl','rb')) if os.path.exists('gens.pkl') else {}
for d in [0.0, 1.0, 2.0, 4.0]:
    if d in gens: continue
    gens[d] = [lm.generate(L, rng, delta=d) for _ in range(N)]; print("delta", d, round(time.time() - t0), "s", flush=True)
    pickle.dump(gens, open('gens.pkl','wb'))
    if time.time() - t0 > 200: raise SystemExit('checkpoint')
R["examples"] = {str(d): " ".join(gens[d][0][0][:60]) for d in gens}
Ts = [10, 25, 50, 100, 200]
R["length"] = {str(d): {str(T): float(np.mean([zscore(t[:T + 1]) > 4 for t, _ in gens[d]])) for T in Ts} for d in gens}
R["mean_z"] = {str(d): {str(T): float(np.mean([zscore(t[:T + 1]) for t, _ in gens[d]])) for T in Ts} for d in gens}
R["log_ppl"] = {str(d): float(np.mean([p for _, p in gens[d]])) for d in gens}
hum = [w for b in HUMAN for w in words(f"corpus/{b}.txt")]; win = [hum[i:i + L + 1] for i in range(0, len(hum) - L, L)]
hz = [zscore(w) for w in win]; R["human"] = dict(n=len(win), fpr_z4=float(np.mean(np.array(hz) > 4)), max_z=float(max(hz)), mean_z=float(np.mean(hz)))
# attack 1: random word substitution on delta=2 texts
sub = {}
for rate in [0, .1, .2, .3, .4, .5, .6]:
    det, zs = [], []
    for t, _ in gens[2.0]:
        t = list(t); idx = np.where(rng.random(len(t)) < rate)[0]
        for i, j in zip(idx, rng.choice(len(lm.vocab), size=len(idx), p=lm.uni_p)): t[i] = lm.vocab[j]
        z = zscore(t); zs.append(z); det.append(z > 4)
    sub[str(rate)] = dict(detect=float(np.mean(det)), mean_z=float(np.mean(zs)))
R["substitution"] = sub
# attack 2: the 'emoji trick' - model writes a filler token after every word (green lists keyed on the filler), filler then deleted
emo = [lm.generate(L, rng, delta=2.0, key_word="\u263a") for _ in range(100)]
R["emoji_trick"] = dict(detect=float(np.mean([zscore(t) > 4 for t, _ in emo])), mean_z=float(np.mean([zscore(t) for t, _ in emo])),
                        z_if_filler_kept=float(np.mean([sum(__import__('wmlab').green("\u263a", w) for w in t) / len(t) for t, _ in emo])))
json.dump(R, open("results.json", "w"), indent=1); print(json.dumps({k: v for k, v in R.items() if k != "examples"}, indent=1)); print(R["examples"]["2.0"][:300])
