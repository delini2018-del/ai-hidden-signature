import json, math, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight"})
B, O, R_, G, P = "#2563eb", "#ea580c", "#dc2626", "#6b7280", "#16a34a"
R = json.load(open("results.json")); SRC = "Source: author's experiment. Word-level language model trained on 8 public-domain novels; green-list watermark after Kirchenbauer et al. (2023). Code: see repo."
def foot(fig, s=SRC): fig.text(0.01, -0.03, s, fontsize=8.5, color=G)
Ts = [10, 25, 50, 100, 200]
fig, ax = plt.subplots(figsize=(9, 5))
for d, c, lab in [("1.0", G, "Weak watermark (bonus 1)"), ("2.0", B, "Medium watermark (bonus 2)"), ("4.0", R_, "Strong watermark (bonus 4)")]:
    ax.plot(Ts, [100*R["length"][d][str(T)] for T in Ts], marker="o", lw=2.5, color=c, label=lab)
ax.plot(Ts, [100*R["length"]["0.0"][str(T)] for T in Ts], marker="o", lw=1.5, ls="--", color="k", label="No watermark (false alarms)")
ax.set(xscale="log", xticks=Ts, xlabel="Length of text checked (words)", ylabel="Detected as watermarked (%)", ylim=(-3, 103), title="A watermark needs words to speak")
ax.set_xticklabels([str(t) for t in Ts]); ax.legend(frameon=False, fontsize=10, loc="lower right"); foot(fig); fig.savefig("charts/01_length.png"); plt.close()
fig, ax = plt.subplots(figsize=(9, 5)); ds = ["0.0", "1.0", "2.0", "4.0"]; ppl = [math.exp(R["log_ppl"][d]) for d in ds]
bb = ax.bar(["None", "Weak (1)", "Medium (2)", "Strong (4)"], ppl, color=[G, "#93c5fd", B, R_], width=.55)
for b, v in zip(bb, ppl): ax.text(b.get_x()+b.get_width()/2, v+.6, f"{v:.1f}", ha="center")
ax.set(ylim=(0, 52), xlabel="Watermark strength", ylabel="Perplexity under the original model\n(higher = less natural text)", title="A louder watermark costs text quality"); foot(fig); fig.savefig("charts/02_quality.png"); plt.close()
S = R["substitution"]; rates = sorted(S, key=float)
fig, ax = plt.subplots(figsize=(9, 5)); ax.plot([100*float(r) for r in rates], [100*S[r]["detect"] for r in rates], marker="o", lw=2.5, color=B)
for r in rates: ax.annotate(f"{100*S[r]['detect']:.0f}%", (100*float(r), 100*S[r]["detect"]), textcoords="offset points", xytext=(6, 6), fontsize=10)
ax.set(xlabel="Share of words randomly replaced (%)", ylabel="Still detected (%)", ylim=(-3, 108), title="It survives light editing, then falls off a cliff"); foot(fig, SRC.replace("Code", "Medium watermark, 200-word texts. Code")); fig.savefig("charts/03_substitution.png"); plt.close()
E = R["emoji_trick"]; z_kept = (E["z_if_filler_kept"] - .5) * 200 / math.sqrt(200 * .25)
fig, ax = plt.subplots(figsize=(8.5, 5)); bb = ax.bar(["Text as generated\n(with filler after every word)", "Same text after\nfillers are deleted"], [z_kept, E["mean_z"]], color=[B, R_], width=.5)
for b, v in zip(bb, [z_kept, E["mean_z"]]): ax.text(b.get_x()+b.get_width()/2, v+.2, f"z = {v:.1f}", ha="center", fontsize=12)
ax.axhline(4, ls="--", color=G); ax.text(1.35, 4.15, "detection threshold", color=G, fontsize=10, ha="right")
ax.set(ylim=(0, 10), ylabel="Watermark strength score (z)", title="The filler trick: the signature is there, then it isn't"); foot(fig, SRC.replace("Code", "Medium watermark, 100 texts of 200 words. Code")); fig.savefig("charts/04_filler_trick.png"); plt.close()
print("z_kept", round(z_kept, 2))
