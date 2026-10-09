# Every AI Answer Could Carry a Hidden Signature. One Simple Trick Erases It.

Code for the Medium article. A word-level trigram language model (8 Gutenberg novels) with the green-list watermark of Kirchenbauer et al. (ICML 2023), plus tests of detection, false positives, quality cost and two attacks.

| Result | Number |
|---|---|
| Detection at 10 / 50 / 100 words (medium watermark) | 0% / 82.5% / 100% |
| False positives on 1,346 human passages | 0 (max z 3.1) |
| Perplexity: none / weak / medium / strong | 34.2 / 35.4 / 38.7 / 44.5 |
| Detection after replacing 20% / 40% / 50% of words | 98.5% / 35% / 4.5% |
| Filler trick (filler after every word, then deleted) | 0% detected, z 8.0 → 0.0 |

## Run it
```bash
pip install -r requirements.txt
bash run_all.sh
```

Change `KEY` in `wmlab.py` to use your own secret key.

License: MIT. Novels: public domain (Project Gutenberg).
