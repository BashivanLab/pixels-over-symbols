# DSA

Commands run from the repository root; set up the environment and download the data first (see the [main README](../../README.md)).

The DSA compares the hidden states of the native and substituted models on the MTurk trials (Sec. 4.3):

```bash
python scripts/evaluate.py substitution --features   # cache the hidden states
python scripts/dsa/run_dsa.py                        # distances behind Figures 4, 22, 23 and 24
python scripts/dsa/sensitivity.py                    # the 19 settings of the sensitivity analysis
python scripts/dsa/sensitivity_report.py             # its table and the counts in the appendix
python scripts/dsa/whitening.py                      # the check behind the sentence on PCA whitening
```

The hidden states of all 26 DSA systems are in the release, so the first command only fills in what is missing.
PCA and the operator fits run on multithreaded BLAS, and the alignment of every pair starts from a random initialization drawn in pair order.
The distances are therefore bitwise reproducible only with the same thread counts (the scripts set 16 and 12) and the same list of systems (`analysis/dsa/systems.py`), which is why that list keeps four ViT systems no figure shows.
