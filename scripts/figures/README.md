# Figures

Commands run from the repository root; set up the environment and download the data first (see the [main README](../../README.md)).

```bash
bash scripts/figures/make_all.sh
```

redraws every data figure of the paper from `data/evaluations` in a few minutes on a CPU.
The panels go to `figures/`, and the numbers the text quotes are printed in `figures/logs/`.
The RSA results depend on Python's hash seed, so the scripts insist on `PYTHONHASHSEED=0` (`make_all.sh` sets it).

| Figure | Files under `figures/` |
|---|---|
| 2 | `seq_type/{correlation,accuracy,legend}.png` |
| 3 | `transplant/final/global/{rho_mh,accuracy}.png`, `transplant/final/legend.png` |
| 4 | `dsa/final/{seq_abs,seq_res,legend}.png` |
| 10 | `metric_alignment/{rho_acc,rho_kappa}.png`, `seq_type/error_consistency.png` |
| 11 | `seq_type/RSA/human_baseline_{ct,cs}.png`, `metric_alignment/{rho_task_rsa,rho_stim_rsa}.png` |
| 12 | `metric_alignment/{cat,id,pos}_{cons,acc,ec}.png` |
| 13 | `ab_match_levels/{correlation,accuracy}.png`, `seq_type/legend.png` |
| 14 | `transplant/acc_matched/global/{rho_mh,accuracy}.png`, `transplant/acc_matched/legend.png` |
| 15 | `noise_dose/{rho_mh,accuracy}.png` |
| 16 | `transplant/dm/global/{rho_mh,accuracy}.png`, `transplant/dm/legend.png` |
| 17 | `vision_type/`, `seq_size/` and `encoder_size/`: `{correlation,accuracy}*.png` and `legend.png` |
| 18 | `seq_type/feature_level/{category,identity,position}/{correlation,accuracy}.png` |
| 19 | `seq_type/task_level/per_task_type_{NBack,InterDMS,CtxDMS,DMS}/{correlation,accuracy}.png` |
| 20 | `transplant/final/global/{cs,ct}_RSA.png` |
| 21 | `seq_type/RSA_ct_panels/*.png` |
| 22 | `dsa/final/{pre_abs,pre_res,legend}.png` |
| 23 | `dsa/dm/{pre_abs,pre_res,seq_abs,seq_res,legend}.png` |
| 24 | `dsa/final/{heatmap,mds}_{pre,seq}_pc50.png` |

Figure 1 and the stimulus and task illustrations (Figures 5 to 9) were made by hand.
