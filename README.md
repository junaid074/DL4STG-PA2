# AI651 Assignment 2

Roll number: 25280044  
Repo: https://github.com/junaid074/DL4STG-PA2

## Question 1
River forecasting design — written in `PA2_Report.pdf` (six required headings).  
Optional companion notebook in `Question1/` (ungraded).

## Question 2
```bash
cd Question2
# train (A–D, seeds 0 and 1)
python -m q2.train --model A --seed 0 --epochs 30 --root .
# … repeat for A/1, B/0, B/1, C/0, C/1, D/0, D/1

# evaluate + figures
python run_eval.py
```

Checkpoints: `Question2/checkpoints/`  
Results: `Question2/results/`  
Predictions written before experiments: `Question2/predictions_before_experiments.md`

Final model: symmetrised Model D (dilated ConvLSTM, speed-aug training).

## Report
`PA2_Report.tex` / `PA2_Report.pdf`
