# AI651 Assignment 2

Roll: 25280044  
GitHub: https://github.com/junaid074/DL4STG-PA2

## What is where
- `PA2_Report.pdf` / `PA2_Report.tex` — both questions
- `Question1/` — optional companion notebook (ungraded)
- `Question2/` — Moving MNIST ConvLSTM code and results

## Question 2 quick start
```bash
cd Question2
python -m q2.train --model A --seed 0 --epochs 30 --root .
# same for A/1, B/0, B/1, C/0, C/1, D/0, D/1
python run_eval.py
```

Checkpoints in `Question2/checkpoints/`, figures in `Question2/results/`.  
Q2.5 and Q2.11 notes (written before those experiments): `Question2/predictions_before_experiments.md`.

Final model in the report: symmetrised D.
