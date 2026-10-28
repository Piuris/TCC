#!/usr/bin/env bash
set -euo pipefail
export LOKY_MAX_CPU_COUNT=$(nproc) PYTHONWARNINGS=ignore
PY=.venv/Scripts/python
FRAC=0.2

S="--dataset paysim_sem_saldo --sample-frac $FRAC"
$PY -m fraude eda $S
$PY -m fraude experiment $S --n-iter 10 --no-resume
$PY -m fraude experiment $S --models xgboost lightgbm --n-iter 50 --out-name experiments_busca50 --no-resume
$PY -m fraude ensemble $S --no-resume
$PY -m fraude compare $S --repeats 3
$PY -m fraude train-final $S
$PY -m fraude explain $S
$PY -m fraude drift $S --windows 6 --label-delay 0
$PY -m fraude drift $S --windows 6 --label-delay 1

S="--dataset paysim --sample-frac $FRAC"
$PY -m fraude eda $S
$PY -m fraude experiment $S --n-iter 10 --no-resume
$PY -m fraude train-final $S
$PY -m fraude explain $S
$PY -m fraude drift $S --windows 6 --label-delay 0

echo "PAYSIM CONCLUIDO"
