#!/usr/bin/env bash
set -e
bash download_data.sh
python run.py   # re-run until it prints results (it checkpoints generations)
python make_charts.py
