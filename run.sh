#!/bin/bash

## Model Training
python main.py --lamb 0.01 --num_factors 10 --datadir data_fund --outdir output_fund/multi-task-lamb0.01-n10
python main.py --lamb 0.01 --num_factors 20 --datadir data_fund --outdir output_fund/multi-task-lamb0.01-n20

# model with or without gat
for NUM in 10 12 14 16 18
do
    python main.py --lamb 0.01 --num_factors $NUM --datadir data_fund --outdir output_fund/multi-task-lamb0.01-n${NUM}-nogat --disable_gat
    python main.py --lamb 0.01 --num_factors $NUM --datadir data_fund --outdir output_fund/multi-task-lamb0.01-n${NUM}
done

# regularization
for LAMB in 0.0001 0.001 0.01 0.1 1.0
do
    python main.py --lamb $LAMB --num_factors 10 --datadir data_fund --outdir output_fund/multi-task-lamb${LAMB}-n10
done

# single-task
python main.py --lamb 0.01 --num_factors 10 --datadir data_fund --outdir output_fund/single-task-lamb0.01-n10 --next_ret_only

## Covariance Estimation
python est_factor_ret.py
python est_factor_cov.py
python est_resid_vola.py

for OUTDIR in output_fund/*
do
    echo $OUTDIR
    python est_factor_ret.py --outdir $OUTDIR --replace
    python est_factor_cov.py --outdir $OUTDIR
    python est_resid_vola.py --outdir $OUTDIR --replace
done
