#!/bin/bash

python -m projektpraktikum_i.discretization.experiments_it plot-error \
    --n-start 1 \
    --n-stop 7 \
    --n-num 20 \
    --eps 10e-6 \
    --omega 1.5 \
    --save-to error_plot.pdf
    
python -m projektpraktikum_i.discretization.experiments_it plot-error \
    --n-start 1 \
    --n-stop 7 \
    --n-num 20 \
    --eps 10e-10 \
    --omega 1.5 \
    --save-to error_plot_better_eps.pdf
        
python -m projektpraktikum_i.discretization.experiments_it plot-optimal-eps \
    --n-start 3 \
    --n-stop 7 \
    --n-num 20 \
    --omega 1.5 \
    --save-to optimal_eps_plot.pdf
    
python -m projektpraktikum_i.discretization.experiments_it plot-optimal-omega \
    --n-start 20 \
    --n-stop 320 \
    --max-iter 32 \
    --save-to optimal_omega_plot.pdf
    
python -m projektpraktikum_i.discretization.experiments_it plot-compare \
    --n-start 3 \
    --n-stop 8 \
    --fixed-eps 1e-3 \
    --save-to comparison_plot.pdf
    
python -m projektpraktikum_i.discretization.experiments_it plot-time-comparison \
    --num-runs 10 \
    --n-start 3 \
    --n-stop 6 \
    --save-to runtime_comparison_plot.pdf