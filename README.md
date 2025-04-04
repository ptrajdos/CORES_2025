# Cascade of one-class classifier ensemble and dynamic naive Bayes classifier applied to the myoelectric-based upper limb prosthesis control with contaminated channels detection

## System Requirements

Requires: __Python==3.9.7__
All required packages may be installed via __pip__.
Tested on: __Ubuntu Linux 22.04__, __macOS  Sequoia 15.3.1__

## Setup
To download test data, create the virtual envirionment, and install required packages type:
```
make create_env
```
To clean virtual environment type:
```
make clean
```

The experimental results will be located in: __./experiments\_results__

## Experiments

To run all experiments type:
```
make run
```

Results will be placed in: __./experiments\_results/results\_channel\_nb\_weights__.
Directory structure:

  + Single set
    + *A[1-9]_Force_Exp_low_windowed.pickle* -- raw results (for a single set) as numpy arrays
    + *A[1-9]_Force_Exp_low_windowed.pdf* -- raw results (for a single set) as boxplots. It include results for all quality criteria, all classifiers, all SNR levels, and signal spoilers.
    + *A[1-9]_Force_Exp_low_windowed.md* -- raw results (for a single set) in tabular form (average and standard deviation). It include results for all quality criteria, all classifiers, all SNR levels.
    + *A[1-9]_Force_Exp_low_windowed_snr_m2.pdf* -- trends in quality criteria (median, Q1, Q3) over all SNR levels. Results for all quality criteria, signal spoilers, and base one-class classifiers.
  + Ranking over all sets
    + *ALL_snr_m2_ranks.pdf" -- Average ranks plots for different SNR values, criteria, ensemble sizes.
    + *ALL_snr_m2_ranks.md" -- Average ranks tables and statistical tests for different SNR values, criteria, ensemble sizes. 
