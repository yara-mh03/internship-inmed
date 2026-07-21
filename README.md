# internship-inmed

## Poisson HMM State Analysis of Neural Population Activity

Fits Hidden Markov Models (HMM) with Poisson emissions to spike-count data from a single recording session, 
to identify discrete population "states" across trials aligned to task events (sample, delay, go-cue).

## What it does
1. Loads spike data and behavioral trial info from a `.mat` file (one session).
2. Selects trials by outcome type (correct/incorrect, left/right) and stimulation condition.
3. Bins spikes into a trial × time-bin × neuron population matrix.
4. Fits Poisson HMMs for k = 2–10 states (via the [ssm](https://github.com/lindermanlab/ssm) package), each with multiple random restarts.
5. Selects the best number of states k using:
   - 5-fold cross-validated held-out log-likelihood
   - Split-half stability of emission profiles (cosine similarity)
   - Marginal log-likelihood gain / elbow detection
   - Dwell-time, fragmentation, and effective-state diagnostics

## How to run
Update the `file_name` and `output_dir` paths at the top of the script before running — they're currently hardcoded to a local machine.

## Requirements
- Python 3.x
- numpy, scipy, matplotlib, scikit-learn
- [ssm](https://github.com/lindermanlab/ssm) (Linderman lab state-space models package)

## Input data format
Expects a `.mat` file with a `unit` struct array, where each unit has `SpikeTimes`, `Trial_idx_of_spike`, `Trial_info.Trial_range_to_analyze`, and a `Behavior` struct with trial type, stim, and event-onset (cue/sample/delay) vectors.

## Output
PNG plots saved to `output_dir`, one set per session, covering model selection diagnostics and state/emission visualizations across k = 2–10.
