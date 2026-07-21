import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import ssm
from sklearn.model_selection import KFold, train_test_split
from scipy.optimize import linear_sum_assignment
from scipy.stats import kruskal, mannwhitneyu

np.random.seed(42)

# ----------------
# Load one session
# ----------------

file_name = "/HI125_022817_units.mat"
session_name = os.path.splitext(os.path.basename(file_name))[0]  # e.g. "HI127_031517_units"
plot_counter = 0

output_dir = r""
os.makedirs(output_dir, exist_ok=True)   # creates the folder if it doesn't exist

data = scipy.io.loadmat(file_name, struct_as_record=False, squeeze_me=True)
units = data["unit"]

print("Number of units:", len(units))

# ----------------
# Choose condition
# ----------------

trial_codes_to_use = [1, 2, 3, 4]  # 1=correct right, 2=correct left, 3=incorrect right, 4=incorrect left
stim_condition = "no stim"          # "no stim", "stim", or "all"


# ----------------
# HMM parameters
# ----------------

bin_size = 0.1   # 100 ms bins
t_start  = -3.4  # seconds relative to cue onset
t_end    = 1.0   # 1 second after go-cue to capture the response period

# ----------------
# Get behavior info
# ----------------

u0 = units[0]

trial_type  = u0.Behavior.Trial_types_of_response_vector
photo_stim  = u0.Behavior.stim_trial_vector

cue_onset    = u0.Behavior.Cue_start
delay_onset  = u0.Behavior.Delay_start
sample_onset = u0.Behavior.Sample_start
sample_onset_rel = float(np.median(sample_onset - cue_onset))  # ≈ -3.15s
delay_onset_rel  = float(np.median(delay_onset  - cue_onset))  # ≈ -2.00s

# We use the intersection of valid trial ranges across all units,
# because different units can have slightly different recording start/end points.
# Using only u0's range risks including trials where some units have no data.
first_trial = max(int(u.Trial_info.Trial_range_to_analyze[0]) for u in units)
last_trial  = min(int(u.Trial_info.Trial_range_to_analyze[1]) for u in units)

# ----------------
# Select trials
# ----------------

selected_trials      = []
selected_trial_types = []
selected_stim_labels = []

for tr in range(first_trial, last_trial + 1):

    if trial_type[tr - 1] not in trial_codes_to_use:
        continue

    if stim_condition == "no stim" and photo_stim[tr - 1] != 0:
        continue

    if stim_condition == "stim" and photo_stim[tr - 1] not in [2, 3, 4]:
        continue

    selected_trials.append(tr)
    selected_trial_types.append(trial_type[tr - 1])

    if photo_stim[tr - 1] == 0:
        selected_stim_labels.append("no stim")
    elif photo_stim[tr - 1] in [2, 3, 4]:
        selected_stim_labels.append("stim")
    else:
        selected_stim_labels.append("other")

selected_trial_types = np.array(selected_trial_types)
selected_stim_labels = np.array(selected_stim_labels)

print("Selected trials:", len(selected_trials))
print("Trial type counts:", {tc: np.sum(selected_trial_types == tc) for tc in trial_codes_to_use})
print("Stim counts:", {s: np.sum(selected_stim_labels == s) for s in np.unique(selected_stim_labels)})

# ----------------
# Build population matrix
# ----------------
# X shape: (n_trials * n_bins, n_units)
# Each block of n_bins rows = one trial

bin_edges = np.arange(t_start, t_end + bin_size, bin_size)
n_bins  = len(bin_edges) - 1
n_units = len(units)

all_trials_data = []

for tr in selected_trials:
    trial_matrix = np.zeros((n_bins, n_units))

    for unit_idx, u in enumerate(units):
        spike_times = u.SpikeTimes
        trial_idx   = u.Trial_idx_of_spike

        # align spikes to cue onset
        spikes = spike_times[trial_idx == tr] - cue_onset[tr - 1]

        counts, _ = np.histogram(spikes, bins=bin_edges)
        trial_matrix[:, unit_idx] = counts

    all_trials_data.append(trial_matrix)

X = np.vstack(all_trials_data)

print("Population matrix shape:", X.shape)

# ----------------
# Shared setup — trial sorting and type annotations
# ----------------

type_labels = {1: "Correct R", 2: "Correct L", 3: "Incorrect R", 4: "Incorrect L"}
type_colors = {1: "#2166ac", 2: "#d6604d", 3: "#4dac26",  4: "#8B008B"}

sort_idx     = np.argsort(selected_trial_types, kind="stable")
sorted_types = selected_trial_types[sort_idx]
stack        = np.array(all_trials_data)   # (n_trials, n_bins, n_units)
stack_sorted = stack[sort_idx]

boundaries = []
for tc in [1, 2, 3, 4]:
    idx = np.where(sorted_types == tc)[0]
    if len(idx):
        boundaries.append((idx[0], idx[-1] + 1, tc))

patches        = [Patch(color=type_colors[tc], label=type_labels[tc]) for tc in [1, 2, 3, 4]]
n_trials_total = len(selected_trials)

# ----------------
# Fit Poisson HMM — SSM (Linderman lab)
# ----------------
# SSM takes a list of 2D arrays (one per trial), each of shape (n_bins, n_units).
# This is cleaner than hmmlearn's concatenated matrix + lengths approach.

X_list = [trial.astype(int) for trial in all_trials_data]   # SSM requires integer counts

# ----------------------------------------------------------------
# Model selection: fit k = 2 to 10, plot diagnostics
# ----------------------------------------------------------------
# For each k we store:
#   - decoded state matrix (trials × bins)
#   - emission rates lambda (k × n_units)
#   - transition matrix (k × k)
# Then we plot:
#   (1) State raster for each k (sorted by trial type)
#   (2) Emission rates for each k
#   (3) Transition matrix for each k

k_range = range(2, 11)

# ----------------------------------------------------------------
# Multi-restart EM fitting helper
# ----------------------------------------------------------------

N_RESTARTS = 5       # used for the main k=2..10 fits — these drive all downstream plots/diagnostics, worth the cost
N_RESTARTS_CV = 2    # used inside CV loops only, 2 is enough to avoid a bad local optimum 


def fit_hmm_with_restarts(data, k, n_units, n_restarts=N_RESTARTS, num_iters=100):
    """Fit a Poisson HMM with several random restarts, return the model
    with the highest training log-likelihood."""
    best_model = None
    best_ll = -np.inf
    for seed in range(n_restarts):
        np.random.seed(seed)  # reseed before each restart for reproducibility
        m_try = ssm.HMM(K=k, D=n_units, observations="poisson")
        m_try.fit(data, method="em", num_iters=num_iters, verbose=False)
        train_ll = sum(m_try.log_likelihood(trial) for trial in data)
        if train_ll > best_ll:
            best_ll = train_ll
            best_model = m_try
    return best_model, best_ll


print("\n--- Fitting models k=2 to 10 ---")

all_models   = {}   # fitted ssm.HMM object
all_states   = {}   # state_matrix (n_trials × n_bins)
all_lambdas  = {}   # lambdas array (k × n_units)
all_transmats = {}  # transition matrix (k × k)

for k in k_range:
    print(f"  Fitting k={k} ({N_RESTARTS} restarts)...", end=" ", flush=True)
    m, best_train_ll = fit_hmm_with_restarts(X_list, k, n_units, n_restarts=N_RESTARTS, num_iters=100)

    hs = np.concatenate([m.most_likely_states(trial) for trial in X_list])
    all_models[k]    = m
    all_states[k]    = hs.reshape(len(selected_trials), n_bins)
    all_lambdas[k]   = np.exp(m.observations.log_lambdas)
    all_transmats[k] = m.transitions.transition_matrix

    print(f"best train LL={best_train_ll:.2f}")

# ----------------------------------------------------------------
# Cross-validation: held-out log-likelihood for k = 2 to 10
# ----------------------------------------------------------------
# Standard k-fold CV, trials shuffled before splitting.
#
# How the shuffling works: 
# X_list is a plain Python list of whole trial matrices, each shaped (n_bins, n_units).
# KFold.split(X_list) only ever sees len(X_list) = n_trials — it has no
# visibility into what's inside each element, so shuffling can only permute
# WHICH whole trials land in train vs test. It cannot touch bin order or
# neuron order within a trial, since it never looks below the list's top
# level. 
#
# fit_hmm_with_restarts is reused inside the CV loop so CV models are
# fitted consistently with the main fitting loop. num_iters=50 (vs 100 in
# the main loop) and N_RESTARTS_CV (vs N_RESTARTS) keep runtime manageable

n_folds = 5

kf_random = KFold(n_splits=n_folds, shuffle=True, random_state=42)

print(f"\n--- Random {n_folds}-fold CV across k=2 to 10 ---")

cv_mean = {}
cv_std  = {}

for k in k_range:
    fold_ll = []
    for train_idx, test_idx in kf_random.split(X_list):
        train_data = [X_list[i] for i in train_idx]
        test_data  = [X_list[i] for i in test_idx]
        m_cv, _ = fit_hmm_with_restarts(
            train_data, k, n_units, n_restarts=N_RESTARTS_CV, num_iters=50
        )
        test_ll   = sum(m_cv.log_likelihood(trial) for trial in test_data)
        test_bins = sum(trial.shape[0] for trial in test_data)
        fold_ll.append(test_ll / test_bins)
    cv_mean[k] = float(np.mean(fold_ll))
    cv_std[k]  = float(np.std(fold_ll))
    print(f"  k={k}  held-out LL/bin = {cv_mean[k]:.4f} ± {cv_std[k]:.4f}")

best_k = max(cv_mean, key=cv_mean.get)
print(f"Best k (CV): {best_k}")

# ---- Plot: CV log-likelihood vs k ----

plt.figure(figsize=(8, 4.5))
plt.errorbar(list(cv_mean.keys()), list(cv_mean.values()),
             yerr=list(cv_std.values()), marker="o", color="seagreen",
             linewidth=2, markersize=7, capsize=4)
plt.axvline(best_k, color="red", linestyle="--", linewidth=1.5, label=f"Best k = {best_k}")
plt.xlabel("Number of HMM states (k)", fontsize=12)
plt.ylabel("Held-out log-likelihood per bin", fontsize=12)
plt.title(f"Random {n_folds}-fold cross-validation\n(higher = better generalization)", fontsize=12)
plt.xticks(list(k_range))
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"plot_{session_name}_{plot_counter}.png"), dpi=120, bbox_inches="tight")
plt.close("all")
plot_counter += 1

# ---- Significance check ----

print("\n--- Significance of consecutive k improvements ---")
print(f"{'k -> k+1':12s} {'Δ mean LL/bin':>15s} {'combined std':>14s} {'significant?':>13s}")

last_significant_k = k_range.start
for k in list(k_range)[:-1]:
    k_next       = k + 1
    delta        = cv_mean[k_next] - cv_mean[k]
    combined_std = np.sqrt(cv_std[k]**2 + cv_std[k_next]**2)
    is_sig       = abs(delta) > combined_std
    if is_sig:
        last_significant_k = k_next
    print(f"  {k:2d} -> {k_next:2d}   {delta:15.4f} {combined_std:14.4f} {'yes' if is_sig else 'no':>13s}")

print(f"\nLast k significant: k={last_significant_k}")

# ----------------------------------------------------------------
# Delta log-likelihood: LL(k) - LL(k-1), with elbow detection
# ----------------------------------------------------------------

k_list = list(k_range)
delta_ll = [np.nan]  # no delta defined for the first k
delta_ll_err = [np.nan]  # propagated uncertainty of the difference

for i in range(1, len(k_list)):
    k_curr, k_prev = k_list[i], k_list[i - 1]
    delta_ll.append(cv_mean[k_curr] - cv_mean[k_prev])
    # uncertainty of a difference of two independent estimates adds in quadrature
    delta_ll_err.append(float(np.sqrt(cv_std[k_curr] ** 2 + cv_std[k_prev] ** 2)))

# Elbow candidate (STRICT): smallest k (from the 2nd delta onward) after
# which |delta_ll| stays within its own uncertainty for ALL subsequent k
# i.e. the gain is no longer reliably distinguishable from fold noise,
elbow_k_strict = None
for i in range(1, len(k_list)):
    remaining_deltas = delta_ll[i:]
    remaining_errs = delta_ll_err[i:]
    if all(abs(d) <= e for d, e in zip(remaining_deltas, remaining_errs)):
        elbow_k_strict = k_list[i - 1]  # last k BEFORE the gains become negligible
        break

# Elbow candidate (RELAXED): smallest k after which the gain stays within
# noise for the next LOOKAHEAD steps (rather than all remaining k forever).
LOOKAHEAD = 2
elbow_k_relaxed = None
for i in range(1, len(k_list)):
    window_deltas = delta_ll[i:i + LOOKAHEAD]
    window_errs = delta_ll_err[i:i + LOOKAHEAD]
    if len(window_deltas) > 0 and all(abs(d) <= e for d, e in zip(window_deltas, window_errs)):
        elbow_k_relaxed = k_list[i - 1]
        break

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(k_list[1:], delta_ll[1:], yerr=delta_ll_err[1:],
            marker="o", color="steelblue", linewidth=2, markersize=7, capsize=4,
            label=r"$\Delta$LL$(k) = $LL$(k) - $LL$(k-1)$")
ax.axhline(0, color="gray", linestyle=":", linewidth=1)
if elbow_k_strict is not None:
    ax.axvline(elbow_k_strict, color="red", linestyle="--", linewidth=1.5,
               label=f"Strict elbow (holds for all k\u2265): k={elbow_k_strict}")
if elbow_k_relaxed is not None:
    ax.axvline(elbow_k_relaxed, color="darkorange", linestyle="--", linewidth=1.5,
               label=f"Relaxed elbow (holds for next {LOOKAHEAD}): k={elbow_k_relaxed}")
ax.set_xlabel("Number of HMM states (k)", fontsize=12)
ax.set_ylabel(r"$\Delta$ Held-out log-likelihood per bin", fontsize=12)
ax.set_title("Marginal gain in held-out log-likelihood per added state\n"
             "(elbow = smallest k after which gains stay within fold noise)", fontsize=12)
ax.set_xticks(k_list[1:])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"plot_{session_name}_{plot_counter}.png"), dpi=120, bbox_inches="tight")
plt.close("all")
plot_counter += 1

print("\n--- Delta log-likelihood (marginal gain per added state) ---")
print(f"{'k':>3} | {'LL(k)-LL(k-1)':>14} | {'fold-noise (±)':>14}")
print("-" * 38)
for i in range(1, len(k_list)):
    print(f"{k_list[i]:>3} | {delta_ll[i]:>14.4f} | {delta_ll_err[i]:>14.4f}")
print()
if elbow_k_strict is not None:
    print(f"Strict elbow (gain stays within noise for ALL subsequent k): k = {elbow_k_strict}")
else:
    print("Strict elbow: none found through k=10 (gain exceeds noise again at some later k).")
if elbow_k_relaxed is not None:
    print(f"Relaxed elbow (gain stays within noise for next {LOOKAHEAD} k values): k = {elbow_k_relaxed}")
else:
    print(f"Relaxed elbow: none found through k=10 even allowing a {LOOKAHEAD}-step lookahead.")

# ----------------------------------------------------------------
# Diagnostics: dwell time, fragmentation, effective # of states
# ----------------------------------------------------------------

MIN_DWELL_BINS = 3        # a "real" visit must last at least this many bins
WEAK_SELF_TRANSITION_THRESH = 0.85  # self-transition below this = flicker-prone state
EMISSION_SIMILARITY_THRESH = 0.97   # cosine similarity above this = duplicate states


def get_runs(state_seq):
    """Return list of (state, run_length_in_bins) for one 1D sequence."""
    runs = []
    if len(state_seq) == 0:
        return runs
    current = state_seq[0]
    length = 1
    for s in state_seq[1:]:
        if s == current:
            length += 1
        else:
            runs.append((current, length))
            current = s
            length = 1
    runs.append((current, length))
    return runs


def dwell_time_stats(state_matrix_k, k, bin_size_s, min_dwell_bins=MIN_DWELL_BINS):
    """state_matrix_k: (n_trials, n_bins) decoded states for one k."""
    n_trials_k = state_matrix_k.shape[0]
    all_runs = []
    for trial_idx in range(n_trials_k):
        all_runs.extend(get_runs(state_matrix_k[trial_idx]))

    runs_by_state = {s: [] for s in range(k)}
    for state, length in all_runs:
        runs_by_state[state].append(length)

    mean_dwell_bins = {
        s: (np.mean(lengths) if lengths else 0.0)
        for s, lengths in runs_by_state.items()
    }

    all_run_lengths = np.array([length for _, length in all_runs])
    frag_fraction_overall = (
        np.mean(all_run_lengths < min_dwell_bins) if len(all_run_lengths) else np.nan
    )

    total_bins = state_matrix_k.size
    occupancy_frac = {s: np.sum(state_matrix_k == s) / total_bins for s in range(k)}

    n_states_used = sum(1 for s in range(k) if occupancy_frac[s] > 0.01)
    n_states_stable = sum(
        1 for s in range(k)
        if occupancy_frac[s] > 0.01 and mean_dwell_bins[s] >= min_dwell_bins
    )

    return {
        "mean_dwell_bins": mean_dwell_bins,
        "mean_dwell_ms": {s: v * bin_size_s * 1000 for s, v in mean_dwell_bins.items()},
        "occupancy_frac": occupancy_frac,
        "frag_fraction_overall": frag_fraction_overall,
        "n_states_used": n_states_used,
        "n_states_stable": n_states_stable,
    }


def transmat_diagnostics(transmat_k, weak_thresh=WEAK_SELF_TRANSITION_THRESH):
    diag = np.diag(transmat_k)
    n_weak = int(np.sum(diag < weak_thresh))
    return {
        "self_transition_mean": float(np.mean(diag)),
        "self_transition_min": float(np.min(diag)),
        "n_weak_self_transition_states": n_weak,
    }


def effective_n_states(lambdas_k, corr_thresh=EMISSION_SIMILARITY_THRESH):
    """Merge states whose emission-rate vectors are near-duplicates
    (cosine similarity > corr_thresh) and return the deduplicated count."""
    k = lambdas_k.shape[0]
    norm = lambdas_k / (np.linalg.norm(lambdas_k, axis=1, keepdims=True) + 1e-12)
    sim = norm @ norm.T
    np.fill_diagonal(sim, 0)

    parent = list(range(k))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(k):
        for j in range(i + 1, k):
            if sim[i, j] > corr_thresh:
                union(i, j)

    n_effective = len(set(find(x) for x in range(k)))
    max_pairwise_sim = float(np.max(sim)) if k > 1 else 0.0
    return n_effective, max_pairwise_sim


print(f"\n--- Diagnostics (dwell time, fragmentation, effective k) across k=2 to 10 ---")

diag_summary = {}
for k in k_range:
    dwell = dwell_time_stats(all_states[k], k, bin_size_s=bin_size)
    trans = transmat_diagnostics(all_transmats[k])
    n_eff, max_sim = effective_n_states(all_lambdas[k])

    diag_summary[k] = {
        "n_states_used": dwell["n_states_used"],
        "n_states_stable": dwell["n_states_stable"],
        "n_effective_states": n_eff,
        "max_pairwise_emission_similarity": max_sim,
        "frag_fraction_overall": dwell["frag_fraction_overall"],
        "mean_self_transition": trans["self_transition_mean"],
        "min_self_transition": trans["self_transition_min"],
        "n_weak_self_transition_states": trans["n_weak_self_transition_states"],
        "mean_dwell_ms_overall": float(np.mean(list(dwell["mean_dwell_ms"].values()))),
    }

# ---- Print compact summary table ----
header = (f"{'k':>3} | {'used':>4} | {'eff':>4} | {'stable':>6} | {'frag%':>6} | "
          f"{'meanSelfTr':>10} | {'minSelfTr':>9} | {'weakTr':>6} | {'dwell(ms)':>9} | {'CV_LL/bin':>10}")
print(header)
print("-" * len(header))
for k in k_range:
    s = diag_summary[k]
    print(f"{k:>3} | {s['n_states_used']:>4} | {s['n_effective_states']:>4} | "
          f"{s['n_states_stable']:>6} | {100*s['frag_fraction_overall']:>5.1f}% | "
          f"{s['mean_self_transition']:>10.3f} | {s['min_self_transition']:>9.3f} | "
          f"{s['n_weak_self_transition_states']:>6} | {s['mean_dwell_ms_overall']:>9.1f} | "
          f"{cv_mean[k]:>10.4f}")

# ---- Plot: CV log-likelihood next to fragmentation / effective-k diagnostics ----

ks = list(k_range)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

ax = axes[0, 0]
means = [cv_mean[k] for k in ks]
stds = [cv_std[k] for k in ks]
ax.errorbar(ks, means, yerr=stds, marker="o", color="seagreen",
            linewidth=2, markersize=6, capsize=4)
ax.set_xlabel("k")
ax.set_ylabel("Held-out LL / bin")
ax.set_title("Cross-validated log-likelihood\n(rarely flattens by itself)")
ax.grid(alpha=0.3)

ax = axes[0, 1]
n_eff_list = [diag_summary[k]["n_effective_states"] for k in ks]
n_used_list = [diag_summary[k]["n_states_used"] for k in ks]
n_stable_list = [diag_summary[k]["n_states_stable"] for k in ks]
ax.plot(ks, ks, "k--", alpha=0.4, label="k (nominal)")
ax.plot(ks, n_used_list, "o-", label="states occupied (>1% bins)")
ax.plot(ks, n_eff_list, "s-", label="effective states (dedup by emission sim.)")
ax.plot(ks, n_stable_list, "^-", label="stable states (occupied & dwell \u2265 min)")
ax.set_xlabel("k (nominal)")
ax.set_ylabel("count")
ax.set_title("Where does nominal k diverge from real structure?")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 0]
frag_list = [diag_summary[k]["frag_fraction_overall"] for k in ks]
ax2 = ax.twinx()
ax.plot(ks, frag_list, "o-", color="crimson", label="fraction of runs < min dwell")
ax2.plot(ks, [diag_summary[k]["n_weak_self_transition_states"] for k in ks],
         "s-", color="navy", label="# states w/ self-transition < 0.85")
ax.set_xlabel("k")
ax.set_ylabel("fragmented-run fraction", color="crimson")
ax2.set_ylabel("# flicker-prone states", color="navy")
ax.set_title("Fragmentation & flicker-prone states vs k")
ax.grid(alpha=0.3)

ax = axes[1, 1]
dwell_ms_list = [diag_summary[k]["mean_dwell_ms_overall"] for k in ks]
ax.plot(ks, dwell_ms_list, "o-", color="darkorange")
ax.set_xlabel("k")
ax.set_ylabel("mean dwell time (ms)")
ax.set_title("Mean state dwell time vs k\n(should stay well above bin size if real)")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"plot_{session_name}_{plot_counter}.png"), dpi=120, bbox_inches="tight")
plt.close("all")
plot_counter += 1


# ----------------------------------------------------------------
# Statistical state merging (Mazzucato et al. 2015 J Neurosci;
# reused in Recanatesi et al. 2022 Neuron, STAR Methods "Single
# neuron multistability")
# ----------------------------------------------------------------
# The papers' method is per-NEURON: for each neuron, test via
# Kruskal-Wallis whether its firing rate differs across states at
# all, then (if so) do pairwise post-hoc comparisons with Bonferroni
# correction. Any pair of states NOT significantly different for
# that neuron are candidates to be treated as the same firing level
# for that neuron. This is what the papers use to count each
# neuron's number of distinguishable firing rates ("multistability").
#
# It does not, by itself, say "merge whole population states m and
# n into one HMM state" -- that requires aggregating the per-neuron
# result across the population. This aggregation step (the
# POP_MERGE_FRAC rule below) is our own extension, not something
# specified in the papers -- flagged clearly so it isn't mistaken
# for their methodology.
#
# We use the model's posterior state probabilities (not just the
# hard Viterbi labels) to decide which bins "belong" to a state, at
# the same 80% confidence threshold the papers use for calling a
# state detected.

POSTERIOR_THRESH = 0.8   # matches the papers' 80% posterior-probability threshold
MIN_SAMPLES_PER_STATE = 5   # minimum # bins needed to trust a per-neuron test for a state
ALPHA = 0.05
POP_MERGE_FRAC = 0.9     # merge states m,n if >= 90% of testable neurons show no
                          # significant firing-rate difference between them (our choice --
                          # not specified in the papers, which only test per-neuron)
POP_MERGE_MIN_NEURONS = max(3, n_units // 2)   # require enough neurons tested before trusting the fraction

# k_report was already set above (= last_significant_k) during the split-half check;
# reused here so the merge is applied to the same currently-reported k


def per_neuron_state_samples(model, data, k, n_units, thresh=POSTERIOR_THRESH):
    """For each neuron and state, collect the spike counts of bins where
    that state's posterior probability exceeds `thresh` (papers' Eq. 3
    logic, using the model's own posteriors rather than re-deriving them)."""
    samples = {i: {m: [] for m in range(k)} for i in range(n_units)}
    for trial in data:
        Ez, _, _ = model.expected_states(trial)   # (T, k) posterior probabilities
        for m in range(k):
            mask = Ez[:, m] >= thresh
            if not np.any(mask):
                continue
            bins_spikes = trial[mask, :]
            for i in range(n_units):
                samples[i][m].extend(bins_spikes[:, i].tolist())
    return samples


def neuron_state_merge_matrix(samples_i, k, alpha=ALPHA, min_samples=MIN_SAMPLES_PER_STATE):
    """k x k boolean matrix for one neuron: True where a pair of states is
    statistically indistinguishable in firing rate (Kruskal-Wallis omnibus,
    then Bonferroni-corrected pairwise Mann-Whitney post-hoc -- the
    nonparametric-rank equivalent of the papers' procedure). Diagonal is
    always True; off-diagonal defaults to False (not merged) unless there
    is a valid test that failed to find a difference -- absence of
    evidence is not treated as evidence of sameness."""
    merge = np.zeros((k, k), dtype=bool)
    np.fill_diagonal(merge, True)
    valid = [m for m in range(k) if len(samples_i[m]) >= min_samples]
    if len(valid) < 2:
        return merge
    try:
        _, p_omni = kruskal(*[samples_i[m] for m in valid])
    except ValueError:
        p_omni = 1.0
    if p_omni >= alpha:
        for a in range(len(valid)):
            for b in range(a + 1, len(valid)):
                merge[valid[a], valid[b]] = merge[valid[b], valid[a]] = True
        return merge
    n_pairs = len(valid) * (len(valid) - 1) // 2
    alpha_bonf = alpha / n_pairs
    for a in range(len(valid)):
        for b in range(a + 1, len(valid)):
            m_a, m_b = valid[a], valid[b]
            try:
                _, p = mannwhitneyu(samples_i[m_a], samples_i[m_b], alternative="two-sided")
            except ValueError:
                p = 1.0
            if p >= alpha_bonf:
                merge[m_a, m_b] = merge[m_b, m_a] = True
    return merge


def population_merge_fraction(model, data, k, n_units):
    """For every pair of states, the fraction of neurons (among those with
    enough data to test) for which that pair is statistically indistinguishable."""
    samples = per_neuron_state_samples(model, data, k, n_units)
    frac = np.zeros((k, k))
    counted = np.zeros((k, k))
    for i in range(n_units):
        merge_i = neuron_state_merge_matrix(samples[i], k)
        valid_i = [m for m in range(k) if len(samples[i][m]) >= MIN_SAMPLES_PER_STATE]
        for a in range(len(valid_i)):
            for b in range(a + 1, len(valid_i)):
                m_a, m_b = valid_i[a], valid_i[b]
                counted[m_a, m_b] += 1
                counted[m_b, m_a] += 1
                if merge_i[m_a, m_b]:
                    frac[m_a, m_b] += 1
                    frac[m_b, m_a] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(counted > 0, frac / counted, 0.0)
    return frac, counted


def merge_states_from_fraction(frac, counted, k, frac_thresh=POP_MERGE_FRAC, min_neurons=POP_MERGE_MIN_NEURONS):
    """Union-find over state pairs that clear both the fraction and
    minimum-neurons-tested thresholds. Returns old_state -> new_state map."""
    parent = list(range(k))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for m in range(k):
        for n in range(m + 1, k):
            if counted[m, n] >= min_neurons and frac[m, n] >= frac_thresh:
                union(m, n)

    roots = sorted(set(find(m) for m in range(k)))
    root_to_new = {r: idx for idx, r in enumerate(roots)}
    return {m: root_to_new[find(m)] for m in range(k)}


print(f"\n--- Statistical state merging (k={k_report}) ---")

frac_matrix, counted_matrix = population_merge_fraction(
    all_models[k_report], X_list, k_report, n_units
)
merge_map = merge_states_from_fraction(frac_matrix, counted_matrix, k_report)
k_merged = len(set(merge_map.values()))

print(f"States before merging: {k_report}  ->  after merging: {k_merged}")
for old_state, new_state in sorted(merge_map.items()):
    print(f"  original state {old_state}  ->  merged state {new_state}")

# ---- Plot: pairwise merge-fraction heatmap ----

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(frac_matrix, cmap="RdYlGn", vmin=0, vmax=1)
ax.set_xticks(range(k_report)); ax.set_yticks(range(k_report))
ax.set_xlabel("State"); ax.set_ylabel("State")
ax.set_title(f"Fraction of neurons unable to distinguish\nstate pairs (k={k_report})\n"
             f"green = candidates for merging (\u2265{POP_MERGE_FRAC:.0%} threshold)", fontsize=11)
for r in range(k_report):
    for c in range(k_report):
        if r != c:
            ax.text(c, r, f"{frac_matrix[r, c]:.2f}", ha="center", va="center", fontsize=8)
plt.colorbar(im, ax=ax, fraction=0.046, label="fraction indistinguishable")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"plot_{session_name}_{plot_counter}.png"), dpi=120, bbox_inches="tight")
plt.close("all")
plot_counter += 1

# ---- Build merged state raster and merged emission profiles ----

merged_states = np.vectorize(merge_map.get)(all_states[k_report])   # relabel, same (n_trials, n_bins) shape

# merged emission profile = occupancy-weighted average of the original
# states' lambdas that were folded into each merged state
occupancy = {m: np.sum(all_states[k_report] == m) for m in range(k_report)}
merged_lambdas = np.zeros((k_merged, n_units))
merged_weight = np.zeros(k_merged)
for old_state, new_state in merge_map.items():
    w = occupancy[old_state]
    merged_lambdas[new_state] += all_lambdas[k_report][old_state] * w
    merged_weight[new_state] += w
merged_lambdas /= merged_weight[:, None]

# ---- Plot: merged state raster (same layout as the per-k rasters below) ----

sm = merged_states[sort_idx]

fig, ax = plt.subplots(figsize=(11, 6))
im = ax.imshow(
    sm, aspect="auto", interpolation="nearest",
    extent=[t_start, t_end, len(selected_trials), 0],
    cmap="tab10" if k_merged <= 10 else "viridis",
    vmin=0, vmax=max(k_merged - 1, 1)
)
ax.axvline(0, color="black", linestyle="--", linewidth=2)
ax.axvline(delay_onset_rel, color="black", linestyle="--", linewidth=2)
ax.axvline(sample_onset_rel, color="black", linestyle="--", linewidth=2)
for start, end, tc in boundaries:
    if start > 0:
        ax.axhline(start, color="white", linewidth=1.5)
    mid = (start + end) / 2
    ax.text(-0.02, mid, type_labels[tc], va="center", ha="right",
            fontsize=9, color=type_colors[tc], fontweight="bold",
            transform=ax.get_yaxis_transform())
plt.colorbar(im, ax=ax, label="merged HMM state", ticks=range(k_merged), fraction=0.025)
ax.set_xlabel("Time from go-cue onset (s)")
ax.set_ylabel("Trial (sorted by type)", fontsize=11)
ax.set_yticks([])
ax.set_title(f"Poisson HMM states after statistical merging \u2014 "
             f"k={k_report} \u2192 {k_merged}, all trials, no stim", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"plot_{session_name}_{plot_counter}.png"), dpi=120, bbox_inches="tight")
plt.close("all")
plot_counter += 1

# ---- Plot: merged emission profiles ----

fig, ax = plt.subplots(figsize=(10, 4))
colors_k = plt.cm.tab10(np.linspace(0, 1, k_merged))
for s in range(k_merged):
    ax.plot(merged_lambdas[s], color=colors_k[s], linewidth=1.8,
            label=f"Merged state {s}  (\u03bb\u0304={merged_lambdas[s].mean():.2f})")
ax.set_xlabel("Neuron index", fontsize=11)
ax.set_ylabel("\u03bb (expected spikes per bin)", fontsize=11)
ax.set_title(f"Emission rates per merged state \u2014 k={k_report} \u2192 {k_merged}", fontsize=12)
ax.legend(fontsize=8, ncol=2, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, n_units - 0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"plot_{session_name}_{plot_counter}.png"), dpi=120, bbox_inches="tight")
plt.close("all")
plot_counter += 1

# ---- Plot 1: State rasters for each k (sorted by trial type) ----
# One figure per k, same layout as the main state raster above.
# Trials are sorted by type so structure is visible across k values.

for k in k_range:
    sm = all_states[k][sort_idx]   # sort by trial type

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(
        sm,
        aspect="auto",
        interpolation="nearest",
        extent=[t_start, t_end, len(selected_trials), 0],
        cmap="tab10" if k <= 10 else "viridis",
        vmin=0,
        vmax=k - 1
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=2)
    ax.axvline(delay_onset_rel, color="black", linestyle="--", linewidth=2)
    ax.axvline(sample_onset_rel, color="black", linestyle="--", linewidth=2)
    for start, end, tc in boundaries:
        if start > 0:
            ax.axhline(start, color="white", linewidth=1.5)
        mid = (start + end) / 2
        ax.text(-0.02, mid, type_labels[tc], va="center", ha="right",
                fontsize=9, color=type_colors[tc], fontweight="bold",
                transform=ax.get_yaxis_transform())
    plt.colorbar(im, ax=ax, label="HMM state", ticks=range(k), fraction=0.025)
    ax.set_xlabel("Time from go-cue onset (s)")
    ax.set_ylabel("Trial (sorted by type)", fontsize=11)
    ax.set_yticks([])
    ax.set_title(f"Poisson HMM states — k={k}, all trials, no stim", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"plot_{session_name}_{plot_counter}.png"), dpi=120, bbox_inches="tight")
    plt.close("all")
    plot_counter += 1

# ---- Plot 2: Emission rates for each k ----
# Each state's lambda profile across neurons.
# States are colour-coded; a legend identifies them.

for k in k_range:
    lam = all_lambdas[k]   # (k, n_units)

    fig, ax = plt.subplots(figsize=(10, 4))
    colors_k = plt.cm.tab10(np.linspace(0, 1, k))
    for s in range(k):
        ax.plot(lam[s], color=colors_k[s], linewidth=1.8,
                label=f"State {s}  (λ̄={lam[s].mean():.2f})")
    ax.set_xlabel("Neuron index", fontsize=11)
    ax.set_ylabel("λ (expected spikes per bin)", fontsize=11)
    ax.set_title(f"Emission rates per state — k={k}", fontsize=12)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, n_units - 0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"plot_{session_name}_{plot_counter}.png"), dpi=120, bbox_inches="tight")
    plt.close("all")
    plot_counter += 1

# ---- Plot 3: Transition matrices for each k ----
# Shown as heatmaps; diagonal = self-transition (persistence),
# off-diagonal = switching probability.

n_cols = 3
n_rows = int(np.ceil(len(k_range) / n_cols))
fig, axs = plt.subplots(n_rows, n_cols,
                         figsize=(n_cols * 4, n_rows * 3.5))
axs = axs.flatten()

for i, k in enumerate(k_range):
    ax = axs[i]
    tm = all_transmats[k]
    im = ax.imshow(tm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_title(f"k={k}", fontsize=11)
    ax.set_xlabel("To state", fontsize=9)
    ax.set_ylabel("From state", fontsize=9)
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    # annotate cells with probability values
    for r in range(k):
        for c in range(k):
            ax.text(c, r, f"{tm[r, c]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if tm[r, c] > 0.5 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)

# hide unused subplots
for j in range(i + 1, len(axs)):
    axs[j].set_visible(False)

plt.suptitle("Transition matrices — k=2 to 10\n"
             "Diagonal = state persistence, off-diagonal = switching probability",
             fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"plot_{session_name}_{plot_counter}.png"), dpi=120, bbox_inches="tight")
plt.close("all")
plot_counter += 1
