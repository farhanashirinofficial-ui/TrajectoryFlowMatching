"""Frozen A-I mortality protocol. Execute only after explicit experiment approval.

Only six fixed development CSVs are inputs. No TFM loading or training occurs.
Calibration regression is a diagnostic, never a probability recalibration step.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import warnings

# Preserve the verified Windows DLL import order.
import torch
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import expit
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / 'analysis/outputs/tfm_phase1_midpoint_v1'
INPUTS = {
    split: {
        'features': PHASE1 / f'{split}_patient_features.csv',
        'cohort': PHASE1 / f'{split}_development_cohort.csv',
        'clinical': ROOT / f'data/eICU_sepsis_{split}.csv',
    } for split in ('train', 'val')
}
ID = 'HADM_ID'
OUTCOME = 'HOSP_MORT'
VITALS = ['hr_normalized', 'map_normalized']
TIME = 'time_scaled_v1'
STATISTICS = ['mean', 'std', 'min', 'max', 'first', 'last', 'slope']
CONVENTIONAL = [f'{v}_{s}' for v in VITALS for s in STATISTICS]
HIDDEN = [f'flow_hidden_{i:03d}_{s}' for s in ('mean', 'std') for i in range(256)]
NOISE = ['noise_amplitude_abs_mean', 'noise_amplitude_abs_std']
SIGNED = ['noise_amplitude_raw_mean', 'noise_amplitude_raw_std']
METADATA = ['n_observations', 'n_intervals']
FORBIDDEN = {ID, OUTCOME, 'ICU_MORT', 'apache_outcome_prob', 'label', *METADATA}
BLOCKS = {'apache': ['apache'], 'conventional': CONVENTIONAL, 'hidden': HIDDEN, 'noise': NOISE}
GROUPS = {
    'A': ['apache'], 'B': ['conventional'], 'C': ['hidden'], 'D': ['noise'],
    'E': ['hidden', 'noise'], 'F': ['apache', 'hidden'], 'G': ['apache', 'hidden', 'noise'],
    'H': ['apache', 'conventional'], 'I': ['apache', 'conventional', 'hidden', 'noise'],
}
DIMENSIONS = {'A': 2, 'B': 14, 'C': 16, 'D': 2, 'E': 18, 'F': 18, 'G': 20, 'H': 16, 'I': 34}
COMPARISONS = [('I', 'H', 'primary'), ('G', 'A', 'important_secondary'),
               ('F', 'A', 'secondary'), ('G', 'F', 'secondary'),
               ('E', 'C', 'secondary'), ('C', 'B', 'secondary')]
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
SEED = 42
N_BOOTSTRAPS = 2000
METRICS = ['log_loss', 'auroc', 'average_precision', 'brier_score',
           'mean_predicted_risk', 'observed_mortality', 'calibration_intercept', 'calibration_slope']
COMPARISON_METRICS = ['log_loss', 'auroc', 'average_precision', 'brier_score']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def write_csv(path, frame):
    frame.to_csv(path, index=False, mode='x')


class ApacheBranch(BaseEstimator, TransformerMixin):
    """Never mutate raw input: impute/scale score, retain unscaled indicator."""
    @staticmethod
    def values(X):
        values = np.asarray(X, dtype=np.float64).reshape(-1).copy()
        require(not np.isinf(values).any(), 'Infinite APACHE')
        require(not ((values < 0) & (values != -1)).any(), 'Unrecognized negative APACHE code')
        unavailable = np.isnan(values) | (values == -1)
        values[unavailable] = np.nan
        return values, unavailable

    def fit(self, X, y=None):
        values, unavailable = self.values(X)
        require((~unavailable).any(), 'No available APACHE in fitting fold')
        self.median_ = float(np.median(values[~unavailable]))
        imputed = np.where(unavailable, self.median_, values).reshape(-1, 1)
        self.scaler_ = StandardScaler().fit(imputed)
        return self

    def transform(self, X):
        values, unavailable = self.values(X)
        imputed = np.where(unavailable, self.median_, values).reshape(-1, 1)
        return np.column_stack([self.scaler_.transform(imputed).ravel(), unavailable.astype(float)])


def pipeline(group, C):
    transformers = []
    for block in GROUPS[group]:
        if block == 'apache':
            transform = ApacheBranch()
        elif block == 'hidden':
            transform = Pipeline([
                ('input_scaler', StandardScaler()),
                ('pca', PCA(n_components=16, svd_solver='full', whiten=False)),
                ('score_scaler', StandardScaler()),
            ])
        else:
            transform = StandardScaler()
        columns = BLOCKS[block]
        require(not FORBIDDEN.intersection(columns), 'Forbidden predictor')
        transformers.append((block, transform, columns))
    return Pipeline([
        ('preprocess', ColumnTransformer(transformers, remainder='drop', sparse_threshold=0)),
        ('classifier', LogisticRegression(penalty='l2', solver='lbfgs', fit_intercept=True,
             class_weight=None, tol=1e-6, max_iter=5000, warm_start=False, C=C)),
    ])


def conventional_features(clinical, ids, metadata):
    require(list(clinical.columns) == [ID, *VITALS, TIME], 'Clinical allowlist mismatch')
    require(clinical[ID].notna().all(), 'Missing clinical ID')
    selected = clinical[clinical[ID].isin(ids)]
    require(set(selected[ID]) == set(ids), 'Eligible clinical ID mapping mismatch')
    require(np.isfinite(selected[[*VITALS, TIME]].to_numpy(dtype=float)).all(), 'Nonfinite clinical values')
    rows = []
    for patient_id, group in selected.groupby(ID, sort=False):
        ordered = group.sort_values(TIME)
        t = ordered[TIME].to_numpy(dtype=np.float64)
        require(len(t) >= 9 and (np.diff(t) > 0).all(), 'Invalid observation count/time ordering')
        require(len(t) == metadata.loc[patient_id, 'n_observations'], 'Observation count mismatch')
        require(len(t)-4 == metadata.loc[patient_id, 'n_intervals'], 'Interval count mismatch')
        centered = t - t.mean()
        denominator = np.dot(centered, centered)
        require(denominator > 0 and np.isfinite(denominator), 'Invalid slope denominator')
        row = {ID: patient_id}
        for vital in VITALS:
            x = ordered[vital].to_numpy(dtype=np.float64)
            summaries = [x.mean(), x.std(ddof=0), x.min(), x.max(), x[0], x[-1],
                         np.dot(centered, x-x.mean()) / denominator]
            row.update({f'{vital}_{name}': float(value) for name, value in zip(STATISTICS, summaries)})
        rows.append(row)
    result = pd.DataFrame(rows).set_index(ID).loc[ids, CONVENTIONAL]
    require(np.isfinite(result.to_numpy()).all(), 'Nonfinite conventional features')
    return result


def load_development():
    loaded = {}
    for split, expected_count in (('train', 2261), ('val', 292)):
        paths = INPUTS[split]
        for path in paths.values():
            require(path.is_file() and path.suffix == '.csv', 'Missing fixed development CSV')
            require('test' not in str(path.relative_to(ROOT)).lower(), 'Prohibited input path')
        features = pd.read_csv(paths['features'], dtype={ID: 'string'}, float_precision='round_trip')
        cohort = pd.read_csv(paths['cohort'], dtype={ID: 'string'}, float_precision='round_trip')
        require(list(cohort.columns) == [ID, OUTCOME, 'apache'], 'Cohort schema mismatch')
        expected_features = {ID, *METADATA, *HIDDEN, *NOISE, *SIGNED}
        require(set(features.columns) == expected_features and len(features.columns) == 519, 'Feature schema mismatch')
        for frame in (features, cohort):
            require(len(frame) == expected_count, 'Primary cohort count mismatch')
            require(frame[ID].notna().all() and frame[ID].is_unique, 'Missing/duplicate ID')
        require(features[ID].equals(cohort[ID]), 'Cohort order differs from feature order')
        require(cohort[OUTCOME].isin([0, 1]).all(), 'Missing/nonbinary outcome')
        require(np.isfinite(features[HIDDEN + NOISE + SIGNED + METADATA].to_numpy(dtype=float)).all(), 'Nonfinite TFM values')
        require((features[NOISE].to_numpy() >= 0).all(), 'Negative absolute amplitude')
        require(not np.isinf(cohort['apache'].to_numpy(dtype=float)).any(), 'Infinite APACHE')
        ids = features[ID].tolist()
        clinical = pd.read_csv(paths['clinical'], usecols=[ID, *VITALS, TIME],
                               dtype={ID: 'string'}, float_precision='round_trip')[[ID, *VITALS, TIME]]
        conventional = conventional_features(clinical, ids, features.set_index(ID)[METADATA])
        X = pd.concat([cohort.set_index(ID)[['apache']],
                       features.set_index(ID)[HIDDEN + NOISE], conventional], axis=1)
        require(X.index.tolist() == ids and not FORBIDDEN.intersection(X.columns), 'Predictor alignment/allowlist failure')
        require(X.columns.is_unique, 'Duplicate predictor column')
        y = pd.Series(cohort[OUTCOME].to_numpy(dtype=np.int64), index=X.index, name=OUTCOME)
        loaded[split] = (X, y)
    require(set(loaded['train'][0].index).isdisjoint(loaded['val'][0].index), 'Development ID overlap')
    return loaded


def probability(model, X):
    p = model.predict_proba(X)[:, 1]
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid probabilities')
    return p


def fit_checked(model, X, y, group):
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(X, y)
    classifier = model.named_steps['classifier']
    require(classifier.coef_.shape == (1, DIMENSIONS[group]), 'Transformed dimension mismatch')
    require(np.isfinite(classifier.coef_).all() and np.isfinite(classifier.intercept_).all(), 'Nonfinite fitted coefficients')
    require((classifier.n_iter_ < 5000).all(), 'Iteration limit reached')
    return int(classifier.n_iter_.max())


def fit_analysis(name, development, directory):
    directory.mkdir(exist_ok=False)
    X, y = development['train']
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y))
    assignment = np.full(len(X), -1, dtype=int)
    for fold, (train_idx, held_idx) in enumerate(folds):
        require(len(np.unique(y.iloc[train_idx])) == 2 and len(np.unique(y.iloc[held_idx])) == 2, 'Single-class CV fold')
        require(not np.intersect1d(train_idx, held_idx).size, 'CV fold overlap')
        assignment[held_idx] = fold
    require((assignment >= 0).all(), 'Incomplete CV assignment')
    write_csv(directory / 'cv_fold_assignments.csv', pd.DataFrame({ID: X.index, 'held_out_fold': assignment}))
    rows, selection, refits, models = [], [], [], {}
    for group in GROUPS:
        means = {}
        for C in C_GRID:
            losses = []
            for fold, (train_idx, held_idx) in enumerate(folds):
                fitted = pipeline(group, C)
                iterations = fit_checked(fitted, X.iloc[train_idx], y.iloc[train_idx], group)
                loss = float(log_loss(y.iloc[held_idx], probability(fitted, X.iloc[held_idx]), labels=[0, 1]))
                losses.append(loss)
                rows.append(dict(group=group, C=C, fold=fold, held_out_log_loss=loss, iterations=iterations, converged=True))
            means[C] = float(np.mean(losses))
        best = min(means.values())
        chosen = min(C for C in C_GRID if means[C] <= best + 1e-6)
        for C in C_GRID:
            selection.append(dict(group=group, C=C, mean_held_out_log_loss=means[C], selected=(C == chosen)))
        fitted = pipeline(group, chosen)
        iterations = fit_checked(fitted, X, y, group)
        models[group] = fitted
        refits.append(dict(group=group, selected_C=chosen, training_patients=len(y),
                           training_deaths=int(y.sum()), iterations=iterations, converged=True,
                           predictor_count=DIMENSIONS[group]))
        print(f'{name}: {group} selection/refit complete; converged; iterations={iterations}', flush=True)
    write_csv(directory / 'cv_fold_results.csv', pd.DataFrame(rows))
    write_csv(directory / 'selected_c_cv_results.csv', pd.DataFrame(selection))
    write_csv(directory / 'refit_summary.csv', pd.DataFrame(refits))
    return models


def calibration(y, p):
    """Joint unpenalized intercept/slope diagnostic on prediction logits."""
    epsilon = np.finfo(np.float64).eps
    bounded = np.clip(p, epsilon, 1-epsilon)
    logits = np.log(bounded) - np.log1p(-bounded)
    if np.ptp(logits) <= 1e-12:
        return np.nan, np.nan, 'unidentified_constant_prediction'
    design = np.column_stack([np.ones(len(y)), logits])

    def objective(beta):
        eta = design @ beta
        loss = np.mean(np.logaddexp(0, eta) - y * eta)
        gradient = design.T @ (expit(eta)-y) / len(y)
        return loss, gradient

    fitted = minimize(objective, np.array([0., 1.]), jac=True, method='L-BFGS-B',
                      options={'maxiter': 5000, 'gtol': 1e-8, 'ftol': 1e-12})
    beta = fitted.x
    if not fitted.success or not np.isfinite(beta).all():
        return np.nan, np.nan, 'optimization_failed'
    q = expit(design @ beta)
    information = (design.T * (q*(1-q))) @ design
    if np.linalg.matrix_rank(information) < 2 or np.linalg.norm(fitted.jac, ord=np.inf) > 1e-5:
        return np.nan, np.nan, 'unidentified_or_nonconverged'
    return float(beta[0]), float(beta[1]), 'ok'


def metrics(y, p):
    intercept, slope, status = calibration(y, p)
    return dict(log_loss=float(log_loss(y, p, labels=[0, 1])), auroc=float(roc_auc_score(y, p)),
                average_precision=float(average_precision_score(y, p)), brier_score=float(brier_score_loss(y, p)),
                mean_predicted_risk=float(np.mean(p)), observed_mortality=float(np.mean(y)),
                calibration_intercept=intercept, calibration_slope=slope), status


def benefit(metric, baseline, augmented):
    # Positive always favors the augmented model.
    return baseline-augmented if metric in ('log_loss', 'brier_score') else augmented-baseline


def interval(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan, np.nan, 0
    lower, upper = np.percentile(finite, [2.5, 97.5])
    return float(lower), float(upper), len(finite)


def evaluate(name, development, models, directory):
    X, target = development['val']
    y = target.to_numpy()
    predictions = {group: probability(models[group], X) for group in GROUPS}
    prediction_table = pd.DataFrame({ID: X.index, OUTCOME: y, **{f'probability_{g}': p for g, p in predictions.items()}})
    write_csv(directory / 'validation_predictions.csv', prediction_table)
    point, statuses, bins = {}, {}, []
    for group, p in predictions.items():
        point[group], statuses[group] = metrics(y, p)
        # Stable order resolves tied predictions by original cohort row order.
        for number, indices in enumerate(np.array_split(np.argsort(p, kind='stable'), 5), 1):
            bins.append(dict(group=group, bin=number, patients=len(indices), deaths=int(y[indices].sum()),
                             mean_predicted_risk=float(p[indices].mean()), observed_mortality=float(y[indices].mean()),
                             min_predicted_risk=float(p[indices].min()), max_predicted_risk=float(p[indices].max())))
    write_csv(directory / 'calibration_bins.csv', pd.DataFrame(bins))
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(y), size=(N_BOOTSTRAPS, len(y)))
    np.save(directory / 'bootstrap_patient_indices.npy', samples, allow_pickle=False)
    bootstrap = {g: {m: [] for m in METRICS} for g in GROUPS}
    bootstrap_rows, discarded = [], []
    for replicate, indices in enumerate(samples):
        by = y[indices]
        if len(np.unique(by)) != 2:
            discarded.append(replicate)
            continue
        for group, p in predictions.items():
            values, status = metrics(by, p[indices])
            bootstrap_rows.append(dict(replicate=replicate, group=group, calibration_status=status, **values))
            for metric, value in values.items():
                bootstrap[group][metric].append(value)
        if (replicate+1) % 100 == 0:
            print(f'{name}: paired bootstrap {replicate+1}/{N_BOOTSTRAPS}', flush=True)
    write_csv(directory / 'bootstrap_metrics.csv', pd.DataFrame(bootstrap_rows))
    results = []
    for group in GROUPS:
        for metric in METRICS:
            lower, upper, valid = interval(bootstrap[group][metric])
            results.append(dict(group=group, metric=metric, estimate=point[group][metric], ci_lower=lower, ci_upper=upper,
                                bootstrap_valid=valid, bootstrap_unavailable=N_BOOTSTRAPS-len(discarded)-valid,
                                validation_patients=len(y), deaths=int(y.sum()), calibration_status=statuses[group]))
    write_csv(directory / 'validation_metrics.csv', pd.DataFrame(results))
    comparisons = []
    for augmented, baseline, priority in COMPARISONS:
        for metric in COMPARISON_METRICS:
            deltas = benefit(metric, np.asarray(bootstrap[baseline][metric]), np.asarray(bootstrap[augmented][metric]))
            lower, upper, valid = interval(deltas)
            comparisons.append(dict(augmented=augmented, baseline=baseline, priority=priority, metric=metric,
                                    delta=benefit(metric, point[baseline][metric], point[augmented][metric]),
                                    ci_lower=lower, ci_upper=upper, bootstrap_valid=valid,
                                    positive_favors='augmented',
                                    delta_definition='baseline_minus_augmented' if metric in ('log_loss', 'brier_score') else 'augmented_minus_baseline'))
    write_csv(directory / 'paired_comparisons.csv', pd.DataFrame(comparisons))
    write_json(directory / 'bootstrap_summary.json', dict(attempted=N_BOOTSTRAPS, accepted=N_BOOTSTRAPS-len(discarded),
               discarded_single_class_replicates=discarded, seed=SEED, paired_across_all_groups=True,
               calibration_point_status=statuses, ci='percentile 2.5/97.5; conditional on fitted models',
               calibration_failures='See bootstrap_metrics.csv and per-metric unavailable counts'))


def protocol():
    return dict(endpoint=OUTCOME, groups=GROUPS, dimensions=DIMENSIONS, blocks=BLOCKS,
        conventional=dict(statistics=STATISTICS, all_observations=True, weighting='equal', ddof=0,
            time=TIME, slope='sum((t-mean(t))*(x-mean(x)))/sum((t-mean(t))**2)'),
        apache='-1/missing unavailable inside branch; fitting-fold median; standardized score; unscaled indicator',
        hidden='512 -> StandardScaler -> PCA(16, full, whiten=False) -> StandardScaler',
        noise=NOISE, conventional_scaling='StandardScaler', combined_scaling='branch only; no overall scaler',
        classifier=dict(penalty='l2', solver='lbfgs', fit_intercept=True, class_weight=None,
                        tol=1e-6, max_iter=5000, warm_start=False),
        cv=dict(C_grid=C_GRID, folds=5, stratified=True, shuffle=True, random_state=SEED,
                selection='lowest mean held-out log loss', tie_tolerance=1e-6, tie_rule='smaller C'),
        metrics=METRICS, primary_metric='log_loss', comparisons=COMPARISONS,
        calibration='joint unpenalized intercept/slope on epsilon-clipped logits; five equal-frequency bins; no recalibration',
        bootstrap=dict(attempts=N_BOOTSTRAPS, seed=SEED, paired=True, discard='single-class samples',
                       ci='percentile 95%; no training uncertainty', diagnostics='calibration failures recorded'),
        complete_case='Common APACHE-available cohort across A-I; independently refit all CV/preprocessing/models',
        primary_counts=dict(train=2261, val=292), complete_case_counts=dict(train=2250, val=292),
        primary_improvement='logloss(H)-logloss(I)>0 with paired CI entirely above zero: development evidence only',
        exclusions=sorted(FORBIDDEN), signed_noise_excluded=SIGNED, threshold_optimization=False,
        post_validation_tuning=False, resampling_or_class_weights=False,
        interpretation=['Available observation spans, not an early-prediction landmark',
                        'TFM is norepinephrine-conditioned; conventional comparator is not fully information-matched',
                        'Frozen TFM learned from full training cohort; CV is for downstream regularization selection',
                        'Validation contributed to TFM checkpoint selection; all validation results are developmental'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'analysis/outputs/mortality_frozen_v1')
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    require(destination.is_relative_to((ROOT / 'analysis/outputs').resolve()), 'Output must stay within analysis/outputs')
    require(not destination.exists(), 'Refusing existing output directory, including partial runs')
    require('test' not in str(destination.relative_to(ROOT)).lower(), 'Prohibited output path')
    sources = {f'{split}_{kind}': path for split, paths in INPUTS.items() for kind, path in paths.items()}
    input_hashes = {name: digest(path) for name, path in sources.items()}
    development = load_development()
    complete_case = {}
    for split, (X, y) in development.items():
        mask = X['apache'].notna() & X['apache'].ne(-1)
        complete_case[split] = (X.loc[mask].copy(), y.loc[mask].copy())
        require(len(complete_case[split][0]) == {'train': 2250, 'val': 292}[split], 'Unexpected complete-case count')
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / 'protocol.json', protocol())
    write_json(destination / 'provenance.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        script=dict(path=str(Path(__file__).resolve()), sha256=digest(Path(__file__))),
        sources={name: dict(path=str(path), sha256=input_hashes[name]) for name, path in sources.items()},
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      scipy=scipy.__version__, sklearn=sklearn.__version__, torch=str(torch.__version__)),
        threads=1, feature_origin='Existing frozen Phase 1 CSVs; checkpoint not opened',
        completion='Only COMPLETE.json indicates successful completion'))
    analyses = {'primary': development, 'complete_case': complete_case}
    # Fit/select BOTH analyses before generating any validation predictions.
    with threadpool_limits(limits=1):
        models = {name: fit_analysis(name, data, destination / name) for name, data in analyses.items()}
        for name, data in analyses.items():
            evaluate(name, data, models[name], destination / name)
    require(all(digest(path) == input_hashes[name] for name, path in sources.items()), 'Input changed during execution')
    output_hashes = {str(path.relative_to(destination)): digest(path)
                     for path in sorted(destination.rglob('*')) if path.is_file()}
    write_json(destination / 'COMPLETE.json', dict(status='complete', input_hashes_unchanged=True,
        groups=list(GROUPS), analyses=list(analyses), primary_comparison='I versus H',
        checks=['counts', 'unique/disjoint IDs', 'binary outcome', 'input allowlists', 'fold pairing',
                'finite predictions', 'convergence', 'unchanged input hashes'], output_sha256=output_hashes))
    print(f'Completed: {destination}', flush=True)


if __name__ == '__main__':
    main()
