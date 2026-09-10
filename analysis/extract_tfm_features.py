"""Deterministic TFM development features from separate train/validation CSVs.

Never reads the bundled pickle: unpickling would materialize its test split.
CSV exports must preserve original split membership, rows and normalized values.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'checkpoints/Noise_MLP_Cond_memory_Module_SDE_Memory_3_eICU_Sepsis_NoAPACHE_DataModule/best_model.ckpt'
ID = 'HADM_ID'
X = ['hr_normalized', 'map_normalized']
COND = ['norepi_inf_scaled']
TIME = 'time_scaled_v1'
ALLOWED = [ID, *X, *COND, TIME]
FEATURES = [f'flow_hidden_{i:03d}' for i in range(256)] + ['noise_amplitude_raw', 'noise_amplitude_abs']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    require(path.suffix.lower() == '.csv', 'Separate development CSVs required')
    frame = pd.read_csv(path, usecols=ALLOWED, dtype={ID: 'string'})
    require(not frame.empty and frame[ID].notna().all(), 'Empty data or missing IDs')
    require(np.isfinite(frame[[*X, *COND, TIME]].to_numpy(dtype=np.float64)).all(), 'Nonfinite input')
    return frame[ALLOWED]


def load_model():
    sys.path.insert(0, str(ROOT / 'src'))
    from model.mlp_noise import Noise_MLP_Cond_Memory_Module
    # Only the fixed, user-trusted checkpoint; no environment-wide load override.
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    hp = checkpoint['hyper_parameters']
    expected = dict(memory=3, dim=2, treatment_cond=1, w=256, time_varying=True,
                    conditional=True, implementation='SDE', sigma=0.1, clip=0.01)
    for key, value in expected.items():
        require(hp.get(key) == value, f'Unexpected checkpoint setting: {key}')
    require(checkpoint['epoch'] == 199 and checkpoint['global_step'] == 167200, 'Checkpoint identity mismatch')
    model = Noise_MLP_Cond_Memory_Module(**hp)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    model.requires_grad_(False)
    require(tuple(model.flow_model.net[0].weight.shape) == (256, 29), 'Flow input dimension')
    require(tuple(model.noise_model.net[6].weight.shape) == (1, 256), 'Noise output dimension')
    return model, expected


def patient_inputs(group, dm):
    ordered = group.sort_values(TIME)
    coords = ordered[X].to_numpy(dtype=np.float32)
    times = ordered[TIME].to_numpy(dtype=np.float32)
    treatment = ordered[COND].to_numpy(dtype=np.float32)
    require((np.diff(times) > 0).all(), 'Duplicate/non-increasing float32 times')
    require(((times >= 0) & (times <= 1)).all(), 'Scaled time outside [0,1]; no silent clamp')
    x0, classes, x1, t0, t1 = dm.create_pairs(group)
    n = len(ordered) - 4
    require(classes.shape == (n, 1, 7), 'Original conditioning shape mismatch')
    classes = classes[:, 0, :]
    history = np.stack([coords[i-3:i].reshape(6) for i in range(3, len(ordered)-1)])
    expected_classes = np.concatenate([treatment[3:-1], history], axis=1)
    for actual, expected in ((x0, coords[3:-1]), (x1, coords[4:]),
                             (classes, expected_classes), (t0, times[3:-1]), (t1, times[4:])):
        np.testing.assert_array_equal(actual, expected)
    midpoint = np.float32(0.5) * x0 + np.float32(0.5) * x1
    model_time = t0 + np.float32(0.5) * (t1 - t0)
    inputs = np.concatenate([midpoint, classes, model_time[:, None]], axis=1)
    require(inputs.shape == (n, 10) and np.isfinite(inputs).all(), 'Invalid network input')
    return torch.from_numpy(inputs), t0, t1


@torch.no_grad()
def probe(model, inputs):
    baseline = model.flow_model.forward_train(inputs)
    captured = []

    def capture(module, args, output):
        captured.append(output.detach().clone())  # Returns None: no output replacement.

    handle = model.flow_model.net[5].register_forward_hook(capture)
    try:
        output = model.flow_model.forward_train(inputs)
        repeated = model.flow_model.forward_train(inputs)
    finally:
        handle.remove()
    raw = model.noise_model.forward_train(inputs)
    raw_again = model.noise_model.forward_train(inputs)
    require(len(captured) == 2 and captured[0].shape == (len(inputs), 256), 'Hidden shape')
    require(output.shape == (len(inputs), 10) and raw.shape == (len(inputs), 1), 'Output shape')
    for tensor in (baseline, output, repeated, *captured, raw, raw_again):
        require(torch.isfinite(tensor).all().item(), 'Nonfinite network output')
    require(torch.equal(baseline, output), 'Hook changed model output')
    require(torch.equal(output, repeated) and torch.equal(captured[0], captured[1])
            and torch.equal(raw, raw_again), 'Non-repeatable features')
    return torch.cat([captured[0], raw, raw.abs()], dim=1).cpu().numpy()


def extract(split, frame, model, dm, destination):
    counts = frame.groupby(ID).size()
    eligible = counts[counts > dm.min_timept].index
    require(len(eligible) > 0, f'No eligible patients in {split}')
    summaries = []
    total = 0
    with (destination / f'{split}_interval_features.csv').open('x', newline='') as stream:
        for number, (patient_id, group) in enumerate(frame[frame[ID].isin(eligible)].groupby(ID), 1):
            inputs, t0, t1 = patient_inputs(group, dm)
            values = np.concatenate([probe(model, chunk) for chunk in inputs.split(256)], axis=0)
            intervals = pd.DataFrame(values, columns=FEATURES)
            intervals.insert(0, 't1', t1)
            intervals.insert(0, 't0', t0)
            intervals.insert(0, 'source_observation_index', np.arange(3, len(group)-1))
            intervals.insert(0, ID, patient_id)
            intervals.to_csv(stream, index=False, header=(number == 1))
            values64 = values.astype(np.float64)
            row = {ID: patient_id, 'n_observations': len(group), 'n_intervals': len(values)}
            for statistic, aggregate in (('mean', values64.mean(axis=0)), ('std', values64.std(axis=0, ddof=0))):
                require(np.isfinite(aggregate).all(), 'Nonfinite patient summary')
                row.update({f'{name}_{statistic}': float(value) for name, value in zip(FEATURES, aggregate)})
            summaries.append(row)
            total += len(values)
            if number % 100 == 0:
                print(f'{split}: {number}/{len(eligible)} patients', flush=True)
    pd.DataFrame(summaries).to_csv(destination / f'{split}_patient_features.csv', index=False, mode='x')
    require(total == int((counts.loc[eligible] - 4).sum()), 'Interval count mismatch')
    return dict(raw_patients=len(counts), eligible_patients=len(eligible),
                excluded_patients=len(counts)-len(eligible), intervals=total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-csv', required=True, type=Path)
    parser.add_argument('--val-csv', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path, help='Must not already exist')
    args = parser.parse_args()
    require(args.train_csv.resolve() != args.val_csv.resolve(), 'Distinct split files required')
    require(not args.output_dir.exists(), 'Output directory already exists')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    development = {'train': read_csv(args.train_csv), 'val': read_csv(args.val_csv)}
    require(set(development['train'][ID]).isdisjoint(set(development['val'][ID])), 'Patient overlap')
    model, settings = load_model()
    from data.datamodule import clinical_DataModule
    # Only constructor and pure pairing; never setup(), dataloaders or split diagnostics.
    dm = clinical_DataModule(memory=3, train_consecutive=False, x_headings=X, cond_headings=COND, t_headings=TIME)
    require(dm.min_timept == 8, 'Eligibility threshold mismatch')
    original_state = {key: value.clone() for key, value in model.state_dict().items()}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='incomplete_until_COMPLETE.json_exists',
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    checkpoint=str(CHECKPOINT), checkpoint_sha256=sha256(CHECKPOINT),
                    epoch=199, global_step=167200, model_settings=settings,
                    data_settings=dict(memory=3, train_consecutive=False, minimum_observations=9),
                    network_columns=[*X, *COND, TIME], history='3 prior HR/MAP rows, oldest first',
                    probe='u=0.5, no Gaussian jitter, no rollout or Brownian draw',
                    hidden_layer='flow_model.net[5]', aggregation='equal interval mean and population std ddof=0',
                    noise='signed raw output and absolute amplitude; not calibrated uncertainty',
                    torch_version=str(torch.__version__), device='cpu', script_sha256=sha256(Path(__file__)),
                    sources={split: dict(path=str(path.resolve()), sha256=sha256(path))
                             for split, path in (('train', args.train_csv), ('val', args.val_csv))})
    with (args.output_dir / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
    counts = {split: extract(split, frame, model, dm, args.output_dir) for split, frame in development.items()}
    require(all(torch.equal(original_state[key], value) for key, value in model.state_dict().items()), 'Model state changed')
    with (args.output_dir / 'COMPLETE.json').open('x') as stream:
        json.dump(dict(status='complete', counts=counts,
                       checks='shapes, finite values, repeatability, original pairing/history, unchanged output/state'), stream, indent=2)
    print(json.dumps(counts, indent=2))
    print(f'Completed: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
