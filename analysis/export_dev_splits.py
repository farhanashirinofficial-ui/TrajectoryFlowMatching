"""One-time, verified export of the existing train and validation DataFrames."""

from contextlib import ExitStack
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import uuid

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/eICU_sepsis_tfm.pkl"
OUTPUTS = {
    "train": ROOT / "data/eICU_sepsis_train.csv",
    "val": ROOT / "data/eICU_sepsis_val.csv",
}


def verify(source, path, missing_marker):
    restored = pd.read_csv(
        path,
        dtype=source.dtypes.to_dict(),
        float_precision="round_trip",
        keep_default_na=False,
        na_values=[missing_marker],
        encoding="utf-8",
    )
    if len(restored) != len(source):
        raise ValueError("Row count mismatch")
    if list(restored.columns) != list(source.columns):
        raise ValueError("Column names/order mismatch")
    pd.testing.assert_frame_equal(
        source.reset_index(drop=True),
        restored,
        check_exact=True,
        check_dtype=True,
        check_names=False,
    )


def main():
    for path in OUTPUTS.values():
        print(path, flush=True)
    try:
        # On Windows, rename is atomic per file and refuses existing targets.
        # Two files cannot be committed as one filesystem transaction; rollback
        # removes this run's first output if publishing the second fails.
        if os.name != "nt":
            raise RuntimeError("This utility requires Windows rename semantics")
        if any(path.exists() for path in OUTPUTS.values()):
            raise FileExistsError("An output already exists")

        # Because the source is a single bundled pickle, deserialization
        # necessarily reconstructs the bundled object in memory. This script
        # intentionally never accesses the test entry.
        bundled = pd.read_pickle(SOURCE)
        development = {"train": bundled["train"], "val": bundled["val"]}
        del bundled

        for frame in development.values():
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("Expected a development DataFrame")
        print(f"train row count: {len(development['train'])}", flush=True)
        print(f"validation row count: {len(development['val'])}", flush=True)

        # Preserve missingness without treating literal strings such as NA or
        # empty strings as missing. Check marker collisions only in development.
        missing_marker = f"__CSV_MISSING_{uuid.uuid4().hex}__"
        for frame in development.values():
            if frame.eq(missing_marker).any().any():
                raise ValueError("Missing-value marker collision")

        with ExitStack() as cleanup, ExitStack() as rollback:
            temporary = {}
            for split in ("train", "val"):
                with NamedTemporaryFile(
                    mode="w", encoding="utf-8", newline="",
                    dir=OUTPUTS[split].parent,
                    prefix=f".{OUTPUTS[split].stem}_", suffix=".tmp.csv",
                    delete=False,
                ) as stream:
                    temporary[split] = Path(stream.name)
                    cleanup.callback(temporary[split].unlink, missing_ok=True)
                    development[split].to_csv(
                        stream, index=False, na_rep=missing_marker
                    )

            for split in ("train", "val"):
                verify(development[split], temporary[split], missing_marker)

            # Both verifications have passed. Never use replace(): an existing
            # final file must survive even if it appeared during verification.
            for split in ("train", "val"):
                temporary[split].rename(OUTPUTS[split])
                rollback.callback(OUTPUTS[split].unlink, missing_ok=True)
            rollback.pop_all()
    except Exception:
        # Context managers remove temporary files and this run's published
        # outputs on failure. Pre-existing outputs are never registered for cleanup.
        # Do not print exceptions: they can contain source values.
        print("verification failed", flush=True)
        return 1

    print("verification passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
