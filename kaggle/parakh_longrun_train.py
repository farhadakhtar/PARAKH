"""PARAKH consistency model - long run. Large corpus, 100-epoch ceiling.

    !python parakh_longrun_train.py --records 500000 --epochs 100 --hours 8

Reuses the model and encoders from ``parakh_consistency_train.py``; only the
scale, the schedule and the instrumentation differ.

100 IS A CEILING, NOT A TARGET
-------------------------------
The run stops at 100 epochs, or when validation loss stops improving, or when
the wall-clock budget expires - whichever comes first. An epoch count chosen
because it is a round number is not a training protocol, and training past the
best epoch does not make a model better, only later.

Best weights are restored from the best validation epoch regardless of where
the run ends, so an interrupted or budget-capped run still yields the same
model the full run would have.

HOW OVER- AND UNDERFITTING ARE ACTUALLY DETECTED
-------------------------------------------------
Not by watching one loss curve. Three signals, reported every epoch:

* **gap** = val_loss - train_loss. Rising steadily while val_loss climbs is
  overfitting. Near zero with both losses high is underfitting - the model
  lacks capacity or the features do not carry the signal.
* **val_loss trajectory** - the epoch of the best value, and how many epochs
  have passed without beating it.
* **the real metric, periodically.** Per-channel defect AUC is computed every
  ``--eval-every`` epochs, because loss and the job are different things. An
  earlier version of this model reached 0.97 reconstruction accuracy and 0.538
  defect AUC. Watching loss alone would have called that run a success.

WHAT IT STILL CANNOT CLAIM
--------------------------
Fraud. Eight hours of compute does not create a label. This trains a
consistency model on injected data-quality defects; scale improves the
feature, not the claim.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from parakh_consistency_train import (
    CATEGORICAL_FIELDS,
    add_durations,
    baseline_metrics,
    build_model,
    encode,
    make_demo_frame,
    mask_inputs,
    set_seeds,
    surprise,
)


def human(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def make_large_frame(n: int, seed: int, chunk: int = 100_000) -> pd.DataFrame:
    """Build a large corpus in chunks with distinct seeds.

    Chunked so memory stays bounded, and each chunk gets its own seed so the
    corpus is not one small pattern repeated - which would inflate every
    metric by letting the holdout contain near-copies of training rows.
    """
    frames = []
    produced = 0
    index = 0
    while produced < n:
        size = min(chunk, n - produced)
        frames.append(make_demo_frame(n=size, seed=seed + index * 7919))
        produced += size
        index += 1
        print(f"  generated {produced:,}/{n:,}", flush=True)
    frame = pd.concat(frames, ignore_index=True)
    return frame


class ExpressiveLogger:
    """Per-epoch diagnostics: losses, the gap, LR, timing and an ETA.

    Implemented as a Keras callback built lazily, so importing this module
    does not require TensorFlow.
    """

    @staticmethod
    def build(total_epochs: int, deadline: Optional[float], eval_hook=None,
              eval_every: int = 10):
        from tensorflow import keras

        class _Callback(keras.callbacks.Callback):
            def __init__(self) -> None:
                super().__init__()
                self.started = time.time()
                self.epoch_started = self.started
                self.best = float("inf")
                self.best_epoch = 0
                self.history: List[Dict[str, float]] = []
                self.stopped_reason: Optional[str] = None

            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_started = time.time()

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                train = float(logs.get("loss", float("nan")))
                val = float(logs.get("val_loss", float("nan")))
                gap = val - train
                elapsed = time.time() - self.started
                per_epoch = elapsed / (epoch + 1)
                remaining = (total_epochs - epoch - 1) * per_epoch

                improved = val < self.best - 1e-6
                if improved:
                    self.best, self.best_epoch = val, epoch + 1
                stale = epoch + 1 - self.best_epoch

                try:
                    lr = float(keras.backend.get_value(
                        self.model.optimizer.learning_rate))
                except Exception:
                    lr = float("nan")

                # A rising gap alongside a rising val_loss is the overfitting
                # signature; a flat near-zero gap with high loss is capacity
                # starvation. Naming it per epoch beats reading a curve later.
                if stale == 0:
                    verdict = "improving"
                elif gap > 0.15 * max(abs(train), 1e-9) and stale >= 3:
                    verdict = "OVERFIT?"
                elif gap < 0.02 * max(abs(train), 1e-9) and stale >= 5:
                    verdict = "UNDERFIT?"
                else:
                    verdict = f"stale x{stale}"

                # 'state' and 'status' both truncate to 'stat' at 4 chars,
                # which made two different heads print under one label.
                short = {"state": "st", "district": "dist",
                         "implementing_agency": "agcy", "status": "stus"}
                accuracies = " ".join(
                    f"{short.get(name, name[:4])}="
                    f"{logs.get(f'val_out_{name}_accuracy', float('nan')):.3f}"
                    for name in CATEGORICAL_FIELDS
                )
                print(
                    f"ep {epoch+1:3d}/{total_epochs}  "
                    f"loss {train:8.4f}  val {val:8.4f}  gap {gap:+7.4f}  "
                    f"lr {lr:.2e}  {human(time.time()-self.epoch_started):>7}  "
                    f"eta {human(remaining):>8}  best@{self.best_epoch:3d}  "
                    f"{verdict:10s}  {accuracies}",
                    flush=True,
                )
                self.history.append({
                    "epoch": epoch + 1, "loss": train, "val_loss": val,
                    "gap": gap, "lr": lr, "best_epoch": self.best_epoch,
                    "verdict": verdict, "elapsed_s": elapsed,
                })

                if eval_hook and (epoch + 1) % eval_every == 0:
                    print("      --- real-metric check "
                          "(loss is not the job) ---", flush=True)
                    for line in eval_hook():
                        print(f"      {line}", flush=True)

                if deadline and time.time() > deadline:
                    self.stopped_reason = (
                        f"wall-clock budget reached at epoch {epoch+1}")
                    print(f"\n[BUDGET] {self.stopped_reason}. "
                          f"Best weights are from epoch {self.best_epoch}.",
                          flush=True)
                    self.model.stop_training = True

        return _Callback()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--records", type=int, default=500_000)
    parser.add_argument("--epochs", type=int, default=100,
                        help="ceiling, not a target")
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--holdout", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--out", type=str, default="parakh_longrun_out")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seeds(args.seed)
    from tensorflow import keras
    from sklearn.metrics import roc_auc_score

    started = time.time()
    deadline = started + args.hours * 3600

    print("=" * 100)
    print("PARAKH consistency model - LONG RUN")
    print(f"  records   {args.records:,}" if not args.csv else f"  csv       {args.csv}")
    print(f"  epochs    {args.epochs} (ceiling; early stopping patience "
          f"{args.patience})")
    print(f"  budget    {args.hours}h  -> deadline {time.strftime('%H:%M:%S', time.localtime(deadline))}")
    print(f"  batch     {args.batch_size}   width {args.width}   seed {args.seed}")
    print("=" * 100, flush=True)

    print("\n[1/5] corpus", flush=True)
    if args.csv:
        frame = pd.read_csv(args.csv, low_memory=False)
        print(f"  loaded {len(frame):,} rows", flush=True)
    else:
        frame = make_large_frame(args.records, args.seed)
    frame = add_durations(frame)
    channels = (frame["defect_channel"].astype(str)
                if "defect_channel" in frame.columns else None)
    print(f"  {len(frame):,} rows, {len(frame.columns)} columns", flush=True)
    if channels is not None:
        counts = channels.value_counts()
        for name, count in counts.items():
            print(f"    {name:22s} {count:8,}  {count/len(frame):6.2%}", flush=True)

    print("\n[2/5] split and encode", flush=True)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(frame))
    cut = int(len(frame) * args.holdout)
    test_frame = frame.iloc[order[:cut]]
    train_frame = frame.iloc[order[cut:]]
    train = encode(train_frame)
    test = encode(test_frame, train["vocabularies"], train["stats"])
    print(f"  train {len(train_frame):,}   holdout {len(test_frame):,}", flush=True)
    for name in CATEGORICAL_FIELDS:
        print(f"    {name:22s} vocab {len(train['vocabularies'][name]):5d}",
              flush=True)

    print("\n[3/5] model", flush=True)
    model = build_model(train, args.width, args.seed)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={**{f"out_{n}": "sparse_categorical_crossentropy"
                 for n in CATEGORICAL_FIELDS}, "out_numeric": "mse"},
        metrics={f"out_{n}": "accuracy" for n in CATEGORICAL_FIELDS})
    print(f"  parameters {model.count_params():,}", flush=True)

    inputs = mask_inputs(train, rng)
    targets = {**{f"out_{n}": train["categorical"][n] for n in CATEGORICAL_FIELDS},
               "out_numeric": train["numeric"]}

    def real_metric_check() -> List[str]:
        """Per-channel defect AUC - the number that actually matters."""
        if channels is None:
            return ["no defect_channel column; cannot check the real metric"]
        scores = surprise(model, test, np.random.default_rng(args.seed))
        holdout = channels.loc[test_frame.index]
        lines = []
        for channel in sorted(set(holdout) - {"clean"}):
            truth = (holdout == channel).to_numpy().astype(int)
            if truth.sum() < 30:
                continue
            keep = holdout.isin([channel, "clean"]).to_numpy()
            auc = float(roc_auc_score(truth[keep], scores[keep]))
            lines.append(f"{channel:22s} n={int(truth.sum()):7,}  AUC {auc:.4f}")
        return lines or ["no channel had enough holdout positives"]

    logger = ExpressiveLogger.build(args.epochs, deadline, real_metric_check,
                                    args.eval_every)

    print("\n[4/5] training", flush=True)
    print("-" * 100, flush=True)
    history = model.fit(
        inputs, targets,
        validation_split=0.15,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        verbose=0,
        callbacks=[
            logger,
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.patience,
                restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=max(3, args.patience // 3), min_lr=1e-6, verbose=1),
            keras.callbacks.ModelCheckpoint(
                str(out / "best.keras"), monitor="val_loss",
                save_best_only=True, verbose=0),
            keras.callbacks.CSVLogger(str(out / "epochs.csv")),
        ],
    )
    print("-" * 100, flush=True)

    print("\n[5/5] final evaluation", flush=True)
    evaluated = model.evaluate(
        mask_inputs(test, rng),
        {**{f"out_{n}": test["categorical"][n] for n in CATEGORICAL_FIELDS},
         "out_numeric": test["numeric"]},
        verbose=0, return_dict=True)

    base = baseline_metrics(train, test)
    print("\nreconstruction (diagnostic - NOT the success metric)")
    reconstruction = {}
    for name in CATEGORICAL_FIELDS:
        accuracy = float(evaluated.get(f"out_{name}_accuracy", float("nan")))
        reconstruction[name] = {"accuracy": accuracy,
                                "mode_baseline": base[name],
                                "lift": accuracy - base[name]}
        print(f"  {name:22s} {accuracy:.4f}  base {base[name]:.4f}  "
              f"lift {accuracy-base[name]:+.4f}")

    print("\nDEFECT DETECTION - the result")
    detection = {}
    for line in real_metric_check():
        print(f"  {line}")
        parts = line.split()
        if len(parts) >= 4 and parts[-2] == "AUC":
            detection[parts[0]] = float(parts[-1])

    epochs_run = len(logger.history)
    gaps = [h["gap"] for h in logger.history]
    if not gaps:
        fit_verdict = "no epoch completed"
    elif logger.best_epoch >= epochs_run - 1:
        fit_verdict = ("UNDERFIT or UNDER-TRAINED: validation was still "
                       "improving when the run ended. Raise the ceiling or "
                       "the budget.")
    elif gaps[-1] > 0.15 * abs(logger.history[-1]["loss"]):
        fit_verdict = (f"OVERFITTING after epoch {logger.best_epoch}: the "
                       "train/val gap widened. Best weights were restored, so "
                       "the saved model is the epoch-"
                       f"{logger.best_epoch} model.")
    else:
        fit_verdict = (f"CONVERGED at epoch {logger.best_epoch} of "
                       f"{epochs_run}; the gap stayed controlled.")
    print(f"\nfit: {fit_verdict}")

    report = {
        "manifest": {
            "framework": "tensorflow", "seed": args.seed,
            "n_records": int(len(frame)), "n_train": int(len(train_frame)),
            "n_holdout": int(len(test_frame)),
            "epochs_ceiling": args.epochs, "epochs_run": epochs_run,
            "best_epoch": logger.best_epoch,
            "early_stopped": epochs_run < args.epochs,
            "stopped_reason": logger.stopped_reason,
            "batch_size": args.batch_size, "width": args.width,
            "n_parameters": int(model.count_params()),
            "wall_clock_s": time.time() - started,
            "budget_hours": args.hours,
        },
        "reconstruction": reconstruction,
        "defect_detection": detection,
        "fit_verdict": fit_verdict,
        "epoch_history": logger.history,
        "_claim": ("Detects INJECTED DATA-QUALITY DEFECTS in a synthetic "
                   "corpus. Not a fraud metric. Scale improves the feature, "
                   "never the claim."),
    }
    json.dump(report, open(out / "metrics.json", "w"), indent=2)
    model.save(out / "final.keras")
    pd.DataFrame(logger.history).to_csv(out / "epoch_log.csv", index=False)

    print(f"\ntotal {human(time.time()-started)}   written to {out}/")
    print("\nNOTE: consistency, not fraud. Nothing here has seen a fraud label.")


if __name__ == "__main__":
    main()
