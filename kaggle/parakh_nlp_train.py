"""PARAKH work-text encoder - Kaggle-ready trainer (TensorFlow/Keras).

Self-contained. No PARAKH imports.

    !pip -q install transformers
    !python parakh_nlp_train.py --encoder muril --csv /kaggle/input/parakh/records.csv

Encoders run under PyTorch (transformers v5 removed the TF classes); the
prediction heads stay in Keras. See :func:`embed_pretrained`.

TWO WAYS TO SUPPLY THE WEIGHTS
------------------------------
1. **HuggingFace hub** (default). Needs kernel internet, which on Kaggle needs
   a phone-verified account. Without it the kernel reports
   ``[Errno -3] Temporary failure in name resolution`` even though
   ``enable_internet: true`` was accepted - the flag is stored and the network
   is still denied.

2. **A local directory** via ``--model-path``, for an attached Kaggle Model or
   Dataset. No internet required.

   Note that Kaggle's own ``google/muril`` is published as a **TF2/TF-Hub
   SavedModel**, not a HuggingFace checkpoint, so it does NOT load through
   ``AutoModel.from_pretrained``. ``--model-path`` expects a HuggingFace-layout
   directory (config.json + tokenizer files + weights). To use the Kaggle copy,
   upload an HF snapshot as a Dataset instead, or verify the phone number and
   use route 1.
    !python parakh_nlp_train.py --encoder muril --demo
    !python parakh_nlp_train.py --encoder scratch --demo   # ablation only

WHY PRETRAINED WEIGHTS ARE NOT OPTIONAL HERE
---------------------------------------------
The first version of this script trained a BiLSTM from scratch. That was
wrong, and the reason is worth stating because it decides the architecture.

The whole justification for replacing Stage 3's TF-IDF is cross-lingual
synonymy: Indian public-works registers are code-mixed, and TF-IDF cannot
know that "sadak" and "road", or "vidyalaya" and "school", denote the same
thing. But a model trained from scratch on a corpus of short work names
cannot know it either. That equivalence is not recoverable from 20,000
strings of four words each - it is a fact about Hindi and English, learned
from large text, and it has to arrive in the weights.

A from-scratch encoder therefore cannot deliver the one capability the model
exists to add. It is kept behind ``--encoder scratch`` as an ablation, and
the script says so rather than quietly reporting its accuracy as if the
comparison were meaningful.

    muril      google/muril-base-cased - BERT over 17 Indian languages plus
               English, trained on transliterated text too, which is what
               code-mixed registers actually look like. The default.
    indicbert  ai4bharat/indic-bert - smaller, ALBERT-based.
    scratch    Ablation. Cannot learn synonymy; present to demonstrate that.

THE SYNONYMY PROBE
------------------
Accuracy on downstream heads is an indirect test. :func:`synonymy_probe`
tests the claim directly - it embeds known cross-lingual pairs and reports
whether they land closer together than unrelated controls. It needs no labels
and no training, so it runs before the heads and tells you immediately
whether the encoder is worth the compute.

WHAT THIS IS NOT
----------------
Not a fraud detector, not a risk model. It learns what a work description
implies about the work, and every downstream use is grouping and comparison -
deciding which works are peers - never scoring a record as suspicious.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TEXT_FIELD = "work_name"
TARGETS = ("implementing_agency", "cost_band", "status")
MAX_TOKENS = 20000
SEQUENCE_LENGTH = 32

#: Open-weight multilingual encoders, all on the HuggingFace hub.
#:
#: Which one is best for this corpus is an empirical question, not a matter of
#: taste, and `--compare-encoders` answers it by running the synonymy probe
#: across all of them. Picking one by reputation is how you end up with a
#: 471M-parameter model that is worse at transliteration than a 236M one.
ENCODERS = {
    # Indian-language specialists. Trained on Devanagari AND romanised text,
    # which is what a works register actually contains.
    "muril": "google/muril-base-cased",
    "muril-large": "google/muril-large-cased",
    "indicbert": "ai4bharat/indic-bert",
    "indicbert-v2": "ai4bharat/IndicBERTv2-MLM-only",
    # General multilingual. Broader language coverage, usually weaker on
    # Hindi transliteration - the probe will show whether that holds here.
    "xlmr": "xlm-roberta-base",
    "xlmr-large": "xlm-roberta-large",
    "mbert": "bert-base-multilingual-cased",
    # Sentence encoders. Trained for semantic similarity rather than masked
    # language modelling, so they often win a similarity probe outright.
    "labse": "sentence-transformers/LaBSE",
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "distiluse": "sentence-transformers/distiluse-base-multilingual-cased-v2",
    # Ablation: proves a from-scratch model cannot learn synonymy at all.
    "scratch": None,
}

#: Rough parameter counts, for reading the bench against its cost. A model
#: that wins by 0.01 separation at 4x the size has not really won.
ENCODER_SIZE_M = {
    "muril": 236, "muril-large": 506, "indicbert": 33, "indicbert-v2": 278,
    "xlmr": 270, "xlmr-large": 560, "mbert": 178,
    "labse": 471, "minilm": 118, "distiluse": 135,
}

#: Cross-lingual/transliterated pairs that mean the same thing in an Indian
#: works register, and control pairs that do not. The probe is only
#: informative because both are present: an encoder that maps everything
#: close together would score well on synonyms alone.
SYNONYM_PAIRS = [
    ("road", "sadak"),
    ("school", "vidyalaya"),
    ("building", "bhavan"),
    ("construction", "nirman"),
    ("water", "jal"),
    ("drain", "nali"),
    ("work", "kary"),
]
CONTROL_PAIRS = [
    ("road", "vidyalaya"),
    ("school", "nali"),
    ("water", "bhavan"),
    ("drain", "sadak"),
    ("construction", "jal"),
]


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)


# --- demo data ------------------------------------------------------------

WORK_KINDS = [
    ("Construction of CC road", "PWD", "road"),
    ("Nirman of sadak", "PWD", "road"),
    ("Repair of village road", "PWD", "road"),
    ("Gram sadak marammat kary", "PWD", "road"),
    ("Construction of anganwadi centre", "WCD", "building"),
    ("Building of primary school block", "EDU", "building"),
    ("Vidyalaya bhavan nirman", "EDU", "building"),
    ("Prathmik school bhavan", "EDU", "building"),
    ("Installation of hand pump", "PHED", "water"),
    ("Jal supply pipeline laying", "PHED", "water"),
    ("Peyjal yojana pipeline", "PHED", "water"),
    ("Construction of drainage", "MUNI", "drain"),
    ("Nali nirman kary", "MUNI", "drain"),
    ("Solar street light installation", "ENERGY", "light"),
    ("Street light repair work", "ENERGY", "light"),
]


def make_demo_frame(n: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Demo corpus of code-mixed descriptions.

    The agency is deliberately NOT a deterministic function of the phrasing -
    each agency is reachable from several surface forms, including ones in a
    different language. An earlier version made the mapping one-to-one, every
    model scored 1.0000, and the comparison measured nothing.
    """
    rng = np.random.default_rng(seed)
    suffix = ["at ward {}", "in village {}", "phase {}", "block {}", ""]
    rows = []
    for _ in range(n):
        base, agency, kind = WORK_KINDS[rng.integers(len(WORK_KINDS))]
        tail = suffix[rng.integers(len(suffix))]
        text = f"{base} {tail.format(rng.integers(1, 40))}".strip()
        # 8% of rows are misfiled to a different agency, so the target keeps
        # irreducible noise and cannot be saturated by any model.
        if rng.random() < 0.08:
            agency = rng.choice(["PWD", "WCD", "EDU", "PHED", "MUNI", "ENERGY"])
        cost = float(np.exp(rng.normal(
            {"road": 14.2, "building": 14.8, "water": 12.9,
             "drain": 13.3, "light": 12.4}[kind], 0.9)))
        rows.append({
            "work_name": text,
            "implementing_agency": agency,
            "sanction_amount": cost,
            "status": rng.choice(["COMPLETED", "ONGOING", "SANCTIONED"],
                                 p=[0.6, 0.3, 0.1]),
        })
    return pd.DataFrame(rows)


def add_cost_band(frame: pd.DataFrame, n_bands: int = 5) -> pd.DataFrame:
    """Quantile cost bands, so classes stay balanced and accuracy is readable."""
    out = frame.copy()
    amount = pd.to_numeric(out.get("sanction_amount"), errors="coerce")
    out["cost_band"] = pd.qcut(amount, n_bands, labels=False, duplicates="drop")
    out["cost_band"] = out["cost_band"].fillna(-1).astype(int).astype(str)
    return out


def clean_text(value: object) -> str:
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9ऀ-ॿ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# --- encoders -------------------------------------------------------------


def embed_pretrained(
    texts: Sequence[str], model_name: str, batch_size: int = 64
) -> np.ndarray:
    """Mean-pooled contextual embeddings from a pretrained transformer.

    **PyTorch, not TensorFlow.** The first version used ``TFAutoModel`` and
    every one of the ten encoders failed identically on Kaggle:

        ImportError: cannot import name 'TFAutoModel' from 'transformers'

    HuggingFace dropped TensorFlow support in transformers v5, so the TF
    classes no longer exist. PyTorch is the supported path and is preinstalled
    on Kaggle. Only this function changed - the heads that consume these
    vectors stay in Keras, because they train on plain numpy arrays and have
    no reason to care which library produced them.

    Mean pooling over the attention mask rather than the CLS vector: CLS is
    only meaningful after next-sentence-style pretraining or fine-tuning, and
    these are four-word fragments being used frozen.

    The encoder is frozen and runs under ``inference_mode``. On short,
    formulaic text the pretrained representation is already what is wanted,
    and fine-tuning 236M parameters on 20,000 phrases would overfit long
    before it improved anything.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    # model_name may be a hub id or a local directory. from_pretrained accepts
    # both, so the only difference is whether the network is touched.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    out: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer(batch, padding=True, truncation=True,
                                max_length=SEQUENCE_LENGTH,
                                return_tensors="pt").to(device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            out.append(pooled.float().cpu().numpy())
            if start % (batch_size * 20) == 0:
                print(f"  embedded "
                      f"{min(start + batch_size, len(texts))}/{len(texts)}",
                      flush=True)

    # Free the encoder before the next candidate loads. Ten models at
    # 100-500M parameters each will exhaust GPU memory otherwise, and the
    # bench would report OOM for every encoder after the first few - a
    # failure that looks like a model problem and is really a cleanup bug.
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return np.concatenate(out, axis=0)


def synonymy_probe(model_name: str) -> Dict[str, object]:
    """Does the encoder actually place cross-lingual synonyms together?

    A direct test of the only capability that justifies replacing TF-IDF.
    Needs no labels and no training, so it runs first: if synonym pairs are
    no closer than unrelated controls, the encoder adds nothing here and the
    rest of the run is wasted compute.
    """
    pairs = SYNONYM_PAIRS + CONTROL_PAIRS
    flat = [w for pair in pairs for w in pair]
    vectors = embed_pretrained(flat, model_name)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9

    similarities = [
        float(vectors[2 * i] @ vectors[2 * i + 1]) for i in range(len(pairs))
    ]
    synonym = similarities[: len(SYNONYM_PAIRS)]
    control = similarities[len(SYNONYM_PAIRS) :]
    separation = float(np.mean(synonym) - np.mean(control))

    return {
        "mean_synonym_similarity": float(np.mean(synonym)),
        "mean_control_similarity": float(np.mean(control)),
        "separation": separation,
        "per_pair": {f"{a}~{b}": s for (a, b), s in zip(pairs, similarities)},
        "verdict": (
            "Encoder separates cross-lingual synonyms from controls; the "
            "capability TF-IDF lacks is present."
            if separation > 0.05 else
            "NO SEPARATION. Synonyms are no closer than unrelated words, so "
            "this encoder does not supply the cross-lingual matching that "
            "motivated replacing TF-IDF."
        ),
    }


def compare_encoders(names: Sequence[str]) -> Dict[str, object]:
    """Run the synonymy probe across several encoders and rank them.

    The probe needs no labels and no training, so comparing ten encoders costs
    ten forward passes over about two dozen words. That is minutes, against
    the hours a wrong choice costs downstream - which makes guessing the
    encoder strictly worse than measuring it.

    A model that fails to load (gated weights, no TF port, out of memory) is
    recorded as an error rather than skipped silently: "did not run" and
    "ran and scored badly" are different results.
    """
    results: Dict[str, object] = {}
    for name in names:
        model_name = ENCODERS.get(name)
        if not model_name:
            continue
        print(f"\n--- probing {name} ({model_name}) ...", flush=True)
        try:
            probe = synonymy_probe(model_name)
            probe["params_m"] = ENCODER_SIZE_M.get(name)
            results[name] = probe
            print(f"    separation {probe['separation']:+.4f}  "
                  f"(syn {probe['mean_synonym_similarity']:.4f} / "
                  f"ctrl {probe['mean_control_similarity']:.4f})", flush=True)
        except Exception as exc:  # noqa: BLE001 - report, never hide
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)

    scored = {k: v for k, v in results.items() if "separation" in v}
    ranked = sorted(scored.items(), key=lambda kv: -kv[1]["separation"])

    print("\n" + "=" * 74)
    print(f"{'encoder':16s} {'params':>7s} {'separation':>11s} "
          f"{'synonym':>9s} {'control':>9s}")
    print("-" * 74)
    for name, probe in ranked:
        size = probe.get("params_m")
        print(f"{name:16s} {(str(size)+'M' if size else '?'):>7s} "
              f"{probe['separation']:+11.4f} "
              f"{probe['mean_synonym_similarity']:9.4f} "
              f"{probe['mean_control_similarity']:9.4f}")
    print("=" * 74)

    if not ranked:
        recommendation = ("NO ENCODER LOADED. Every candidate errored - check "
                          "network access and `pip install transformers`.")
    elif ranked[0][1]["separation"] <= 0.05:
        recommendation = (
            "NO ENCODER PASSES. The best separation is "
            f"{ranked[0][1]['separation']:+.4f}, at or below the 0.05 gate. "
            "None of these supplies the cross-lingual matching that motivated "
            "replacing TF-IDF, so keep TF-IDF and do not train heads."
        )
    else:
        best, probe = ranked[0]
        cheaper = [
            (n, p) for n, p in ranked[1:]
            if p["separation"] > 0.05
            and (p.get("params_m") or 1e9) < (probe.get("params_m") or 1e9)
            and probe["separation"] - p["separation"] < 0.02
        ]
        recommendation = f"USE {best} (separation {probe['separation']:+.4f})."
        if cheaper:
            name, alt = cheaper[0]
            recommendation += (
                f" But {name} is within 0.02 separation at "
                f"{alt.get('params_m')}M vs {probe.get('params_m')}M - prefer "
                "it unless the margin matters, since a smaller encoder is "
                "cheaper to serve and easier to keep frozen."
            )

    print("\n" + recommendation)
    return {"results": results,
            "ranking": [n for n, _ in ranked],
            "recommendation": recommendation}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--encoder", choices=sorted(ENCODERS), default="muril")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--out", type=str, default="parakh_nlp_out")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument(
        "--compare-encoders", nargs="*", default=None, metavar="NAME",
        help="probe several encoders and rank them, then exit. No argument "
             "means every pretrained encoder in the registry.")
    parser.add_argument(
        "--model-path", type=str, default=None, metavar="DIR",
        help="Load the encoder from a local HuggingFace-layout directory "
             "instead of the hub. Use when the kernel has no internet. "
             "Overrides --encoder.")
    args = parser.parse_args()

    set_seeds(args.seed)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # A local path overrides the registry: if the caller has the weights on
    # disk, there is nothing to look up and nothing to download.
    if args.model_path:
        import os
        if not os.path.isdir(args.model_path):
            raise SystemExit(f"--model-path {args.model_path!r} is not a directory")
        if not os.path.exists(os.path.join(args.model_path, "config.json")):
            raise SystemExit(
                f"{args.model_path!r} has no config.json, so it is not a "
                "HuggingFace-layout checkpoint. Kaggle's google/muril is a "
                "TF-Hub SavedModel and will not load here - upload an HF "
                "snapshot as a Dataset, or enable kernel internet.")
        ENCODERS[args.encoder] = args.model_path
        print(f"loading encoder from local path: {args.model_path}")

    if args.compare_encoders is not None:
        names = args.compare_encoders or [
            k for k, v in ENCODERS.items() if v is not None]
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        bench = compare_encoders(names)
        json.dump(bench, open(out / "encoder_bench.json", "w"), indent=2)
        print(f"\nwritten to {out}/encoder_bench.json")
        return

    model_name = ENCODERS[args.encoder]
    report: Dict[str, object] = {"encoder": args.encoder,
                                 "encoder_model": model_name, "targets": {}}

    if args.encoder == "scratch":
        print("\n*** ABLATION MODE ***\n"
              "A from-scratch encoder cannot learn that 'sadak' means 'road' - "
              "that fact is not present in the corpus. Any result below is a "
              "lower bound and must not be reported as the neural result.\n")
        report["_ablation_warning"] = (
            "from-scratch encoder cannot learn cross-lingual synonymy")

    frame = (pd.read_csv(args.csv, low_memory=False) if args.csv
             else make_demo_frame(seed=args.seed))
    if TEXT_FIELD not in frame.columns:
        raise SystemExit(f"input must carry a {TEXT_FIELD!r} column")

    frame = add_cost_band(frame)
    frame["_text"] = frame[TEXT_FIELD].map(clean_text)
    frame = frame[frame["_text"].str.len() > 0].reset_index(drop=True)

    targets = [t for t in TARGETS if t in frame.columns]
    if not targets:
        raise SystemExit(f"input carries none of {TARGETS}")

    if model_name and not args.skip_probe:
        print(f"\nsynonymy probe on {model_name} ...")
        probe = synonymy_probe(model_name)
        report["synonymy_probe"] = probe
        print(f"  synonyms  {probe['mean_synonym_similarity']:.4f}")
        print(f"  controls  {probe['mean_control_similarity']:.4f}")
        print(f"  separation {probe['separation']:+.4f}")
        print(f"  -> {probe['verdict']}\n")

    codes, classes = {}, {}
    for name in targets:
        categorical = frame[name].astype(str).astype("category")
        codes[name] = categorical.cat.codes.to_numpy("int32")
        classes[name] = list(categorical.cat.categories)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(frame))
    cut = int(len(frame) * args.holdout)
    test_rows, train_rows = order[:cut], order[cut:]
    texts = frame["_text"].to_numpy()
    train_text, test_text = texts[train_rows], texts[test_rows]

    # --- incumbent baseline: TF-IDF + logistic regression -----------------
    vectoriser = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                 min_df=2, max_features=MAX_TOKENS)
    train_vectors = vectoriser.fit_transform(train_text)
    test_vectors = vectoriser.transform(test_text)
    baseline: Dict[str, float] = {}
    for name in targets:
        clf = LogisticRegression(max_iter=400)
        clf.fit(train_vectors, codes[name][train_rows])
        baseline[name] = float(accuracy_score(
            codes[name][test_rows], clf.predict(test_vectors)))

    # --- neural heads -----------------------------------------------------
    if model_name:
        print("embedding corpus with pretrained encoder ...")
        features = embed_pretrained(list(texts), model_name)
        inputs = keras.Input(shape=(features.shape[1],), name="embedding")
        hidden = layers.Dense(128, activation="relu")(inputs)
        hidden = layers.Dropout(0.2, seed=args.seed)(hidden)
        train_x, test_x = features[train_rows], features[test_rows]
    else:
        vectorise = layers.TextVectorization(
            max_tokens=MAX_TOKENS, output_sequence_length=SEQUENCE_LENGTH)
        vectorise.adapt(tf.constant(train_text))
        inputs = keras.Input(shape=(1,), dtype=tf.string, name="text")
        hidden = vectorise(inputs)
        hidden = layers.Embedding(MAX_TOKENS, 64, mask_zero=True)(hidden)
        hidden = layers.Bidirectional(layers.LSTM(64))(hidden)
        hidden = layers.Dropout(0.2, seed=args.seed)(hidden)
        hidden = layers.Dense(128, activation="relu")(hidden)
        train_x, test_x = tf.constant(train_text), tf.constant(test_text)

    outputs = {f"out_{n}": layers.Dense(len(classes[n]), activation="softmax",
                                        name=f"out_{n}")(hidden)
               for n in targets}
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={f"out_{n}": "sparse_categorical_crossentropy" for n in targets},
        metrics={f"out_{n}": "accuracy" for n in targets})

    history = model.fit(
        train_x, {f"out_{n}": codes[n][train_rows] for n in targets},
        validation_split=0.15, epochs=args.epochs,
        batch_size=args.batch_size, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True)])

    evaluated = model.evaluate(
        test_x, {f"out_{n}": codes[n][test_rows] for n in targets},
        verbose=0, return_dict=True)

    print("\ntarget                 neural   tfidf+lr   winner")
    neural_wins = decisive = 0
    for name in targets:
        neural = float(evaluated.get(f"out_{name}_accuracy", float("nan")))
        gap = abs(neural - baseline[name])
        # A target both models saturate, or separate by less than run-to-run
        # wobble, compares nothing and is excluded from the verdict.
        degenerate = (neural > 0.995 and baseline[name] > 0.995) or gap < 0.005
        winner = "tie/degenerate" if degenerate else (
            "neural" if neural > baseline[name] else "TF-IDF")
        if not degenerate:
            decisive += 1
            neural_wins += winner == "neural"
        report["targets"][name] = {
            "neural_accuracy": neural, "tfidf_lr_accuracy": baseline[name],
            "delta": neural - baseline[name], "winner": winner,
            "decisive": not degenerate, "n_classes": len(classes[name])}
        print(f"  {name:20s} {neural:.4f}   {baseline[name]:.4f}   {winner}")

    if decisive == 0:
        verdict = (f"NO DECISIVE TARGET across {len(targets)}. This run "
                   "compared nothing; re-run on real work names.")
    elif neural_wins:
        verdict = (f"Neural wins {neural_wins}/{decisive} decisive target(s). "
                   "Adopt only if the margin justifies losing TF-IDF's "
                   "per-token interpretability, which Stage 3 explanations "
                   "depend on.")
    else:
        verdict = (f"KEEP TF-IDF. It won or tied all {decisive} decisive "
                   "target(s) and it is interpretable.")
    report["verdict"] = verdict
    report["manifest"] = {
        "framework": "tensorflow", "seed": args.seed,
        "n_train": int(len(train_rows)), "n_holdout": int(len(test_rows)),
        "epochs_max": args.epochs, "epochs_run": len(history.history["loss"]),
        "n_head_parameters": int(model.count_params()),
        "encoder_frozen": bool(model_name),
    }
    print("\n" + verdict)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "work_text_heads.keras")
    json.dump(report, open(out / "metrics.json", "w"), indent=2)
    print(f"written to {out}/")
    print("\nNOTE: this model groups works. It does not score risk and has "
          "never seen a fraud label.")


if __name__ == "__main__":
    main()
