#!/usr/bin/env python3
"""Export the gamma XGBoost bundle for native inference."""

import argparse
import json
import pickle
import struct
from pathlib import Path

MAGIC = b"PLXG"
VERSION = 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open("rb") as handle:
        bundle = pickle.load(handle)
    model = bundle["model"]
    if not hasattr(model, "get_booster"):
        raise TypeError(f"expected XGBoost estimator, got {type(model)!r}")

    booster = model.get_booster()
    booster_json = json.loads(booster.save_raw(raw_format="json").decode("utf-8"))
    learner = booster_json["learner"]
    tree_model = learner["gradient_booster"]["model"]
    trees = tree_model["trees"]
    learner_params = learner["learner_model_param"]
    base_score = float(learner_params["base_score"])
    learning_rate = float(model.get_params().get("learning_rate") or 0.3)

    with args.model_output.open("wb") as out:
        out.write(struct.pack("<4sIIIIff", MAGIC, VERSION, len(bundle["feat_cols"]),
                              len(trees), 0, base_score, learning_rate))
        for tree in trees:
            left = tree["left_children"]
            out.write(struct.pack("<I", len(left)))
            for i in range(len(left)):
                out.write(struct.pack(
                    "<iiifBB", int(left[i]), int(tree["right_children"][i]),
                    int(tree["split_indices"][i]), float(tree["split_conditions"][i]),
                    int(tree["default_left"][i]), int(tree["split_type"][i])))
            out.write(struct.pack("<" + "f" * len(tree["base_weights"]),
                                  *(float(value) for value in tree["base_weights"])))

    feature_names = list(bundle["feat_cols"])
    if len(feature_names) != 88:
        raise ValueError(f"expected 88 features, got {len(feature_names)}")

    manifest = {
        "format": "libplacebo-xgboost-binary",
        "version": VERSION,
        "schema": "libplacebo-ml-gamma-v1",
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "output": "gamma",
        "tree_count": len(trees),
        "base_score": base_score,
        "learning_rate": learning_rate,
        "clip": {"minimum": 0.5, "maximum": 1.5},
        "training": {
            "mode": str(bundle.get("mode")) if bundle.get("mode") is not None else None,
            "mae": float(bundle["mae"]) if bundle.get("mae") is not None else None,
            "corr": float(bundle["corr"]) if bundle.get("corr") is not None else None,
        },
    }
    args.manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"model={args.model_output}")
    print(f"manifest={args.manifest_output}")
    print(f"features={len(feature_names)}")
    print(f"trees={booster.num_boosted_rounds()}")


if __name__ == "__main__":
    main()
