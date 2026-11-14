"""Run explicit HST preparation, development validation and frozen transfer stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    """Dispatch one intentional action; never chain acquisition, training or scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in (
        "assets",
        "prepare",
        "validate-preparation",
        "freeze",
        "predict",
        "analyze",
        "plot",
    ):
        command = commands.add_parser(action)
        command.add_argument("--config", type=Path, default=Path("configs/hst_study.yaml"))
        if action in {"prepare", "predict", "analyze", "plot"}:
            command.add_argument(
                "--cohort",
                choices=("development", "confirmation"),
                required=action != "prepare",
                default="development",
            )
        if action == "prepare":
            command.add_argument(
                "--initialize", action="store_true", help="Create pending pilot review ledgers only"
            )
        if action == "validate-preparation":
            command.add_argument(
                "--smoke",
                action="store_true",
                help="Use a bounded subset; never satisfies confirmation gate",
            )
    args = parser.parse_args()
    from src.data.hst import load_hst_config

    config = load_hst_config(args.config)
    output = Path(config["output"])
    if args.action == "assets":
        from src.ml.artifacts import inspect_model_assets, load_config, write_json

        if (output / "confirmation_freeze.json").exists():
            raise ValueError("Asset identities are frozen; use the existing verified manifest")
        report = inspect_model_assets(load_config(config["study_config"]))
        write_json(output / "model_assets.json", report)
        print(
            json.dumps(
                {
                    "ready": report["ready"],
                    "verified_checkpoints": len(report["checkpoints"]),
                    "missing": report["missing"],
                },
                indent=2,
            )
        )
        if not report["ready"]:
            raise SystemExit(2)
    elif args.action == "prepare":
        from src.data.hst import initialize_observations, prepare_release

        if args.initialize:
            initialize_observations(config)
            print(
                "Created pending development observations and separate classical references; no qualification."
            )
        else:
            report = prepare_release(config, args.cohort)
            from src.ml.plots import preview_hst_preparation

            preview_hst_preparation(config, args.cohort)
            print(
                json.dumps(
                    {
                        "cohort": args.cohort,
                        "prepared": len(report["observations"]),
                        "ledger": report["ledger"],
                    },
                    indent=2,
                )
            )
    elif args.action == "validate-preparation":
        from src.simulation.hst_bridge import validate_preparation

        report = validate_preparation(config, smoke=args.smoke)
        print(
            json.dumps(
                {
                    "smoke": report["smoke"],
                    "passed": report["passed"],
                    "parents": report["parent_count"],
                    "controls": report["controls"],
                },
                indent=2,
            )
        )
        if not report["numerical_controls_passed"]:
            raise SystemExit(2)
    elif args.action == "freeze":
        from src.data.hst import freeze_confirmation

        report = freeze_confirmation(config)
        print(json.dumps({"stage": "confirmation_frozen", "systems": len(report["system_ids"])}))
    elif args.action == "predict":
        from src.ml.predict import predict_hst

        predict_hst(config, args.cohort)
        print(json.dumps({"stage": "predictions_sealed", "cohort": args.cohort}))
    elif args.action == "analyze":
        from src.ml.analysis import score_hst

        report = score_hst(config, args.cohort)
        print(
            json.dumps(
                {
                    "cohort": args.cohort,
                    "systems": report["system_count"],
                    "contrasts": report["contrasts"],
                },
                indent=2,
            )
        )
    elif args.action == "plot":
        from src.ml.plots import plot_hst

        plot_hst(config, args.cohort)
        print(json.dumps({"stage": "figures_written", "cohort": args.cohort}))


if __name__ == "__main__":
    main()
