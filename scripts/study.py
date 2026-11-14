"""Run the maintained study from the repository root.

Actions import only the implementation they need. In particular, help and
analysis do not initialize a simulator or open training data.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """Dispatch an explicit study stage without chaining final-data access."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("train", "predict", "analyze", "plot"):
        command = commands.add_parser(action)
        command.add_argument("--config", type=Path, default=Path("configs/study.yaml"))
        if action in {"predict", "analyze"}:
            command.add_argument("--paired", action="store_true", help="Secondary renderer study")
        if action in {"analyze", "plot"}:
            command.add_argument(
                "--report-output", type=Path, help="Separate primary reanalysis destination"
            )

    development = commands.add_parser("develop")
    development.add_argument("stage", choices=("reference", "weights", "tune", "review"))
    development.add_argument("--config", type=Path, default=Path("configs/development.yaml"))

    generation = commands.add_parser("generate")
    generation.add_argument("--role", choices=("source_train", "source_development"))
    generation.add_argument("--workers", type=int, default=2)
    generation.add_argument("--paired", choices=("prepare", "render", "assemble"))
    generation.add_argument("--design", type=Path)
    generation.add_argument(
        "--release", type=Path, default=Path("data/simulations/cross-simulator-rapid-v2")
    )
    generation.add_argument("--renderer", choices=("lenstronomy", "autolens"))
    qualification = commands.add_parser("qualify")
    qualification.add_argument(
        "--renderers", nargs="+", default=["lenstronomy"], choices=("lenstronomy", "autolens")
    )
    args = parser.parse_args()

    if args.action == "qualify":
        from src.data.validation import qualify_rendering

        qualify_rendering(args.renderers)
        return

    if args.action == "generate":
        if args.paired:
            from src.simulation import paired

            if args.role:
                parser.error("--role and --paired are separate generation workflows")
            if args.paired == "prepare":
                if args.design is None:
                    parser.error("paired preparation requires --design")
                paired.prepare(args.design)
            elif args.paired == "render":
                if args.renderer is None:
                    parser.error("paired rendering requires --renderer")
                paired.phase(args.release, args.renderer, args.workers)
            else:
                paired.assemble(args.release)
        else:
            if args.role is None:
                parser.error("source generation requires --role")
            from src.simulation.generate import run

            run(args.role, args.workers)
        return

    if args.action == "develop":
        from src.ml.policy import develop

        develop(args.stage, args.config)
        return

    from src.ml.artifacts import load_config

    config = load_config(args.config)
    if args.action == "train":
        from src.ml.train import fit

        fit(config)
    elif args.action == "predict":
        from src.ml.predict import predict, predict_paired

        if args.paired:
            predict_paired()
        else:
            predict(config)
    elif args.action == "analyze":
        from src.ml.analysis import score, score_paired

        if args.paired:
            if args.report_output is not None:
                parser.error("--report-output is for the primary analysis")
            score_paired()
        else:
            score(config, report_output=args.report_output)
    elif args.action == "plot":
        from src.ml.plots import plot

        plot(config, report_output=args.report_output)


if __name__ == "__main__":
    main()
