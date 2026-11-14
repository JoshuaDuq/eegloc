"""Command line: `eegloc fit STUDY` (first pass), then `eegloc refine STUDY` (second pass)."""

import argparse
from collections.abc import Callable, Sequence
from functools import partial
from multiprocessing import Pool

from eegloc import pipeline, report
from eegloc.config import Study


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="eegloc", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="first pass: each participant from the images")
    fit.add_argument("--subjects", nargs="+", help="only these participants of the study")
    refine = commands.add_parser(
        "refine", help="second pass: every participant refitted with the learned cap layout"
    )
    for command in (fit, refine):
        command.add_argument("study", help="the study file (TOML)")
        command.add_argument("--jobs", type=int, default=1, help="participants fitted at once")
    arguments = parser.parse_args(argv)

    study = Study.read(arguments.study)
    if arguments.command == "fit":
        subjects = arguments.subjects or study.subjects
        unknown = set(subjects) - set(study.subjects)
        if unknown:
            parser.error(f"not in the study: {sorted(unknown)}")
        run(partial(pipeline.first_pass, study), subjects, arguments.jobs)
        print(report.summarize(study.output / "first"))
    else:
        learned = pipeline.lessons(study)
        print(f"layout learned from {len(learned.subjects)} participants: {learned.subjects}")
        run(partial(pipeline.second_pass, study, learned), study.subjects, arguments.jobs)
        print(report.summarize(study.output / "second"))


def run(one: Callable[[str], dict], subjects: Sequence[str], jobs: int) -> None:
    """Fit each participant, `jobs` of them at once."""
    with Pool(jobs) as pool:
        for record in pool.imap_unordered(one, subjects):
            print(f"{record['subject']}: {'passed' if record['passed'] else 'failed'}", flush=True)
