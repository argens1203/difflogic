#!/usr/bin/env python3
"""
Main entry point for difflogic experiments.

Usage:
    python main.py --dataset iris -bs 100 -ni 2000 -ef 1000 -k 6 -l 2 --save_model

For debug mode (runs predefined experiments on multiple datasets):
    python main.py --debug
"""
import sys

from experiment.experiment import Experiment
from experiment.helpers.context import MultiContext


def run_debug_experiments():
    """Run predefined debug experiments on multiple datasets."""
    m_ctx = MultiContext()

    m_ctx.add(Experiment.debug(dataset="iris", small=True))
    m_ctx.add(Experiment.debug(dataset="monk1", small=True))
    m_ctx.add(Experiment.debug(dataset="monk2", small=True))
    m_ctx.add(Experiment.debug(dataset="monk3", small=True))
    m_ctx.add(Experiment.debug(dataset="breast_cancer", small=True))
    m_ctx.add(Experiment.debug(dataset="compas", small=True))
    m_ctx.add(Experiment.debug(dataset="lending", small=True))

    m_ctx.display()
    m_ctx.to_csv(filename="results.csv", with_timestamp=True)


if __name__ == "__main__":
    if "--debug" in sys.argv:
        run_debug_experiments()
    else:
        Experiment.run_with_cmd()
