import os
import pytest

SUPPORTED_SOLVERS = {
    'miqp': ('cplex',),
    'milp': ('cplex', 'glpk', 'cbc'),
}

def pytest_configure(config):
#    # register an additional marker
    config.addinivalue_line(
        "markers", "solver(type): mark test to run only on solvers that support a given problem structure"
    )

def pytest_runtest_setup(item):
    engine = os.environ["OPTIMISER_ENGINE"]
    problem_type = [mark.args[0] for mark in item.iter_markers(name="solver")]
    for problem in problem_type:
        if engine not in SUPPORTED_SOLVERS[problem]:
            pytest.skip(f"test requires solver that supports {problem_type} - {engine} not supported")
