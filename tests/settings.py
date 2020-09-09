import os

os.environ["OPTIMISER_ENGINE"] = "cplex"  #'ipopt'
os.environ[
    "OPTIMISER_ENGINE_EXECUTABLE"
] = "/Applications/CPLEX_Studio1210/cplex/bin/x86-64_osx/cplex"

import pytest
import numpy as np

from c3x.enomo.models import (
    EnergyStorage,
    EnergySystem,
    Demand,
    Generation,
    Tariff,
    LocalTariff,
)
from c3x.enomo.energy_optimiser import (
    BTMEnergyOptimiser,
    OptimiserObjectiveSet,
    OptimiserObjective,
)

# #Capacity Tariffs
# charge_prices = np.array(([0.085] * 96))
# # charge_prices[range(2, 96, 8)] = 1
# discharge_prices = np.array(([0.09] * 48 + [0.08] * 48))
# # discharge_prices[range(1, 96, 8)] = 1

N_INTERVALS = 48
flat_tariff = np.array([1.0] * N_INTERVALS)
flat_tariff_dct = dict(enumerate(flat_tariff))
flat_load = np.array([1.0] * N_INTERVALS)
flat_generation = np.array([-1.0] * N_INTERVALS)


def to_dict(array):
    return dict(enumerate(array))


def create_battery():
    return EnergyStorage(
        max_capacity=4.0,
        depth_of_discharge_limit=0,
        charging_power_limit=2.0,
        discharging_power_limit=-2.0,
        charging_efficiency=0.9,
        discharging_efficiency=0.9,
        throughput_cost=0.0,
    )
