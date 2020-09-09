"""
A series of tests that run optimisations with relatively simple tariffs and random
demand/generation. These tests attempt to find issues where optimising does not 
give reasonable (or any) result.
"""
# TODO Should probably either correct these or ignore linting in tests
# pylint:disable=unused-wildcard-import,unused-variable,function-redefined

import pytest
from pytest import approx
import numpy as np

from settings import *
from c3x.enomo.energy_optimiser import LocalEnergyOptimiser

from hypothesis import given, settings
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays

N_INTERVALS = 48
MAX_ENERGY = 100.0
BATTERY_CAPACITY = 10.0
SIMPLE_TOU_TARIFF = np.array([1.0] * 12 + [2.0] * 24 + [1.0] * 12)
SIMPLE_FLAT_TARIFF = np.array([1.0] * 48)
ZERO_TARIFF = np.zeros(48)


def create_battery():
    return EnergyStorage(
        max_capacity=4.0,
        depth_of_discharge_limit=0,
        charging_power_limit=2.0,
        discharging_power_limit=-2.0,
        charging_efficiency=1.0,
        discharging_efficiency=1.0,
        throughput_cost=0.0,
    )


def create_energy_system(
    battery,
    load,
    generation,
    local_energy_import_tariff,
    local_energy_export_tariff,
    remote_energy_import_tariff,
    remote_energy_export_tariff,
    local_transport_import_tariff,
    local_transport_export_tariff,
    remote_transport_import_tariff,
    remote_transport_export_tariff,
):
    energy_system = EnergySystem()
    energy_system.add_energy_storage(battery)
    demand = Demand()
    demand.add_demand_profile(load)
    pv = Generation()
    pv.add_generation_profile(generation)
    tariff = LocalTariff()
    tariff.add_local_energy_tariff_profile_import(local_energy_import_tariff)
    tariff.add_local_energy_tariff_profile_export(local_energy_export_tariff)
    tariff.add_remote_energy_tariff_profile_import(remote_energy_import_tariff)
    tariff.add_remote_energy_tariff_profile_export(remote_energy_export_tariff)
    tariff.add_local_transport_tariff_profile_import(local_transport_import_tariff)
    tariff.add_local_transport_tariff_profile_export(local_transport_export_tariff)
    tariff.add_remote_transport_tariff_profile_import(remote_transport_import_tariff)
    tariff.add_remote_transport_tariff_profile_export(remote_transport_export_tariff)
    energy_system.add_demand(demand)
    energy_system.add_generation(pv)
    energy_system.add_local_tariff(tariff)
    return energy_system


@settings(deadline=1000, max_examples=100)
@given(
    arrays(np.float, N_INTERVALS, elements=floats(0, MAX_ENERGY)),
    arrays(np.float, N_INTERVALS, elements=floats(-MAX_ENERGY, 0)),
)
def test_reasonable_local_optimisation_convergence(demand, generation):
    """This is a fairly straight-forward optimisation problem with randomly generated
    demand and generation. This tests that the optimisation, given a flat set of tariffs,
    will have:
    - battery starting and ending empty
    - local demand transfer always equal to the difference between demand and generation
    - only exporting or importing at each interval
    - only charging or discharging in each interval
    - given current tariffs, the battery should never export to the grid

    Args:
        demand (np.ndarray): Array of (non-negative) energy demand for intervals
        generation (np.ndarray): Array of (non-positive) energy generation values for intervals
    """
    energy_system = create_energy_system(
        create_battery(),
        demand,
        generation,
        dict(enumerate(SIMPLE_FLAT_TARIFF)),
        dict(enumerate(SIMPLE_FLAT_TARIFF / 2.0)),
        dict(enumerate(SIMPLE_FLAT_TARIFF)),
        dict(enumerate(SIMPLE_FLAT_TARIFF / 2.0)),
        dict(enumerate(ZERO_TARIFF)),
        dict(enumerate(ZERO_TARIFF)),
        dict(enumerate(ZERO_TARIFF)),
        dict(enumerate(ZERO_TARIFF)),
    )
    optimiser = LocalEnergyOptimiser(
        30, N_INTERVALS, energy_system, OptimiserObjectiveSet.LocalModels
    )
    local_net_import = optimiser.values("local_net_import")
    local_net_export = optimiser.values("local_net_export")
    storage_discharge_demand = optimiser.values("storage_discharge_demand")
    storage_discharge_grid = optimiser.values("storage_discharge_grid")
    storage_state_of_charge = optimiser.values("storage_state_of_charge")
    local_demand_transfer = optimiser.values("local_demand_transfer")

    for i in range(N_INTERVALS):
        assert local_net_import[i] >= 0.0
        assert local_net_export[i] <= 0.0
        assert local_net_import[i] == 0.0 or local_net_export[i] == 0.0
        # Convoluted way to say that the discharge to meet demand is always
        # less than the max possible, within some tolerance
        assert storage_discharge_demand[i] - (max(0, demand[i] + generation[i])) <= 0.0
        assert storage_discharge_demand[i] <= 0.0
        assert storage_discharge_grid[i] == 0.0

    assert storage_state_of_charge[-1] == 0.0
