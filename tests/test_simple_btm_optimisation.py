"""
A set of tests to ensure simple optimisation functionality is preserved.
"""
# TODO Should probably either correct these or ignore linting in tests
# pylint:disable=unused-wildcard-import,unused-variable,function-redefined
from settings import *
import pytest


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


def create_energy_system(battery, load, generation, import_tariff, export_tariff):
    energy_system = EnergySystem()
    energy_system.add_energy_storage(battery)
    demand = Demand()
    demand.add_demand_profile(load)
    pv = Generation()
    pv.add_generation_profile(generation)
    tariff = Tariff()
    tariff.add_tariff_profile_export(to_dict(export_tariff))
    tariff.add_tariff_profile_import(to_dict(import_tariff))
    energy_system.add_demand(demand)
    energy_system.add_generation(pv)
    energy_system.add_tariff(tariff)
    return energy_system


def test_no_action_for_equal_export_import_price():
    energy_system = create_energy_system(
        create_battery(),
        flat_load * 0.0,
        flat_generation * 0.0,
        flat_tariff,
        flat_tariff,
    )
    optimiser = BTMEnergyOptimiser(
        30, N_INTERVALS, energy_system, OptimiserObjectiveSet.FinancialOptimisation
    )
    # test

    np.testing.assert_array_equal(
        optimiser.values("storage_state_of_charge"), np.zeros(N_INTERVALS)
    )


@pytest.mark.parametrize(
    "charging_efficiency,discharging_efficiency,should_charge",
    [(1.0, 1.0, True), (0.9, 0.9, True), (0.8, 0.8, True), (0.7, 0.7, False)],
)
def test_charging_in_interval_with_excess_generation(
    charging_efficiency: float, discharging_efficiency: float, should_charge: bool
):
    """Test that the system under financial optimisation charges in the first period 
    provided that the round trip efficiency is above the tariff rate difference (50%)
    
    Args:
        charging_efficiency (float): Battery charging efficiency
        discharging_efficiency (float): Battery discharge efficiency
        should_charge (bool): Flag to indicate whether the efficiency is high enough that 
            the system should optimise for discharging to meet load
    """
    generation = np.array([-1.0] + [0.0] * (N_INTERVALS - 1))
    load = np.array([0.0] + [1.0] + [0.0] * (N_INTERVALS - 2))
    battery = EnergyStorage(
        max_capacity=1.0,
        depth_of_discharge_limit=0,
        charging_power_limit=2.0,
        discharging_power_limit=-2.0,
        charging_efficiency=charging_efficiency,
        discharging_efficiency=discharging_efficiency,
        throughput_cost=0.0,
    )
    energy_system = create_energy_system(
        battery, load, generation, flat_tariff, flat_tariff / 2.0
    )
    optimiser = BTMEnergyOptimiser(
        30, N_INTERVALS, energy_system, OptimiserObjectiveSet.FinancialOptimisation
    )

    # We should charge in the first interval for zero export, then discharge in the second
    # The amount discharged depends on efficiency
    if should_charge:
        np.testing.assert_almost_equal(optimiser.values("btm_net_export")[0], 0.0)
        np.testing.assert_almost_equal(
            optimiser.values("btm_net_import")[1],
            1 - (charging_efficiency * discharging_efficiency),
        )
    else:
        np.testing.assert_almost_equal(optimiser.values("btm_net_export")[0], -1.0)
        np.testing.assert_almost_equal(optimiser.values("btm_net_import")[1], 1.0)
    np.testing.assert_array_almost_equal(
        optimiser.values("btm_net_export")[1:], np.zeros(N_INTERVALS - 1)
    )
    np.testing.assert_almost_equal(optimiser.values("btm_net_import")[0], 0.0)
    np.testing.assert_array_equal(
        optimiser.values("btm_net_import")[2:], np.zeros(N_INTERVALS - 2)
    )


def test_charge_discharge_across_multiple_intervals():
    """
    Test that we spread charge and discharge periods across intervals with the same 
    generation/demand, when we aren't optimising for greedy charge/discharge
    """
    generation = np.array([-1.0] * (N_INTERVALS // 2) + [0.0] * (N_INTERVALS // 2))
    load = np.array([0.0] * (N_INTERVALS // 2) + [1.0] * (N_INTERVALS // 2))

    battery = EnergyStorage(
        max_capacity=1.0,
        depth_of_discharge_limit=0,
        charging_power_limit=2.0,
        discharging_power_limit=-2.0,
        charging_efficiency=1.0,
        discharging_efficiency=1.0,
        throughput_cost=0.0,
    )
    energy_system = create_energy_system(
        battery, load, generation, flat_tariff, flat_tariff / 2.0
    )
    optimiser = BTMEnergyOptimiser(
        30, N_INTERVALS, energy_system, OptimiserObjectiveSet.FinancialOptimisation
    )

    energy_per_interval = 1.0 / (N_INTERVALS / 2)

    np.testing.assert_array_almost_equal(
        optimiser.values("btm_net_export")[0 : N_INTERVALS // 2],
        -np.ones(N_INTERVALS // 2) + energy_per_interval,
        3,
    )
    np.testing.assert_array_almost_equal(
        optimiser.values("btm_net_export")[N_INTERVALS // 2 :],
        np.zeros(N_INTERVALS // 2),
        3,
    )
    np.testing.assert_array_almost_equal(
        optimiser.values("btm_net_import")[N_INTERVALS // 2 :],
        np.ones(N_INTERVALS // 2) - energy_per_interval,
        3,
    )
    np.testing.assert_array_almost_equal(
        optimiser.values("btm_net_import")[0 : N_INTERVALS // 2],
        np.zeros(N_INTERVALS // 2),
        3,
    )


def test_charge_discharge_greedy():
    """
    Test that we spread charge and discharge greedily when they are included in the
    optimisation
    """
    generation = np.array([-1.0] * (N_INTERVALS // 2) + [0.0] * (N_INTERVALS // 2))
    load = np.array([0.0] * (N_INTERVALS // 2) + [1.0] * (N_INTERVALS // 2))
    battery = EnergyStorage(
        max_capacity=4.0,
        depth_of_discharge_limit=0,
        charging_power_limit=2.0,
        discharging_power_limit=-2.0,
        charging_efficiency=1.0,
        discharging_efficiency=1.0,
        throughput_cost=0.0,
    )
    energy_system = create_energy_system(
        battery, load, generation, flat_tariff, flat_tariff / 2.0
    )
    optimiser = BTMEnergyOptimiser(
        30,
        N_INTERVALS,
        energy_system,
        OptimiserObjectiveSet.FinancialOptimisation
        + [
            OptimiserObjective.GreedyGenerationCharging,
            OptimiserObjective.GreedyDemandDischarging,
        ],
    )

    energy_per_interval = 1.0 / (N_INTERVALS / 2)

    np.testing.assert_array_almost_equal(
        optimiser.values("btm_net_export")[0:4], -np.zeros(4)
    )
    np.testing.assert_array_almost_equal(
        optimiser.values("btm_net_export")[4 : N_INTERVALS // 2],
        np.ones(N_INTERVALS // 2 - 4) * -1,
    )
    np.testing.assert_array_almost_equal(
        optimiser.values("btm_net_export")[N_INTERVALS // 2 :],
        np.zeros(N_INTERVALS // 2),
    )
    np.testing.assert_array_almost_equal(
        optimiser.values("btm_net_import")[0 : N_INTERVALS // 2 + 4],
        -np.zeros(N_INTERVALS // 2 + 4),
    )
    np.testing.assert_array_almost_equal(
        optimiser.values("btm_net_import")[N_INTERVALS // 2 + 4 :],
        np.ones(N_INTERVALS // 2 - 4),
    )

