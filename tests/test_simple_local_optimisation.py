"""
A set of tests to ensure simple optimisation functionality is preserved.
"""
# TODO Should probably either correct these or ignore linting in tests
# pylint:disable=unused-wildcard-import,unused-variable,function-redefined
from settings import *
import pytest

from c3x.enomo.energy_optimiser import LocalEnergyOptimiser

N_INTERVALS = 48
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

@pytest.mark.solver('miqp')
def test_no_action_for_equal_export_import_price():
    """Given a simple tariff structure where transport costs are zero, and import
    tariffs are twice export tariffs, with no distinction between local and remote tariffs,
    the battery should not charge at all.
    """
    energy_system = create_energy_system(
        create_battery(),
        np.array([0.0] * 24 + [5.0] * 24),
        np.array([-5.0] * 24 + [0.0] * 24),
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

    np.testing.assert_array_equal(
        optimiser.values("storage_state_of_charge"), np.zeros(N_INTERVALS)
    )

@pytest.mark.solver('miqp')
def test_simple_arbitrage_for_cheaper_local_energy():
    """When the local energy import tariff is cheaper than the remote energy import tariff,
    the battery should charge off excess local generation.
    """
    energy_system = create_energy_system(
        create_battery(),
        np.array([0.0] * 24 + [5.0] * 24),
        np.array([-5.0] * 24 + [0.0] * 24),
        dict(enumerate(SIMPLE_FLAT_TARIFF / 2.0)),
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
    storage_charge_total = optimiser.values("storage_charge_total")
    storage_discharge_total = optimiser.values("storage_discharge_total")
    for i in range(0, 24):
        assert storage_discharge_total[i] == 0.0
        assert storage_charge_total[i] == pytest.approx(
            energy_system.energy_storage.capacity / 24.0, 0.01
        )
    for i in range(24, 48):
        assert storage_discharge_total[i] == pytest.approx(
            -energy_system.energy_storage.capacity / 24.0, 0.01
        )
        assert storage_charge_total[i] == 0.0

@pytest.mark.solver('miqp')
def test_cannot_remote_export_before_satisfying_local_demand():
    """Test that we cannot create an electrically infeasible solution where the battery
    is discharging into the remote grid (e.g. to take advantage of a high remote export value)
    when there is local demand that must be satisfied first
    """
    energy_storage = create_battery()
    energy_storage.max_capacity = 4.0 * 48
    energy_storage.initial_state_of_charge = energy_storage.max_capacity
    energy_storage.discharging_power_limit = -4.0
    energy_system = create_energy_system(
        energy_storage,
        np.array([2.0] * 48),
        np.array([0.0] * 48),
        dict(enumerate(SIMPLE_FLAT_TARIFF)),
        dict(enumerate(SIMPLE_FLAT_TARIFF / 3.0)),
        dict(enumerate(SIMPLE_FLAT_TARIFF)),
        dict(enumerate(SIMPLE_FLAT_TARIFF / 1.1)),
        dict(enumerate(ZERO_TARIFF)),
        dict(enumerate(ZERO_TARIFF)),
        dict(enumerate(ZERO_TARIFF)),
        dict(enumerate(ZERO_TARIFF)),
    )
    optimiser = LocalEnergyOptimiser(
        30, N_INTERVALS, energy_system, OptimiserObjectiveSet.LocalModels
    )

    np.testing.assert_array_equal(
        optimiser.values("local_net_import"), np.zeros(N_INTERVALS)
    )
    np.testing.assert_array_equal(
        optimiser.values("local_net_export"), np.zeros(N_INTERVALS)
    )
    np.testing.assert_array_equal(
        optimiser.values("storage_discharge_demand"), np.ones(N_INTERVALS) * -2.0
    )
    np.testing.assert_array_equal(
        optimiser.values("local_demand_transfer"), np.zeros(N_INTERVALS)
    )


@pytest.mark.skip(
    "Optimisation fails to converge when remote export price > local export under certain scenarios"
)
def test_export_when_remote_export_value_greater_than_local_export_value():
    """When there is some constant local demand, but enough capacity to satisfy this
    and also export at a higher value, we should still find a solution that maximises value.

    Note: this test currently fails to converge. Setting demand to zero works fine, 
    but any non-zero demand fails to converge.
    """
    energy_storage = create_battery()
    energy_storage.max_capacity = 4.0 * 48
    energy_storage.initial_state_of_charge = energy_storage.max_capacity
    energy_storage.discharging_power_limit = -20.0
    energy_system = create_energy_system(
        energy_storage,
        np.array([1.0] * 48),
        np.array([0.0] * 48),
        dict(enumerate(SIMPLE_FLAT_TARIFF)),
        dict(enumerate(SIMPLE_FLAT_TARIFF * 0.1)),
        dict(enumerate(SIMPLE_FLAT_TARIFF)),
        dict(enumerate(SIMPLE_FLAT_TARIFF * 0.2)),
        dict(enumerate(ZERO_TARIFF)),
        dict(enumerate(ZERO_TARIFF)),
        dict(enumerate(ZERO_TARIFF)),
        dict(enumerate(ZERO_TARIFF)),
    )
    optimiser = LocalEnergyOptimiser(
        30, N_INTERVALS, energy_system, OptimiserObjectiveSet.LocalModels
    )

    np.testing.assert_array_equal(
        optimiser.values("storage_discharge_demand"), np.ones(N_INTERVALS) * -2.0
    )
    np.testing.assert_array_equal(
        optimiser.values("storage_discharge_grid"), np.ones(N_INTERVALS) * -2.0
    )
    np.testing.assert_array_equal(
        optimiser.values("local_demand_transfer"), np.zeros(N_INTERVALS)
    )

@pytest.mark.solver('miqp')
def test_cannot_remote_import_before_satisfying_local_generation():
    """Test that we cannot create an electrically infeasible solution where the battery
    is charging remotely from the grid while there is excess generation.

    This test requires the two electrical feasibility charge constraints to be active
    in order to pass.
    """
    energy_storage = create_battery()
    energy_storage.max_capacity = 4.0 * 48
    energy_storage.discharging_power_limit = -4.0
    energy_system = create_energy_system(
        energy_storage,
        np.array([0.0] * 48),
        np.array([-2.0] * 48),
        dict(enumerate(SIMPLE_FLAT_TARIFF)),
        dict(enumerate(SIMPLE_FLAT_TARIFF / 3.0)),
        dict(enumerate(-SIMPLE_FLAT_TARIFF * 1.0)),
        dict(enumerate(SIMPLE_FLAT_TARIFF / 3.0)),
        dict(enumerate(ZERO_TARIFF)),
        dict(enumerate(ZERO_TARIFF)),
        dict(enumerate(ZERO_TARIFF)),
        dict(enumerate(ZERO_TARIFF)),
    )
    optimiser = LocalEnergyOptimiser(
        30, N_INTERVALS, energy_system, OptimiserObjectiveSet.LocalModelsThirdParty
    )

    np.testing.assert_array_equal(
        optimiser.values("local_net_import"), np.zeros(N_INTERVALS)
    )
    np.testing.assert_array_equal(
        optimiser.values("local_net_export"), np.ones(N_INTERVALS) * -2.0
    )
    np.testing.assert_array_equal(
        optimiser.values("storage_charge_generation"), np.zeros(N_INTERVALS)
    )
    np.testing.assert_array_equal(
        optimiser.values("storage_charge_grid"), np.zeros(N_INTERVALS)
    )
    np.testing.assert_array_equal(
        optimiser.values("storage_state_of_charge"), np.zeros(N_INTERVALS)
    )
