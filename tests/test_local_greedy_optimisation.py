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


def test_local_greedy_solar_optimisation():
    """Storage should preferentially charge from solar earlier
    when tariffs are equal across time periods
    """
    energy_system = create_energy_system(
        create_battery(),
        np.array([0.0] * 24 + [5.0] * 24),
        np.array([-5.0] * 24 + [0.0] * 24),
        dict(enumerate(SIMPLE_FLAT_TARIFF)),
        dict(enumerate(SIMPLE_FLAT_TARIFF / 2.0)),
        dict(enumerate(SIMPLE_FLAT_TARIFF * 1.5)),
        dict(enumerate(SIMPLE_FLAT_TARIFF / 3.0)),
        dict(enumerate(ZERO_TARIFF)),
        dict(enumerate(ZERO_TARIFF)),
        dict(enumerate(ZERO_TARIFF)),
        dict(enumerate(ZERO_TARIFF)),
    )
    optimiser = LocalEnergyOptimiser(
        30,
        N_INTERVALS,
        energy_system,
        OptimiserObjectiveSet.LocalModels
        + [OptimiserObjective.GreedyGenerationCharging],
    )

    storage_charge_generation = optimiser.values("storage_charge_generation")
    storage_discharge_demand = optimiser.values("storage_discharge_demand")
    for i in range(0, 4):
        assert storage_charge_generation[i] == 1.0
    for i in range(4, N_INTERVALS):
        assert storage_charge_generation[i] == 0.0

    for i in range(0, 24):
        assert storage_discharge_demand[i] == 0.0
    for i in range(24, N_INTERVALS):
        assert storage_discharge_demand[i] == pytest.approx(-1.0 / 6.0, 3)


def test_local_greedy_demand_optimisation():
    """Storage should preferentially discharge to meet demand earlier
    when tariffs are equal across time periods
    """
    energy_system = create_energy_system(
        create_battery(),
        np.array([0.0] * 24 + [5.0] * 24),
        np.array([-5.0] * 24 + [0.0] * 24),
        dict(enumerate(SIMPLE_FLAT_TARIFF)),
        dict(enumerate(SIMPLE_FLAT_TARIFF / 2.0)),
        dict(enumerate(SIMPLE_FLAT_TARIFF * 1.5)),
        dict(enumerate(SIMPLE_FLAT_TARIFF / 3.0)),
        dict(enumerate(ZERO_TARIFF)),
        dict(enumerate(ZERO_TARIFF)),
        dict(enumerate(ZERO_TARIFF)),
        dict(enumerate(ZERO_TARIFF)),
    )
    optimiser = LocalEnergyOptimiser(
        30,
        N_INTERVALS,
        energy_system,
        OptimiserObjectiveSet.LocalModels
        + [OptimiserObjective.GreedyDemandDischarging],
    )

    storage_charge_generation = optimiser.values("storage_charge_generation")
    storage_discharge_demand = optimiser.values("storage_discharge_demand")
    for i in range(0, 24):
        assert storage_charge_generation[i] == pytest.approx(1.0 / 6.0, 3)
    for i in range(24, N_INTERVALS):
        assert storage_charge_generation[i] == 0.0

    for i in range(0, 24):
        assert storage_discharge_demand[i] == 0.0
    for i in range(24, 28):
        assert storage_discharge_demand[i] == -1.0
    for i in range(28, N_INTERVALS):
        assert storage_discharge_demand[i] == 0.0
