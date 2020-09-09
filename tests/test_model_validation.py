"""A set of simple tests that validate the values passed into optimiser model instances
"""
# TODO Should probably either correct these or ignore linting in tests
# pylint:disable=unused-wildcard-import,unused-variable,function-redefined

import pytest
import numpy as np
from pydantic import ValidationError
from c3x.enomo.models import (
    Demand,
    Generation,
    Tariff,
    LocalTariff,
    Inverter,
    EnergyStorage,
    Demand,
    Generation,
    EnergySystem,
)


def test_positive_generation_raises_error():
    with pytest.raises(ValueError):
        generation = Generation(np.array([1.0]))


def test_negative_generation_ok():
    generation = Generation(np.array([-1.0]))


def test_zero_generation_ok():
    generation = Generation(np.array([0.0]))


def test_non_numpy_array_generation_raises_error():
    with pytest.raises(ValueError):
        generation = Generation([-1.0])


def test_positive_demand_ok():
    demand = Demand(np.array([1.0]))


def test_negative_raises_error():
    with pytest.raises(ValueError):
        demand = Demand(np.array([-1.0]))


def test_zero_demand_ok():
    demand = Demand(np.array([0.0]))


def test_non_numpy_array_generation_raises_error():
    with pytest.raises(ValueError):
        demand = Demand([1.0])


@pytest.mark.parametrize("tariff_type", ("import_tariff", "export_tariff"))
def test_tariff_accepts_valid_dict(tariff_type):
    tariff = Tariff(**{tariff_type: {1: 1.0, 2: 2, 3: 0}})


@pytest.mark.parametrize("tariff_type", ("import_tariff", "export_tariff"))
def test_tariff_rejects_array(tariff_type):
    with pytest.raises(ValueError):
        tariff = Tariff(**{tariff_type: np.zeros(48)})


@pytest.mark.parametrize("tariff_type", ("import_tariff", "export_tariff"))
def test_tariff_rejects_improper_keys(tariff_type):
    with pytest.raises(ValueError):
        tariff = Tariff(**{tariff_type: {"A": 1.0}})


@pytest.mark.parametrize("tariff_type", ("import_tariff", "export_tariff"))
def test_tariff_rejects_improper_values(tariff_type):
    with pytest.raises(ValueError):
        tariff = Tariff(**{tariff_type: {1: "A"}})


LOCAL_TARIFF_TYPES = (
    "le_import_tariff",
    "le_export_tariff",
    "lt_import_tariff",
    "lt_export_tariff",
    "re_import_tariff",
    "re_export_tariff",
    "rt_import_tariff",
    "rt_export_tariff",
)


@pytest.mark.parametrize("tariff_type", LOCAL_TARIFF_TYPES)
def test_local_tariff_accepts_valid_dict(tariff_type):
    tariff = LocalTariff(**{tariff_type: {1: 1.0, 2: 2, 3: 0}})


@pytest.mark.parametrize("tariff_type", LOCAL_TARIFF_TYPES)
def test_local_tariff_rejects_array(tariff_type):
    with pytest.raises(ValueError):
        tariff = LocalTariff(**{tariff_type: np.zeros(48)})


@pytest.mark.parametrize("tariff_type", LOCAL_TARIFF_TYPES)
def test_local_tariff_rejects_improper_keys(tariff_type):
    with pytest.raises(ValueError):
        tariff = LocalTariff(**{tariff_type: {"A": 1.0}})


@pytest.mark.parametrize("tariff_type", LOCAL_TARIFF_TYPES)
def test_local_tariff_rejects_improper_values(tariff_type):
    with pytest.raises(ValueError):
        tariff = LocalTariff(**{tariff_type: {1: "A"}})


def test_demand_rejects_negative_values():
    with pytest.raises(ValueError):
        demand = Demand(np.array([-1.0]))


def test_demand_accepts_positive_array():
    demand = Demand(np.array([0.0, 1.0]))


def test_demand_accepts_none():
    demand = Demand()


def test_demand_rejects_list():
    with pytest.raises(ValueError):
        demand = Demand([1.0])


def test_generation_rejects_positive_values():
    with pytest.raises(ValueError):
        generation = Generation(np.array([1.0]))


def test_generation_accepts_negative_array():
    generation = Generation(np.array([0.0, -1.0]))


def test_generation_accepts_none():
    generation = Generation()


def test_generation_rejects_list():
    with pytest.raises(ValueError):
        generation = Generation([1.0])


def test_create_valid_inverter():
    inverter = Inverter(5.0, -5.0, 0.9, 0.9, 5.0, 5.0, 1.0, 1.0)


def test_inverter_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        # Invalid charge power limit
        inverter = Inverter(-4.0, 3.0, 1.0, 1.0, 5.0, 5.0, 1.0, 1.0)

    with pytest.raises(ValueError):
        # Invalid discharge power limit
        inverter = Inverter(5.0, 3.0, 1.0, 1.0, 5.0, 5.0, 1.0, 1.0)

    with pytest.raises(ValueError):
        # Invalid charge efficiency
        inverter = Inverter(5.0, 3.0, 1.5, 1.0, 5.0, 5.0, 1.0, 1.0)

    with pytest.raises(ValueError):
        # Invalid discharge efficiency
        inverter = Inverter(5.0, 3.0, 1.0, -1.0, 5.0, 5.0, 1.0, 1.0)


def test_create_valid_storage():
    storage = EnergyStorage(10.0, 0.0, 5.0, -5.0, 1.0, 1.0, 0.0)


def test_energy_storage_validation():
    with pytest.raises(ValueError):
        # Invalid depth of discharge
        storage = EnergyStorage(10.0, 1.5, 5.0, -5.0, 1.0, 1.0, 0.0)

    with pytest.raises(ValueError):
        # Invalid efficiency
        storage = EnergyStorage(10.0, 1.5, 5.0, -5.0, 1.5, 1.0, 0.0)

    with pytest.raises(ValueError):
        # Invalid initial SoC
        storage = EnergyStorage(10.0, 1.5, 5.0, -5.0, 1.0, 1.0, 0.0, 12.0)


def test_energy_system_validation():
    generation = Generation()
    storage = EnergyStorage(10.0, 0.0, 5.0, -5.0, 1.0, 1.0, 0.0)
    inverter = Inverter(5.0, -5.0, 0.9, 0.9, 5.0, 5.0, 1.0, 1.0)
    with pytest.raises(ValidationError):
        system = EnergySystem(None, generation, None, False)

    with pytest.raises(ValidationError):
        system = EnergySystem(generation, None, None, False)

    with pytest.raises(ValidationError):
        system = EnergySystem(generation, None, inverter)

    # These examples should be valid
    system = EnergySystem(None, None, None, False)
    system = EnergySystem(storage, inverter, generation)
    system = EnergySystem(storage, inverter, None)
    system = EnergySystem(None, inverter, generation)

    # TODO This should probably not be valid, but requires a breaking change to the constructor
    system = EnergySystem(storage, None, generation)
