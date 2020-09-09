from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyomo.core import Var

# Since using the data-driven data for the testing, use their battery class
import sys

from c3x.enomo.models import EnergyStorage, EnergySystem, Demand, Generation, Tariff, DispatchRequest, LocalTariff
from c3x.enomo.energy_optimiser import EnergyOptimiser, OptimiserObjectiveSet, LocalEnergyOptimiser


# set up seaborn the way you like
sns.set_style({'axes.linewidth': 1, 'axes.edgecolor': 'black', 'xtick.direction': \
    'out', 'xtick.major.size': 4.0, 'ytick.direction': 'out', 'ytick.major.size': 4.0, \
               'axes.facecolor': 'white', 'grid.color': '.8', 'grid.linestyle': u'-', 'grid.linewidth': 0.5})

############################ Define an Example Optimisation Problem ########################################

battery = EnergyStorage(max_capacity=15.0,
                        depth_of_discharge_limit=0,
                        charging_power_limit=5.0,
                        discharging_power_limit=-5.0,
                        charging_efficiency=1,
                        discharging_efficiency=1,
                        throughput_cost=0.018)

# The load and pv arrays below are in kwh consumed per 15 minutes
test_load = np.array([2.13, 2.09, 2.3, 2.11, 2.2, 2.23, 2.2, 2.15, 2.02, 2.19, 2.19, 2.19, 2.12, 2.15, 2.25, 2.12, 2.21, 2.16,
                      2.26, 2.13, 2.08, 2.15, 2.42, 2.02, 2.3, 2.26, 2.35, 2.55, 3.23, 2.98, 3.49, 3.5, 3.12, 3.52, 3.94, 3.55,
                      3.99, 3.71, 3.38, 3.76, 3.71, 3.78, 3.29, 3.65, 3.61, 3.75, 3.38, 3.66, 3.56, 3.69, 3.3, 3.61, 3.71, 3.82,
                      3.17, 3.69, 3.74, 3.86, 3.57, 3.55, 3.75, 3.6, 3.67, 3.48, 3.51, 3.46, 3.19, 3.38, 3.19, 3.38, 3.04, 3.12,
                      2.91, 3.11, 3.13, 2.77, 2.24, 2.54, 2.24, 2.24, 2.09, 2.33, 2.17, 2.16, 1.97, 2.16, 2.21, 2.18, 2.01, 2.16,
                      2.19, 2.11, 2.17, 2.13, 2.05, 2.19])

test_pv = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.23, 0.52,
                    0.74, 0.71, 0.63, 0.68, 0.97, 0.01, 0.52, 0.83, 0.83, 0.79, 1.22, 1.36, 1.27, 1.42, 1.97, 2.56, 2.91, 3.24,
                    3.8, 4.3, 4.62, 4.84, 4.6, 4.17, 3.77, 3.76, 3.38, 2.64, 1.96, 1.76, 1.85, 2.4, 3.82, 5.13, 4.97, 5.02, 5.43,
                    5.32, 3.56, 1.75, 1.43, 1.65, 1.69, 2.3, 2.71, 2.41, 2.63, 2.6, 1.9, 0.78, 0.13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
test_pv *= -1 #convert solar generation to negative to match convention.

net_load = test_load + test_pv
# split load into import and export
connection_point_import = np.copy(net_load)
connection_point_export = np.copy(net_load)
for j, e in enumerate(net_load):
    if e >= 0:
        connection_point_export[j] = 0
    else:
        connection_point_import[j] = 0

# Include additional data for testing local energy model optimisation
connection_point_export[50] = -1.53
net_load[50] += connection_point_export[50]
import_load_dct = dict(enumerate(connection_point_import))
export_load_dct = dict(enumerate(connection_point_export))

# Local Tariffs are in $ / kwh
# Storage arbitrage to the grid and back
# le_import_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12))
# le_export_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12))
# lt_import_tariff = np.array(([0.0] * 96))
# lt_export_tariff = np.array(([0.0] * 96))
# re_import_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12))
# re_export_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12))
# rt_import_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12))*0.0
# rt_export_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12))*0.0

# # Storage satisfying local network demand.
# le_import_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12))
# le_export_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12))
# lt_import_tariff = np.array(([0.0] * 96))
# lt_export_tariff = np.array(([0.0] * 96))
# re_import_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12))
# re_export_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12))*0.0
# rt_import_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12))*0.0
# rt_export_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12))*0.0

# # Storage satisfying local network demand.
# le_import_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12))
# le_export_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12))
# lt_import_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12)) * 0.1
# lt_export_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12)) * 0.1
# re_import_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12))
# re_export_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12)) * 0.0
# rt_import_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12)) * 0.1
# rt_export_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12)) * 0.1

# Doesnt converge on a solution - Why?
# le_import_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12))
# le_export_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12))
# lt_import_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12)) * 0.1
# lt_export_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12)) * 0.1
# re_import_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12)) * 0.0
# re_export_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12)) * 1.0
# rt_import_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12)) * 0.1
# rt_export_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12)) * 0.1

# Offsetting local load from local solar
# le_import_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12)) * 0.1
# le_export_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12)) * 0.1
# lt_import_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12)) * 0.1
# lt_export_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12)) * 0.1
# re_import_tariff = np.array(([0.5] * 96))
# re_export_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12)) * 0.0
# rt_import_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12)) * 0.1
# rt_export_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12)) * 0.1

# Favouring Cheaper Local Energy (3rd Party Optimisation)
le_import_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12)) * 0.5
le_export_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12)) * 0.5
lt_import_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12)) * 0.1
lt_export_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12)) * 0.1
re_import_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12))
re_export_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12))
rt_import_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12)) * 0.1
rt_export_tariff = np.array(([0.1] * 28 + [0.3] * 8 + [0.2] * 32 + [0.3] * 16 + [0.1] * 12)) * 0.1

le_export_tariff_dct = dict(enumerate(le_export_tariff))
le_import_tariff_dct = dict(enumerate(le_import_tariff))
lt_export_tariff_dct = dict(enumerate(lt_export_tariff))
lt_import_tariff_dct = dict(enumerate(lt_import_tariff))
re_export_tariff_dct = dict(enumerate(re_export_tariff))
re_import_tariff_dct = dict(enumerate(re_import_tariff))
rt_export_tariff_dct = dict(enumerate(rt_export_tariff))
rt_import_tariff_dct = dict(enumerate(rt_import_tariff))



colors = sns.color_palette()
hrs = np.arange(0, len(test_load)) / 4
fig = plt.figure(figsize=(14, 4))
ax1 = fig.add_subplot(1, 1, 1)
l1, = ax1.plot(hrs, 4 * test_load, color=colors[0])
l2, = ax1.plot(hrs, 4 * test_pv, color=colors[1])
l3, = ax1.plot(hrs, 4 * net_load, color=colors[2])
ax1.set_xlabel('hour'), ax1.set_ylabel('kW')
ax1.legend([l1, l2, l3], ['Load', 'PV', 'Connection Point'], ncol=2)
ax1.set_xlim([0, len(test_load) / 4])
fig.tight_layout()

fig.show()

############################ Optimise this Example ########################################

energy_system = EnergySystem()
energy_system.add_energy_storage(battery)
load = Demand()
load.add_demand_profile(connection_point_import)
pv = Generation()
pv.add_generation_profile(connection_point_export)
local_tariff = LocalTariff()
local_tariff.add_local_energy_tariff_profile_export(le_export_tariff_dct)
local_tariff.add_local_energy_tariff_profile_import(le_import_tariff_dct)
local_tariff.add_local_transport_tariff_profile_export(lt_export_tariff_dct)
local_tariff.add_local_transport_tariff_profile_import(lt_import_tariff_dct)
local_tariff.add_remote_energy_tariff_profile_export(re_export_tariff_dct)
local_tariff.add_remote_energy_tariff_profile_import(re_import_tariff_dct)
local_tariff.add_remote_transport_tariff_profile_export(rt_export_tariff_dct)
local_tariff.add_remote_transport_tariff_profile_import(rt_import_tariff_dct)
energy_system.add_demand(load)
energy_system.add_generation(pv)
energy_system.add_local_tariff(local_tariff)


# Dispatch capabilities will be added in a future version
'''dispatch = DispatchRequest()
req = [[4, 7, 12], [(0, 5000), (0, 15000), (0, 28000)]]
req = [[4], [(0, 15000)]]
dispatch.add_dispatch_request_linear_ramp(req)
energy_system.add_dispatch(dispatch)'''

# Invoke the optimiser and optimise
local_energy_models = True
optimiser = LocalEnergyOptimiser(15, 96, energy_system, OptimiserObjectiveSet.LocalModelsThirdParty + OptimiserObjectiveSet.LocalPeakOptimisation)




############################ Analyse the Optimisation ########################################
storage_energy_delta = optimiser.values('storage_charge_grid') +\
                       optimiser.values('storage_charge_generation') +\
                       optimiser.values('storage_discharge_demand') +\
                       optimiser.values('storage_discharge_grid')



colors = sns.color_palette()
hrs = np.arange(0, len(test_load)) / 4
fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(2, 1, 1)
l1, = ax1.plot(hrs, 4 * test_load, color=colors[0])
l2, = ax1.plot(hrs, 4 * test_pv, color=colors[1])
l4, = ax1.plot(hrs, 4 * storage_energy_delta, color=colors[3])
ax1.set_xlabel('hour'), ax1.set_ylabel('kW')
ax1.legend([l1, l2, l4], ['Load', 'PV', 'Storage'], ncol=3)
ax1.set_xlim([0, len(test_load) / 4])
ax3 = fig.add_subplot(2, 1, 2)
l1, = ax3.plot(hrs, storage_energy_delta * 4, color=colors[5])
l2, = ax3.plot(hrs, optimiser.values('storage_state_of_charge'), color=colors[4])
ax3.set_xlabel('hour'), ax3.set_ylabel('action')
ax3.legend([l1, l2], ['battery action (kW)', 'SOC (kWh)'], ncol=2)
ax3.set_xlim([0, len(test_load) / 4])
fig.tight_layout()
plt.show()

net_grid_flow = 4 * optimiser.values('storage_charge_grid') + 4 * optimiser.values('storage_discharge_grid') + 4 * optimiser.values('local_net_import') + 4 * optimiser.values('local_net_export')

fig = plt.figure(figsize=(14, 7))
ax11 = fig.add_subplot(2, 1, 1)
l1, = ax11.plot(hrs, 4 * net_load, color=colors[0])
l2, = ax11.plot(hrs, 4 * optimiser.values('storage_charge_grid'), color=colors[1])
l3, = ax11.plot(hrs, 4 * optimiser.values('storage_charge_generation'), color=colors[2])
l4, = ax11.plot(hrs, 4 * optimiser.values('storage_discharge_demand'), color=colors[3])
l5, = ax11.plot(hrs, 4 * optimiser.values('storage_discharge_grid'), color=colors[4])
l6, = ax11.plot(hrs, 4 * optimiser.values('local_net_import'), color=colors[5])
l7, = ax11.plot(hrs, 4 * optimiser.values('local_net_export'), color=colors[6])
l8, = ax11.plot(hrs, 4 * optimiser.values('local_demand_transfer'), color=colors[8])
ax11.set_xlabel('hour'), ax1.set_ylabel('kW')
ax11.legend([l1, l2, l3, l4, l5, l6, l7, l8], ['Net', 'storage_charge_grid', 'storage_charge_generation', 'storage_discharge_load', 'storage_discharge_grid', 'Net Customer Import', 'Net Customer Export', 'Local Transfer'], ncol=3)
ax11.set_xlim([0, len(test_load) / 4])
ax33 = fig.add_subplot(2, 1, 2)
l33, = ax33.plot(hrs, net_grid_flow, color=colors[0])
ax33.set_xlabel('hour'), ax3.set_ylabel('kW')
ax33.legend([l33], ['Net Grid Flows'], ncol=2)
ax33.set_xlim([0, len(test_load) / 4])
plt.show()