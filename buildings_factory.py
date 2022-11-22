import sys
sys.path.insert(0, "/home/energym/energym/")

import os
from energym.examples.Controller import *
from energym.factory import make
from energym.wrappers.downsample_outputs import DownsampleOutputs
from energym.wrappers.rescale_outputs import RescaleOutputs
from energym.wrappers.rl_wrapper import StableBaselinesRLWrapper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal



buildings_list = ["ApartmentsThermal-v0", "ApartmentsGrid-v0", "Apartments2Thermal-v0",
                  "Apartments2Grid-v0", "OfficesThermostat-v0", "MixedUseFanFCU-v0",
                  "SeminarcenterThermostat-v0", "SeminarcenterFull-v0", "SimpleHouseRad-v0",
                  "SimpleHouseRSla-v0", "SwissHouseRSlaW2W-v0", "SwissHouseRSlaTank-v0"] 
weather_list = ["ESP_CT_Barcelona", "ESP_CT_Barcelona", "ESP_CT_Barcelona",
                "ESP_CT_Barcelona", "GRC_A_Athens", "GRC_A_Athens", 
                "DNK_MJ_Horsens1", "DNK_MJ_Horsens1", "CH_BS_Basel", 
                "CH_BS_Basel", "CH_BS_Basel", "CH_BS_Basel"]

default_controls = [{'P1_T_Tank_sp': [40.0], 'P2_T_Tank_sp': [40.0], 'P3_T_Tank_sp': [40.0],
                     'P4_T_Tank_sp': [40.0], 'Bd_Ch_EVBat_sp': [1.0], 'Bd_DisCh_EVBat_sp': [0.0],
                     'HVAC_onoff_HP_sp': [1.0], 'Bd_T_HP_sp': [45]},
                    {'P1_T_Tank_sp': [45.0], 'P2_T_Tank_sp': [45.0], 'P3_T_Tank_sp':[45.0],
                     'P4_T_Tank_sp': [4.0], 'Bd_Ch_EVBat_sp': [0.0], 'Bd_DisCh_EVBat_sp': [0.0]},
                    {'Bd_Ch_EV1Bat_sp': [0.0], 'Bd_Ch_EV2Bat_sp': [0.0]},
                    {'Bd_Ch_EV1Bat_sp': [0.0], 'Bd_Ch_EV2Bat_sp': [0.0]},
                    {'Bd_Heating_onoff_sp': [1], 'Bd_Cooling_onoff_sp': [0]},
                    {},
                    {},
                    {},
                    {},
                    {},
                    {},
                    {}
                    ]

control_values = [21, 21, 21, 21, 21, 22, 22, 22, 0, 0, 0, 0]
control_frequency = [480, 480, 480, 480, 96, 96, 144, 144, 288, 288, 288, 288]

simulation_days = 100

def get_env(building_name):
    building_idx = buildings_list.index(building_name)
    env = make(building_name, weather=weather_list[building_idx], simulation_days=simulation_days)
    return env

def get_inputs(building_name, _env):
    inputs = _env.get_inputs_names()
    if building_name == "SeminarcenterThermostat-v0": inputs = inputs[1:]
    elif building_name == "OfficesThermostat-v0": inputs = inputs[2:]
    return inputs


controller_list = [lambda inputs, step: LabController(control_list=inputs, lower_tol=0.3, upper_tol=0.8, nighttime_setback=True, nighttime_start=18, nighttime_end=6, nighttime_temp=18).get_control,
                   lambda inputs, step: LabController(control_list=inputs, lower_tol=0.3, upper_tol=0.8, nighttime_setback=True, nighttime_start=18, nighttime_end=6, nighttime_temp=18).get_control,
                   lambda inputs, step: LabController(control_list=inputs, lower_tol=0.3, upper_tol=0.8, nighttime_setback=True, nighttime_start=18, nighttime_end=6, nighttime_temp=18).get_control,
                   lambda inputs, step: LabController(control_list=inputs, lower_tol=0.3, upper_tol=0.8, nighttime_setback=True, nighttime_start=18, nighttime_end=6, nighttime_temp=18).get_control,
                   lambda inputs, step: SimpleController(control_list=inputs, lower_tol=0.3, upper_tol=0.8, nighttime_setback=True, nighttime_start=18, nighttime_end=6, nighttime_temp=18).get_control,
                   lambda inputs, step: MixedUseController(control_list=inputs, lower_tol=0.3, upper_tol=0.8, nighttime_setback=True, nighttime_start=17, nighttime_end=6, nighttime_temp=18).get_control,
                   lambda inputs, step: SimpleController(control_list=inputs, lower_tol=0.3, upper_tol=0.8, nighttime_setback=True, nighttime_start=22, nighttime_end=9, nighttime_temp=17).get_control,
                   lambda inputs, step: SeminarcenterFullController(control_list=inputs, lower_tol=0.3, upper_tol=0.8, nighttime_setback=True, nighttime_start=17, nighttime_end=6, nighttime_temp=18).get_control,
                   lambda inputs, step: {"u": [0.5*(signal.square(0.1*step)+1.0)]},
                   lambda inputs, step: {"u": [0.5*(signal.square(0.1*step)+1.0)]},
                   lambda inputs, step: {"u": [0.5*(signal.square(0.1*step)+1.0)]},
                   lambda inputs, step: {"uHP": [0.5*(signal.square(0.1*step)+1.0)], 'uHP':[0.5*(math.sin(0.01*step)+1.0)]}
                   ]

cols_plot = [[['Z01_T', 'P1_T_Thermostat_sp_out'], ['Ext_T'], ['Fa_Pw_All']],
             [['Z01_T', 'P1_T_Thermostat_sp_out'], ['Ext_T'], ['Fa_Pw_All']],
             [['Z01_T', 'P1_T_Thermostat_sp_out', 'P1_onoff_HP_sp_out'], ['Ext_T'], ['Fa_Pw_All']],
             [['Z01_T', 'P1_T_Thermostat_sp_out'], ['Ext_T'], ['Fa_Pw_All']],
             [['Z17_T', 'Z17_T_Thermostat_sp_out'], ['Ext_T'], ['Fa_Pw_All']],
             [['Z02_T', 'Z02_T_Thermostat_sp_out', 'Bd_Fl_AHU1_sp'], ['Ext_T'], ['Fa_Pw_All']],
             [['Z02_T', 'Z02_T_Thermostat_sp_out'], ['Ext_T'], ['Fa_Pw_All']],
             [['Z22_T', 'Z22_T_Thermostat_sp_out'], ['Ext_T'], ['Bd_onoff_HP1_sp', 'Bd_T_HP1_sp', 'Bd_onoff_HP3_sp', 'Bd_onoff_HP4_sp'], ['Fa_Pw_All']],
             [['temRoo.T', 'temSup.T', 'temRet.T'], ['TOut.T'], ['heaPum.QCon_flow']],
             [['temRoo.T', 'temSup.T', 'temRet.T'], ['TOut.T'], ['heaPum.QCon_flow']],
             [['temRoo.T', 'sla.heatPortEmb[1].T', 'heaPum.TEvaAct'], ['TOut.T'], ['heaPum.QCon_flow']],
             [['temRoo.T', 'sla.heatPortEmb[1].T', 'heaPum.TEvaAct'], ['TOut.T'], ['heaPum.QCon_flow']]
            ]


def collect_offline_data(building_name, iter):
    env = get_env(building_name)
    building_idx = buildings_list.index(building_name)
    inputs = get_inputs(building_name, env)
    controller = controller_list[building_idx]
    offline_data_dir = f"offline_data/{building_name}/"
    os.makedirs(offline_data_dir, exist_ok=True)
    env.reset()
    steps = control_frequency[building_idx]*simulation_days
    outputs = env.step(env.sample_random_action())
    out_list = [outputs]
    hour = 0
    controls = []
    for i in range(steps):
        control = controller(inputs, i)(outputs, control_values[building_idx], hour)
        control.update(default_controls[building_idx])
        controls +=[ {p:control[p][0] for p in control} ]
        print(f"iter: {iter}, step: {i}, controls: {control}")
        outputs = env.step(control)
        _,hour,_,_ = env.get_date()
        out_list.append(outputs)
        if env.time >= env.stop_time: break
    out_df = pd.DataFrame(out_list)
    cmd_df = pd.DataFrame(controls)
    out_df.to_csv(f"{offline_data_dir}/states_{iter}.csv", index=False)
    cmd_df.to_csv(f"{offline_data_dir}/controls_{iter}.csv", index=False)
    
    f, axs = plt.subplots(len(cols_plot[building_idx]),figsize=(10,15))
    colors = ['r', 'b', 'g', 'orange']
    for i, cols in enumerate(cols_plot[building_idx]):
        for j, col in enumerate(cols):
            axs[i].plot(out_df[col], colors[j])
        axs[i].set_ylabel('Temp' if i < 3 else "Power")
        axs[i].set_xlabel('Steps')
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(f"{offline_data_dir}/fig_{iter}.png")
    env.close()
    

if __name__ == "__main__":
    building_name = "OfficesThermostat-v0"
    collect_offline_data(building_name, 0)
    