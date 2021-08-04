#!/usr/bin/env python3
import glob
import os
import re
import sys
import time
import dask.bag as db
from dask_jobqueue import SGECluster
from dask.distributed import Client
import joblib
import numpy as np
import pandas as pd
import xarray as xr

output = 'PM2_5_DRY'
#output = 'o3_6mDM8h'

normal = False # 20 percent intervals
extra = False # additional ones for the emission trend matching
climate_cobenefits = False
top_down_2020_baseline = True

data_dir = sys.argv[1]
out_dir = sys.argv[2]
EMULATORS = None

def get_emulator_files(file_path=data_dir, file_pattern="emulator*"):
    emulator_files = glob.glob(os.sep.join([file_path, file_pattern]))
    return emulator_files


def load_emulator(emulator_file):
    lat, lon = [float(item) for item in re.findall(r"\d+\.\d+", emulator_file)]
    emulator = joblib.load(emulator_file)
    return lat, lon, emulator


def create_dataset(results):
    res = results[0]["res"]
    ind = results[0]["ind"]
    tra = results[0]["tra"]
    agr = results[0]["agr"]
    ene = results[0]["ene"]
    if normal:
        filename = f"RES{res:.1f}_IND{ind:.1f}_TRA{tra:.1f}_AGR{agr:.1f}_ENE{ene:.1f}"
    if extra or top_down_2020_baseline:
        filename = f"RES{res:.3f}_IND{ind:.3f}_TRA{tra:.3f}_AGR{agr:.3f}_ENE{ene:.3f}"
    if climate_cobenefits:
        filename = f"RES{res:.3f}_IND{ind:.3f}_TRA{tra:.3f}_AGR{agr:.3f}_ENE{ene:.3f}"

    lat = [item["lat"] for item in results]
    lon = [item["lon"] for item in results]
    result = [item["result"] for item in results]

    df_results = pd.DataFrame([lat, lon, result]).T
    df_results.columns = ["lat", "lon", output]
    df_results = df_results.set_index(["lat", "lon"]).sort_index()
    ds_custom_output = xr.Dataset.from_dataframe(df_results)
    ds_custom_output.to_netcdf(f"{out_dir}ds_{filename}_{output}.nc")
    print(f"completed for {filename}")


def custom_predicts(custom_input):
    def emulator_wrap(emulator):
        lat, lon, emulator = emulator
        return {
            "lat": lat,
            "lon": lon,
            "res": custom_input[0][0],
            "ind": custom_input[0][1],
            "tra": custom_input[0][2],
            "agr": custom_input[0][3],
            "ene": custom_input[0][4],
            "result": emulator.predict(custom_input)[0],
        }

    global EMULATORS
    if not EMULATORS:
        emulator_files = get_emulator_files()
        EMULATORS = list(map(load_emulator, emulator_files))
    emulators = EMULATORS

    results = list(map(emulator_wrap, emulators))
    create_dataset(results)


def main():
    # dask cluster and client
    n_processes = 1
    n_jobs = 35
    n_workers = n_processes * n_jobs

    cluster = SGECluster(
        interface="ib0",
        walltime="01:00:00",
        memory=f"2 G",
        resource_spec=f"h_vmem=2G",
        scheduler_options={
            "dashboard_address": ":5757",
        },
        job_extra=[
            "-cwd",
            "-V",
            f"-pe smp {n_processes}",
            f"-l disk=1G",
        ],
        local_directory=os.sep.join([os.environ.get("PWD"), "dask-worker-space"]),
    )

    client = Client(cluster)

    cluster.scale(jobs=n_jobs)

    time_start = time.time()

    # custom inputs
    if normal:
        matrix_stacked = np.array(
            np.meshgrid(
                np.linspace(0, 1.5, 16),  # 1.5 and 16 for 0.1, 1.5 and 6 for 0.3, 1.4 and 8 for 0.2
                np.linspace(0, 1.5, 16),
                np.linspace(0, 1.5, 16),
                np.linspace(0, 1.5, 16),
                np.linspace(0, 1.5, 16),
            )
        ).T.reshape(-1, 5)
        custom_inputs_set = set(tuple(map(float, map("{:.1f}".format, item))) for item in matrix_stacked)

        custom_inputs_completed_filenames = glob.glob(f"/nobackup/earlacoa/machinelearning/data_annual/predictions/{output}/ds*{output}*")
        custom_inputs_completed_list = []
        for custom_inputs_completed_filename in custom_inputs_completed_filenames:
            custom_inputs_completed_list.append([float(item) for item in re.findall(r"\d+\.\d+", custom_inputs_completed_filename)])

        custom_inputs_completed_set = set(tuple(item) for item in custom_inputs_completed_list)
        custom_inputs_remaining_set = custom_inputs_set - custom_inputs_completed_set
        custom_inputs = [np.array(item).reshape(1, -1) for item in custom_inputs_remaining_set]
        print(f"custom inputs remaining for {output}: {len(custom_inputs)}")

    if extra:
        custom_inputs_main = [
            np.array([[1.15, 1.27, 0.98, 0.98, 1.36]]), # bottom-up 2010
            np.array([[1.19, 1.30, 1.01, 1.01, 1.46]]), # bottom-up 2011
            np.array([[1.20, 1.30, 1.01, 1.02, 1.39]]), # bottom-up 2012
            np.array([[1.13, 1.29, 1.02, 1.01, 1.29]]), # bottom-up 2013
            np.array([[1.06, 1.12, 0.99, 1.01, 1.12]]), # bottom-up 2014
            np.array([[0.92, 0.84, 0.97, 0.99, 0.94]]), # bottom-up 2016
            np.array([[0.84, 0.81, 0.99, 0.99, 0.89]]), # bottom-up 2017
            np.array([[0.76 , 0.934, 0.735, 0.683, 0.708]]),
            np.array([[0.704, 0.786, 0.73 , 0.659, 0.6  ]]),
            np.array([[0.712, 0.703, 0.725, 0.676, 0.649]]),
            np.array([[0.739, 0.668, 0.701, 0.686, 0.682]]),
            np.array([[0.67 , 0.609, 0.709, 0.621, 0.661]]),
            np.array([[0.744, 0.904, 0.778, 0.678, 0.716]]),
            np.array([[0.771, 0.835, 0.711, 0.685, 0.544]]),
            np.array([[0.647, 0.945, 0.746, 0.588, 0.473]]),
            np.array([[0.657, 0.745, 0.714, 0.613, 0.591]]),
            np.array([[0.582, 0.7  , 0.672, 0.5  , 0.492]]),
            np.array([[0.803, 0.835, 0.742, 0.71 , 0.717]]),
            np.array([[0.721, 0.863, 0.712, 0.74 , 0.709]]),
            np.array([[0.661, 0.674, 0.694, 0.742, 0.715]]),
            np.array([[0.701, 0.642, 0.669, 0.681, 0.679]]),
            np.array([[0.604, 0.399, 0.659, 0.613, 0.724]]),
            np.array([[0.769, 1.009, 0.697, 0.69 , 0.72 ]]),
            np.array([[0.824, 0.759, 0.767, 0.641, 0.429]]),
            np.array([[0.858, 1.092, 0.794, 0.604, 0.475]]),
            np.array([[0.8  , 0.987, 0.648, 0.57 , 0.493]]),
            np.array([[0.867, 0.957, 0.677, 0.558, 0.477]])
        ]
        custom_inputs = []
        for custom_input in custom_inputs_main:
            custom_input_res = np.copy(custom_input)
            custom_input_ind = np.copy(custom_input)
            custom_input_tra = np.copy(custom_input)
            custom_input_agr = np.copy(custom_input)
            custom_input_ene = np.copy(custom_input)
            custom_input_nores = np.copy(custom_input)
            custom_input_noind = np.copy(custom_input)
            custom_input_notra = np.copy(custom_input)
            custom_input_noagr = np.copy(custom_input)
            custom_input_noene = np.copy(custom_input)
            custom_input_resonly = np.copy(custom_input)
            custom_input_indonly = np.copy(custom_input)
            custom_input_traonly = np.copy(custom_input)
            custom_input_agronly = np.copy(custom_input)
            custom_input_eneonly = np.copy(custom_input)

            custom_input_res[0][1:] = 1.0
            custom_input_ind[0][0]  = 1.0
            custom_input_ind[0][2:] = 1.0
            custom_input_tra[0][:2] = 1.0
            custom_input_tra[0][3:] = 1.0
            custom_input_agr[0][:3] = 1.0
            custom_input_agr[0][4:] = 1.0
            custom_input_ene[0][:4] = 1.0

            custom_input_nores[0][0] = 0.0
            custom_input_noind[0][1] = 0.0
            custom_input_notra[0][2] = 0.0
            custom_input_noagr[0][3] = 0.0
            custom_input_noene[0][4] = 0.0
            
            custom_input_resonly[0][1:] = 0.0
            custom_input_indonly[0][0]  = 0.0
            custom_input_indonly[0][2:] = 0.0
            custom_input_traonly[0][:2] = 0.0
            custom_input_traonly[0][3:] = 0.0
            custom_input_agronly[0][:3] = 0.0
            custom_input_agronly[0][4:] = 0.0
            custom_input_eneonly[0][:4] = 0.0

            custom_inputs.append(custom_input)
            custom_inputs.append(custom_input_res)
            custom_inputs.append(custom_input_ind)
            custom_inputs.append(custom_input_tra)
            custom_inputs.append(custom_input_agr)
            custom_inputs.append(custom_input_ene)
            custom_inputs.append(custom_input_nores)
            custom_inputs.append(custom_input_noind)
            custom_inputs.append(custom_input_notra)
            custom_inputs.append(custom_input_noagr)
            custom_inputs.append(custom_input_noene)
            custom_inputs.append(custom_input_resonly)
            custom_inputs.append(custom_input_indonly)
            custom_inputs.append(custom_input_traonly)
            custom_inputs.append(custom_input_agronly)
            custom_inputs.append(custom_input_eneonly)

        # just for emulator_predictions.py as this is required in order to adjust for double emissions
        custom_inputs_temp = custom_inputs.copy()
        for custom_input in custom_inputs_temp:
            custom_input_resonly = np.copy(custom_input)
            custom_input_indonly = np.copy(custom_input)
            custom_input_traonly = np.copy(custom_input)
            custom_input_agronly = np.copy(custom_input)
            custom_input_eneonly = np.copy(custom_input)
            
            custom_input_resonly[0][1:] = 0.0
            custom_input_indonly[0][0]  = 0.0
            custom_input_indonly[0][2:] = 0.0
            custom_input_traonly[0][:2] = 0.0
            custom_input_traonly[0][3:] = 0.0
            custom_input_agronly[0][:3] = 0.0
            custom_input_agronly[0][4:] = 0.0
            custom_input_eneonly[0][:4] = 0.0
            
            custom_inputs.append(custom_input_resonly)
            custom_inputs.append(custom_input_indonly)
            custom_inputs.append(custom_input_traonly)
            custom_inputs.append(custom_input_agronly)
            custom_inputs.append(custom_input_eneonly)

        emission_configs_20percentintervals = []
        for custom_input in custom_inputs:
            emission_config = f'RES{custom_input[0][0]:0.3f}_IND{custom_input[0][1]:0.3f}_TRA{custom_input[0][2]:0.3f}_AGR{custom_input[0][3]:0.3f}_ENE{custom_input[0][4]:0.3f}'
            emission_configs_20percentintervals.append(emission_config)
            
        emission_configs_20percentintervals = list(set(emission_configs_20percentintervals))

        custom_inputs = []
        for emission_config in emission_configs_20percentintervals:
            custom_input = np.array([float(num) for num in re.findall(r'\d.\d+', emission_config)]).reshape(1, -1)
            custom_inputs.append(custom_input)

    if climate_cobenefits:
        custom_inputs_main = [
            np.array([[0.91, 0.95, 0.85, 1.05, 0.96]]), # Base_CLE_2020
            np.array([[0.91, 0.95, 0.85, 1.05, 0.96]]), # Base_MFR_2020
            np.array([[0.91, 0.95, 0.85, 1.05, 0.96]]), # SDS_MFR_2020
            np.array([[0.68, 0.84, 0.71, 1.16, 0.93]]), # Base_CLE_2030
            np.array([[0.33, 0.47, 0.48, 0.81, 0.69]]), # Base_MFR_2030
            np.array([[0.27, 0.45, 0.41, 0.81, 0.55]]), # SDS_MFR_2030
            np.array([[0.57, 0.75, 0.69, 1.2, 0.94]]), # Base_CLE_2040
            np.array([[0.24, 0.41, 0.31, 0.83, 0.73]]), # Base_MFR_2040
            np.array([[0.19, 0.38, 0.22, 0.83, 0.5]]), # SDS_MFR_2040
            np.array([[0.52, 0.72, 0.65, 1.24, 0.91]]), # Base_CLE_2050
            np.array([[0.2, 0.38, 0.29, 0.86, 0.72]]), # Base_MFR_2050
            np.array([[0.18, 0.35, 0.2, 0.86, 0.46]]), # SDS_MFR_2050
        ]
        custom_inputs = []
        for custom_input in custom_inputs_main:
            custom_input_res = np.copy(custom_input)
            custom_input_ind = np.copy(custom_input)
            custom_input_tra = np.copy(custom_input)
            custom_input_agr = np.copy(custom_input)
            custom_input_ene = np.copy(custom_input)
            custom_input_nores = np.copy(custom_input)
            custom_input_noind = np.copy(custom_input)
            custom_input_notra = np.copy(custom_input)
            custom_input_noagr = np.copy(custom_input)
            custom_input_noene = np.copy(custom_input)
            custom_input_resonly = np.copy(custom_input)
            custom_input_indonly = np.copy(custom_input)
            custom_input_traonly = np.copy(custom_input)
            custom_input_agronly = np.copy(custom_input)
            custom_input_eneonly = np.copy(custom_input)

            custom_input_res[0][1:] = 1.0
            custom_input_ind[0][0]  = 1.0
            custom_input_ind[0][2:] = 1.0
            custom_input_tra[0][:2] = 1.0
            custom_input_tra[0][3:] = 1.0
            custom_input_agr[0][:3] = 1.0
            custom_input_agr[0][4:] = 1.0
            custom_input_ene[0][:4] = 1.0

            custom_input_nores[0][0] = 0.0
            custom_input_noind[0][1] = 0.0
            custom_input_notra[0][2] = 0.0
            custom_input_noagr[0][3] = 0.0
            custom_input_noene[0][4] = 0.0
            
            custom_input_resonly[0][1:] = 0.0
            custom_input_indonly[0][0]  = 0.0
            custom_input_indonly[0][2:] = 0.0
            custom_input_traonly[0][:2] = 0.0
            custom_input_traonly[0][3:] = 0.0
            custom_input_agronly[0][:3] = 0.0
            custom_input_agronly[0][4:] = 0.0
            custom_input_eneonly[0][:4] = 0.0

            custom_inputs.append(custom_input)
            custom_inputs.append(custom_input_res)
            custom_inputs.append(custom_input_ind)
            custom_inputs.append(custom_input_tra)
            custom_inputs.append(custom_input_agr)
            custom_inputs.append(custom_input_ene)
            custom_inputs.append(custom_input_nores)
            custom_inputs.append(custom_input_noind)
            custom_inputs.append(custom_input_notra)
            custom_inputs.append(custom_input_noagr)
            custom_inputs.append(custom_input_noene)
            custom_inputs.append(custom_input_resonly)
            custom_inputs.append(custom_input_indonly)
            custom_inputs.append(custom_input_traonly)
            custom_inputs.append(custom_input_agronly)
            custom_inputs.append(custom_input_eneonly)

        # just for emulator_predictions.py as this is required in order to adjust for double emissions
        custom_inputs_temp = custom_inputs.copy()
        for custom_input in custom_inputs_temp:
            custom_input_resonly = np.copy(custom_input)
            custom_input_indonly = np.copy(custom_input)
            custom_input_traonly = np.copy(custom_input)
            custom_input_agronly = np.copy(custom_input)
            custom_input_eneonly = np.copy(custom_input)
            
            custom_input_resonly[0][1:] = 0.0
            custom_input_indonly[0][0]  = 0.0
            custom_input_indonly[0][2:] = 0.0
            custom_input_traonly[0][:2] = 0.0
            custom_input_traonly[0][3:] = 0.0
            custom_input_agronly[0][:3] = 0.0
            custom_input_agronly[0][4:] = 0.0
            custom_input_eneonly[0][:4] = 0.0
            
            custom_inputs.append(custom_input_resonly)
            custom_inputs.append(custom_input_indonly)
            custom_inputs.append(custom_input_traonly)
            custom_inputs.append(custom_input_agronly)
            custom_inputs.append(custom_input_eneonly)

        emission_configs_20percentintervals = []
        for custom_input in custom_inputs:
            emission_config = f'RES{custom_input[0][0]:0.3f}_IND{custom_input[0][1]:0.3f}_TRA{custom_input[0][2]:0.3f}_AGR{custom_input[0][3]:0.3f}_ENE{custom_input[0][4]:0.3f}'
            emission_configs_20percentintervals.append(emission_config)
            
        emission_configs_20percentintervals = list(set(emission_configs_20percentintervals))

        custom_inputs = []
        for emission_config in emission_configs_20percentintervals:
            custom_input = np.array([float(num) for num in re.findall(r'\d.\d+', emission_config)]).reshape(1, -1)
            custom_inputs.append(custom_input)

    if top_down_2020_baseline:
        emission_config_2020_baseline = np.array([0.604, 0.399, 0.659, 0.613, 0.724]) # matching to PM2.5 only, top 1,000
        emission_configs = np.array(
            np.meshgrid(
                np.linspace(emission_config_2020_baseline[0] * 0.50, emission_config_2020_baseline[0], 6), # 10% reduction increments from 2020 baseline up to 50%
                np.linspace(emission_config_2020_baseline[1] * 0.50, emission_config_2020_baseline[1], 6),
                np.linspace(emission_config_2020_baseline[2] * 0.50, emission_config_2020_baseline[2], 6),
                np.linspace(emission_config_2020_baseline[3] * 0.50, emission_config_2020_baseline[3], 6),
                np.linspace(emission_config_2020_baseline[4] * 0.50, emission_config_2020_baseline[4], 6),
            )
        ).T.reshape(-1, 5)   
        custom_inputs = [np.array(item).reshape(1, -1) for item in emission_configs]
        # add a couple more for larger reductions in RES and IND to reach WHO-IT2
        custom_inputs.append(np.array([[0.242, 0.160, 0.659, 0.613, 0.724]]))
        custom_inputs.append(np.array([[0.181, 0.120, 0.659, 0.613, 0.724]]))
        custom_inputs.append(np.array([[0.121, 0.080, 0.659, 0.613, 0.724]]))
        custom_inputs.append(np.array([[0.060, 0.040, 0.659, 0.613, 0.724]]))

        # just for emulator_predictions.py as this is required in order to adjust for double emissions
        custom_inputs_temp = custom_inputs.copy()
        for custom_input in custom_inputs_temp:
            custom_input_resonly = np.copy(custom_input)
            custom_input_indonly = np.copy(custom_input)
            custom_input_traonly = np.copy(custom_input)
            custom_input_agronly = np.copy(custom_input)
            custom_input_eneonly = np.copy(custom_input)
            custom_input_resonly[0][1:] = 0.0
            custom_input_indonly[0][0]  = 0.0
            custom_input_indonly[0][2:] = 0.0
            custom_input_traonly[0][:2] = 0.0
            custom_input_traonly[0][3:] = 0.0
            custom_input_agronly[0][:3] = 0.0
            custom_input_agronly[0][4:] = 0.0
            custom_input_eneonly[0][:4] = 0.0
            custom_inputs.append(custom_input_resonly)
            custom_inputs.append(custom_input_indonly)
            custom_inputs.append(custom_input_traonly)
            custom_inputs.append(custom_input_agronly)
            custom_inputs.append(custom_input_eneonly)

        emission_configs_20percentintervals = []
        for custom_input in custom_inputs:
            emission_config = f'RES{custom_input[0][0]:0.3f}_IND{custom_input[0][1]:0.3f}_TRA{custom_input[0][2]:0.3f}_AGR{custom_input[0][3]:0.3f}_ENE{custom_input[0][4]:0.3f}'
            emission_configs_20percentintervals.append(emission_config)

        emission_configs_20percentintervals = set(emission_configs_20percentintervals)

        custom_inputs_completed_filenames = glob.glob(f"/nobackup/earlacoa/machinelearning/data_annual/predictions/{output}/ds*{output}.nc")
        custom_inputs_completed_list = []
        for custom_inputs_completed_filename in custom_inputs_completed_filenames:
            emission_config = re.findall(r"RES\d+\.\d+_IND\d+\.\d+_TRA\d+\.\d+_AGR\d+\.\d+_ENE\d+\.\d+", custom_inputs_completed_filename)
            if len(emission_config) > 0:
                custom_inputs_completed_list.append(emission_config)

        custom_inputs_completed_set = set(item[0] for item in custom_inputs_completed_list)
        custom_inputs_remaining_set = emission_configs_20percentintervals - custom_inputs_completed_set
        custom_inputs = [np.array([float(n) for n in re.findall(r'\d+.\d+', item)]).reshape(1, -1) for item in custom_inputs_remaining_set]

    # dask bag and process
    custom_inputs = custom_inputs[:5000]
    #custom_inputs = custom_inputs[5000:]

    print(f"predicting for {len(custom_inputs)} custom inputs ...")
    bag_custom_inputs = db.from_sequence(custom_inputs, npartitions=n_workers)
    bag_custom_inputs.map(custom_predicts).compute()

    time_end = time.time() - time_start
    print(
        f"completed in {time_end:0.2f} seconds, or {time_end / 60:0.2f} minutes, or {time_end / 3600:0.2f} hours"
    )
    print(
        f"average time per custom input is {time_end / len(custom_inputs):0.2f} seconds"
    )

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()

