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

#output = 'PM2_5_DRY'
output = 'o3_6mDM8h'
# 'bc_2p5', 'oc_2p5', 'no3_2p5', 'oin_2p5', 'AOD550_sfc', 'bsoaX_2p5', 'nh4_2p5', 'no3_2p5', 'asoaX_2p5'

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
        filename = f"RES{res:.2f}_IND{ind:.2f}_TRA{tra:.2f}_AGR{agr:.2f}_ENE{ene:.2f}"
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
                np.linspace(
                    0, 1.5, 16
                ),  # 1.5 and 16 for 0.1, 1.5 and 6 for 0.3, 1.4 and 8 for 0.2
                np.linspace(0, 1.5, 16),
                np.linspace(0, 1.5, 16),
                np.linspace(0, 1.5, 16),
                np.linspace(0, 1.5, 16),
            )
        ).T.reshape(-1, 5)
        custom_inputs_set = set(
            tuple(map(float, map("{:.1f}".format, item))) for item in matrix_stacked
        )

        custom_inputs_completed_filenames = glob.glob(
            f"/nobackup/earlacoa/machinelearning/data_annual/predictions/{output}/ds*{output}*"
        )
        custom_inputs_completed_list = []
        for custom_inputs_completed_filename in custom_inputs_completed_filenames:
            custom_inputs_completed_list.append(
                [
                    float(item)
                    for item in re.findall(r"\d+\.\d+", custom_inputs_completed_filename)
                ]
            )

        custom_inputs_completed_set = set(
            tuple(item) for item in custom_inputs_completed_list
        )
        custom_inputs_remaining_set = custom_inputs_set - custom_inputs_completed_set
        custom_inputs = [
            np.array(item).reshape(1, -1) for item in custom_inputs_remaining_set
        ]
        print(f"custom inputs remaining for {output}: {len(custom_inputs)}")

    if extra:
        custom_inputs = [
            np.array([[1.15, 1.27, 0.98, 0.98, 1.36]]), # bottom-up 2010
            np.array([[1.19, 1.30, 1.01, 1.01, 1.46]]), # bottom-up 2011
            np.array([[1.20, 1.30, 1.01, 1.02, 1.39]]), # bottom-up 2012
            np.array([[1.13, 1.29, 1.02, 1.01, 1.29]]), # bottom-up 2013
            np.array([[1.06, 1.12, 0.99, 1.01, 1.12]]), # bottom-up 2014
            np.array([[0.92, 0.84, 0.97, 0.99, 0.94]]), # bottom-up 2016
            np.array([[0.84, 0.81, 0.99, 0.99, 0.89]]), # bottom-up 2017
            np.array([[0.95, 0.99, 0.71, 0.88, 0.69]]), # top-down, 2016, both
            np.array([[0.89, 0.90, 0.79, 0.74, 0.59]]), # top-down, 2017, both
            np.array([[0.71, 0.91, 0.84, 0.53, 0.54]]), # top-down, 2018, both
            np.array([[0.72, 0.88, 0.73, 0.71, 0.63]]), # top-down, 2019, both
            np.array([[0.64, 0.79, 0.63, 0.56, 0.44]]), # top-down, 2020, both
            np.array([[0.96, 0.93, 0.68, 0.87, 0.76]]), # top-down, 2016, either
            np.array([[0.96, 0.92, 0.75, 0.83, 0.46]]), # top-down, 2017, either
            np.array([[0.86, 1.08, 0.82, 0.52, 0.49]]), # top-down, 2018, either
            np.array([[0.87, 0.94, 0.71, 0.59, 0.50]]), # top-down, 2019, either
            np.array([[0.79, 0.79, 0.60, 0.42, 0.44]]), # top-down, 2020, either
            np.array([[0.76, 1.08, 0.56, 0.77, 0.86]]), # top-down, 2016, o3_6mDM8h
            np.array([[0.94, 0.67, 0.74, 0.72, 0.37]]), # top-down, 2017, o3_6mDM8h
            np.array([[0.93, 1.11, 0.93, 0.64, 0.40]]), # top-down, 2018, o3_6mDM8h
            np.array([[0.94, 1.12, 0.61, 0.48, 0.35]]), # top-down, 2019, o3_6mDM8h
            np.array([[0.94, 0.99, 0.66, 0.50, 0.43]]), # top-down, 2020, o3_6mDM8h
            np.array([[1.01, 0.82, 0.77, 0.94, 0.73]]), # top-down, 2016, PM2_5_DRY
            np.array([[0.88, 0.88, 0.75, 0.91, 0.70]]), # top-down, 2017, PM2_5_DRY
            np.array([[0.79, 0.85, 0.74, 0.85, 0.83]]), # top-down, 2018, PM2_5_DRY
            np.array([[0.83, 0.71, 0.82, 0.89, 0.79]]), # top-down, 2019, PM2_5_DRY
            np.array([[0.94, 0.38, 0.65, 0.74, 0.72]]), # top-down, 2020, PM2_5_DRY
        ]

    if climate_cobenefits:
        custom_inputs = [
            np.array([[1.000, 1.000, 1.000, 1.000, 1.000]]), # Base_CLE_2015
            np.array([[0.934, 0.937, 0.876, 1.054, 0.964]]), # Base_CLE_2020
            np.array([[0.934, 1.0, 1.0, 1.0, 1.0]]), # Base_CLE_2020 - RES
            np.array([[1.0, 0.937, 1.0, 1.0, 1.0]]), # Base_CLE_2020 - IND
            np.array([[1.0, 1.0, 0.876, 1.0, 1.0]]), # Base_CLE_2020 - TRA
            np.array([[1.0, 1.0, 1.0, 1.054, 1.0]]), # Base_CLE_2020 - AGR
            np.array([[1.0, 1.0, 1.0, 1.0, 0.964]]), # Base_CLE_2020 - ENE
            np.array([[0.934, 0.937, 0.876, 1.054, 0.964]]), # Base_MFR_2020
            np.array([[0.934, 1.0, 1.0, 1.0, 1.0]]), # Base_MFR_2020 - RES
            np.array([[1.0, 0.937, 1.0, 1.0, 1.0]]), # Base_MFR_2020 - IND
            np.array([[1.0, 1.0, 0.876, 1.0, 1.0]]), # Base_MFR_2020 - TRA
            np.array([[1.0, 1.0, 1.0, 1.054, 1.0]]), # Base_MFR_2020 - AGR
            np.array([[1.0, 1.0, 1.0, 1.0, 0.964]]), # Base_MFR_2020 - ENE
            np.array([[0.934, 0.937, 0.876, 1.054, 0.964]]), # SDS_MFR_2020
            np.array([[0.934, 1.0, 1.0, 1.0, 1.0]]), # SDS_MFR_2020 - RES
            np.array([[1.0, 0.937, 1.0, 1.0, 1.0]]), # SDS_MFR_2020 - IND
            np.array([[1.0, 1.0, 0.876, 1.0, 1.0]]), # SDS_MFR_2020 - TRA
            np.array([[1.0, 1.0, 1.0, 1.054, 1.0]]), # SDS_MFR_2020 - AGR
            np.array([[1.0, 1.0, 1.0, 1.0, 0.964]]), # SDS_MFR_2020 - ENE
            np.array([[0.839, 0.880, 0.788, 1.105, 0.947]]), # Base_CLE_2025
            np.array([[0.839, 1.0, 1.0, 1.0, 1.0]]), # Base_CLE_2025 - RES
            np.array([[1.0, 0.880, 1.0, 1.0, 1.0]]), # Base_CLE_2025 - IND
            np.array([[1.0, 1.0, 0.788, 1.0, 1.0]]), # Base_CLE_2025 - TRA
            np.array([[1.0, 1.0, 1.0, 1.105, 1.0]]), # Base_CLE_2025 - AGR
            np.array([[1.0, 1.0, 1.0, 1.0, 0.947]]), # Base_CLE_2025 - ENE
            np.array([[0.536, 0.495, 0.630, 0.787, 0.647]]), # Base_MFR_2025
            np.array([[0.536, 1.0, 1.0, 1.0, 1.0]]), # Base_MFR_2025 - RES
            np.array([[1.0, 0.495, 1.0, 1.0, 1.0]]), # Base_MFR_2025 - IND
            np.array([[1.0, 1.0, 0.630, 1.0, 1.0]]), # Base_MFR_2025 - TRA
            np.array([[1.0, 1.0, 1.0, 0.787, 1.0]]), # Base_MFR_2025 - AGR
            np.array([[1.0, 1.0, 1.0, 1.0, 0.647]]), # Base_MFR_2025 - ENE
            np.array([[0.507, 0.483, 0.598, 0.787, 0.557]]), # SDS_MFR_2025
            np.array([[0.507, 1.0, 1.0, 1.0, 1.0]]), # SDS_MFR_2025 - RES
            np.array([[1.0, 0.483, 1.0, 1.0, 1.0]]), # SDS_MFR_2025 - IND
            np.array([[1.0, 1.0, 0.598, 1.0, 1.0]]), # SDS_MFR_2025 - TRA
            np.array([[1.0, 1.0, 1.0, 0.787, 1.0]]), # SDS_MFR_2025 - AGR
            np.array([[1.0, 1.0, 1.0, 1.0, 0.557]]), # SDS_MFR_2025 - ENE
            np.array([[0.769, 0.853, 0.760, 1.159, 0.935]]), # Base_CLE_2030
            np.array([[0.769, 1.0, 1.0, 1.0, 1.0]]), # Base_CLE_2030 - RES
            np.array([[1.0, 0.853, 1.0, 1.0, 1.0]]), # Base_CLE_2030 - IND
            np.array([[1.0, 1.0, 0.760, 1.0, 1.0]]), # Base_CLE_2030 - TRA
            np.array([[1.0, 1.0, 1.0, 1.159, 1.0]]), # Base_CLE_2030 - AGR
            np.array([[1.0, 1.0, 1.0, 1.0, 0.935]]), # Base_CLE_2030 - ENE
            np.array([[0.409, 0.469, 0.540, 0.810, 0.661]]), # Base_MFR_2030
            np.array([[0.409, 1.0, 1.0, 1.0, 1.0]]), # Base_MFR_2030 - RES
            np.array([[1.0, 0.469, 1.0, 1.0, 1.0]]), # Base_MFR_2030 - IND
            np.array([[1.0, 1.0, 0.540, 1.0, 1.0]]), # Base_MFR_2030 - TRA
            np.array([[1.0, 1.0, 1.0, 0.810, 1.0]]), # Base_MFR_2030 - AGR
            np.array([[1.0, 1.0, 1.0, 1.0, 0.661]]), # Base_MFR_2030 - ENE
            np.array([[0.353, 0.449, 0.483, 0.810, 0.517]]), # SDS_MFR_2030
            np.array([[0.353, 1.0, 1.0, 1.0, 1.0]]), # SDS_MFR_2030 - RES
            np.array([[1.0, 0.449, 1.0, 1.0, 1.0]]), # SDS_MFR_2030 - IND
            np.array([[1.0, 1.0, 0.483, 1.0, 1.0]]), # SDS_MFR_2030 - TRA
            np.array([[1.0, 1.0, 1.0, 0.810, 1.0]]), # SDS_MFR_2030 - AGR
            np.array([[1.0, 1.0, 1.0, 1.0, 0.517]]), # SDS_MFR_2030 - ENE
            np.array([[0.732, 0.821, 0.748, 1.180, 0.938]]), # Base_CLE_2035
            np.array([[0.732, 1.0, 1.0, 1.0, 1.0]]), # Base_CLE_2035 - RES
            np.array([[1.0, 0.821, 1.0, 1.0, 1.0]]), # Base_CLE_2035 - IND
            np.array([[1.0, 1.0, 0.748, 1.0, 1.0]]), # Base_CLE_2035 - TRA
            np.array([[1.0, 1.0, 1.0, 1.180, 1.0]]), # Base_CLE_2035 - AGR
            np.array([[1.0, 1.0, 1.0, 1.0, 0.938]]), # Base_CLE_2035 - ENE
            np.array([[0.344, 0.438, 0.466, 0.821, 0.674]]), # Base_MFR_2035
            np.array([[0.344, 1.0, 1.0, 1.0, 1.0]]), # Base_MFR_2035 - RES
            np.array([[1.0, 0.438, 1.0, 1.0, 1.0]]), # Base_MFR_2035 - IND
            np.array([[1.0, 1.0, 0.466, 1.0, 1.0]]), # Base_MFR_2035 - TRA
            np.array([[1.0, 1.0, 1.0, 0.821, 1.0]]), # Base_MFR_2035 - AGR
            np.array([[1.0, 1.0, 1.0, 1.0, 0.674]]), # Base_MFR_2035 - ENE
            np.array([[0.296, 0.414, 0.394, 0.821, 0.494]]), # SDS_MFR_2035
            np.array([[0.296, 1.0, 1.0, 1.0, 1.0]]), # SDS_MFR_2035 - RES
            np.array([[1.0, 0.414, 1.0, 1.0, 1.0]]), # SDS_MFR_2035 - IND
            np.array([[1.0, 1.0, 0.394, 1.0, 1.0]]), # SDS_MFR_2035 - TRA
            np.array([[1.0, 1.0, 1.0, 0.821, 1.0]]), # SDS_MFR_2035 - AGR
            np.array([[1.0, 1.0, 1.0, 1.0, 0.494]]), # SDS_MFR_2035 - ENE
            np.array([[0.681, 0.775, 0.707, 1.245, 0.897]]), # Base_CLE_2050
            np.array([[0.681, 1.0, 1.0, 1.0, 1.0]]), # Base_CLE_2050 - RES
            np.array([[1.0, 0.775, 1.0, 1.0, 1.0]]), # Base_CLE_2050 - IND
            np.array([[1.0, 1.0, 0.707, 1.0, 1.0]]), # Base_CLE_2050 - TRA
            np.array([[1.0, 1.0, 1.0, 1.245, 1.0]]), # Base_CLE_2050 - AGR
            np.array([[1.0, 1.0, 1.0, 1.0, 0.897]]), # Base_CLE_2050 - ENE
            np.array([[0.221, 0.383, 0.377, 0.860, 0.678]]), # Base_MFR_2050
            np.array([[0.221, 1.0, 1.0, 1.0, 1.0]]), # Base_MFR_2050 - RES
            np.array([[1.0, 0.383, 1.0, 1.0, 1.0]]), # Base_MFR_2050 - IND
            np.array([[1.0, 1.0, 0.377, 1.0, 1.0]]), # Base_MFR_2050 - TRA
            np.array([[1.0, 1.0, 1.0, 0.860, 1.0]]), # Base_MFR_2050 - AGR
            np.array([[1.0, 1.0, 1.0, 1.0, 0.678]]), # Base_MFR_2050 - ENE
            np.array([[0.196, 0.351, 0.272, 0.860, 0.433]]), # SDS_MFR_2050
            np.array([[0.196, 1.0, 1.0, 1.0, 1.0]]), # SDS_MFR_2050 - RES
            np.array([[1.0, 0.351, 1.0, 1.0, 1.0]]), # SDS_MFR_2050 - IND
            np.array([[1.0, 1.0, 0.272, 1.0, 1.0]]), # SDS_MFR_2050 - TRA
            np.array([[1.0, 1.0, 1.0, 0.860, 1.0]]), # SDS_MFR_2050 - AGR
            np.array([[1.0, 1.0, 1.0, 1.0, 0.433]]), # SDS_MFR_2050 - ENE
            np.array([[0.0, 0.937, 0.876, 1.054, 0.964]]), # Base_CLE_2020 - NO RES
            np.array([[0.934, 0.0, 0.876, 1.054, 0.964]]), # Base_CLE_2020 - NO IND
            np.array([[0.934, 0.937, 0.0, 1.054, 0.964]]), # Base_CLE_2020 - NO TRA
            np.array([[0.934, 0.937, 0.876, 0.0, 0.964]]), # Base_CLE_2020 - NO AGR
            np.array([[0.934, 0.937, 0.876, 1.054, 0.0]]), # Base_CLE_2020 - NO ENE
            np.array([[0.0, 0.937, 0.876, 1.054, 0.964]]), # Base_MFR_2020 - NO RES
            np.array([[0.934, 0.0, 0.876, 1.054, 0.964]]), # Base_MFR_2020 - NO IND
            np.array([[0.934, 0.937, 0.0, 1.054, 0.964]]), # Base_MFR_2020 - NO TRA
            np.array([[0.934, 0.937, 0.876, 0.0, 0.964]]), # Base_MFR_2020 - NO AGR
            np.array([[0.934, 0.937, 0.876, 1.054, 0.0]]), # Base_MFR_2020 - NO ENE
            np.array([[0.0, 0.937, 0.876, 1.054, 0.964]]), # SDS_MFR_2020 - NO RES
            np.array([[0.934, 0.0, 0.876, 1.054, 0.964]]), # SDS_MFR_2020 - NO IND
            np.array([[0.934, 0.937, 0.0, 1.054, 0.964]]), # SDS_MFR_2020 - NO TRA
            np.array([[0.934, 0.937, 0.876, 0.0, 0.964]]), # SDS_MFR_2020 - NO AGR
            np.array([[0.934, 0.937, 0.876, 1.054, 0.0]]), # SDS_MFR_2020 - NO ENE
            np.array([[0.0, 0.880, 0.788, 1.105, 0.947]]), # Base_CLE_2025 - NO RES
            np.array([[0.839, 0.0, 0.788, 1.105, 0.947]]), # Base_CLE_2025 - NO IND
            np.array([[0.839, 0.880, 0.0, 1.105, 0.947]]), # Base_CLE_2025 - NO TRA
            np.array([[0.839, 0.880, 0.788, 0.0, 0.947]]), # Base_CLE_2025 - NO AGR
            np.array([[0.839, 0.880, 0.788, 1.105, 0.0]]), # Base_CLE_2025 - NO ENE
            np.array([[0.0, 0.495, 0.630, 0.787, 0.647]]), # Base_MFR_2025 - NO RES
            np.array([[0.536, 0.0, 0.630, 0.787, 0.647]]), # Base_MFR_2025 - NO IND
            np.array([[0.536, 0.495, 0.0, 0.787, 0.647]]), # Base_MFR_2025 - NO TRA
            np.array([[0.536, 0.495, 0.630, 0.0, 0.647]]), # Base_MFR_2025 - NO AGR
            np.array([[0.536, 0.495, 0.630, 0.787, 0.0]]), # Base_MFR_2025 - NO ENE
            np.array([[0.0, 0.483, 0.598, 0.787, 0.557]]), # SDS_MFR_2025 - NO RES
            np.array([[0.507, 0.0, 0.598, 0.787, 0.557]]), # SDS_MFR_2025 - NO IND
            np.array([[0.507, 0.483, 0.0, 0.787, 0.557]]), # SDS_MFR_2025 - NO TRA
            np.array([[0.507, 0.483, 0.598, 0.0, 0.557]]), # SDS_MFR_2025 - NO AGR
            np.array([[0.507, 0.483, 0.598, 0.787, 0.0]]), # SDS_MFR_2025 - NO ENE
            np.array([[0.0, 0.853, 0.760, 1.159, 0.935]]), # Base_CLE_2030 - NO RES
            np.array([[0.769, 0.0, 0.760, 1.159, 0.935]]), # Base_CLE_2030 - NO IND
            np.array([[0.769, 0.853, 0.0, 1.159, 0.935]]), # Base_CLE_2030 - NO TRA
            np.array([[0.769, 0.853, 0.760, 0.0, 0.935]]), # Base_CLE_2030 - NO AGR
            np.array([[0.769, 0.853, 0.760, 1.159, 0.0]]), # Base_CLE_2030 - NO ENE
            np.array([[0.0, 0.469, 0.540, 0.810, 0.661]]), # Base_MFR_2030 - NO RES
            np.array([[0.409, 0.0, 0.540, 0.810, 0.661]]), # Base_MFR_2030 - NO IND
            np.array([[0.409, 0.469, 0.0, 0.810, 0.661]]), # Base_MFR_2030 - NO TRA
            np.array([[0.409, 0.469, 0.540, 0.0, 0.661]]), # Base_MFR_2030 - NO AGR
            np.array([[0.409, 0.469, 0.540, 0.810, 0.0]]), # Base_MFR_2030 - NO ENE
            np.array([[0.0, 0.449, 0.483, 0.810, 0.517]]), # SDS_MFR_2030 - NO RES
            np.array([[0.353, 0.0, 0.483, 0.810, 0.517]]), # SDS_MFR_2030 - NO IND
            np.array([[0.353, 0.449, 0.0, 0.810, 0.517]]), # SDS_MFR_2030 - NO TRA
            np.array([[0.353, 0.449, 0.483, 0.0, 0.517]]), # SDS_MFR_2030 - NO AGR
            np.array([[0.353, 0.449, 0.483, 0.810, 0.0]]), # SDS_MFR_2030 - NO ENE
            np.array([[0.0, 0.821, 0.748, 1.180, 0.938]]), # Base_CLE_2035 - NO RES
            np.array([[0.732, 0.0, 0.748, 1.180, 0.938]]), # Base_CLE_2035 - NO IND
            np.array([[0.732, 0.821, 0.0, 1.180, 0.938]]), # Base_CLE_2035 - NO TRA
            np.array([[0.732, 0.821, 0.748, 0.0, 0.938]]), # Base_CLE_2035 - NO AGR
            np.array([[0.732, 0.821, 0.748, 1.180, 0.0]]), # Base_CLE_2035 - NO ENE
            np.array([[0.0, 0.438, 0.466, 0.821, 0.674]]), # Base_MFR_2035 - NO RES
            np.array([[0.344, 0.0, 0.466, 0.821, 0.674]]), # Base_MFR_2035 - NO IND
            np.array([[0.344, 0.438, 0.0, 0.821, 0.674]]), # Base_MFR_2035 - NO TRA
            np.array([[0.344, 0.438, 0.466, 0.0, 0.674]]), # Base_MFR_2035 - NO AGR
            np.array([[0.344, 0.438, 0.466, 0.821, 0.0]]), # Base_MFR_2035 - NO ENE
            np.array([[0.0, 0.414, 0.394, 0.821, 0.494]]), # SDS_MFR_2035 - NO RES
            np.array([[0.296, 0.0, 0.394, 0.821, 0.494]]), # SDS_MFR_2035 - NO IND
            np.array([[0.296, 0.414, 0.0, 0.821, 0.494]]), # SDS_MFR_2035 - NO TRA
            np.array([[0.296, 0.414, 0.394, 0.0, 0.494]]), # SDS_MFR_2035 - NO AGR
            np.array([[0.296, 0.414, 0.394, 0.821, 0.0]]), # SDS_MFR_2035 - NO ENE
            np.array([[0.0, 0.775, 0.707, 1.245, 0.897]]), # Base_CLE_2050 - NO RES
            np.array([[0.681, 0.0, 0.707, 1.245, 0.897]]), # Base_CLE_2050 - NO IND
            np.array([[0.681, 0.775, 0.0, 1.245, 0.897]]), # Base_CLE_2050 - NO TRA
            np.array([[0.681, 0.775, 0.707, 0.0, 0.897]]), # Base_CLE_2050 - NO AGR
            np.array([[0.681, 0.775, 0.707, 1.245, 0.0]]), # Base_CLE_2050 - NO ENE
            np.array([[0.0, 0.383, 0.377, 0.860, 0.678]]), # Base_MFR_2050 - NO RES
            np.array([[0.221, 0.0, 0.377, 0.860, 0.678]]), # Base_MFR_2050 - NO IND
            np.array([[0.221, 0.383, 0.0, 0.860, 0.678]]), # Base_MFR_2050 - NO TRA
            np.array([[0.221, 0.383, 0.377, 0.0, 0.678]]), # Base_MFR_2050 - NO AGR
            np.array([[0.221, 0.383, 0.377, 0.860, 0.0]]), # Base_MFR_2050 - NO ENE
            np.array([[0.0, 0.351, 0.272, 0.860, 0.433]]), # SDS_MFR_2050 - NO RES
            np.array([[0.196, 0.0, 0.272, 0.860, 0.433]]), # SDS_MFR_2050 - NO IND
            np.array([[0.196, 0.351, 0.0, 0.860, 0.433]]), # SDS_MFR_2050 - NO TRA
            np.array([[0.196, 0.351, 0.272, 0.0, 0.433]]), # SDS_MFR_2050 - NO AGR
            np.array([[0.196, 0.351, 0.272, 0.860, 0.0]]), # SDS_MFR_2050 - NO ENE
        ]

    if top_down_2020_baseline:
        emission_config_2020_baseline = np.array([0.64, 0.79, 0.63, 0.56, 0.44])
        emission_configs = np.array(
            np.meshgrid(
                np.linspace(emission_config_2020_baseline[0] - 0.40, emission_config_2020_baseline[0], 5),
                np.linspace(emission_config_2020_baseline[1] - 0.40, emission_config_2020_baseline[1], 5),
                np.linspace(emission_config_2020_baseline[2] - 0.40, emission_config_2020_baseline[2], 5),
                np.linspace(emission_config_2020_baseline[3] - 0.40, emission_config_2020_baseline[3], 5),
                np.linspace(emission_config_2020_baseline[4] - 0.40, emission_config_2020_baseline[4], 5),
            )
        ).T.reshape(-1, 5)
        custom_inputs = [np.array(emission_config).reshape(1, -1) for emission_config in emission_configs]

    # dask bag and process
    custom_inputs = custom_inputs[
        0:5000
    ]  # run in 1,000 chunks over 30 cores, each chunk taking 1 hour
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

