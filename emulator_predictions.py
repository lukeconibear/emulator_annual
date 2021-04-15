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

output = 'PM2_5_DRY' # 'PM2_5_DRY', 'o3_6mDM8h', 'bc_2p5', 'oc_2p5', 'no3_2p5', 'oin_2p5', 'AOD550_sfc', 'bsoaX_2p5', 'nh4_2p5', 'no3_2p5', 'asoaX_2p5'
normal = False # 20 percent intervals
extra = False # additional ones for the emission trend matching
climate_cobenefits = True

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
    if extra:
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
            np.array([[1.15, 1.27, 0.98, 0.98, 1.36]]), # bottom_up_2010vs2015 = RES1.15_IND1.27_TRA0.98_AGR0.98_ENE1.36
            np.array([[1.19, 1.30, 1.01, 1.01, 1.46]]), # bottom_up_2011vs2015 = RES1.19_IND1.30_TRA1.01_AGR1.01_ENE1.46
            np.array([[1.20, 1.30, 1.01, 1.02, 1.39]]), # bottom_up_2012vs2015 = RES1.20_IND1.30_TRA1.01_AGR1.02_ENE1.39
            np.array([[1.13, 1.29, 1.02, 1.01, 1.29]]), # bottom_up_2013vs2015 = RES1.13_IND1.29_TRA1.02_AGR1.01_ENE1.29
            np.array([[1.06, 1.12, 0.99, 1.01, 1.12]]), # bottom_up_2014vs2015 = RES1.06_IND1.12_TRA0.99_AGR1.01_ENE1.12
            np.array([[0.92, 0.84, 0.97, 0.99, 0.94]]), # bottom_up_2016vs2015 = RES0.92_IND0.84_TRA0.97_AGR0.99_ENE0.94
            np.array([[0.84, 0.81, 0.99, 0.99, 0.89]]), # bottom_up_2017vs2015 = RES0.84_IND0.81_TRA0.99_AGR0.99_ENE0.89
            np.array([[0.91, 1.04, 0.88, 0.88, 0.52]]), # top_down = RES0.91_IND1.04_TRA0.88_AGR0.88_ENE0.52
        ]

    if climate_cobenefits:
        custom_inputs = [
            np.array([[0.934, 0.937, 0.876, 1.054, 0.964]]), # Base_CLE_2020
            np.array([[0.934, 0.937, 0.876, 1.054, 0.964]]), # Base_MFR_2020
            np.array([[0.934, 0.937, 0.876, 1.054, 0.964]]), # SDS_MFR_2020
            np.array([[0.839, 0.88 , 0.788, 1.105, 0.947]]), # Base_CLE_2025
            np.array([[0.536, 0.495, 0.63 , 0.787, 0.647]]), # Base_MFR_2025
            np.array([[0.507, 0.483, 0.598, 0.787, 0.557]]), # SDS_MFR_2025
            np.array([[0.769, 0.853, 0.76 , 1.159, 0.935]]), # Base_CLE_2030
            np.array([[0.409, 0.469, 0.54 , 0.81 , 0.661]]), # Base_MFR_2030
            np.array([[0.353, 0.449, 0.483, 0.81 , 0.517]]), # SDS_MFR_2030
            np.array([[0.732, 0.821, 0.748, 1.18 , 0.938]]), # Base_CLE_2035
            np.array([[0.344, 0.438, 0.466, 0.821, 0.674]]), # Base_MFR_2035
            np.array([[0.296, 0.414, 0.394, 0.821, 0.494]]), # SDS_MFR_2035
            np.array([[0.681, 0.775, 0.707, 1.245, 0.897]]), # Base_CLE_2050
            np.array([[0.221, 0.383, 0.377, 0.86 , 0.678]]), # Base_MFR_2050
            np.array([[0.196, 0.351, 0.272, 0.86 , 0.433]]), # SDS_MFR_2050
        ]

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
