#!/usr/bin/env python3
import os
import time
import xarray as xr
import numpy as np
import xesmf as xe
import glob
import dask.bag as db
from dask_jobqueue import SGECluster
from dask.distributed import Client

#output = "PM2_5_DRY"
output = "o3_6mDM8h"

normal = False # 20 percent intervals
extra = True # additional ones for the emission trend matching
climate_cobenefits = False
top_down_2020_baseline = False

with xr.open_dataset(f"/nobackup/earlacoa/machinelearning/data_annual/prefecture_scaling_factors_{output}.nc") as ds:
    scaling_factor = ds["scaling_factor"]


def scale(emission_config):
    with xr.open_dataset(f"/nobackup/earlacoa/machinelearning/data_annual/predictions/{output}/ds_{emission_config}_{output}_popgrid_0.25deg.nc") as ds:
        ds = ds[output]

    ds_scaled = ds * scaling_factor
    ds_scaled = ds_scaled.to_dataset(name=output)
    ds_scaled.to_netcdf(f"/nobackup/earlacoa/machinelearning/data_annual/predictions/{output}_scaled/ds_{emission_config}_{output}_popgrid_0.25deg_scaled.nc")


def main():
    # dask cluster and client
    n_processes = 1
    n_jobs = 35
    n_workers = n_processes * n_jobs

    cluster = SGECluster(
        interface="ib0",
        walltime="01:00:00",
        memory=f"64 G",
        resource_spec=f"h_vmem=64G",
        scheduler_options={
            "dashboard_address": ":5757",
        },
        job_extra=[
            "-cwd",
            "-V",
            f"-pe smp {n_processes}",
            f"-l disk=32G",
        ],
        local_directory=os.sep.join([os.environ.get("PWD"), "dask-worker-scale-space"]),
    )

    client = Client(cluster)

    cluster.scale(jobs=n_jobs)

    time_start = time.time()

    # scale custom outputs
    if normal:
        emission_configs = np.array(
            np.meshgrid(
                np.linspace(0.0, 1.4, 8),
                np.linspace(0.0, 1.4, 8),
                np.linspace(0.0, 1.4, 8),
                np.linspace(0.0, 1.4, 8),
                np.linspace(0.0, 1.4, 8),
            )
        ).T.reshape(-1, 5)
        emission_configs_20percentintervals = []
        for emission_config in emission_configs:
            emission_configs_20percentintervals.append(f'RES{round(emission_config[0], 1)}_IND{round(emission_config[1], 1)}_TRA{round(emission_config[2], 1)}_AGR{round(emission_config[3], 1)}_ENE{round(emission_config[4], 1)}')

    if extra:
        custom_inputs_main = [
            np.array([[1.15, 1.27, 0.98, 0.98, 1.36]]), # bottom-up 2010
            np.array([[1.19, 1.30, 1.01, 1.01, 1.46]]), # bottom-up 2011
            np.array([[1.20, 1.30, 1.01, 1.02, 1.39]]), # bottom-up 2012
            np.array([[1.13, 1.29, 1.02, 1.01, 1.29]]), # bottom-up 2013
            np.array([[1.06, 1.12, 0.99, 1.01, 1.12]]), # bottom-up 2014
            np.array([[0.92, 0.84, 0.97, 0.99, 0.94]]), # bottom-up 2016
            np.array([[0.84, 0.81, 0.99, 0.99, 0.89]]), # bottom-up 2017
            np.array([[0.93, 1.00, 0.77, 0.84, 0.73]]), # top-down, 2016, both
            np.array([[0.87, 0.86, 0.75, 0.72, 0.66]]), # top-down, 2017, both
            np.array([[0.75, 0.87, 0.75, 0.64, 0.63]]), # top-down, 2018, both
            np.array([[0.77, 0.81, 0.71, 0.67, 0.66]]), # top-down, 2019, both
            np.array([[0.67, 0.71, 0.66, 0.60, 0.62]]), # top-down, 2020, both
            np.array([[0.93, 0.98, 0.73, 0.84, 0.72]]), # top-down, 2016, either
            np.array([[0.94, 0.94, 0.76, 0.79, 0.60]]), # top-down, 2017, either
            np.array([[0.81, 1.04, 0.74, 0.63, 0.52]]), # top-down, 2018, either
            np.array([[0.84, 0.87, 0.72, 0.71, 0.59]]), # top-down, 2019, either
            np.array([[0.72, 0.80, 0.68, 0.54, 0.51]]), # top-down, 2020, either
            np.array([[0.93, 0.97, 0.76, 0.84, 0.73]]), # top-down, 2016, PM2_5_DRY
            np.array([[0.90, 0.94, 0.75, 0.84, 0.74]]), # top-down, 2017, PM2_5_DRY
            np.array([[0.82, 0.85, 0.72, 0.81, 0.73]]), # top-down, 2018, PM2_5_DRY
            np.array([[0.82, 0.78, 0.75, 0.83, 0.72]]), # top-down, 2019, PM2_5_DRY
            np.array([[0.77, 0.58, 0.69, 0.72, 0.71]]), # top-down, 2020, PM2_5_DRY
            np.array([[0.77, 1.01, 0.70, 0.69, 0.72]]), # top-down, 2016, o3_6mDM8h
            np.array([[0.82, 0.76, 0.77, 0.64, 0.43]]), # top-down, 2017, o3_6mDM8h
            np.array([[0.86, 1.09, 0.79, 0.60, 0.48]]), # top-down, 2018, o3_6mDM8h
            np.array([[0.80, 0.99, 0.65, 0.57, 0.49]]), # top-down, 2019, o3_6mDM8h
            np.array([[0.87, 0.96, 0.68, 0.56, 0.48]]), # top-down, 2020, o3_6mDM8h
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
            
            custom_input_res[0][1:] = 1.0
            custom_input_ind[0][0] = 1.0
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
            
            custom_input_res[0][1:] = 1.0
            custom_input_ind[0][0] = 1.0
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

        emission_configs_20percentintervals = []
        for custom_input in custom_inputs:
            emission_config = f'RES{custom_input[0][0]:0.3f}_IND{custom_input[0][1]:0.3f}_TRA{custom_input[0][2]:0.3f}_AGR{custom_input[0][3]:0.3f}_ENE{custom_input[0][4]:0.3f}'
            emission_configs_20percentintervals.append(emission_config)

    if top_down_2020_baseline:
        emission_config_2020_baseline = np.array([0.77, 0.58, 0.69, 0.72, 0.71]) # matching to PM2.5 only, top 1,000
        emission_configs = np.array(
            np.meshgrid(
                np.linspace(emission_config_2020_baseline[0] * 0.50, emission_config_2020_baseline[0], 6), # 10% reduction increments from 2020 baseline up to 50%
                np.linspace(emission_config_2020_baseline[1] * 0.50, emission_config_2020_baseline[1], 6),
                np.linspace(emission_config_2020_baseline[2] * 0.50, emission_config_2020_baseline[2], 6),
                np.linspace(emission_config_2020_baseline[3] * 0.50, emission_config_2020_baseline[3], 6),
                np.linspace(emission_config_2020_baseline[4] * 0.50, emission_config_2020_baseline[4], 6),
            )
        ).T.reshape(-1, 5)
        emission_configs_20percentintervals = []
        for emission_config in emission_configs:
            emission_configs_20percentintervals.append(f'RES{round(emission_config[0], 2)}_IND{round(emission_config[1], 2)}_TRA{round(emission_config[2], 2)}_AGR{round(emission_config[3], 2)}_ENE{round(emission_config[4], 2)}')


    emission_configs_completed = glob.glob(f"/nobackup/earlacoa/machinelearning/data_annual/predictions/{output}_scaled/ds*{output}_popgrid_0.25deg_scaled.nc")
    emission_configs_completed = [f"{item[79:-36]}" for item in emission_configs_completed]

    emission_configs_20percentintervals_remaining_set = set(emission_configs_20percentintervals) - set(emission_configs_completed)
    emission_configs_remaining = [item for item in emission_configs_20percentintervals_remaining_set]
    print(f"custom outputs remaining for {output}: {len(emission_configs_remaining)} - 20% intervals with {int(100 * len(emission_configs_20percentintervals_remaining_set) / len(emission_configs_20percentintervals))}% remaining")


    # dask bag and process
    emission_configs_remaining = emission_configs_remaining[0:5000]  # run in 5,000 chunks over 30 cores, each chunk taking 2 minutes
    print(f"predicting for {len(emission_configs_remaining)} custom outputs ...")
    bag_emission_configs = db.from_sequence(emission_configs_remaining, npartitions=n_workers)
    bag_emission_configs.map(scale).compute()

    time_end = time.time() - time_start
    print(f"completed in {time_end:0.2f} seconds, or {time_end / 60:0.2f} minutes, or {time_end / 3600:0.2f} hours")
    print(f"average time per custom output is {time_end / len(emission_configs_remaining):0.2f} seconds")

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()

