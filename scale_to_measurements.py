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

normal = True # 20 percent intervals
extra = False # additional ones for the emission trend matching
climate_cobenefits = False

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
        emission_configs_20percentintervals = [
            'RES1.15_IND1.27_TRA0.98_AGR0.98_ENE1.36', # bottom_up_2010vs2015 = RES1.15_IND1.27_TRA0.98_AGR0.98_ENE1.36
            'RES1.19_IND1.30_TRA1.01_AGR1.01_ENE1.46', # bottom_up_2011vs2015 = RES1.19_IND1.30_TRA1.01_AGR1.01_ENE1.46
            'RES1.20_IND1.30_TRA1.01_AGR1.02_ENE1.39', # bottom_up_2012vs2015 = RES1.20_IND1.30_TRA1.01_AGR1.02_ENE1.39
            'RES1.13_IND1.29_TRA1.02_AGR1.01_ENE1.29', # bottom_up_2013vs2015 = RES1.13_IND1.29_TRA1.02_AGR1.01_ENE1.29
            'RES1.06_IND1.12_TRA0.99_AGR1.01_ENE1.12', # bottom_up_2014vs2015 = RES1.06_IND1.12_TRA0.99_AGR1.01_ENE1.12
            'RES0.92_IND0.84_TRA0.97_AGR0.99_ENE0.94', # bottom_up_2016vs2015 = RES0.92_IND0.84_TRA0.97_AGR0.99_ENE0.94
            'RES0.84_IND0.81_TRA0.99_AGR0.99_ENE0.89', # bottom_up_2017vs2015 = RES0.84_IND0.81_TRA0.99_AGR0.99_ENE0.89
            'RES0.91_IND1.04_TRA0.88_AGR0.88_ENE0.52', # top_down = RES0.91_IND1.04_TRA0.88_AGR0.88_ENE0.52
        ]

    if climate_cobenefits:
        emission_configs_20percentintervals = [
            'RES0.934_IND0.937_TRA0.876_AGR1.054_ENE0.964', # Base_CLE_2020
            'RES0.934_IND0.937_TRA0.876_AGR1.054_ENE0.964', # Base_MFR_2020
            'RES0.934_IND0.937_TRA0.876_AGR1.054_ENE0.964', # SDS_MFR_2020
            'RES0.839_IND0.880_TRA0.788_AGR1.105_ENE0.947', # Base_CLE_2025
            'RES0.536_IND0.495_TRA0.630_AGR0.787_ENE0.647', # Base_MFR_2025
            'RES0.507_IND0.483_TRA0.598_AGR0.787_ENE0.557', # SDS_MFR_2025
            'RES0.769_IND0.853_TRA0.760_AGR1.159_ENE0.935', # Base_CLE_2030
            'RES0.409_IND0.469_TRA0.540_AGR0.810_ENE0.661', # Base_MFR_2030
            'RES0.353_IND0.449_TRA0.483_AGR0.810_ENE0.517', # SDS_MFR_2030
            'RES0.732_IND0.821_TRA0.748_AGR1.180_ENE0.938', # Base_CLE_2035
            'RES0.344_IND0.438_TRA0.466_AGR0.821_ENE0.674', # Base_MFR_2035
            'RES0.296_IND0.414_TRA0.394_AGR0.821_ENE0.494', # SDS_MFR_2035
            'RES0.681_IND0.775_TRA0.707_AGR1.245_ENE0.897', # Base_CLE_2050
            'RES0.221_IND0.383_TRA0.377_AGR0.860_ENE0.678', # Base_MFR_2050
            'RES0.196_IND0.351_TRA0.272_AGR0.860_ENE0.433', # SDS_MFR_2050
        ]

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

