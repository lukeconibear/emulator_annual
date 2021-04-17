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

output = "PM2_5_DRY"
#output = "o3_6mDM8h"

with xr.open_dataset(f"/nobackup/earlacoa/machinelearning/data_annual/prefecture_scaling_factors_{output}.nc") as ds:
    scaling_factor = ds["__xarray_dataarray_variable__"]


def scale(emission_config):
    with xr.open_dataset(f"/nobackup/earlacoa/machinelearning/data_annual/predictions/{output}/ds_{emission_config}_{output}_popgrid_0.25deg.nc") as ds:
        ds = ds[output]

    ds_scaled = ds * scaling_factor
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
        local_directory=os.sep.join([os.environ.get("PWD"), "dask-worker-space"]),
    )

    client = Client(cluster)

    cluster.scale(jobs=n_jobs)

    time_start = time.time()

    # scale custom outputs
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

