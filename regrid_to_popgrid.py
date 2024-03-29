#!/usr/bin/env python3
import os
import time
import xarray as xr
import xesmf as xe
import glob
import dask.bag as db
from dask_jobqueue import SGECluster
from dask.distributed import Client

with xr.open_dataset("/nobackup/earlacoa/health/data/gpw-v4-population-count-adjusted-to-2015-unwpp-country-totals-2015-qtr-deg.nc") as ds:
    pop_2015 = ds["pop"]


pop_lat = pop_2015["lat"].values
pop_lon = pop_2015["lon"].values

pop_grid = xr.Dataset(
    {
        "lat": (["lat"], pop_lat),
        "lon": (["lon"], pop_lon),
    }
)

#output = "PM2_5_DRY"
output = "o3_6mDM8h"


def regrid_to_pop(custom_output):
    with xr.open_dataset(custom_output) as ds:
        ds = ds[output]

    regridder = xe.Regridder(ds, pop_grid, "bilinear", reuse_weights=True)
    ds_regrid = regridder(ds)
    ds_regrid.to_netcdf(custom_output[0:-3] + "_popgrid_0.25deg.nc")


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

    # regrid custom outputs to pop grid
    custom_outputs = glob.glob(
        f"/nobackup/earlacoa/machinelearning/data_annual/predictions/{output}/ds*{output}.nc"
    )
    custom_outputs_completed = glob.glob(
        f"/nobackup/earlacoa/machinelearning/data_annual/predictions/{output}/ds*{output}_popgrid_0.25deg.nc"
    )
    custom_outputs_completed = [
        f"{item[0:-19]}.nc" for item in custom_outputs_completed
    ]
    custom_outputs_remaining_set = set(custom_outputs) - set(custom_outputs_completed)
    custom_outputs_remaining = [item for item in custom_outputs_remaining_set]
    print(f"custom outputs remaining for {output}: {len(custom_outputs_remaining)}")

    # dask bag and process
    custom_outputs_remaining = custom_outputs_remaining[
        0:5000
    ]  # run in 5,000 chunks over 30 cores, each chunk taking 2 minutes
    print(f"predicting for {len(custom_outputs_remaining)} custom outputs ...")
    bag_custom_outputs = db.from_sequence(
        custom_outputs_remaining, npartitions=n_workers
    )
    bag_custom_outputs.map(regrid_to_pop).compute()

    time_end = time.time() - time_start
    print(
        f"completed in {time_end:0.2f} seconds, or {time_end / 60:0.2f} minutes, or {time_end / 3600:0.2f} hours"
    )
    print(
        f"average time per custom output is {time_end / len(custom_outputs_remaining):0.2f} seconds"
    )

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()

