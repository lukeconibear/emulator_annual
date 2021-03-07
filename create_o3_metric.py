#!/usr/bin/env python3
import os
import sys
import numpy as np
import xarray as xr
import xesmf as xe
from dask_jobqueue import SGECluster
from dask.distributed import Client
import joblib
import dask.bag as db

sims = ["t1", "t2", "t3"]  # three at a time


def create_ozone_metric(sim):
    """seasonal (maximum 6-month mean), daily maximum, 8-hour, O3 concentration (6mDM8h) for GBD2017"""
    path = f"/nobackup/earlacoa/machinelearning/data_annual/o3_6mDM8h/"

    with xr.open_dataset(
        f"{path}wrfout_d01_global_0.25deg_2015_o3_{sim}.nc",
        chunks={"lat": "auto", "lon": "auto"},
    ) as ds:
        o3 = ds["o3"]

    # first: 24, 8-hour, rolling mean, O3 concentrations
    o3_6mDM8h_8hrrollingmean = o3.rolling(time=8).construct("window").mean("window")

    # second: find the max of these each day (daily maximum, 8-hour)
    o3_6mDM8h_dailymax = (
        o3_6mDM8h_8hrrollingmean.sortby("time").resample(time="24H").max()
    )

    # third: 6-month mean - to account for different times when seasonal maximums e.g. different hemispheres
    o3_6mDM8h_6monthmean = o3_6mDM8h_dailymax.resample(time="6M").mean()

    # fourth: maximum of these
    o3_6mDM8h = o3_6mDM8h_6monthmean.max(dim="time").compute()

    o3_6mDM8h.to_netcdf(f"{path}wrfout_d01_global_0.25deg_2015_o3_6mDM8h_{sim}.nc")


def main():
    # dask cluster and client
    number_processes = 1
    number_jobs = 35
    number_workers = number_processes * number_jobs

    cluster = SGECluster(
        interface="ib0",
        walltime="04:00:00",
        memory=f"12 G",
        resource_spec=f"h_vmem=12G",
        scheduler_options={
            "dashboard_address": ":2727",
        },
        job_extra=[
            "-cwd",
            "-V",
            f"-pe smp {number_processes}",
            f"-l disk=1G",
        ],
        local_directory=os.sep.join([os.environ.get("PWD"), "dask-worker-space"]),
    )

    client = Client(cluster)
    cluster.scale(jobs=number_jobs)

    # main processing
    print("processing ...")
    results = []
    bag = db.from_sequence(sims, npartitions=number_workers)
    results = bag.map(create_ozone_metric).compute()
    print("complete")

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
