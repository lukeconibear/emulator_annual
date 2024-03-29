#!/usr/bin/env python3
import os
import sys
import xarray as xr
import glob
import dask.bag as db
import numpy as np
import geopandas as gpd
from cutshapefile import transform_from_latlon, rasterize
from dask_jobqueue import SGECluster
from dask.distributed import Client
import joblib

#output = "o3_6mDM8h"
output = 'PM2_5_DRY'

with xr.open_dataset(
    "/nobackup/earlacoa/health/data/gpw-v4-population-count-adjusted-to-2015-unwpp-country-totals-2015-qtr-deg.nc"
) as ds:
    pop_2015 = ds["pop"].load()

pop_lat = pop_2015["lat"].values
pop_lon = pop_2015["lon"].values

region = 'china'
shapefile = '/nobackup/earlacoa/health/data/china_taiwan_hongkong_macao.shp'
# region = 'gba'
# shapefile = '/nobackup/earlacoa/health/data/gba.shp'
#region = "china_north"
#shapefile = "/nobackup/earlacoa/health/data/CHN_north.shp"
# region = 'china_north_east'
# shapefile = '/nobackup/earlacoa/health/data/CHN_north_east.shp'
# region = 'china_east'
# shapefile = '/nobackup/earlacoa/health/data/CHN_east.shp'
# region = 'china_south_central'
# shapefile = '/nobackup/earlacoa/health/data/CHN_south_central.shp'
# region = 'china_south_west'
# shapefile = '/nobackup/earlacoa/health/data/CHN_south_west.shp'
# region = 'china_north_west'
# shapefile = '/nobackup/earlacoa/health/data/CHN_north_west.shp'

shp = gpd.read_file(shapefile)
shapes = [(shape, n) for n, shape in enumerate(shp.geometry)]
clip = rasterize(shapes, pop_2015.coords, longitude="lon", latitude="lat")
pop_2015_clipped = pop_2015.where(clip == 0, other=np.nan)


def popweight_outputs_for_input(custom_input):
    sim = (
        "RES"
        + str(np.round(custom_input[0][0], decimals=1))
        + "_IND"
        + str(np.round(custom_input[0][1], decimals=1))
        + "_TRA"
        + str(np.round(custom_input[0][2], decimals=1))
        + "_AGR"
        + str(np.round(custom_input[0][3], decimals=1))
        + "_ENE"
        + str(np.round(custom_input[0][4], decimals=1))
    )
    filename = f"ds_{sim}_{output}_popgrid_0.25deg_adjusted_scaled.nc"
    print(f"processing for {filename} ...")
    with xr.open_dataset(
        f"/nobackup/earlacoa/machinelearning/data_annual/predictions/{output}_adjusted_scaled/{filename}"
    ) as ds:
        ds_custom_output = ds[output].load()

    clip = rasterize(shapes, ds_custom_output.coords, longitude="lon", latitude="lat")
    ds_custom_output_clipped = ds_custom_output.where(clip == 0, other=np.nan)
    return filename, np.nansum((ds_custom_output_clipped * pop_2015) / pop_2015_clipped.sum())


def main():
    # dask cluster and client
    n_processes = 1
    n_jobs = 35
    n_workers = n_processes * n_jobs

    cluster = SGECluster(
        interface="ib0",
        walltime="48:00:00",
        memory=f"12 G",
        resource_spec=f"h_vmem=12G",
        scheduler_options={
            "dashboard_address": ":5757",
        },
        job_extra=[
            "-cwd",
            "-V",
            f"-pe smp {n_processes}",
            f"-l disk=1G",
        ],
        local_directory=os.sep.join([os.environ.get("PWD"), "dask-worker-space_popweighted_region"]),
    )

    client = Client(cluster)

    cluster.scale(jobs=n_jobs)

    # main processing
    matrix_stacked = np.array(
        np.meshgrid(
            np.linspace(0, 1.4, 8),
            np.linspace(0, 1.4, 8),
            np.linspace(0, 1.4, 8),
            np.linspace(0, 1.4, 8),
            np.linspace(0, 1.4, 8),
        )
    ).T.reshape(-1, 5)

    custom_inputs = [np.array(item).reshape(1, -1) for item in matrix_stacked]

    print(f"processing for {output} over {region} ...")
    outputs_popweighted = []
    bag_custom_inputs = db.from_sequence(custom_inputs, npartitions=n_workers)
    outputs_popweighted = bag_custom_inputs.map(popweight_outputs_for_input).compute()

    print("saving ...")
    joblib.dump(
        outputs_popweighted,
        f"/nobackup/earlacoa/machinelearning/data_annual/popweighted/popweighted_{region}_{output}_0.25deg_adjusted_scaled.joblib",
    )

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()

