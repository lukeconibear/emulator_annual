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

normal = False # 20 percent intervals
extra = False # additional ones for the emission trend matching
climate_cobenefits = True

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
            'RES1.15_IND1.27_TRA0.98_AGR0.98_ENE1.36', # bottom-up 2010
            'RES1.19_IND1.30_TRA1.01_AGR1.01_ENE1.46', # bottom-up 2011
            'RES1.20_IND1.30_TRA1.01_AGR1.02_ENE1.39', # bottom-up 2012
            'RES1.13_IND1.29_TRA1.02_AGR1.01_ENE1.29', # bottom-up 2013
            'RES1.06_IND1.12_TRA0.99_AGR1.01_ENE1.12', # bottom-up 2014
            'RES0.92_IND0.84_TRA0.97_AGR0.99_ENE0.94', # bottom-up 2016
            'RES0.84_IND0.81_TRA0.99_AGR0.99_ENE0.89', # bottom-up 2017
            'RES0.95_IND0.99_TRA0.71_AGR0.88_ENE0.69', # top-down, 2016, both
            'RES0.89_IND0.90_TRA0.79_AGR0.74_ENE0.59', # top-down, 2017, both
            'RES0.71_IND0.91_TRA0.84_AGR0.53_ENE0.54', # top-down, 2018, both
            'RES0.72_IND0.88_TRA0.73_AGR0.71_ENE0.63', # top-down, 2019, both
            'RES0.64_IND0.79_TRA0.63_AGR0.56_ENE0.44', # top-down, 2020, both
            'RES0.96_IND0.93_TRA0.68_AGR0.87_ENE0.76', # top-down, 2016, either
            'RES0.96_IND0.92_TRA0.75_AGR0.83_ENE0.46', # top-down, 2017, either
            'RES0.86_IND1.08_TRA0.82_AGR0.52_ENE0.49', # top-down, 2018, either
            'RES0.87_IND0.94_TRA0.71_AGR0.59_ENE0.50', # top-down, 2019, either
            'RES0.79_IND0.79_TRA0.60_AGR0.42_ENE0.44', # top-down, 2020, either
            'RES0.76_IND1.08_TRA0.56_AGR0.77_ENE0.86', # top-down, 2016, o3_6mDM8h
            'RES0.94_IND0.67_TRA0.74_AGR0.72_ENE0.37', # top-down, 2017, o3_6mDM8h
            'RES0.93_IND1.11_TRA0.93_AGR0.64_ENE0.40', # top-down, 2018, o3_6mDM8h
            'RES0.94_IND1.12_TRA0.61_AGR0.48_ENE0.35', # top-down, 2019, o3_6mDM8h
            'RES0.94_IND0.99_TRA0.66_AGR0.50_ENE0.43', # top-down, 2020, o3_6mDM8h
            'RES1.01_IND0.82_TRA0.77_AGR0.94_ENE0.73', # top-down, 2016, PM2_5_DRY
            'RES0.88_IND0.88_TRA0.75_AGR0.91_ENE0.70', # top-down, 2017, PM2_5_DRY
            'RES0.79_IND0.85_TRA0.74_AGR0.85_ENE0.83', # top-down, 2018, PM2_5_DRY
            'RES0.83_IND0.71_TRA0.82_AGR0.89_ENE0.79', # top-down, 2019, PM2_5_DRY
            'RES0.94_IND0.38_TRA0.65_AGR0.74_ENE0.72', # top-down, 2020, PM2_5_DRY
        ]

    if climate_cobenefits:
        emission_configs_20percentintervals = [
            'RES0.934_IND0.937_TRA0.876_AGR1.054_ENE0.964', # Base_CLE_2020
            'RES0.934_IND1.000_TRA1.000_AGR1.000_ENE1.000', # Base_CLE_2020 - RES
            'RES1.000_IND0.937_TRA1.000_AGR1.000_ENE1.000', # Base_CLE_2020 - IND
            'RES1.000_IND1.000_TRA0.876_AGR1.000_ENE1.000', # Base_CLE_2020 - TRA
            'RES1.000_IND1.000_TRA1.000_AGR1.054_ENE1.000', # Base_CLE_2020 - AGR
            'RES1.000_IND1.000_TRA1.000_AGR1.000_ENE0.964', # Base_CLE_2020 - ENE
            'RES0.934_IND0.937_TRA0.876_AGR1.054_ENE0.964', # Base_MFR_2020
            'RES0.934_IND1.000_TRA1.000_AGR1.000_ENE1.000', # Base_MFR_2020 - RES
            'RES1.000_IND0.937_TRA1.000_AGR1.000_ENE1.000', # Base_MFR_2020 - IND
            'RES1.000_IND1.000_TRA0.876_AGR1.000_ENE1.000', # Base_MFR_2020 - TRA
            'RES1.000_IND1.000_TRA1.000_AGR1.054_ENE1.000', # Base_MFR_2020 - AGR
            'RES1.000_IND1.000_TRA1.000_AGR1.000_ENE0.964', # Base_MFR_2020 - ENE
            'RES0.934_IND0.937_TRA0.876_AGR1.054_ENE0.964', # SDS_MFR_2020
            'RES0.934_IND1.000_TRA1.000_AGR1.000_ENE1.000', # SDS_MFR_2020 - RES
            'RES1.000_IND0.937_TRA1.000_AGR1.000_ENE1.000', # SDS_MFR_2020 - IND
            'RES1.000_IND1.000_TRA0.876_AGR1.000_ENE1.000', # SDS_MFR_2020 - TRA
            'RES1.000_IND1.000_TRA1.000_AGR1.054_ENE1.000', # SDS_MFR_2020 - AGR
            'RES1.000_IND1.000_TRA1.000_AGR1.000_ENE0.964', # SDS_MFR_2020 - ENE
            'RES0.839_IND0.880_TRA0.788_AGR1.105_ENE0.947', # Base_CLE_2025
            'RES0.839_IND1.000_TRA1.000_AGR1.000_ENE1.000', # Base_CLE_2025 - RES
            'RES1.000_IND0.880_TRA1.000_AGR1.000_ENE1.000', # Base_CLE_2025 - IND
            'RES1.000_IND1.000_TRA0.788_AGR1.000_ENE1.000', # Base_CLE_2025 - TRA
            'RES1.000_IND1.000_TRA1.000_AGR1.105_ENE1.000', # Base_CLE_2025 - AGR
            'RES1.000_IND1.000_TRA1.000_AGR1.000_ENE0.947', # Base_CLE_2025 - ENE
            'RES0.536_IND0.495_TRA0.630_AGR0.787_ENE0.647', # Base_MFR_2025
            'RES0.536_IND1.000_TRA1.000_AGR1.000_ENE1.000', # Base_MFR_2025 - RES
            'RES1.000_IND0.495_TRA1.000_AGR1.000_ENE1.000', # Base_MFR_2025 - IND
            'RES1.000_IND1.000_TRA0.630_AGR1.000_ENE1.000', # Base_MFR_2025 - TRA
            'RES1.000_IND1.000_TRA1.000_AGR0.787_ENE1.000', # Base_MFR_2025 - AGR
            'RES1.000_IND1.000_TRA1.000_AGR1.000_ENE0.647', # Base_MFR_2025 - ENE
            'RES0.507_IND0.483_TRA0.598_AGR0.787_ENE0.557', # SDS_MFR_2025
            'RES0.507_IND1.000_TRA1.000_AGR1.000_ENE1.000', # SDS_MFR_2025 - RES
            'RES1.000_IND0.483_TRA1.000_AGR1.000_ENE1.000', # SDS_MFR_2025 - IND
            'RES1.000_IND1.000_TRA0.598_AGR1.000_ENE1.000', # SDS_MFR_2025 - TRA
            'RES1.000_IND1.000_TRA1.000_AGR0.787_ENE1.000', # SDS_MFR_2025 - AGR
            'RES1.000_IND1.000_TRA1.000_AGR1.000_ENE0.557', # SDS_MFR_2025 - ENE
            'RES0.769_IND0.853_TRA0.760_AGR1.159_ENE0.935', # Base_CLE_2030
            'RES0.769_IND1.000_TRA1.000_AGR1.000_ENE1.000', # Base_CLE_2030 - RES
            'RES1.000_IND0.853_TRA1.000_AGR1.000_ENE1.000', # Base_CLE_2030 - IND
            'RES1.000_IND1.000_TRA0.760_AGR1.000_ENE1.000', # Base_CLE_2030 - TRA
            'RES1.000_IND1.000_TRA1.000_AGR1.159_ENE1.000', # Base_CLE_2030 - AGR
            'RES1.000_IND1.000_TRA1.000_AGR1.000_ENE0.935', # Base_CLE_2030 - ENE
            'RES0.409_IND0.469_TRA0.540_AGR0.810_ENE0.661', # Base_MFR_2030
            'RES0.409_IND1.000_TRA1.000_AGR1.000_ENE1.000', # Base_MFR_2030 - RES
            'RES1.000_IND0.469_TRA1.000_AGR1.000_ENE1.000', # Base_MFR_2030 - IND
            'RES1.000_IND1.000_TRA0.540_AGR1.000_ENE1.000', # Base_MFR_2030 - TRA
            'RES1.000_IND1.000_TRA1.000_AGR0.810_ENE1.000', # Base_MFR_2030 - AGR
            'RES1.000_IND1.000_TRA1.000_AGR1.000_ENE0.661', # Base_MFR_2030 - ENE
            'RES0.353_IND0.449_TRA0.483_AGR0.810_ENE0.517', # SDS_MFR_2030
            'RES0.353_IND1.000_TRA1.000_AGR1.000_ENE1.000', # SDS_MFR_2030 - RES
            'RES1.000_IND0.449_TRA1.000_AGR1.000_ENE1.000', # SDS_MFR_2030 - IND
            'RES1.000_IND1.000_TRA0.483_AGR1.000_ENE1.000', # SDS_MFR_2030 - TRA
            'RES1.000_IND1.000_TRA1.000_AGR0.810_ENE1.000', # SDS_MFR_2030 - AGR
            'RES1.000_IND1.000_TRA1.000_AGR1.000_ENE0.517', # SDS_MFR_2030 - ENE
            'RES0.732_IND0.821_TRA0.748_AGR1.180_ENE0.938', # Base_CLE_2035
            'RES0.732_IND1.000_TRA1.000_AGR1.000_ENE1.000', # Base_CLE_2035 - RES
            'RES1.000_IND0.821_TRA1.000_AGR1.000_ENE1.000', # Base_CLE_2035 - IND
            'RES1.000_IND1.000_TRA0.748_AGR1.000_ENE1.000', # Base_CLE_2035 - TRA
            'RES1.000_IND1.000_TRA1.000_AGR1.180_ENE1.000', # Base_CLE_2035 - AGR
            'RES1.000_IND1.000_TRA1.000_AGR1.000_ENE0.938', # Base_CLE_2035 - ENE
            'RES0.344_IND0.438_TRA0.466_AGR0.821_ENE0.674', # Base_MFR_2035
            'RES0.344_IND1.000_TRA1.000_AGR1.000_ENE1.000', # Base_MFR_2035 - RES
            'RES1.000_IND0.438_TRA1.000_AGR1.000_ENE1.000', # Base_MFR_2035 - IND
            'RES1.000_IND1.000_TRA0.466_AGR1.000_ENE1.000', # Base_MFR_2035 - TRA
            'RES1.000_IND1.000_TRA1.000_AGR0.821_ENE1.000', # Base_MFR_2035 - AGR
            'RES1.000_IND1.000_TRA1.000_AGR1.000_ENE0.674', # Base_MFR_2035 - ENE
            'RES0.296_IND0.414_TRA0.394_AGR0.821_ENE0.494', # SDS_MFR_2035
            'RES0.296_IND1.000_TRA1.000_AGR1.000_ENE1.000', # SDS_MFR_2035 - RES
            'RES1.000_IND0.414_TRA1.000_AGR1.000_ENE1.000', # SDS_MFR_2035 - IND
            'RES1.000_IND1.000_TRA0.394_AGR1.000_ENE1.000', # SDS_MFR_2035 - TRA
            'RES1.000_IND1.000_TRA1.000_AGR0.821_ENE1.000', # SDS_MFR_2035 - AGR
            'RES1.000_IND1.000_TRA1.000_AGR1.000_ENE0.494', # SDS_MFR_2035 - ENE
            'RES0.681_IND0.775_TRA0.707_AGR1.245_ENE0.897', # Base_CLE_2050
            'RES0.681_IND1.000_TRA1.000_AGR1.000_ENE1.000', # Base_CLE_2050 - RES
            'RES1.000_IND0.775_TRA1.000_AGR1.000_ENE1.000', # Base_CLE_2050 - IND
            'RES1.000_IND1.000_TRA0.707_AGR1.000_ENE1.000', # Base_CLE_2050 - TRA
            'RES1.000_IND1.000_TRA1.000_AGR1.245_ENE1.000', # Base_CLE_2050 - AGR
            'RES1.000_IND1.000_TRA1.000_AGR1.000_ENE0.897', # Base_CLE_2050 - ENE
            'RES0.221_IND0.383_TRA0.377_AGR0.860_ENE0.678', # Base_MFR_2050
            'RES0.221_IND1.000_TRA1.000_AGR1.000_ENE1.000', # Base_MFR_2050 - RES
            'RES1.000_IND0.383_TRA1.000_AGR1.000_ENE1.000', # Base_MFR_2050 - IND
            'RES1.000_IND1.000_TRA0.377_AGR1.000_ENE1.000', # Base_MFR_2050 - TRA
            'RES1.000_IND1.000_TRA1.000_AGR0.860_ENE1.000', # Base_MFR_2050 - AGR
            'RES1.000_IND1.000_TRA1.000_AGR1.000_ENE0.678', # Base_MFR_2050 - ENE
            'RES0.196_IND0.351_TRA0.272_AGR0.860_ENE0.433', # SDS_MFR_2050
            'RES0.196_IND1.000_TRA1.000_AGR1.000_ENE1.000', # SDS_MFR_2050 - RES
            'RES1.000_IND0.351_TRA1.000_AGR1.000_ENE1.000', # SDS_MFR_2050 - IND
            'RES1.000_IND1.000_TRA0.272_AGR1.000_ENE1.000', # SDS_MFR_2050 - TRA
            'RES1.000_IND1.000_TRA1.000_AGR0.860_ENE1.000', # SDS_MFR_2050 - AGR
            'RES1.000_IND1.000_TRA1.000_AGR1.000_ENE0.433', # SDS_MFR_2050 - ENE
            'RES0.000_IND0.937_TRA0.876_AGR1.054_ENE0.964', # Base_CLE_2020 - NO RES
            'RES0.934_IND0.000_TRA0.876_AGR1.054_ENE0.964', # Base_CLE_2020 - NO IND
            'RES0.934_IND0.937_TRA0.000_AGR1.054_ENE0.964', # Base_CLE_2020 - NO TRA
            'RES0.934_IND0.937_TRA0.876_AGR0.000_ENE0.964', # Base_CLE_2020 - NO AGR
            'RES0.934_IND0.937_TRA0.876_AGR1.054_ENE0.000', # Base_CLE_2020 - NO ENE
            'RES0.000_IND0.937_TRA0.876_AGR1.054_ENE0.964', # Base_MFR_2020 - NO RES
            'RES0.934_IND0.000_TRA0.876_AGR1.054_ENE0.964', # Base_MFR_2020 - NO IND
            'RES0.934_IND0.937_TRA0.000_AGR1.054_ENE0.964', # Base_MFR_2020 - NO TRA
            'RES0.934_IND0.937_TRA0.876_AGR0.000_ENE0.964', # Base_MFR_2020 - NO AGR
            'RES0.934_IND0.937_TRA0.876_AGR1.054_ENE0.000', # Base_MFR_2020 - NO ENE
            'RES0.000_IND0.937_TRA0.876_AGR1.054_ENE0.964', # SDS_MFR_2020 - NO RES
            'RES0.934_IND0.000_TRA0.876_AGR1.054_ENE0.964', # SDS_MFR_2020 - NO IND
            'RES0.934_IND0.937_TRA0.000_AGR1.054_ENE0.964', # SDS_MFR_2020 - NO TRA
            'RES0.934_IND0.937_TRA0.876_AGR0.000_ENE0.964', # SDS_MFR_2020 - NO AGR
            'RES0.934_IND0.937_TRA0.876_AGR1.054_ENE0.000', # SDS_MFR_2020 - NO ENE
            'RES0.000_IND0.880_TRA0.788_AGR1.105_ENE0.947', # Base_CLE_2025 - NO RES
            'RES0.839_IND0.000_TRA0.788_AGR1.105_ENE0.947', # Base_CLE_2025 - NO IND
            'RES0.839_IND0.880_TRA0.000_AGR1.105_ENE0.947', # Base_CLE_2025 - NO TRA
            'RES0.839_IND0.880_TRA0.788_AGR0.000_ENE0.947', # Base_CLE_2025 - NO AGR
            'RES0.839_IND0.880_TRA0.788_AGR1.105_ENE0.000', # Base_CLE_2025 - NO ENE
            'RES0.000_IND0.495_TRA0.630_AGR0.787_ENE0.647', # Base_MFR_2025 - NO RES
            'RES0.536_IND0.000_TRA0.630_AGR0.787_ENE0.647', # Base_MFR_2025 - NO IND
            'RES0.536_IND0.495_TRA0.000_AGR0.787_ENE0.647', # Base_MFR_2025 - NO TRA
            'RES0.536_IND0.495_TRA0.630_AGR0.000_ENE0.647', # Base_MFR_2025 - NO AGR
            'RES0.536_IND0.495_TRA0.630_AGR0.787_ENE0.000', # Base_MFR_2025 - NO ENE
            'RES0.000_IND0.483_TRA0.598_AGR0.787_ENE0.557', # SDS_MFR_2025 - NO RES
            'RES0.507_IND0.000_TRA0.598_AGR0.787_ENE0.557', # SDS_MFR_2025 - NO IND
            'RES0.507_IND0.483_TRA0.000_AGR0.787_ENE0.557', # SDS_MFR_2025 - NO TRA
            'RES0.507_IND0.483_TRA0.598_AGR0.000_ENE0.557', # SDS_MFR_2025 - NO AGR
            'RES0.507_IND0.483_TRA0.598_AGR0.787_ENE0.000', # SDS_MFR_2025 - NO ENE
            'RES0.000_IND0.853_TRA0.760_AGR1.159_ENE0.935', # Base_CLE_2030 - NO RES
            'RES0.769_IND0.000_TRA0.760_AGR1.159_ENE0.935', # Base_CLE_2030 - NO IND
            'RES0.769_IND0.853_TRA0.000_AGR1.159_ENE0.935', # Base_CLE_2030 - NO TRA
            'RES0.769_IND0.853_TRA0.760_AGR0.000_ENE0.935', # Base_CLE_2030 - NO AGR
            'RES0.769_IND0.853_TRA0.760_AGR1.159_ENE0.000', # Base_CLE_2030 - NO ENE
            'RES0.000_IND0.469_TRA0.540_AGR0.810_ENE0.661', # Base_MFR_2030 - NO RES
            'RES0.409_IND0.000_TRA0.540_AGR0.810_ENE0.661', # Base_MFR_2030 - NO IND
            'RES0.409_IND0.469_TRA0.000_AGR0.810_ENE0.661', # Base_MFR_2030 - NO TRA
            'RES0.409_IND0.469_TRA0.540_AGR0.000_ENE0.661', # Base_MFR_2030 - NO AGR
            'RES0.409_IND0.469_TRA0.540_AGR0.810_ENE0.000', # Base_MFR_2030 - NO ENE
            'RES0.000_IND0.449_TRA0.483_AGR0.810_ENE0.517', # SDS_MFR_2030 - NO RES
            'RES0.353_IND0.000_TRA0.483_AGR0.810_ENE0.517', # SDS_MFR_2030 - NO IND
            'RES0.353_IND0.449_TRA0.000_AGR0.810_ENE0.517', # SDS_MFR_2030 - NO TRA
            'RES0.353_IND0.449_TRA0.483_AGR0.000_ENE0.517', # SDS_MFR_2030 - NO AGR
            'RES0.353_IND0.449_TRA0.483_AGR0.810_ENE0.000', # SDS_MFR_2030 - NO ENE
            'RES0.000_IND0.821_TRA0.748_AGR1.180_ENE0.938', # Base_CLE_2035 - NO RES
            'RES0.732_IND0.000_TRA0.748_AGR1.180_ENE0.938', # Base_CLE_2035 - NO IND
            'RES0.732_IND0.821_TRA0.000_AGR1.180_ENE0.938', # Base_CLE_2035 - NO TRA
            'RES0.732_IND0.821_TRA0.748_AGR0.000_ENE0.938', # Base_CLE_2035 - NO AGR
            'RES0.732_IND0.821_TRA0.748_AGR1.180_ENE0.000', # Base_CLE_2035 - NO ENE
            'RES0.000_IND0.438_TRA0.466_AGR0.821_ENE0.674', # Base_MFR_2035 - NO RES
            'RES0.344_IND0.000_TRA0.466_AGR0.821_ENE0.674', # Base_MFR_2035 - NO IND
            'RES0.344_IND0.438_TRA0.000_AGR0.821_ENE0.674', # Base_MFR_2035 - NO TRA
            'RES0.344_IND0.438_TRA0.466_AGR0.000_ENE0.674', # Base_MFR_2035 - NO AGR
            'RES0.344_IND0.438_TRA0.466_AGR0.821_ENE0.000', # Base_MFR_2035 - NO ENE
            'RES0.000_IND0.414_TRA0.394_AGR0.821_ENE0.494', # SDS_MFR_2035 - NO RES
            'RES0.296_IND0.000_TRA0.394_AGR0.821_ENE0.494', # SDS_MFR_2035 - NO IND
            'RES0.296_IND0.414_TRA0.000_AGR0.821_ENE0.494', # SDS_MFR_2035 - NO TRA
            'RES0.296_IND0.414_TRA0.394_AGR0.000_ENE0.494', # SDS_MFR_2035 - NO AGR
            'RES0.296_IND0.414_TRA0.394_AGR0.821_ENE0.000', # SDS_MFR_2035 - NO ENE
            'RES0.000_IND0.775_TRA0.707_AGR1.245_ENE0.897', # Base_CLE_2050 - NO RES
            'RES0.681_IND0.000_TRA0.707_AGR1.245_ENE0.897', # Base_CLE_2050 - NO IND
            'RES0.681_IND0.775_TRA0.000_AGR1.245_ENE0.897', # Base_CLE_2050 - NO TRA
            'RES0.681_IND0.775_TRA0.707_AGR0.000_ENE0.897', # Base_CLE_2050 - NO AGR
            'RES0.681_IND0.775_TRA0.707_AGR1.245_ENE0.000', # Base_CLE_2050 - NO ENE
            'RES0.000_IND0.383_TRA0.377_AGR0.860_ENE0.678', # Base_MFR_2050 - NO RES
            'RES0.221_IND0.000_TRA0.377_AGR0.860_ENE0.678', # Base_MFR_2050 - NO IND
            'RES0.221_IND0.383_TRA0.000_AGR0.860_ENE0.678', # Base_MFR_2050 - NO TRA
            'RES0.221_IND0.383_TRA0.377_AGR0.000_ENE0.678', # Base_MFR_2050 - NO AGR
            'RES0.221_IND0.383_TRA0.377_AGR0.860_ENE0.000', # Base_MFR_2050 - NO ENE
            'RES0.000_IND0.351_TRA0.272_AGR0.860_ENE0.433', # SDS_MFR_2050 - NO RES
            'RES0.196_IND0.000_TRA0.272_AGR0.860_ENE0.433', # SDS_MFR_2050 - NO IND
            'RES0.196_IND0.351_TRA0.000_AGR0.860_ENE0.433', # SDS_MFR_2050 - NO TRA
            'RES0.196_IND0.351_TRA0.272_AGR0.000_ENE0.433', # SDS_MFR_2050 - NO AGR
            'RES0.196_IND0.351_TRA0.272_AGR0.860_ENE0.000', # SDS_MFR_2050 - NO ENE
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

