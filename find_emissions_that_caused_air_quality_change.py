#!/usr/bin/env python3
import os
import time
import glob
import joblib
from itertools import islice
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import xesmf as xe
import dask.bag as db
from dask_jobqueue import SGECluster
from dask.distributed import Client

#output = "PM2_5_DRY"
output = "o3_6mDM8h_ppb"

year_range = '2015-2014'

if output == "o3_6mDM8h_ppb":
    emulator_output = "o3_6mDM8h"
else:
    emulator_output = output

use_10percent_intervals = True
use_20percent_intervals = False 
if use_10percent_intervals:
    sub_folder = year_range
    n_jobs = 20
    walltime='01:00:00' # also change for .bash)
    emission_configs = np.array(
        np.meshgrid(
            np.linspace(0.2, 1.3, 12),
            np.linspace(0.2, 1.3, 12),
            np.linspace(0.2, 1.3, 12),
            np.linspace(0.2, 1.3, 12),
            np.linspace(0.2, 1.3, 12),
        )
    ).T.reshape(-1, 5)
elif use_20percent_intervals:
    sub_folder = f'{year_range}_20percentintervals'
    n_jobs = 5
    walltime='00:05:00' # also change for .bash)
    emission_configs = np.array(
        np.meshgrid(
            np.linspace(0.3, 1.3, 6),
            np.linspace(0.3, 1.3, 6),
            np.linspace(0.3, 1.3, 6),
            np.linspace(0.3, 1.3, 6),
            np.linspace(0.3, 1.3, 6),
        )
    ).T.reshape(-1, 5)

df_obs = pd.read_csv(
    f"/nobackup/earlacoa/machinelearning/data_annual/china_measurements_corrected/df_obs_o3_6mDM8h_ppb_PM2_5_DRY.csv",
    index_col="datetime",
    parse_dates=True,
)

# stations left
obs_files = glob.glob(f"/nobackup/earlacoa/machinelearning/data_annual/china_measurements_corrected/*.nc")
obs_files = [f"{obs_file[-8:-3]}" for obs_file in obs_files]
obs_files_completed = glob.glob(f"/nobackup/earlacoa/machinelearning/data_annual/find_emissions_that_match_change_air_quality/{sub_folder}_scaled/*{output}*")
obs_files_completed = [f"{item[-12:-7]}" for item in obs_files_completed]
obs_files_remaining_set = set(obs_files) - set(obs_files_completed)
obs_files_remaining = [item for item in obs_files_remaining_set]
print(f"custom outputs remaining for {output}: {len(obs_files_remaining)}")

station_id = obs_files_remaining[0]
lat = df_obs.loc[df_obs.station_id == station_id].station_lat.unique()[0]
lon = df_obs.loc[df_obs.station_id == station_id].station_lon.unique()[0]

obs_change_abs = {}
obs_change_per = {}
baselines = {}
targets = {}
target_diffs = {}

change_per = 100 * ((
    df_obs.loc[df_obs.station_id == station_id][output]["2014"].values[0]
    / df_obs.loc[df_obs.station_id == station_id][output]["2015"].values[0]
) - 1)
change_abs = (
    df_obs.loc[df_obs.station_id == station_id][output]["2014"].values[0]
    - df_obs.loc[df_obs.station_id == station_id][output]["2015"].values[0]
)

obs_change_abs.update({f"{station_id}_{output}": change_abs})
obs_change_per.update({f"{station_id}_{output}": change_per})

with xr.open_dataset(
    f"/nobackup/earlacoa/machinelearning/data_annual/predictions/{emulator_output}_scaled/ds_RES1.0_IND1.0_TRA1.0_AGR1.0_ENE1.0_{emulator_output}_popgrid_0.25deg.nc"
)[emulator_output] as ds:
    baseline = (
        ds.sel(lat=lat, method="nearest").sel(lon=lon, method="nearest").values
    )

baselines.update({f"{station_id}_{output}": baseline})

target_abs = baseline + change_abs
target_per = baseline * (1 + (change_per / 100))
target = np.mean([target_abs, target_per])
targets.update({f"{station_id}_{output}": target})

target_diffs.update({f"{station_id}_{output}": target - baseline}) 
    
       
def filter_emission_configs(emission_config):
    station_diffs_abs = {}
    station_diffs_per = {}
    target_diffs_abs = {}
    target_diffs_per = {}
    
    inputs = emission_config.reshape(-1, 5)
    filename = f"RES{inputs[0][0]:.1f}_IND{inputs[0][1]:.1f}_TRA{inputs[0][2]:.1f}_AGR{inputs[0][3]:.1f}_ENE{inputs[0][4]:.1f}"
    with xr.open_dataset(
        f"/nobackup/earlacoa/machinelearning/data_annual/predictions/{emulator_output}_scaled/ds_{filename}_{emulator_output}_popgrid_0.25deg.nc"
    )[emulator_output] as ds:
        prediction = (
            ds.sel(lat=lat, method="nearest").sel(lon=lon, method="nearest").values
        )

    target_diff_abs = targets[f"{station_id}_{output}"] - prediction
    target_diff_per = (100 * (prediction / targets[f"{station_id}_{output}"])) - 100

    if abs(target_diff_per) < 1:  # +/- 1% of target
        target_diffs_abs.update({filename: target_diff_abs})
        target_diffs_per.update({filename: target_diff_per})

    station_diffs_abs.update({f"{station_id}_{output}": target_diffs_abs})
    station_diffs_per.update({f"{station_id}_{output}": target_diffs_per})

    return station_diffs_abs, station_diffs_per


def main():
    # dask cluster and client
    n_processes = 1
    n_workers = n_processes * n_jobs

    cluster = SGECluster(
        interface="ib0",
        walltime=walltime,
        memory=f"32 G",
        resource_spec=f"h_vmem=32G",
        scheduler_options={
            "dashboard_address": ":5757",
        },
        job_extra=[
            "-cwd",
            "-V",
            f"-pe smp {n_processes}",
            f"-l disk=32G",
        ],
        local_directory=os.sep.join([os.environ.get("PWD"), "dask-find-emis-space"]),
    )

    client = Client(cluster)

    cluster.scale(jobs=n_jobs)

    time_start = time.time()

    # dask bag over emission_configs
    print(f"predicting over {len(emission_configs)} emission configs for {station_id} ...")   
    bag_emission_configs = db.from_sequence(emission_configs, npartitions=n_workers)
    results = bag_emission_configs.map(filter_emission_configs).compute()   
    
    station_diffs_abs = [result[0] for result in results]
    station_diffs_per = [result[1] for result in results]
    key = [key for key in baselines.keys()][0]
    station_diffs_abs = [station_diff_abs for station_diff_abs in station_diffs_abs if len(station_diff_abs[key]) > 0]
    station_diffs_per = [station_diff_per for station_diff_per in station_diffs_per if len(station_diff_per[key]) > 0]
    
    merged_per = {}
    for station_diff_per in station_diffs_per:
        merged_per = {**merged_per, **station_diff_per[key]}


    merged_abs = {}
    for station_diff_abs in station_diffs_abs:
        merged_abs = {**merged_abs, **station_diff_abs[key]}


    station_diffs_per = {key: merged_per}   
    station_diffs_abs = {key: merged_abs}
    
    joblib.dump(obs_change_abs, f"/nobackup/earlacoa/machinelearning/data_annual/find_emissions_that_match_change_air_quality/{sub_folder}_scaled/obs_change_abs_{output}_{station_id}.joblib")
    joblib.dump(obs_change_per, f"/nobackup/earlacoa/machinelearning/data_annual/find_emissions_that_match_change_air_quality/{sub_folder}_scaled/obs_change_per_{output}_{station_id}.joblib")
    joblib.dump(baselines, f"/nobackup/earlacoa/machinelearning/data_annual/find_emissions_that_match_change_air_quality/{sub_folder}_scaled/baselines_{output}_{station_id}.joblib")
    joblib.dump(targets, f"/nobackup/earlacoa/machinelearning/data_annual/find_emissions_that_match_change_air_quality/{sub_folder}_scaled/targets_{output}_{station_id}.joblib")
    joblib.dump(target_diffs, f"/nobackup/earlacoa/machinelearning/data_annual/find_emissions_that_match_change_air_quality/{sub_folder}_scaled/target_diffs_{output}_{station_id}.joblib")
    joblib.dump(station_diffs_abs, f"/nobackup/earlacoa/machinelearning/data_annual/find_emissions_that_match_change_air_quality/{sub_folder}_scaled/station_diffs_abs_{output}_{station_id}.joblib")
    joblib.dump(station_diffs_per, f"/nobackup/earlacoa/machinelearning/data_annual/find_emissions_that_match_change_air_quality/{sub_folder}_scaled/station_diffs_per_{output}_{station_id}.joblib")

    time_end = time.time() - time_start
    print(f"completed in {time_end:0.2f} seconds, or {time_end / 60:0.2f} minutes, or {time_end / 3600:0.2f} hours")

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
