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

output = 'PM2_5_DRY'
# 'PM2_5_DRY', 'o3_6mDM8h'

df_obs = pd.read_csv(
    f'/nobackup/earlacoa/machinelearning/data_annual/china_measurements_corrected/df_obs_o3_6mDM8h_ppb_PM2_5_DRY.csv',
    index_col='datetime',
    parse_dates=True
)

outputs = ['o3_6mDM8h_ppb', 'PM2_5_DRY']
obs_files = glob.glob(f'/nobackup/earlacoa/machinelearning/data_annual/china_measurements_corrected/*.nc')

matrix_stacked = np.array(np.meshgrid(
    np.linspace(0.2, 1.3, 12), # np.linspace(0.0, 1.5, 16) for 0.0-1.5
    np.linspace(0.2, 1.3, 12), # np.linspace(0.2, 1.3, 12) for 0.2-1.3
    np.linspace(0.2, 1.3, 12), # removing edges of parameter space 0.0, 0.1, 1.4, 1.5
    np.linspace(0.2, 1.3, 12),
    np.linspace(0.2, 1.3, 12)
)).T.reshape(-1, 5)

obs_change_abs = {}
obs_change_per = {}
baselines = {}
targets = {}
target_diffs = {}
station_diffs_abs = {}
station_diffs_per = {}

def targets_per_station(output, obs_file):
    station_id = obs_file[76:-3]
    lat = df_obs.loc[df_obs.station_id == station_id].station_lat.unique()[0]
    lon = df_obs.loc[df_obs.station_id == station_id].station_lon.unique()[0]
    
    change_per = 100 * ((df_obs.loc[df_obs.station_id == station_id][output]['2017'].values[0] / \
                         df_obs.loc[df_obs.station_id == station_id][output]['2015'].values[0]) - 1)
    change_abs = df_obs.loc[df_obs.station_id == station_id][output]['2017'].values[0] - \
                 df_obs.loc[df_obs.station_id == station_id][output]['2015'].values[0]

    obs_change_abs.update({f'{station_id}_{output}': change_abs})
    obs_change_per.update({f'{station_id}_{output}': change_per})

    if output == 'o3_6mDM8h_ppb':
        emulator_output = 'o3_6mDM8h'
    else:
        emulator_output = output
        
    with xr.open_dataset(
        f'/nobackup/earlacoa/machinelearning/data_annual/predictions/{emulator_output}/ds_RES1.0_IND1.0_TRA1.0_AGR1.0_ENE1.0_{emulator_output}_popgrid_0.25deg.nc'
    )[emulator_output] as ds:
        baseline = ds.sel(lat=lat, method='nearest').sel(lon=lon, method='nearest').values
                
    baselines.update({f'{station_id}_{output}': baseline})

    target_abs = baseline + change_abs
    target_per = baseline * (1 + (change_per / 100))
    target = np.mean([target_abs, target_per])
    targets.update({f'{station_id}_{output}': target})
    
    target_diffs.update({f'{station_id}_{output}': target - baseline})

    target_diffs_abs = {}
    target_diffs_per = {}
    
    for matrix in matrix_stacked:
        inputs = matrix.reshape(-1, 5)        
        filename = f'RES{inputs[0][0]:.1f}_IND{inputs[0][1]:.1f}_TRA{inputs[0][2]:.1f}_AGR{inputs[0][3]:.1f}_ENE{inputs[0][4]:.1f}'
        with xr.open_dataset(
            f'/nobackup/earlacoa/machinelearning/data_annual/predictions/{emulator_output}/ds_{filename}_{emulator_output}_popgrid_0.25deg.nc'
        )[emulator_output] as ds:
            prediction = ds.sel(lat=lat, method='nearest').sel(lon=lon, method='nearest').values

        target_diff_abs = targets[f'{station_id}_{output}'] - prediction
        target_diff_per = (100 * (prediction / targets[f'{station_id}_{output}'])) - 100
        
        if abs(target_diff_per) < 1: # +/- 1% of target
            target_diffs_abs.update({filename: target_diff_abs})
            target_diffs_per.update({filename: target_diff_per})

    station_diffs_abs.update({f'{station_id}_{output}': target_diffs_abs})
    station_diffs_per.update({f'{station_id}_{output}': target_diffs_per})

    return obs_change_abs, obs_change_per, baselines, targets, target_diffs, station_diffs_per, station_diffs_abs


def main():
    # dask cluster and client
    n_processes = 1
    n_jobs = 35
    n_workers = n_processes * n_jobs

    cluster = SGECluster(
        interface='ib0',
        walltime='01:00:00',
        memory=f'64 G',
        resource_spec=f'h_vmem=64G',
        scheduler_options={
            'dashboard_address': ':5757',
        },
#        project='admiralty',
        job_extra = [
            '-cwd',
            '-V',
            f'-pe smp {n_processes}',
            f'-l disk=32G',
        ],
        local_directory = os.sep.join([
            os.environ.get('PWD'),
            'dask-worker-space'
        ])
    )

    client = Client(cluster)

    cluster.scale(jobs=n_jobs)

    time_start = time.time()

    # process targets for each station
    obs_files = glob.glob(f'/nobackup/earlacoa/machinelearning/data_annual/china_measurements_corrected/*.nc')
    obs_files = [f'{item[-8:-3]}' for item in obs_files]
    obs_files_completed = glob.glob(f'/nobackup/earlacoa/machinelearning/data_annual/find_emissions_that_match_change_air_quality/2015-2017/*{output}*')  
    obs_files_completed = [f'{item[-12:-7]}' for item in obs_files_completed]
    obs_files_remaining_set = set(obs_files) - set(obs_files_completed)
    obs_files_remaining = [item for item in obs_files_remaining_set]
    print(f'custom outputs remaining for {output}: {len(obs_files_remaining)}')

    # dask bag and process
    obs_files_remaining = obs_files_remaining[0:2] # run in 5,000 chunks over 30 cores, each chunk taking 2 minutes
    print(f'predicting for {len(obs_files_remaining)} custom outputs ...')
    bag_obs_files = db.from_sequence(obs_files_remaining, npartitions=n_workers)
    results = bag_obs_files.map(regrid_to_pop).compute()
    print('saving ...')
    joblib.dump(results, f'/nobackup/earlacoa/machinelearning/data_annual/find_emissions_that_match_change_air_quality/2015-2017/all_dicts_{output}_{station_id}.joblib')

    time_end = time.time() - time_start
    print(f'completed in {time_end:0.2f} seconds, or {time_end / 60:0.2f} minutes, or {time_end / 3600:0.2f} hours')
    print(f'average time per custom output is {time_end / len(obs_files_remaining):0.2f} seconds')

    client.close()
    cluster.close()

if __name__ == '__main__':
    main()
