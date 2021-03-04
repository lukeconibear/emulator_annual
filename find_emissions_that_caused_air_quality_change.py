import pandas as pd
import numpy as np
import glob
import joblib
import xarray as xr
import geopandas as gpd
from itertools import islice

path = '/nobackup/earlacoa/machinelearning/data_annual'

df_obs = pd.read_csv(
    f'{path}/china_measurements_corrected/df_obs_o3_6mDM8h_ppb_PM2_5_DRY.csv',
    index_col='datetime',
    parse_dates=True
)

outputs = ['o3_6mDM8h_ppb', 'PM2_5_DRY']
obs_files = glob.glob(f'{path}/china_measurements_corrected/*.nc')

obs_change_abs = {}
obs_change_per = {}
emulators = {}
baselines = {}
targets = {}
target_diffs = {}

print('processing targets per station')
for output in outputs:
    for obs_file in obs_files:
        station_id = obs_file[76:-3] # [47:-3] on viper, [76:-3] on arc4
        print(f'    for {output} - {station_id}') 
        lat = round(df_obs.loc[df_obs.station_id == station_id].station_lat.unique()[0] * 4) / 4
        lon = round(df_obs.loc[df_obs.station_id == station_id].station_lon.unique()[0] * 4) / 4
        change_per = 100 * ((df_obs.loc[df_obs.station_id == station_id][output]['2017'].values[0] / \
                             df_obs.loc[df_obs.station_id == station_id][output]['2015'].values[0]) - 1)
        change_abs = df_obs.loc[df_obs.station_id == station_id][output]['2017'].values[0] - \
                     df_obs.loc[df_obs.station_id == station_id][output]['2015'].values[0]
        if output == 'o3_6mDM8h_ppb':
            emulator_output_name = 'o3_6mDM8h'
        else:
            emulator_output_name = output           
        try:
            emulator = joblib.load(f'{path}/emulators/{emulator_output_name}/emulator_{emulator_output_name}_{lat}_{lon}.joblib')
            emulators.update({f'{station_id}_{output}': emulator})
            obs_change_abs.update({f'{station_id}_{output}': change_abs})
            obs_change_per.update({f'{station_id}_{output}': change_per})
            baseline = emulator.predict(np.array([[1.0, 1.0, 1.0, 1.0, 1.0]]))
            baselines.update({f'{station_id}_{output}': baseline[0]})
            target_abs = baseline + change_abs
            target_per = baseline * (1 + (change_per / 100))
            target = np.mean([target_abs, target_per])
            targets.update({f'{station_id}_{output}': target})
        except:
            FileNotFoundError


# delete targets that are nan
for key in [key for key in targets.keys()]:
    if np.isnan(targets[key]):
        del targets[key]


# add the target diffs
for key in [key for key in targets.keys()]:
    target_diffs.update({key: targets[key] - baselines[key]})


# ensure matching count
for key in list(set([key for key in emulators.keys()]) - set([key for key in targets.keys()])):
    del emulators[key]


joblib.dump(targets, f'{path}/find_emissions_that_match_change_air_quality/2015-2017/targets.joblib')
joblib.dump(baselines, f'{path}/find_emissions_that_match_change_air_quality/2015-2017/baselines.joblib')
joblib.dump(target_diffs, f'{path}/find_emissions_that_match_change_air_quality/2015-2017/target_diffs.joblib')
joblib.dump(emulators, f'{path}/find_emissions_that_match_change_air_quality/2015-2017/emulators.joblib')
print('    completed')

print('filtering target predictions')
matrix_stacked = np.array(np.meshgrid(
    np.linspace(0.3, 1.2, 10), # 1.5 and 16 for 0.1, 1.5 and 6 for 0.3, 1.4 and 8 for 0.2
    np.linspace(0.3, 1.2, 10), # removing edges of parameter space 0.0, 0.1, 1.4, 1.5
    np.linspace(0.3, 1.2, 10), # also removing unlikely reductions in emissions of > -40% or +30%
    np.linspace(0.3, 1.2, 10),
    np.linspace(0.3, 1.2, 10)
)).T.reshape(-1, 5)

station_diffs_abs = {}
station_diffs_per = {}

for station_id, emulator in emulators.items():
    target_diffs_abs = {}
    target_diffs_per = {}
    for matrix in matrix_stacked:
        inputs = matrix.reshape(-1, 5)
        filename = f'RES{inputs[0][0]:.1f}_IND{inputs[0][1]:.1f}_TRA{inputs[0][2]:.1f}_AGR{inputs[0][3]:.1f}_ENE{inputs[0][4]:.1f}'
        target_diff_abs = targets[station_id] - emulator.predict(inputs)[0]
        target_diff_per = (100 * (emulator.predict(inputs)[0] / targets[station_id])) - 100
        if abs(target_diff_per) < 1: # +/- 1% of target
            target_diffs_abs.update({filename: target_diff_abs})
            target_diffs_per.update({filename: target_diff_per})
    
    station_diffs_abs.update({station_id: target_diffs_abs})
    station_diffs_per.update({station_id: target_diffs_per})


keys = [list(station_diffs_per[station].keys())for station in station_diffs_per.keys()]
keys_flatten = [item for sublist in keys for item in sublist]
keys_unique = {}
for key in keys_flatten:
    if key not in keys_unique:
        keys_unique.update({key: 1})
    elif key in keys_unique:
        keys_unique.update({key: keys_unique[key] + 1})


joblib.dump(station_diffs_per, f'{path}/find_emissions_that_match_change_air_quality/2015-2017/station_diffs_per_1percent.joblib')
joblib.dump(station_diffs_abs, f'{path}/find_emissions_that_match_change_air_quality/2015-2017/station_diffs_abs_1percent.joblib')
print('    completed')

