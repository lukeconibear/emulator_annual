#!/usr/bin/env python3
import os
import time
import sys
import glob
import xarray as xr
import numpy as np
import dask.bag as db
import geopandas as gpd
import pandas as pd
from rasterio import features
from affine import Affine
from dask_jobqueue import SGECluster
from dask.distributed import Client

output = 'PM2_5_DRY'
# 'PM2_5_DRY', 'o3_6mDM8h'

# -----------
# functions
def import_npz(npz_file, namespace):
    data = np.load(npz_file)
    for var in data:
        if data[var].dtype == np.dtype('float64'):
            namespace[var] = data[var].astype('float32')
        else:
            namespace[var] = data[var]


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def transform_from_latlon(lat, lon):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale


def rasterize(shapes, coords, latitude='latitude', longitude='longitude', fill=np.nan, **kwargs):
    transform = transform_from_latlon(coords[latitude], coords[longitude])
    out_shape = (len(coords[latitude]), len(coords[longitude]))
    raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
    spatial_coords = {latitude: coords[latitude], longitude: coords[longitude]}
    return xr.DataArray(raster, coords=spatial_coords, dims=(latitude, longitude))


def shapefile_hia(hia, measure, region, shapefile_file, hia_path, lat, lon, **kwargs):
    shp = gpd.read_file(shapefile_file)
    if region == 'country':
        ids = list(shp['ID_0'].values)
        names = list(shp['NAME_ENGLI'].values)
    elif (region == 'state') or (region == 'province'):
        ids = list(shp['GID_1'].values)
        names = list(shp['NAME_1'].values)
    elif (region == 'city') or (region == 'prefecture'):
        ids = list(shp['GID_2'].values)
        names = list(shp['NAME_2'].values)

    names = [name.replace(' ', '_') for name in names]
    df = pd.DataFrame(np.asarray(ids))
    df.columns = ['id']
    df['name'] = pd.Series(np.asarray(names))
    region_list = kwargs.get('region_list', None)
    if region_list != None:
        df = df.loc[df['id'].isin(region_list),:]
        ids = np.array(region_list)

    hia_list = [key for key, value in hia.items() if measure in key and 'total' in key and not 'yl' in key]
    hia_list.insert(0, 'pop')
    if (measure == 'ncdlri') or (measure == '5cod'):
        hia_list.insert(1, 'pm25_popweighted')
    elif measure == '6cod':
        hia_list.insert(1, 'apm25_popweighted')
    elif measure == 'copd':
        hia_list.insert(1, 'o3_popweighted')

    # loop through variables and regions
    for variable in hia_list:
        df[variable] = pd.Series(np.nan)
        for i in ids:
            # create list of tuples (shapely.geometry, id) to allow for many different polygons within a .shp file
            if region == 'country':
                shapes = [(shape, n) for n, shape in enumerate(shp[shp.ID_0 == i].geometry)]
            elif (region == 'state') or (region == 'province'):
                shapes = [(shape, n) for n, shape in enumerate(shp[shp.GID_1 == i].geometry)]
            elif (region == 'city') or (region == 'prefecture'):
                shapes = [(shape, n) for n, shape in enumerate(shp[shp.GID_2 == i].geometry)]
                
            # create dataarray for each variable
            da = xr.DataArray(hia[variable], coords=[lat, lon], dims=['lat', 'lon'])
            # create the clip for the shapefile
            clip = rasterize(shapes, da.coords, longitude='lon', latitude='lat')
            # clip the dataarray
            da_clip = da.where(clip==0, other=np.nan)
            # assign to dataframe
            if variable == 'pop':
                df.loc[df.id == i, variable] = np.nansum(da_clip.values)

            elif 'popweighted' in variable:
                df.loc[df.id == i, variable] = np.nansum(da_clip.values) / df.loc[df.id == i, 'pop'].values[0]

            elif 'rate' not in variable:
                df.loc[df.id == i, variable] = np.nansum(da_clip.values)

            else:
                df.loc[df.id == i, variable] = np.nanmean(da_clip.values)

    return df


def outcome_per_age(pop_z_2015, dict_ages, age, dict_bm, outcome, metric, pm25_clipped, dict_gemm):
    return pop_z_2015 * dict_ages['cf_age_fraction_' + age + '_grid'] \
           * (dict_bm['i_' + outcome + '_ncd_both_' + metric + '_' + age] \
              + dict_bm['i_' + outcome + '_lri_both_' + metric + '_' + age]) \
           * (1 - 1 / (np.exp(np.log(1 + pm25_clipped \
                                     / dict_gemm['gemm_health_nonacc_alpha_' + age]) \
                              / (1 + np.exp((dict_gemm['gemm_health_nonacc_mu_' + age] \
                                             - pm25_clipped) \
                                            / dict_gemm['gemm_health_nonacc_pi_' + age])) \
                              * dict_gemm['gemm_health_nonacc_theta_' + age])))

def outcome_total(hia_ncdlri, outcome, metric):
    return sum([value for key, value in hia_ncdlri.items() if outcome + '_ncdlri_' + metric in key])

def dalys_age(hia_ncdlri, metric, age):
    return hia_ncdlri['yll_ncdlri_' + metric + '_' + age] + hia_ncdlri['yld_ncdlri_' + metric + '_' + age]

def dalys_total(hia_ncdlri, metric):
    return sum([value for key, value in hia_ncdlri.items() if 'dalys_ncdlri_' + metric in key])

def rates_total(hia_ncdlri, outcome, metric, pop_z_2015):
    return hia_ncdlri[outcome + '_ncdlri_' + metric + '_total'] * ( 100000 / pop_z_2015)

def calc_hia_gemm_ncdlri(pm25, pop_z_2015, dict_ages, dict_bm, dict_gemm):
    """ health impact assessment using the GEMM for NCD+LRI """
    """ inputs are exposure to annual-mean PM2.5 on a global grid at 0.25 degrees """
    """ estimated for all ages individually, with outcomes separately """
    """ risks include China cohort """
    """ call example: calc_hia_gemm_ncdlri(pm25_ctl, pop_z_2015, dict_ages, dict_bm, dict_gemm) """
    # inputs
    ages = ['25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59',
            '60_64', '65_69', '70_74', '75_79', '80up']
    outcomes = ['mort', 'yll', 'yld']
    metrics = ['mean', 'upper', 'lower']
    lcc = 2.4 # no cap at 84 ugm-3
    pm25_clipped = (pm25 - lcc).clip(min=0)
    # health impact assessment
    hia_ncdlri = {}
    hia_ncdlri.update({'pop' : pop_z_2015})
    hia_ncdlri.update({'pm25_popweighted' : pop_z_2015 * pm25})
    for outcome in outcomes:
        for metric in metrics:
            for age in ages:
                # outcome_per_age
                hia_ncdlri.update({ outcome + '_ncdlri_' + metric + '_' + age : outcome_per_age(pop_z_2015, dict_ages, age, dict_bm, outcome, metric, pm25_clipped, dict_gemm) })

            # outcome_total
            hia_ncdlri.update({ outcome + '_ncdlri_' + metric + '_total' : outcome_total(hia_ncdlri, outcome, metric) })

    for metric in metrics:
        for age in ages:
            # dalys_age
            hia_ncdlri.update({ 'dalys_ncdlri_' + metric + '_' + age : dalys_age(hia_ncdlri, metric, age) })

        # dalys_total
        hia_ncdlri.update({ 'dalys_ncdlri_' + metric + '_total' : dalys_total(hia_ncdlri, metric) })

    for outcome in ['mort', 'yll', 'yld', 'dalys']:
        for metric in metrics:
            # rates_total
            hia_ncdlri.update({ outcome + '_rate_ncdlri_' + metric + '_total' : rates_total(hia_ncdlri, outcome, metric, pop_z_2015) })

    return hia_ncdlri


def calc_hia_gbd2017_o3(o3, pop_z_2015, dict_ages, dict_bm, dict_o3):
    """ health impact assessment using the GBD2017 function for O3 """
    """ inputs are exposure to annual-mean, daily maximum, 8-hour, O3 concentrations (ADM8h) on a global grid at 0.25 degrees """
    """ estimated for all ages individually """
    """ call example: calc_hia_gbd2017_o3(o3_ctl, pop_z_2015, dict_ages, dict_bm, dict_o3) """
    # inputs
    ages = ['25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64',
            '65_69', '70_74', '75_79', '80up']
    outcomes = ['mort', 'yll', 'yld']
    metrics = ['mean', 'upper', 'lower']
    # health impact assessment
    hia_o3 = {}
    hia_o3.update({'pop' : pop_z_2015})
    hia_o3.update({'o3_popweighted' : pop_z_2015 * o3})

    # attributable fraction
    dict_af = {}
    for metric in metrics:
        dict_af.update({ metric : np.array([[dict_o3['af_o3_copd_' + metric][find_nearest(dict_o3['o3_conc'], o3[lat][lon])] for lon in range(o3.shape[1])] for lat in range(o3.shape[0])]) })

    for outcome in outcomes:
        for metric in metrics:
            for age in ages:
                # mort, yll, yld - age
                hia_o3.update({ outcome + '_copd_' + metric + '_' + age :
                                 pop_z_2015 * dict_ages['cf_age_fraction_' + age + '_grid']
                                 * dict_bm['i_' + outcome + '_copd_both_' + metric + '_' + age]
                                 * dict_af[metric] })

            # mort, yll, yld - total
            hia_o3.update({ outcome + '_copd_' + metric + '_total' :
                             sum([value for key, value in hia_o3.items()
                                  if outcome + '_copd_' + metric in key]) })

    # dalys - age
    for metric in metrics:
        for age in ages:
            hia_o3.update({ 'dalys_copd_' + metric + '_' + age :
                             hia_o3['yll_copd_' + metric + '_' + age]
                             + hia_o3['yld_copd_' + metric + '_' + age] })
        # dalys - total
        hia_o3.update({ 'dalys_copd_' + metric + '_total' :
                         sum([value for key, value in hia_o3.items()
                              if 'dalys_copd_' + metric in key]) })

    # rates - total
    for outcome in ['mort', 'yll', 'yld', 'dalys']:
        for metric in metrics:
            hia_o3.update({ outcome + '_rate_copd_' + metric + '_total' :
                             hia_o3[outcome + '_copd_' + metric + '_total']
                             * ( 100000 / pop_z_2015) })

    return hia_o3


def health_impact_assessment_pm25(custom_output):
    with xr.open_dataset(f'/nobackup/earlacoa/machinelearning/data_annual/predictions/{output}/ds_{custom_output}_{output}_popgrid_0.25deg.nc') as ds:
        pm25 = ds['PM2_5_DRY'].values
        lon = ds.lon.values
        lat = ds.lat.values


    xx, yy = np.meshgrid(lon, lat)

    hia_ncdlri = calc_hia_gemm_ncdlri(pm25, pop_z_2015, dict_ages, dict_bm, dict_gemm)
    np.savez_compressed(f'/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/PM2_5_DRY/hia_ncdlri_{custom_output}.npz', hia_ncdlri=hia_ncdlri)

    region_list = [49, 102, 132, 225]
    df_country_hia_ncdlri = shapefile_hia(hia_ncdlri, 'ncdlri', 'country', '/nobackup/earlacoa/health/data/gadm28_adm0.shp', '/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/PM2_5_DRY/', lat, lon, region_list=region_list)
    df_country_hia_ncdlri.to_csv(f'/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/PM2_5_DRY/df_country_hia_PM2_5_DRY_{custom_output}.csv')

    df_province_hia_ncdlri = shapefile_hia(hia_ncdlri, 'ncdlri', 'province', '/nobackup/earlacoa/health/data/gadm36_CHN_1.shp', '/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/PM2_5_DRY/', lat, lon)
    df_province_hia_ncdlri.to_csv(f'/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/PM2_5_DRY/df_province_hia_PM2_5_DRY_{custom_output}.csv')

    region_list = ['CHN.6.2_1', 'CHN.6.3_1', 'CHN.6.4_1', 'CHN.6.6_1', 'CHN.6.7_1', 'CHN.6.15_1', 'CHN.6.19_1', 'CHN.6.20_1', 'CHN.6.21_1']
    df_prefecture_hia_ncdlri  = shapefile_hia(hia_ncdlri, 'ncdlri', 'prefecture', '/nobackup/earlacoa/health/data/gadm36_CHN_2.shp', '/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/PM2_5_DRY/', lat, lon, region_list=region_list)
    df_prefecture_hia_ncdlri.to_csv(f'/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/PM2_5_DRY/df_prefecture_hia_PM2_5_DRY_{custom_output}.csv')


def health_impact_assessment_o3(custom_output):
    with xr.open_dataset('/nobackup/earlacoa/machinelearning/data_annual/wrfout_combined-domains_global_0.25deg_2015_o3_6mDM8h_ppb.nc') as ds:
        o3_6mDM8h_ctl = ds['__xarray_dataarray_variable__'].values
        lon = ds.lon.values
        lat = ds.lat.values

    xx, yy = np.meshgrid(lon, lat)

    with xr.open_dataset('/nobackup/earlacoa/machinelearning/data_annual/predictions/o3/ds_RES1.0_IND1.0_TRA1.0_AGR1.0_ENE1.0_o3_popgrid_0.25deg.nc') as ds:
        o3_emulator_ctl = ds['o3'].values

    with xr.open_dataset(f'/nobackup/earlacoa/machinelearning/data_annual/predictions/{output}/ds_{custom_output}_{output}_popgrid_0.25deg.nc') as ds:
        o3_emulator_custom_output = ds['o3'].values

    fraction = o3_emulator_custom_output / o3_emulator_ctl
    o3_6mDM8h_custom_output = fraction * o3_6mDM8h_ctl

    hia_o3 = calc_hia_gbd2017_o3(o3_6mDM8h_custom_output, pop_z_2015, dict_ages, dict_bm, dict_o3)
    np.savez_compressed(f'/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/o3/hia_o3_{custom_output}.npz', hia_o3=hia_o3)

    region_list = [49, 102, 132, 225]
    df_country_hia_o3 = shapefile_hia(hia_o3, 'copd', 'country', '/nobackup/earlacoa/health/data/gadm28_adm0.shp', '/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/o3/', lat, lon, region_list=region_list)
    df_country_hia_o3.to_csv(f'/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/o3/df_country_hia_o3_{custom_output}.csv')

    df_province_hia_o3 = shapefile_hia(hia_o3, 'copd', 'province', '/nobackup/earlacoa/health/data/gadm36_CHN_1.shp', '/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/o3/', lat, lon)
    df_province_hia_o3.to_csv(f'/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/o3/df_province_hia_o3_{custom_output}.csv')

    region_list = ['CHN.6.2_1', 'CHN.6.3_1', 'CHN.6.4_1', 'CHN.6.6_1', 'CHN.6.7_1', 'CHN.6.15_1', 'CHN.6.19_1', 'CHN.6.20_1', 'CHN.6.21_1']
    df_prefecture_hia_o3  = shapefile_hia(hia_o3, 'copd', 'prefecture', '/nobackup/earlacoa/health/data/gadm36_CHN_2.shp', '/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/o3/', lat, lon, region_list=region_list)
    df_prefecture_hia_o3.to_csv(f'/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/o3/df_prefecture_hia_o3_{custom_output}.csv')

# -----------
# import data
import_npz('/nobackup/earlacoa/health/data/population-count-0.25deg.npz', globals())

with np.load('/nobackup/earlacoa/health/data/GBD2017_population_age_fraction_global_2015_array_0.25deg.npz') as file_age:
    dict_ages = dict(zip([key for key in file_age], [file_age[key].astype('float32') for key in file_age]))

file_bm_list = []
for disease in ['copd', 'ncd', 'lri']:
    file_bm_list.extend(glob.glob('/nobackup/earlacoa/health/data/GBD2017_baseline_mortality*' + disease + '*0.25deg.npz'))

dict_bm = {}
for file_bm_each in file_bm_list:
    file_bm = np.load(file_bm_each)
    dict_bm_each = dict(zip([key for key in file_bm], [file_bm[key].astype('float32') for key in file_bm]))
    dict_bm.update(dict_bm_each)

del file_bm_list, file_bm_each, file_bm, dict_bm_each

with np.load('/nobackup/earlacoa/health/data/GBD2017_O3_attributablefraction.npz') as file_o3:
    dict_o3 = dict(zip([key for key in file_o3], [file_o3[key].astype('float32') for key in file_o3]))

with np.load('/nobackup/earlacoa/health/data/GEMM_healthfunction_part1.npz') as file_gemm_1:
    dict_gemm   = dict(zip([key for key in file_gemm_1], [np.atleast_1d(file_gemm_1[key].astype('float32')) for key in file_gemm_1]))

with np.load('/nobackup/earlacoa/health/data/GEMM_healthfunction_part2.npz') as file_gemm_2:
    dict_gemm_2 = dict(zip([key for key in file_gemm_2], [np.atleast_1d(file_gemm_2[key].astype('float32')) for key in file_gemm_2]))

dict_gemm.update(dict_gemm_2)

# -----------

def main():
    # dask cluster and client
    n_processes = 1
    n_jobs = 35
    n_workers = n_processes * n_jobs

    cluster = SGECluster(
        interface='ib0',
        walltime='01:00:00',
        memory=f'128 G',
        resource_spec=f'h_vmem=128G',
        scheduler_options={
            'dashboard_address': ':5757',
        },
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

    # regrid custom outputs to pop grid
    custom_outputs = glob.glob(f'/nobackup/earlacoa/machinelearning/data_annual/predictions/{output}/ds*{output}_popgrid_0.25deg.nc')
    custom_outputs_completed = glob.glob(f'/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/{output}/hia_{output}_*.npz')
    custom_outputs_remaining_set = set([item.split('/')[-1][3:-1 - len(output) - 19] for item in custom_outputs]) - set([item.split('/')[-1][4 + len(output) + 1:-4] for item in custom_outputs_completed])
    custom_outputs_remaining = [item for item in custom_outputs_remaining_set]
    print(f'custom outputs remaining for {output}: {len(custom_outputs_remaining)}')

    # dask bag and process
    custom_outputs_remaining = custom_outputs_remaining[0:500] # run in 500 chunks over 30 cores, each chunk taking 2 minutes
    print(f'predicting for {len(custom_outputs_remaining)} custom outputs ...')
    bag_custom_outputs = db.from_sequence(custom_outputs_remaining, npartitions=n_workers)
    if output == 'PM2_5_DRY':
        bag_custom_outputs.map(health_impact_assessment_pm25).compute()
    elif output == 'o3':
        bag_custom_outputs.map(health_impact_assessment_o3).compute()

    time_end = time.time() - time_start
    print(f'completed in {time_end:0.2f} seconds, or {time_end / 60:0.2f} minutes, or {time_end / 3600:0.2f} hours')
    print(f'average time per custom output is {time_end / len(custom_outputs_remaining):0.2f} seconds')

    client.close()
    cluster.close()

# -----------

if __name__ == '__main__':
    main()

