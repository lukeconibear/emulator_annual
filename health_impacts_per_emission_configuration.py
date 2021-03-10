#!/usr/bin/env python3
import os
import time
import sys
import glob
import joblib
import xarray as xr
import numpy as np
import dask.bag as db
import geopandas as gpd
import pandas as pd
from dask_jobqueue import SGECluster
from dask.distributed import Client
from numba import njit, typeof, typed, types, jit

output = "PM2_5_DRY"
#output = "o3_6mDM8h"

# -----------
# functions
def shapefile_hia(hia, measure, clips, hia_path, lat, lon, regions):
    df = pd.DataFrame({'name': regions})

    hia_list = [
        key
        for key, value in hia.items()
        if measure in key and "total" in key and not "yl" in key
    ]
    hia_list.insert(0, "pop")
    if (measure == "ncdlri") or (measure == "5cod"):
        hia_list.insert(1, "pm25_popweighted")
    elif measure == "6cod":
        hia_list.insert(1, "apm25_popweighted")
    elif measure == "copd":
        hia_list.insert(1, "o3_popweighted")

    # loop through variables and regions
    for variable in hia_list:
        df[variable] = pd.Series(np.nan)
        for region in regions:
            da = xr.DataArray(hia[variable], coords=[lat, lon], dims=["lat", "lon"])
            clip = clips[region]
            da_clip = da.where(clip==0, other=np.nan) # didn't convert the values in this version to be consistent

            if variable == "pop":
                df.loc[df.name == region, variable] = np.nansum(da_clip.values)

            elif "popweighted" in variable:
                df.loc[df.name == region, variable] = (
                    np.nansum(da_clip.values) / df.loc[df.name == region, "pop"].values[0]
                )

            elif "rate" not in variable:
                df.loc[df.name == region, variable] = np.nansum(da_clip.values)

            else:
                df.loc[df.name == region, variable] = np.nanmean(da_clip.values)

    return df


def dict_to_typed_dict(dict_normal):
    """convert to typed dict for numba"""
    if len(dict_normal[next(iter(dict_normal))].shape) == 1:
        value_shape = types.f4[:]
    elif len(dict_normal[next(iter(dict_normal))].shape) == 2:
        value_shape = types.f4[:, :]

    typed_dict = typed.Dict.empty(types.string, value_shape)
    for key, value in dict_normal.items():
        typed_dict[key] = value

    return typed_dict


def outcome_per_age_ncdlri(pop_z_2015, age_grid, age, bm_ncd, bm_lri, outcome, metric, pm25_clipped, alpha, mu, pi, theta):
    return (pop_z_2015 * age_grid * (bm_ncd + bm_lri) * (1 - 1 / (np.exp(np.log(1 + pm25_clipped / alpha) / (1 + np.exp((mu - pm25_clipped) / pi)) * theta))))


def outcome_total(hia_ncdlri, outcome, metric):
    return sum(
        [
            value
            for key, value in hia_ncdlri.items()
            if f"{outcome}_ncdlri_{metric}" in key
        ]
    )


def dalys_age(hia_ncdlri, metric, age):
    return (
        hia_ncdlri[f"yll_ncdlri_{metric}_{age}"]
        + hia_ncdlri[f"yld_ncdlri_{metric}_{age}"]
    )


def dalys_total(hia_ncdlri, metric):
    return sum(
        [value for key, value in hia_ncdlri.items() if f"dalys_ncdlri_{metric}" in key]
    )


def rates_total(hia_ncdlri, outcome, metric, pop_z_2015):
    return hia_ncdlri[f"{outcome}_ncdlri_{metric}_total"] * (100000 / pop_z_2015)


def calc_hia_gemm_ncdlri(pm25, pop_z_2015, dict_ages, dict_bm, dict_gemm):
    """ health impact assessment using the GEMM for NCD+LRI """
    """ inputs are exposure to annual-mean PM2.5 on a global grid at 0.25 degrees """
    """ estimated for all ages individually, with outcomes separately """
    """ risks include China cohort """
    """ call example: calc_hia_gemm_ncdlri(pm25_ctl, pop_z_2015, dict_ages, dict_bm, dict_gemm) """
    # inputs
    ages = [
        "25_29",
        "30_34",
        "35_39",
        "40_44",
        "45_49",
        "50_54",
        "55_59",
        "60_64",
        "65_69",
        "70_74",
        "75_79",
        "80up",
    ]
    outcomes = ["mort", "yll", "yld"]
    metrics = ["mean", "upper", "lower"]
    lcc = 2.4  # no cap at 84 ugm-3
    pm25_clipped = (pm25 - lcc).clip(min=0)
    # health impact assessment
    hia_ncdlri = {}
    hia_ncdlri.update({"pop": pop_z_2015})
    hia_ncdlri.update({"pm25_popweighted": pop_z_2015 * pm25})
    for outcome in outcomes:
        for metric in metrics:
            for age in ages:
                # outcome_per_age
                hia_ncdlri.update(
                    {
                        f"{outcome}_ncdlri_{metric}_{age}": outcome_per_age_ncdlri(
                            pop_z_2015,
                            dict_ages[f"cf_age_fraction_{age}_grid"],
                            age,
                            dict_bm[f"i_{outcome}_ncd_both_{metric}_{age}"],
                            dict_bm[f"i_{outcome}_lri_both_{metric}_{age}"],
                            outcome,
                            metric,
                            pm25_clipped,
                            dict_gemm[f"gemm_health_nonacc_alpha_{age}"],
                            dict_gemm[f"gemm_health_nonacc_mu_{age}"],
                            dict_gemm[f"gemm_health_nonacc_pi_{age}"],
                            dict_gemm[f"gemm_health_nonacc_theta_{age}"]
                        )
                    }
                )

            # outcome_total
            hia_ncdlri.update(
                    {f"{outcome}_ncdlri_{metric}_total": outcome_total(hia_ncdlri, outcome, metric)}
            )

    for metric in metrics:
        for age in ages:
            # dalys_age
            hia_ncdlri.update(
                    {f"dalys_ncdlri_{metric}_{age}": dalys_age(hia_ncdlri, metric, age)}
            )

        # dalys_total
        hia_ncdlri.update(
            {f"dalys_ncdlri_{metric}_total": dalys_total(hia_ncdlri, metric)}
        )

    for outcome in ["mort", "yll", "yld", "dalys"]:
        for metric in metrics:
            # rates_total
            hia_ncdlri.update(
                    {f"{outcome}_rate_ncdlri_{metric}_total": rates_total(hia_ncdlri, outcome, metric, pop_z_2015)}
            )

    return hia_ncdlri


def create_attribute_fraction(value, dict_af):
    return dict_af[f'{value}']


create_attribute_fraction = np.vectorize(create_attribute_fraction)


def calc_hia_gbd2017_o3(o3, pop_z_2015, dict_ages, dict_bm, dict_af):
    """ health impact assessment using the GBD2017 function for O3 """
    """ inputs are exposure to annual-mean, daily maximum, 8-hour, O3 concentrations (ADM8h) on a global grid at 0.25 degrees """
    """ estimated for all ages individually """
    """ call example: calc_hia_gbd2017_o3(o3_ctl, pop_z_2015, dict_ages, dict_bm, dict_af) """
    # inputs
    ages = [
        "25_29",
        "30_34",
        "35_39",
        "40_44",
        "45_49",
        "50_54",
        "55_59",
        "60_64",
        "65_69",
        "70_74",
        "75_79",
        "80up",
    ]
    outcomes = ["mort", "yll", "yld"]
    metrics = ["mean", "upper", "lower"]
    # health impact assessment
    hia_o3 = {}
    hia_o3.update({"pop": pop_z_2015})
    hia_o3.update({"o3_popweighted": pop_z_2015 * o3})

    # attributable fraction
    o3_rounded = np.nan_to_num(np.around(o3, 1)) # 1dp for the nearest af
    af = {}
    for metric in metrics:
        af.update({
            metric: create_attribute_fraction(o3_rounded, dict_af[metric])
        })

    for outcome in outcomes:
        for metric in metrics:
            for age in ages:
                # mort, yll, yld - age
                hia_o3.update(
                    {
                        f"{outcome}_copd_{metric}_{age}": pop_z_2015
                        * dict_ages[f"cf_age_fraction_{age}_grid"]
                        * dict_bm[f"i_{outcome}_copd_both_{metric}_{age}"]
                        * af[metric]
                    }
                )

            # mort, yll, yld - total
            hia_o3.update(
                {
                    f"{outcome}_copd_{metric}_total": sum(
                        [
                            value
                            for key, value in hia_o3.items()
                            if f"{outcome}_copd_{metric}" in key
                        ]
                    )
                }
            )

    # dalys - age
    for metric in metrics:
        for age in ages:
            hia_o3.update(
                {
                    f"dalys_copd_{metric}_{age}": hia_o3[f"yll_copd_{metric}_{age}"]
                    + hia_o3[f"yld_copd_{metric}_{age}"]
                }
            )
        # dalys - total
        hia_o3.update(
            {
                f"dalys_copd_{metric}_total": sum(
                    [
                        value
                        for key, value in hia_o3.items()
                        if f"dalys_copd_{metric}" in key
                    ]
                )
            }
        )

    # rates - total
    for outcome in ["mort", "yll", "yld", "dalys"]:
        for metric in metrics:
            hia_o3.update(
                {
                    f"{outcome}_rate_copd_{metric}_total": hia_o3[f"{outcome}_copd_{metric}_total"] 
                    * (100_000 / pop_z_2015)
                }
            )

    return hia_o3



def health_impact_assessment_pm25(custom_output):
    with xr.open_dataset(
        f"/nobackup/earlacoa/machinelearning/data_annual/predictions/{output}/ds_{custom_output}_{output}_popgrid_0.25deg.nc"
    ) as ds:
        pm25 = ds["PM2_5_DRY"].values
        lon = ds.lon.values
        lat = ds.lat.values

    xx, yy = np.meshgrid(lon, lat)

    hia_ncdlri = calc_hia_gemm_ncdlri(pm25, pop_z_2015, dict_ages, dict_bm, dict_gemm)
    np.savez_compressed(
        f"/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/{output}/hia_{output}_{custom_output}.npz",
        hia_ncdlri=hia_ncdlri,
    )

    countries = ['China', 'Hong Kong', 'Macao', 'Taiwan']
    provinces = ['Anhui', 'Beijing', 'Chongqing', 'Fujian', 'Gansu', 'Guangdong', 'Guangxi', 'Guizhou', 'Hainan', 'Hebei', 'Heilongjiang', 'Henan', 'Hubei', 'Hunan', 'Jiangsu', 'Jiangxi', 'Jilin', 'Liaoning', 'Nei Mongol', 'Ningxia Hui', 'Qinghai', 'Shaanxi', 'Shandong', 'Shanghai', 'Shanxi', 'Sichuan', 'Tianjin', 'Xinjiang Uygur', 'Xizang', 'Yunnan', 'Zhejiang']
    prefectures = ['Dongguan', 'Foshan', 'Guangzhou', 'Huizhou', 'Jiangmen', 'Shenzhen', 'Zhaoqing', 'Zhongshan', 'Zhuhai']
    
    df_country_hia_ncdlri = shapefile_hia(
        hia_ncdlri,
        "ncdlri",
        clips,
        f"/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/{output}/",
        lat,
        lon,
        countries,
    )
    df_country_hia_ncdlri.to_csv(
        f"/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/{output}/df_country_hia_{output}_{custom_output}.csv"
    )

    df_province_hia_ncdlri = shapefile_hia(
        hia_ncdlri,
        "ncdlri",
        clips,
        f"/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/{output}/",
        lat,
        lon,
        provinces,
    )
    df_province_hia_ncdlri.to_csv(
        f"/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/{output}/df_province_hia_{output}_{custom_output}.csv"
    )

    df_prefecture_hia_ncdlri = shapefile_hia(
        hia_ncdlri,
        "ncdlri",
        clips,
        f"/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/{output}/",
        lat,
        lon,
        prefectures,
    )
    df_prefecture_hia_ncdlri.to_csv(
        f"/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/{output}/df_prefecture_hia_{output}_{custom_output}.csv"
    )


def health_impact_assessment_o3(custom_output):
    with xr.open_dataset(
        f"/nobackup/earlacoa/machinelearning/data_annual/predictions/{output}/ds_{custom_output}_{output}_popgrid_0.25deg.nc"
    ) as ds:
        o3_6mDM8h = ds["o3_6mDM8h"].values
        lon = ds.lon.values
        lat = ds.lat.values

    xx, yy = np.meshgrid(lon, lat)

    hia_o3 = calc_hia_gbd2017_o3(o3_6mDM8h, pop_z_2015, dict_ages, dict_bm, dict_af)
    np.savez_compressed(
        f"/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/{output}/hia_{output}_{custom_output}.npz",
        hia_o3=hia_o3,
    )

    countries = ['China', 'Hong Kong', 'Macao', 'Taiwan']
    provinces = ['Anhui', 'Beijing', 'Chongqing', 'Fujian', 'Gansu', 'Guangdong', 'Guangxi', 'Guizhou', 'Hainan', 'Hebei', 'Heilongjiang', 'Henan', 'Hubei', 'Hunan', 'Jiangsu', 'Jiangxi', 'Jilin', 'Liaoning', 'Nei Mongol', 'Ningxia Hui', 'Qinghai', 'Shaanxi', 'Shandong', 'Shanghai', 'Shanxi', 'Sichuan', 'Tianjin', 'Xinjiang Uygur', 'Xizang', 'Yunnan', 'Zhejiang']
    prefectures = ['Dongguan', 'Foshan', 'Guangzhou', 'Huizhou', 'Jiangmen', 'Shenzhen', 'Zhaoqing', 'Zhongshan', 'Zhuhai']
    
    df_country_hia_o3 = shapefile_hia(
        hia_o3,
        "copd",
        clips,
        f"/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/{output}/",
        lat,
        lon,
        countries,
    )
    df_country_hia_o3.to_csv(
        f"/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/{output}/df_country_hia_{output}_{custom_output}.csv"
    )

    df_province_hia_o3 = shapefile_hia(
        hia_o3,
        "copd",
        clips,
        f"/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/{output}/",
        lat,
        lon,
        provinces,
    )
    df_province_hia_o3.to_csv(
        f"/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/{output}/df_province_hia_{output}_{custom_output}.csv"
    )

    df_prefecture_hia_o3 = shapefile_hia(
        hia_o3,
        "copd",
        clips,
        f"/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/{output}/",
        lat,
        lon,
        prefectures,
    )
    df_prefecture_hia_o3.to_csv(
        f"/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/{output}/df_prefecture_hia_{output}_{custom_output}.csv"
    )


# -----------
# import data
clips = joblib.load('/nobackup/earlacoa/machinelearning/data_annual/clips.joblib')

with np.load("/nobackup/earlacoa/health/data/population-count-0.25deg.npz") as ds:
    pop_z_2015 = ds["pop_z_2015"]
    pop_xx = ds["pop_yy"]
    pop_yy = ds["pop_yy"]

with np.load(
    "/nobackup/earlacoa/health/data/GBD2017_population_age_fraction_global_2015_array_0.25deg.npz"
) as file_age:
    dict_ages = dict(
        zip(
            [key for key in file_age],
            [file_age[key].astype("float32") for key in file_age],
        )
    )

file_bm_list = []
for disease in ["copd", "ncd", "lri"]:
    file_bm_list.extend(
        glob.glob(
            "/nobackup/earlacoa/health/data/GBD2017_baseline_mortality*"
            + disease
            + "*0.25deg.npz"
        )
    )

dict_bm = {}
for file_bm_each in file_bm_list:
    file_bm = np.load(file_bm_each)
    dict_bm_each = dict(
        zip(
            [key for key in file_bm],
            [file_bm[key].astype("float32") for key in file_bm],
        )
    )
    dict_bm.update(dict_bm_each)

del file_bm_list, file_bm_each, file_bm, dict_bm_each

dict_af = {}
dict_af.update({'mean':  joblib.load('/nobackup/earlacoa/health/data/o3_dict_af_mean.joblib')})
dict_af.update({'lower': joblib.load('/nobackup/earlacoa/health/data/o3_dict_af_lower.joblib')})
dict_af.update({'upper': joblib.load('/nobackup/earlacoa/health/data/o3_dict_af_upper.joblib')})

with np.load(
    "/nobackup/earlacoa/health/data/GEMM_healthfunction_part1.npz"
) as file_gemm_1:
    dict_gemm = dict(
        zip(
            [key for key in file_gemm_1],
            [np.atleast_1d(file_gemm_1[key].astype("float32")) for key in file_gemm_1],
        )
    )

with np.load(
    "/nobackup/earlacoa/health/data/GEMM_healthfunction_part2.npz"
) as file_gemm_2:
    dict_gemm_2 = dict(
        zip(
            [key for key in file_gemm_2],
            [np.atleast_1d(file_gemm_2[key].astype("float32")) for key in file_gemm_2],
        )
    )

dict_gemm.update(dict_gemm_2)
#return clips, pop_z_2015, pop_xx, pop_yy, dict_ages, dict_bm, dict_af, dict_gemm
#clips, pop_z_2015, pop_xx, pop_yy, dict_ages, dict_bm, dict_af, dict_gemm = import_data()

# convert to typed dicts
#dict_ages = dict_to_typed_dict(dict_ages)
#dict_gemm = dict_to_typed_dict(dict_gemm)
#dict_bm = dict_to_typed_dict(dict_bm)
# -----------


def main():
    # dask cluster and client
    n_processes = 1
    n_jobs = 20
    n_workers = n_processes * n_jobs

    cluster = SGECluster(
        interface="ib0",
        walltime="01:00:00",
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
        local_directory=os.sep.join([os.environ.get("PWD"), "dask-worker-space"]),
    )

    client = Client(cluster)

    cluster.scale(jobs=n_jobs)

    time_start = time.time()

    # find remaining inputs
    custom_outputs = glob.glob(
        f"/nobackup/earlacoa/machinelearning/data_annual/predictions/{output}/ds*{output}_popgrid_0.25deg.nc"
    )
    custom_outputs_completed = glob.glob(
        f"/nobackup/earlacoa/machinelearning/data_annual/health_impact_assessments/{output}/hia_{output}_*.npz"
    )
    custom_outputs_remaining_set = set(
        [item.split("/")[-1][3 : -1 - len(output) - 19] for item in custom_outputs]
    ) - set(
        [
            item.split("/")[-1][4 + len(output) + 1 : -4]
            for item in custom_outputs_completed
        ]
    )
    custom_outputs_remaining = [item for item in custom_outputs_remaining_set]
    print(f"custom outputs remaining for {output}: {len(custom_outputs_remaining)}")

    # dask bag and process
    # run in 10 chunks over 10 cores, each chunk taking 2 minutes
    custom_outputs_remaining = custom_outputs_remaining[0:2000] 
    print(f"predicting for {len(custom_outputs_remaining)} custom outputs ...")
    bag_custom_outputs = db.from_sequence(
        custom_outputs_remaining, npartitions=n_workers
    )
    if output == "PM2_5_DRY":
        bag_custom_outputs.map(health_impact_assessment_pm25).compute()
    elif output == "o3_6mDM8h":
        bag_custom_outputs.map(health_impact_assessment_o3).compute()

    time_end = time.time() - time_start
    print(
        f"completed in {time_end:0.2f} seconds, or {time_end / 60:0.2f} minutes, or {time_end / 3600:0.2f} hours"
    )
    print(
        f"average time per custom output is {time_end / len(custom_outputs_remaining):0.2f} seconds"
    )

    client.close()
    cluster.close()


# -----------

if __name__ == "__main__":
    main()

