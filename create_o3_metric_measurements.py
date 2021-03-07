import datetime
import glob
import pandas as pd
import xarray as xr

obs_files = glob.glob(
    "/nobackup/earlacoa/machinelearning/data_annual/china_measurements_corrected/*.nc"
)

df_obs = pd.DataFrame(
    {
        "station_id": [],
        "station_lat": [],
        "station_lon": [],
        "name": [],
        "prefecture": [],
        "o3_6mDM8h_ppb": [],
        "PM2_5_DRY": [],
        "datetime": [],
    }
)


def create_metrics(obs_file, df_obs):
    ds = xr.open_dataset(obs_file)

    o3 = ds["O3"]
    pm25 = ds["PM2.5"]

    years = ["2015", "2016", "2017", "2018", "2019"]

    for year in years:
        # pm25
        pm25_year = pm25.sel(time=year)
        pm25_year_mean = pm25_year.mean(dim="time").values

        # o3
        o3_year = o3.sel(time=year)
        # first: 24, 8-hour, rolling mean, O3 concentrations
        o3_6mDM8h_8hrrollingmean = (
            o3_year.rolling(time=8).construct("window").mean("window")
        )
        # second: find the max of these each day (daily maximum, 8-hour)
        o3_6mDM8h_dailymax = (
            o3_6mDM8h_8hrrollingmean.sortby("time").resample(time="24H").max()
        )
        # third: 6-month mean - to account for different times when seasonal maximums e.g. different hemispheres
        o3_6mDM8h_6monthmean = o3_6mDM8h_dailymax.resample(time="6M").mean()
        # fourth: maximum of these
        o3_6mDM8h = o3_6mDM8h_6monthmean.max(dim="time")
        # convert units from µg m-3 to ppb - using 1 ppb = 1.9957 µg m-3
        o3_6mDM8h_ppb = o3_6mDM8h.values / 1.9957

        df = pd.DataFrame(
            {
                "station_id": [ds.station],
                "station_lat": [ds.lat],
                "station_lon": [ds.lon],
                "name": [ds.name],
                "prefecture": [ds.city],
                "o3_6mDM8h_ppb": [o3_6mDM8h_ppb],
                "PM2_5_DRY": [pm25_year_mean],
                "datetime": [pd.to_datetime(datetime.datetime(int(year), 1, 1))],
            }
        )

        df_obs = df_obs.append(df)

    ds.close()

    return df_obs


for obs_file in obs_files:
    df_obs = create_metrics(obs_file, df_obs)


df_obs = df_obs.set_index("datetime").sort_index()

df_obs.to_csv(
    "/nobackup/earlacoa/machinelearning/data_annual/china_measurements_corrected/df_obs_o3_6mDM8h_ppb_PM2_5_DRY.csv"
)
