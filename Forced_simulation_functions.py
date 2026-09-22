####################### FUNCIONES PARA TRATAR LAS SIMULACIONES ACOPLADAS ############################

import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm

def n_I_rainy_days(ds, variable):

    """ Función que calcula el número de días lluviosos y su intensidad por año así como la cantidad de precipitación caída durante los días no lluviosos """


    ds['masked_rainy'] = xr.where(ds.pr > 1, 1, 0)
    ds['masked_nonrainy'] = xr.where(ds.pr <= 1, 1, 0)
    ds['pr_rainy'] = xr.where(ds.pr > 1, ds.pr, 0)
    ds['pr_rainy_cum_yr'] = ds.pr_rainy.groupby('time.year').sum(dim = 'time', skipna = True)
    ds['n_rainy'] = ds.masked_rainy.groupby('time.year').sum(dim = 'time', skipna = True)
    ds['I_rainy'] = ds.pr_rainy_cum_yr / ds.n_rainy
    ds['pr_nonrainy_cum_yr'] = xr.where(ds.pr <= 1, ds.pr, 0).groupby('time.year').sum(dim = 'time', skipna = True)
    ds['pr_cum_yr'] = ds.pr.groupby('time.year').sum(dim = 'time', skipna = True)

    return ds




def WAM_moderate_heavy_extreme_difquant_forced(ds, ds_perc, variable):

    """ Función que calcula el número de días con eventos moderados, fuertes y extremos con percentiles calculados con todos los miembros de una simulación a lo largo de todos los tiempos, variable es la precipitación en mm/day. 
    Para la simulación acoplada utilizamos todos los tiempos.
    Para las simulationes forzadas, el umbral se calcula con la simulación de control y se aplica a las variantes DA y OM."""

    only_wet_days = xr.where((ds[variable] > 1), ds[variable], np.nan)

    # También saco el año de incio


    if 'member' not in list(ds.dims):
        perc_95 = only_wet_days.quantile(0.95, dim = ('time'), skipna = True)
        perc_75 = only_wet_days.quantile(0.75, dim = ('time'), skipna = True)
        
        ds['n_extreme'] = ((only_wet_days > perc_95.values)).groupby('time.year').sum(dim = 'time', skipna = True)
        ds['n_heavy'] = ((only_wet_days >= perc_75.values)).groupby('time.year').sum(dim = 'time', skipna = True)
        ds['n_moderate'] = ((only_wet_days < perc_75.values)).groupby('time.year').sum(dim = 'time', skipna = True)


    else:
        perc_95 = only_wet_days.quantile(0.95, dim = ('time', 'member'), skipna = True)
        perc_75 = only_wet_days.quantile(0.75, dim = ('time', 'member'), skipna = True)
        
        ds['n_extreme'] = ((only_wet_days > perc_95.values)).groupby('time.year').sum(dim = 'time', skipna = True)
        ds['n_heavy'] = ((only_wet_days >= perc_75.values)).groupby('time.year').sum(dim = 'time', skipna = True)
        ds['n_moderate'] = ((only_wet_days < perc_75.values)).groupby('time.year').sum(dim = 'time', skipna = True)


    return ds



def sea_mask_creator(ds):  
    from shapely.geometry import Point
    from shapely.prepared import prep
    import cartopy.feature as cfeature

    """ Función que crea un sea-mask a partir de los xarrays atmosféricos de la simulación forzada """

    lon = ds.lon.values
    lat = ds.lat.values

    lon2d, lat2d = np.meshgrid(lon, lat)

    # Alisar la malla para iteracion
    flat_points = np.column_stack((lon2d.ravel(), lat2d.ravel()))

    # Obtener los polígonos de tierra de cartopy
    land_geom = cfeature.NaturalEarthFeature('physical', 'land', '110m').geometries()
    prepared_land = [prep(geom) for geom in land_geom]

    # Creación de la sea mask (1 = mar, 0 = tierra)
    mask = []
    for lon_val, lat_val in tqdm(flat_points, desc="Creating sea mask"):
        point = Point(lon_val, lat_val)
        is_land = any(prep_geom.contains(point) for prep_geom in prepared_land)
        mask.append(0 if is_land else 1)

    # Reshapear al tamaño lat x lon original
    mask_array = np.array(mask).reshape(lat.size, lon.size)

    # Poner todo en un xarray
    sea_mask = xr.DataArray(
        mask_array,
        coords={'lat': lat, 'lon': lon},
        dims=['lat', 'lon'],
        name='sea_mask',
        attrs={'description': '1=sea, 0=land'}
    )

    return sea_mask