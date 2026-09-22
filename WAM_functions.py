###### SCRIPT CON LAS FUNCIONES RELATIVAS AL WAM PARA LAS SIMULACIONES ACOPLADAS #################

import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm

def area_weighted_mean(da, lat_name="lat", lon_name="lon"):

    
    # latitude in radians
    lat = np.deg2rad(da[lat_name])

    # weights proportional to grid-cell area
    weights = np.cos(lat)

    # normalize weights (optional but nice)
    weights = weights / weights.mean()

    # apply weighted mean
    return da.weighted(weights).mean(dim=(lat_name, lon_name))



def WAM_onset_demise_duration(ds, variable):
    
    """ Función que calcula el inicio y fin del monzón africano con el método de Liebman et al 2012
    ds es el xarray en cuestion y 'variable' el nombre en string de la variable precipiacion """

    # Calculo para cada punto de malla la media anual de precipitación
    ds[variable + '_annual_mean'] = ds.mean(dim = 'time', skipna = True)[variable]
    
    # Calculo también la media para cada día del año
    ds_daily_mean = ds[variable].groupby('time.dayofyear').mean(dim='time')

    # Calculo la anomalía cumulada de precipitación
    c = (ds_daily_mean - ds[variable + '_annual_mean']) 

    # Calculo para cada día del año la anomalía cumulada
    C = c.cumsum(dim = 'dayofyear', skipna = True)

    # El mínimo de C se considera el inicio climatológico de la estación humeda y el máximo el fin
    min_index = C.argmin(dim = 'dayofyear',skipna = True)
    climatological_onset = min_index + 1

    # El máximo de C marca el fin
    max_index = C.argmax(dim = 'dayofyear', skipna = True)
    climatological_end = max_index + 1

    # Libero memoria 

    del C, c

        # Añado también el límite inferior y superior para sumar
    ds[variable + '_lim_inf'] = climatological_onset - 50
    ds[variable + '_lim_sup'] = climatological_end + 50

    # Creo un diccionario de xarrays por año
    ds_year = ds.groupby('time.year')
    # Creating a dictionary to hold the subsets
    ds_yearly = {year: group for year, group in ds_year}

    # Tengo que localizar el primer y el último año de mi xarray
    #yr_start = ds.time.values[0].astype('datetime64[Y]').astype(int) + 1970
    #yr_end = ds.time.values[-1].astype('datetime64[Y]').astype(int) + 1970

    yr_start = ds['time.year'].values[0]
    yr_end = ds['time.year'].values[-1]
    
    for i in tqdm(np.arange(yr_start,yr_end+1,1)):

        # Creo una coordenada que contenga el día del año, diferenciando en función del tipo de objeto que tengo 

        if isinstance(ds_yearly[i].time.values[0], np.datetime64):
            ds_yearly[i]['dayofyear'] = (('time',), pd.to_datetime(ds_yearly[i].time).dayofyear.values)
        else:
            ds_yearly[i]['dayofyear'] = (('time',), [t.timetuple().tm_yday for t in ds_yearly[i].time.values])

        a = ds_yearly[i][variable] - ds[variable + '_annual_mean']

        # Pongo ceros
        a_masked = xr.where((ds_yearly[i]['dayofyear'] >= ds_yearly[i][variable + '_lim_inf']) & (ds_yearly[i]['dayofyear'] <= ds_yearly[i][variable + '_lim_sup']), a,0)

        # Ahora hago cumsum
        A = a_masked.cumsum(dim = 'time', skipna = True)

        # El día en que A tiene su mínimo es el comienzo del año y el día en que A llega al máximo es el día de cese
        max_index = A.argmax(dim = 'time', skipna = True)
        min_index = A.argmin(dim = 'time', skipna = True)

        # Libero memoria

        del A, a_masked, a
      
        ds_yearly[i][variable +'_onset'] = min_index + 1
        ds_yearly[i][variable +'_demise'] = max_index + 1


         # También calculo la duración del monzón
        ds_yearly[i][variable +'_duration'] = ds_yearly[i][variable + '_demise'] - ds_yearly[i][variable +'_onset']

        # Los años en los que la duración es negativa ponemos ceros

        ds_yearly[i][variable + '_duration'] = xr.where((ds_yearly[i][variable + '_duration'] >= 0), ds_yearly[i][variable + '_duration'],0)

        # Pongo también a cero los días que están fuera de la estación del monzón

        ds_yearly[i][variable + '_masked'] = xr.where((ds_yearly[i]['dayofyear'] >= ds_yearly[i][variable + '_onset']) & (ds_yearly[i]['dayofyear'] <= ds_yearly[i][variable + '_demise']), ds_yearly[i][variable],0)

        # Elimino las variables innecesarias 

        ds_yearly[i] = ds_yearly[i].drop_vars([variable + '_lim_sup', variable + '_lim_inf'])

        # Reagrupar el xarray en uno concatenandolo
    ds_orig = ds_yearly[yr_start]
        
    for j in tqdm(np.arange(yr_start+1,yr_end+1,1)):
        ds_orig = xr.concat([ds_orig, ds_yearly[j]], dim = 'time')

        # Calculo la media de las medias de la fecha de inicio y de fin por año
    #ds_orig[variable + '_duration_yearly'] = ds_orig[variable + '_duration'].groupby('time.year').mean(skipna = True)
    #ds_orig[variable + '_onset_yearly'] = ds_orig[variable + '_onset'].groupby('time.year').mean(skipna = True)
    #ds_orig[variable + '_demise_yearly'] = ds_orig[variable + '_demise'].groupby('time.year').mean(skipna = True)

        # Calculo la media del onset y del demise
    #ds_orig[variable + '_duration_mean'] = ds_orig[variable + '_duration_yearly'].mean(dim = 'year', skipna = True)
    #ds_orig[variable + '_onset_mean'] = ds_orig[variable + '_onset_yearly'].mean(dim = 'year', skipna = True)
    #ds_orig[variable + '_demise_mean'] = ds_orig[variable + '_demise_yearly'].mean(dim = 'year', skipna = True)
        # Calculo las medias zonales del onset, demise y duración
    #ds_orig[variable + '_duration_zonal_mean'] = ds_orig[variable + '_duration_yearly'].mean(dim = 'lon', skipna = True)
    #ds_orig[variable + '_onset_zonal_mean'] = ds_orig[variable + '_onset_yearly'].mean(dim = 'lon', skipna = True)
    #ds_orig[variable + '_demise_zonal_mean'] = ds_orig[variable + '_demise_yearly'].mean(dim = 'lon', skipna = True)

    # Elimino las variables innecesarias 


    # Retorno el xarray con las variables metidas

    return ds_orig


def WAM_wet_days_int_pr_tot(ds, variable, threshold):
    """ Función que sirve para calcular el número de días lluviosos, la intensidad de
    precipitación en los mismos y la precipitación total en la estación lluviosa según las 
    definiciones de Mohino et al 2024
    
    La variable de entrada debe ser pr_1_masked o cualquier precipitación de otro ensemble
    
    perc75 y perc95 son los percentiles 75 y 95 calculados en cada punto de malla a partir de los días húmedos"""

    # Divido el xarray por años 

    #ds_year = ds.groupby('time.year')

    # Considero día húmedo cuando ha caído más de 1mm
    condition_wet_days = ds[variable] > threshold
    condition_non_rainy_days = (ds[variable] < threshold)
    
    # Calculo el número de wet days 
    n_wet_days = condition_wet_days.sum(dim='time').rename(variable + '_n_wet_days')
    n_non_rainy_days = condition_non_rainy_days.sum(dim='time').rename(variable + '_n_non_rainy_days')

    # Calculo la intensidad de precipitación
    I = (ds[variable].where(condition_wet_days).sum(dim='time') / condition_wet_days.sum(dim='time')).rename(variable + '_I')

    pr_tot = (ds[variable].sum(dim = 'time', skipna = True)).rename(variable + '_tot')



    # Creo un nuevo dataset con estas tres estadísticas
    ds_pr_stats = xr.Dataset({
        variable +'_n_wet_days': n_wet_days,
        variable +'_I': I,
        variable + '_tot': pr_tot,
        variable + '_n_non_rainy_days': n_non_rainy_days
    })
    
    return ds_pr_stats

def WAM_moderate_heavy_extreme(ds, variable, threshold):

    """ Threshold es el número de mm/day necesarios para considerar un día húmedo """
    
    ds[variable + '_wet_days'] = xr.where((ds[variable] >= threshold), ds[variable],np.nan)
    
    condition_extreme = ds[variable + '_wet_days'] > ds[variable + '_perc95'].values
    condition_heavy = (ds[variable + '_wet_days'] > ds[variable + '_perc75'].values) & (ds[variable + '_wet_days'] < ds[variable + '_perc95'].values)
    condition_moderate = (ds[variable + '_wet_days'] < ds[variable + '_perc75'].values)

    n_extreme_rainfall_days = condition_extreme.sum(dim='time').rename(variable + '_n_extreme_days')
    n_heavy_rainfall_days = condition_heavy.sum(dim='time').rename(variable + '_n_heavy_days')
    n_moderate_rainfall_days = condition_moderate.sum(dim='time').rename(variable + '_n_moderate_days')

    ds_pr_stats = xr.Dataset({
        variable +'_n_extreme_rainfall_days': n_extreme_rainfall_days,
        variable +'_n_heavy_rainfall_days': n_heavy_rainfall_days,
        variable + '_n_moderate_rainfall_days': n_moderate_rainfall_days
    })
    return ds_pr_stats


def area_grid(lat, lon):
    """
    Calculate the area of each grid cell
    Area is in square meters

    Input
    -----------
    lat: vector of latitude in degrees
    lon: vector of longitude in degrees

    Output
    -----------
    area: grid-cell area in square-meters with dimensions, [lat,lon]

    Notes
    -----------
    Based on the function in
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """
    from numpy import meshgrid, deg2rad, gradient, cos
    from xarray import DataArray

    xlon, ylat = meshgrid(lon, lat)
    R = earth_radius(ylat)

    dlat = deg2rad(gradient(ylat, axis=0))
    dlon = deg2rad(gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * cos(deg2rad(ylat))

    area = dy * dx

    xda = DataArray(
        area,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={
            "long_name": "area_per_pixel",
            "description": "area per pixel",
            "units": "m^2",
        },
    )
    return xda, dx, dy



def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid
    defined by WGS84

    Input
    ---------
    lat: vector or latitudes in degrees

    Output
    ----------
    r: vector of radius in meters

    Notes
    -----------
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    '''
    from numpy import deg2rad, sin, cos

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)

    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = deg2rad(lat)
    lat_gc = np.arctan( (1-e2)*np.tan(lat) )

    # radius equation
    # see equation 3-107 in WGS84
    r = ((a * (1 - e2)**0.5) / (1 - (e2 * np.cos(lat_gc)**2))**0.5)

    return r

def convert_to_mm_per_day(ds,variable):
    """ Función que convierte la precipitación en kg·m^2/s en mm·day^{-1}
    variable: precipitation """

    ds[variable + '_mm_day'] = ds[variable] * 86400

    # Elimino la variable inicial 

    ds = ds.drop_vars(variable).rename({variable + '_mm_day':variable})

    return ds


def butterworth(time, x, nc, n, padding):
    
    import scipy
    """ Función que filtra con el filtro Butterworth una serie temporal
    time: tiempo
    x: serie de datos equiespaciados en tiempo
    nc: número de puntos de tiempo para calcular la frecuencia de corte
    n: orden del filtro, 4 originalmente
    padding: establece las condiciones de borde
        1. original: condiciones periódicas de contorno i.e. 2019, 2020, 1850 por la izquierda y mirror por la derecha i.e. 2019, 2020, 2020, 2019. Opción de Elsa
        2. mirror: condiciones de espejo por los dos lados i.e. 1851, 1850, 1850, 1851 por la izquierda y 2019, 2020, 2020, 2019 por la derecha
    Elegir una u otra no modifica mucho, solo los extremos de la serie.
    La función devuelve la serie filtrada x_filtered
    """
    fc = 1/nc # Frecuencia de corte
    N = len(x) # Longitud de la serie temporal
    b,a = scipy.signal.butter(n, fc*2, output='ba') # Calculo los parámetros del filtro

    if padding == 'original':

        x_extended = np.append(x,[x,np.flip(x)])   #np.flip cambia el orden de los elementos del array [1850, 1851, 1852] --> [1852, 1851, 1850]


    elif padding == 'mirror':

        x_extended = np.append(np.flip(x),[x,np.flip(x)])

    x_filtered_extended = scipy.signal.filtfilt(b,a,x_extended)

    x_filtered = x_filtered_extended[N:2*N] # Me quedo con la parte central de la serie filtrada

    return x_filtered


# Función para binnear variables en un xarray 


def metrics_fixed_binned_distribution_xarray(ds, control, variable, perc_step):

    from scipy import stats
    import gc

    """ ds : chunked xarray que contiene las dos variables
        control: string con el nombre de la variable a partir de la cual queremos hacer intervalos
        variable: string con la variable que queremos poner como intervalos
        perc_step: un entero el paso del intervalo"""

    # Calculo los intervalos

    # Selecciono del xarray las dos variables

    ds = ds[variable]

    nbins = int(100/perc_step)

    #bin_edges_fb = np.zeros(nbins + 1)
    #bw = (ds[control].max(skipna = True).compute() - ds[control].min(skipna = True).compute())/nbins
    #perc_step = 5
    #for pp in range(0,100,perc_step):
    #    qq = int(pp/perc_step) # Index starting from 0.
    #    lower = ds[control].min(skipna = True).compute()+qq*bw
    #    upper = lower + bw
    #    bin_edges_fb[qq] = lower

    #bin_edges_fb[-1] = upper
    #bin_centers_fb = 0.5*(bin_edges_fb[1:]+bin_edges_fb[:-1])

    # Para construir binns con el mismo número de observaciones tengo que sacarle los quantiles a la distribución

    bin_edges_fb = np.nanquantile(ds[control].values.flatten(), np.arange(0,1 + perc_step,perc_step))
    bin_centers_fb = 0.5*(bin_edges_fb[1:]+bin_edges_fb[:-1])

    ds_sub_binned = ds.groupby_bins(control, bin_edges_fb)

    # Calculo la media, la desviación típica, el standard error y el número de puntos de cada intervalo

    bin_mean = ds_sub_binned.mean(skipna = True).compute()
    bin_std = ds_sub_binned.std(skipna = True).compute()
    bin_npoints = ds_sub_binned.count().compute() # No cuento los NaNs como puntos válidos

    # Elimino la variable para liberar memoria
    del ds_sub_binned
    gc.collect()

    bin_std_err = bin_std / np.sqrt(bin_npoints)

    # Calculo un xarray con todos los bins juntos y lo guardo como netcdf

    #bin_mean.to_netcdf('./Binning_mean_'+control+'_'+variable+'.nc')
    #bin_std.to_netcdf('./Binning_std_'+control+'_'+variable+'.nc')
    #bin_npoints.to_netcdf('./Binning_npoints_'+control+'_'+variable+'.nc')
    #bin_std_err.to_netcdf('./Binning_std_err_'+control+'_'+variable+'.nc')

    # Calculo también el p_value a partir de los datos binneados

    sr_distr, trash = stats.spearmanr(bin_centers_fb[(~np.isnan(bin_mean.values))& (bin_npoints.values>1000)],bin_mean.values[(~np.isnan(bin_mean.values))& (bin_npoints.values>1000)])
    df = len(bin_centers_fb)-2 # degrees of freedom.
    t_value = np.abs(sr_distr)*np.sqrt((df)/(1-sr_distr**2))
    p_value = 1 - stats.t.cdf(t_value,df=df)

    # Y la pendiente y ordenada de la recta e regresión con aquellos que tengan más de 50 puntos

    lsq_res = stats.linregress(bin_centers_fb[(~np.isnan(bin_mean.values))& (bin_npoints.values>1000)], bin_mean.values[(~np.isnan(bin_mean.values))& (bin_npoints.values>1000)])
    slope = lsq_res[0]
    intercept = lsq_res[1]
    std_err_slope = lsq_res[4]

    return bin_centers_fb, bin_mean, bin_std, bin_npoints, bin_std_err, p_value, slope, intercept, std_err_slope



# Funciones para calcular el número de días heavy, extreme etc 

def WAM_moderate_heavy_extreme_difquant(ds, variable, yrstart, threshold):

    """ Función que calcula el número de días con eventos moderados, fuertes y extremos con percentiles calculados de distintas formas
    all_year_all_times: sobre todos los días lluviosos de toda la serie temporal
    all_year_yearly: sobre todos los días lluviosos de cada año i.e. hay un percentil por año
    wet_season_all_times: percentiles calculados sobre la wet season considerando todos los años del fichero
    wet_season_yearly: percentiles calculados sobre los días húmedos de la estación lluviosa cada año
    variable: precipitación 
    threshold es el umbral que utilizamos para definir un día húmedo """

    only_wet_days = xr.where((ds[variable] >= threshold), ds[variable], np.nan)

    # También saco el año de incio

    yrstart = ds.time.dt.year[0]

    if 'member' not in list(ds.dims):
        ds[variable +'_perc95_all_year_all_times'] = only_wet_days.quantile(0.95, dim = ('time'), skipna = True)
        ds[variable +'_perc75_all_year_all_times'] = only_wet_days.quantile(0.75, dim = ('time'), skipna = True)
        ds[variable +'_perc50_all_year_all_times'] = only_wet_days.quantile(0.50, dim = ('time'), skipna = True)
        ds[variable + '_n_extreme_rainfall_days_all_year_all_times'] = ((only_wet_days > ds[variable + '_perc95_all_year_all_times'].values)).groupby('time.year').sum(dim = 'time')
        ds[variable + '_n_heavy_rainfall_days_all_year_all_times'] = ((only_wet_days < ds[variable + '_perc95_all_year_all_times'].values) & (only_wet_days > ds[variable + '_perc75_all_year_all_times'].values)).groupby('time.year').sum(dim = 'time')
        ds[variable + '_n_moderate_rainfall_days_all_year_all_times'] = ((only_wet_days < ds[variable + '_perc75_all_year_all_times'].values)).groupby('time.year').sum(dim = 'time')


    else:

        ds[variable +'_perc95_all_year_all_times'] = only_wet_days.quantile(0.95, dim = ('time','member'), skipna = True)
    #ds[variable +'_perc95_all_year_yearly'] = only_wet_days.groupby('time.year').quantile(0.95, dim = ('time', 'member'), skipna = True)

        ds[variable +'_perc75_all_year_all_times'] = only_wet_days.quantile(0.75, dim = ('time', 'member'), skipna = True)
    #ds[variable +'_perc75_all_year_yearly'] = only_wet_days.groupby('time.year').quantile(0.75, dim = ('time', 'member'), skipna = True)

        ds[variable +'_perc50_all_year_all_times'] = only_wet_days.quantile(0.50, dim = ('time','member'), skipna = True)
    #ds[variable +'_perc50_all_year_yearly'] = only_wet_days.groupby('time.year').quantile(0.50, dim = ('time', 'member'), skipna = True)

        ds[variable + '_n_extreme_rainfall_days_all_year_all_times'] = ((only_wet_days > ds[variable + '_perc95_all_year_all_times'].values)).groupby('time.year').sum(dim = 'time')
    #ds[variable + '_n_extreme_rainfall_days_all_year_yearly'] = (only_wet_days.groupby('time.year').map(lambda group: apply_threshold(group, group.time.dt.year[0] - yrstart, ds[variable + '_perc95_all_year_yearly']))).groupby('time.year').sum(dim = 'time')

        ds[variable + '_n_heavy_rainfall_days_all_year_all_times'] = ((only_wet_days < ds[variable + '_perc95_all_year_all_times'].values) & (only_wet_days > ds[variable + '_perc75_all_year_all_times'].values)).groupby('time.year').sum(dim = 'time')
    #ds[variable + '_n_heavy_rainfall_days_all_year_yearly'] = (only_wet_days.groupby('time.year').map(lambda group: apply_interval(group, group.time.dt.year[0] - yrstart, ds[variable + '_perc95_all_year_yearly'], ds[variable +'_perc75_all_year_yearly']))).groupby('time.year').sum(dim = 'time')

        ds[variable + '_n_moderate_rainfall_days_all_year_all_times'] = ((only_wet_days < ds[variable + '_perc75_all_year_all_times'].values)).groupby('time.year').sum(dim = 'time')
    #ds[variable + '_n_moderate_rainfall_days_all_year_yearly'] = (only_wet_days.groupby('time.year').map(lambda group: apply_threshold(group, group.time.dt.year[0] - yrstart, ds[variable + '_perc50_all_year_yearly']))).groupby('time.year').sum(dim = 'time')

    return ds





def apply_threshold(group, year, thresholds):
    threshold = thresholds[year]
    return group > threshold


def apply_interval(group, year, thresholds_upper, thresholds_lower):
    threshold_upper = thresholds_upper[year]
    threshold_lower = thresholds_lower[year]
    return (group < threshold_upper) & (group > threshold_lower)



def WAM_onset_demise_marteau(ds):

    """ Función que calcula el inicio y fin del monzón con el método de Marteau (2010) flexibilizado como Badji (2022).
    El inicio del monzón es el primer día de un grupo de 4 días consecutivos que acumulan 20mm de lluvia sin que haya un período de sequía (cero lluvia acumulada)
    de más de 10 días en los 20 días siguientes a acabar el grupo de 4 días. El fin del monzón es el último día del grupo de 4 días consecutivos más tardío del año
    que no haya sido precedido por más de 10 días consecutivos de sequía en los 20 días precedentes al primer día del grupo de 4 días
    ds: xarray
    variable: normalmente pr """
    
    ds_year = ds.groupby("time.year")


# Creo un diccionario para guardar los subsets
    ds_yr = {year: group for year, group in ds_year}

    yr_start = list(ds_year.groups.keys())[0]
    yr_end = list(ds_year.groups.keys())[-1]

    # Hago un bucle a lo largo de los años
    for yr in tqdm(np.arange(yr_start, yr_end + 1, 1)):

        ds_yr[yr]


        # Hago las "sumas móviles" para seleccionar los períodos de 4 días con precipitaciones acumuladas superiores a 20 días y los períodos de 10 días consecutivos secos
        # Como no puedo igualar a cero directamente por cuestiones numéricas, pongo una tolerancia de 0.0001

        ds_yr[yr]['pr_cumrolling_4'] = xr.where((ds_yr[yr].pr.rolling(time=4, center=True).sum(skipna = False) > 20), ds_yr[yr].pr.rolling(time=4, center=True).sum(skipna = False), np.nan)
        ds_yr[yr]['pr_cumrolling_10'] = xr.where((ds_yr[yr].pr.rolling(time=10, center=True).sum(skipna = False) == 0),np.nan, ds_yr[yr].pr.rolling(time=10, center=True).sum(skipna = False))


            # Inicializo el índice en la dimensión temporal y las variables WAM onset y WAM demise 

        ds_yr[yr]['time_index'] = xr.Dataset(data_vars = dict(
                                    time_index = (["member","lat","lon","time"], np.tile(np.arange(0, len(ds_yr[yr].time.values),1), (len(ds_yr[yr].member), len(ds_yr[yr].lat), len(ds_yr[yr].lon))).reshape(len(ds_yr[yr].member), len(ds_yr[yr].lat), len(ds_yr[yr].lon), len(ds_yr[yr].time)))),
                                    coords=dict(
                                    time= ds_yr[yr].time.values,
                                    lon = ds_yr[yr].lon.values,
                                    lat = ds_yr[yr].lat.values,
                                    member = ds_yr[yr].member.values),
                                    attrs=dict(description="Time index")).time_index

               # También inicializo la matriz ds_prueba_wam_onset
        ds_yr[yr]['WAM_onset_daily'] = xr.Dataset(data_vars = dict(
                                    time_index = (["member","lat","lon"], np.tile(np.nan, (len(ds_yr[yr].member), len(ds_yr[yr].lat), len(ds_yr[yr].lon))).reshape(len(ds_yr[yr].member), len(ds_yr[yr].lat), len(ds_yr[yr].lon)))),
                                    coords=dict(
                                    time= ds_yr[yr].time.values,
                                    lon = ds_yr[yr].lon.values,
                                    lat = ds_yr[yr].lat.values,
                                    member = ds_yr[yr].member.values),
                                    attrs=dict(description="Time index")).time_index

        ds_yr[yr]['WAM_demise_daily'] = xr.Dataset(data_vars = dict(
                                    time_index = (["member","lat","lon"], np.tile(np.nan, (len(ds_yr[yr].member), len(ds_yr[yr].lat), len(ds_yr[yr].lon))).reshape(len(ds_yr[yr].member), len(ds_yr[yr].lat), len(ds_yr[yr].lon)))),
                                    coords=dict(
                                    time=ds_yr[yr].time.values,
                                    lon = ds_yr[yr].lon.values,
                                    lat = ds_yr[yr].lat.values,
                                    member = ds_yr[yr].member.values),
                                    attrs=dict(description="Time index")).time_index

        # Selecciono los índices en los que hay cuatro días consecutivos que acumulan más de 20 litros

        ds_yr[yr]['time_indices_rainy'] = xr.where((~np.isnan(ds_yr[yr]['pr_cumrolling_4'])), ds_yr[yr]['time_index'],np.nan)

        for i in range(len(ds_yr[yr].member)):
            for j in range(len(ds_yr[yr].lat)):
                for k in range(len(ds_yr[yr].lon)):
                    found_onset = 0
                    indices_rainy = ds_yr[yr].time_indices_rainy.isel(member = i, lat = j, lon = k)[~np.isnan(ds_yr[yr].time_indices_rainy.isel(member = i, lat = j, lon = k))]
                    m = 0 # Indice para recorrer los días que acumulo más de 20 mm
                    while (m<len(indices_rainy)):
                        if (~np.isnan(ds_yr[yr].pr_cumrolling_10.isel(member = i, lat = j, lon = k)[int(indices_rainy[m].values + 7) : int(indices_rainy[m].values + 17)].sum(skipna = False))) & (found_onset == 0) & (indices_rainy[m] - 2 > 120):
                            # Hemos encontrado el inicio del monzón
                            found_onset = 1
                            ds_yr[yr]['WAM_onset_daily'][i,j,k] = int(indices_rainy[m] - 2) # Lo damos en day of year


                        if (~np.isnan(ds_yr[yr].pr_cumrolling_10.isel(member = i, lat = j, lon = k)[int(indices_rainy[m].values - 17) : int(indices_rainy[m].values - 7)].sum(skipna = False))):
                            # Hemos encontrado el fin del monzón
                            ds_yr[yr]['WAM_demise_daily'][i,j,k] = int(indices_rainy[m] + 2) # Lo damos en day of year
                        
                        m = m + 1

        # Me deshago de las variables que tendrán tamaño diference en función del año y que luego no sirven para nada
        ds_yr[yr] = ds_yr[yr].drop_vars(["time_indices_rainy", "pr_cumrolling_4", "pr_cumrolling_10", "time_index"])


    # Reagrupamos el array concatenándolo

    ds_orig = ds_yr[yr_start]

    for j in tqdm(np.arange(yr_start + 1, yr_end + 1, 1)):
        ds_orig = xr.concat([ds_orig, ds_yr[j]], dim = 'time')

    # De el onset y el demise solo me interesa tener un valor por año (aquí los tengo cada año repetidos 365 veces)

    
    ds_orig['WAM_onset'] = ds_orig.WAM_onset_daily.groupby('time.year').mean(skipna = True)
    ds_orig['WAM_demise'] = ds_orig.WAM_demise_daily.groupby('time.year').mean(skipna = True)

    ds_orig = ds_orig.drop_vars(['WAM_onset_daily', 'WAM_demise_daily'])
    
    return ds_orig

