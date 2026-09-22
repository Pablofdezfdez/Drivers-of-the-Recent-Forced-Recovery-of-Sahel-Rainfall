######################## SCRIPT TO COMPUTE THE BAROTROPIC WIND CORRECTION ################################

import xarray as xr
import numpy as np
import cartopy
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append('/home/pfernand/Postdoc/Python_scripts/')
from WAM_functions import *
from Forced_simulation_functions import *
from scipy import stats
import numpy.ma as ma
import matplotlib.mlab as mlab
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
import scipy
import matplotlib.patches as patches
import cftime
from regr_sig import *
from regr_2d_ttest import *
import xskillscore
from matplotlib.lines import Line2D
from scipy.stats import ttest_rel
from matplotlib.ticker import MultipleLocator
import windspharm
from windspharm.xarray import VectorWind



########### Cargo las funciones que me dio Spencer Hill #####################
def uv_col_budg_adj(u_col_int, v_col_int, tendency, source,
                    lat_str="lat", lon_str="lon", time_str="time"):
    """Apply the column tracer budget adjustment method to enforce closure.
    
    For tracers other than dry mass, the `u_col_int` and `v_col_int` field
    must be the column integral of the *product* of the given tracer and the 
    given wind component.  So for MSE (denoted h), these would be 
    \int_0^{p_s}hu dp/g and \int_0^{p_s}hv dp/g.

    `tendency` is the d/dt term appearing in the column budget to be closed.
    For MSE, this would be \partial<\mathcal{E}>/\partialt, where 
    \mathcal{E}=c_v^+gz+L_vq.
    For mass, this is \partial p_s/\partial t.
    
    `source` is the RHS term appearing in the column budget to be closed.
    For MSE, this is "F_net", i.e. TOA radiative + SFC radiative + turbulent
    net fluxes directed into the atmosphere.
    For mass, this is g(E-P).
    
    """
    # Compute the divergence of the column-integrated horizontal flux.
    vecwind_col = windspharm.xarray.VectorWind(u_col_int, v_col_int)
    div_col = vecwind_col.divergence()
    # Compute the budget residual.
    resid = tendency + div_col - source
    # Get the spherical harmonic coefficients of the computed residual.
    # The adjustment computed from this residual is assumed to be purely divergent.
    # The transpose operation is to satisfy the requirements of the pyspharm.grdtospec function.
    div_adj_spec_coeffs = vecwind_col._api.s.grdtospec(resid.transpose(lat_str, lon_str, time_str))
    vort_adj_spec_coeffs = np.zeros_like(div_adj_spec_coeffs)
    # Transform these div and vort spectral coefficients into corresponding 
    # gridded column-integrated zonal flux and meridional flux adjustment fields.
    u_adj_vals, v_adj_vals = vecwind_col._api.s.getuv(vort_adj_spec_coeffs, div_adj_spec_coeffs)
    # Turn those numpy arrays into xarray.DataArrays with the original shape of u and v.
    u_col_adj = (xr.ones_like(u_col_int.transpose(lat_str, lon_str, time_str)) 
                 * u_adj_vals).transpose(*u_col_int.dims)
    v_col_adj = (xr.ones_like(v_col_int.transpose(lat_str, lon_str, time_str)) 
                 * v_adj_vals).transpose(*v_col_int.dims)

    return resid, u_col_adj, v_col_adj


def resid_after_col_adj(u_col_adjusted, v_col_adjusted, tendency, source):
    """Verify that the column budget residual is small after adjustment."""
    vecwind = windspharm.xarray.VectorWind(u_col_adjusted, v_col_adjusted)
    div_col_adjusted = vecwind.divergence()
    return tendency + div_col_adjusted - source



##################### Y MIS FUNCIONES #####################################

def spherical_gradients(da, R=6_371_000.0):
    """
    Cálculo de las derivadas espaciales en una esfera

    - Differncias finitas centradas de segundo orden
    - Differencia finita de primer orden cuando hay un NaN adyacente
    Paramteros
    ----------
    da : xr.DataArray
        dims ("lat", "lon"), coordinates in degrees
    R : float
        Sphere radius (meters)

    Returns
    -------
    dfdx, dfdy : xr.DataArray
        Derivadas in meters⁻¹
    """
    lat_rad = np.deg2rad(da.lat)
    lon_rad = np.deg2rad(da.lon)

    dphi = xr.DataArray(np.gradient(lat_rad), coords={"lat": da.lat}, dims="lat")
    dlambda = xr.DataArray(np.gradient(lon_rad), coords={"lon": da.lon}, dims="lon")
    cosphi = xr.DataArray(np.cos(lat_rad), coords={"lat": da.lat}, dims="lat")
    f = da
    f_ip1 = f.shift(lon=-1)
    f_im1 = f.shift(lon=1)

    centered = (f_ip1 - f_im1) / (2 * dlambda)
    forward  = (f_ip1 - f) / dlambda
    backward = (f - f_im1) / dlambda
    use_center  = f_ip1.notnull() & f_im1.notnull() # Sustituir para los no-nulos
    use_forward = (~use_center) & f_ip1.notnull()
    use_backward= (~use_center) & (~f_ip1.notnull()) & f_im1.notnull()
    dfdlambda = xr.where(
        use_center,
        centered,
        xr.where(use_forward, forward,
                 xr.where(use_backward, backward, np.nan))
    )
    dfdx = dfdlambda / (R * cosphi)
    dfdx = dfdx.where(np.abs(cosphi) > 1e-12)
    # Para las latitudes    
    f_jp1 = f.shift(lat=-1)
    f_jm1 = f.shift(lat=1)
    centered = (f_jp1 - f_jm1) / (2 * dphi)
    forward  = (f_jp1 - f) / dphi
    backward = (f - f_jm1) / dphi
    use_center  = f_jp1.notnull() & f_jm1.notnull()
    use_forward = (~use_center) & f_jp1.notnull()
    use_backward= (~use_center) & (~f_jp1.notnull()) & f_jm1.notnull()
    dfdphi = xr.where(
        use_center,
        centered,
        xr.where(use_forward, forward,
                 xr.where(use_backward, backward, np.nan))
    )
    dfdy = dfdphi / R
    return dfdx.rename("dfdx"), dfdy.rename("dfdy")



def vertical_integral(ds, masked, slp, g=9.80665, rho_w = 1000, dim="presnivs"):
    """
    Función que calcula la integral vertical de presión en todos los niveles

    Parameters
    ----------
    f : xr.DataArray
        Variable a integrar
    presnivs : xr.DataArray
        Presión entre dos niveles
    g : float
        Gravedad (m/s²)
    rho_w : float
        Densidad del agua (kg/m3)
    dim : str
        Nombre de la dimensión vertical

    Da de vuelta
    -------
    xr.DataArray
        Mass-weighted integral vertical
    """
    # Calcular espesor en presión
    #dp = presnivs.diff(dim)  

    # Calculo los niveles de presión 

    dp = np.zeros(79)
    dp[0:78] = ds.presnivs.diff(dim = 'presnivs').values/2 + ds.presnivs[0:78].values - (ds.presnivs[0:78].values - (ds.presnivs.diff(dim = 'presnivs')).values/2)
    dp[78] = -(ds.presnivs[0:78].values + (ds.presnivs.diff(dim = 'presnivs')).values/2)[-1]
    dp_xr = xr.DataArray(dp, dims = ['presnivs'], coords={'presnivs': ds.presnivs.values})

    # Le voy a poner nans a aquellos puntos que estén por debajo de slp (topografía)
    if masked == 0:
        ds_masked = ds
    else:
        ds_masked = ds.where(ds.presnivs <= slp)
    
    #if 'time' in ds.dims:
    #    presnivs_broadcasted = ds.presnivs.broadcast_like(xr.DataArray(np.zeros((len(ds.time), len(ds.lon), len(ds.lat), len(ds.presnivs))), dims=["time", "lon", "lat", "presnivs"]))
    #    ds_masked = xr.where(presnivs_broadcasted>slp, np.nan, ds)
    #else:
    #    presnivs_broadcasted = ds.presnivs.broadcast_like(xr.DataArray(np.zeros((len(ds.lon), len(ds.lat), len(ds.presnivs))), dims=["lon", "lat", "presnivs"]))
    #    ds_masked = xr.where(presnivs_broadcasted>slp, np.nan, ds)

        

    # Multiplicar por variable y sumar para integrar
    #integral = (ds_masked * dp_xr).sum(dim=dim, skipna=True)
    integral = xr.dot(ds_masked.fillna(0), dp_xr, dims = 'presnivs')
    # Ponderar por la masa dividiendo por gravedad
    integral_weighted = integral / (g * rho_w)
    return integral_weighted


def mask_ground_data(ds,slp):

    """Función para enmascarar los datos que estén por debajo de la slp"""

    return ds.where(ds.presnivs <= slp)
    
def time_tendency_centered(da, time_dim="time"):
    """
    Calculo de la tendencia usando diferencias finitas.
    
    Parameters
    ----------
    da : xr.DataArray
        Input data array with time dimension.
    time_dim : str
        Name of time coordinate.

    Returns
    -------
    xr.DataArray
        Time derivative d(da)/dt with same shape as input.
    """

    # Asegurarse de que datetime es numérico
    t = da[time_dim]

    # Convertir el tiempo en segundos
    if np.issubdtype(t.dtype, np.datetime64):
        t_sec = (t - t[0]) / np.timedelta64(1, "s")
    else:
        t_sec = t

    t_sec = xr.DataArray(t_sec, coords={time_dim: t}, dims=time_dim)

    # Calcular forward y backward para el primer y último putnos
    dt_forward = t_sec.diff(time_dim)
    dt_backward = dt_forward.shift({time_dim: 1})

    # Calcular diferencias finitas centradas en el interior
    dadt = (
        da.shift({time_dim: -1}) - da.shift({time_dim: 1})
    ) / (2*86400) 

    # Forward difference (primer punto)
    first = (da.isel({time_dim: 1}) - da.isel({time_dim: 0})) / 86400

    # Backward difference (último punto)
    last = (da.isel({time_dim: -1}) - da.isel({time_dim: -2})) / 86400

    # Insert boundary values
    dadt[{time_dim: 0}] = first
    dadt[{time_dim: -1}] = last

    dadt.name = f"d{da.name}_dt" if da.name else "time_tendency"

    return dadt

def divergence_spectral(u, v):
    
    from windspharm.xarray import VectorWind
    w = VectorWind(u, v)
    return w.divergence()

def gradient_spectral(chi):
    from windspharm.xarray import VectorWind
    
    u = xr.zeros_like(chi)
    v = xr.zeros_like(chi)

    w = VectorWind(u, v)
    
    dphi_dx, dphi_dy = w.gradient(chi)

    return dphi_dx, dphi_dy




############# CARGO LOS FICHEROS #################### 

ds_ctrlghg1980 = xr.open_dataset('/thredds/tgcc/work/fernandb/Hard_links/CTRLAER1980mb/DA/CTRLAER1980mb_19800701_20790930_1D_histday_world.nc').rename({'time_counter':'time'})

ds_ctrlghg2004 = xr.open_dataset('/thredds/tgcc/work/fernandb/Hard_links/CTRLAER2004mb/DA/CTRLAER2004mb_19800701_20790930_1D_histday_world.nc').rename({'time_counter':'time'})

ds_ghgda2004 = xr.open_dataset('/thredds/tgcc/work/fernandb/Hard_links/AERDA2004mb/DA/AERDA2004mb_19800701_20500930_1D_histday_world.nc').rename({'time_counter':'time'})

ds_ghgom2004 = xr.open_dataset('/thredds/tgcc/work/fernandb/Hard_links/AEROM2004mb/DA/AEROM2004mb_19800701_20790930_1D_histday_world.nc').rename({'time_counter':'time'})


##### Les meto la presión de superficie como variable ########

ds_ctrlghg1980['psol'] = xr.open_dataset('/thredds/tgcc/store/fernandb/CTRLAER1980mb/DA/CTRLAER1980mb_19800701_20790930_1D_histday_psol.nc').rename({'time_counter':'time'}).psol

ds_ctrlghg2004['psol'] = xr.open_dataset('/thredds/tgcc/store/fernandb/CTRLAER2004mb/DA/CTRLAER2004mb_19800701_20790930_1D_histday_psol.nc').rename({'time_counter':'time'}).psol

ds_ghgda2004['psol'] = xr.open_dataset('/thredds/tgcc/store/fernandb/AERDA2004mb/DA/AERDA2004mb_19800701_20500930_1D_histday_psol.nc').rename({'time_counter':'time'}).psol

ds_ghgom2004['psol'] = xr.open_dataset('/thredds/tgcc/store/fernandb/AEROM2004mb/DA/AEROM2004mb_19800701_20790930_1D_histday_psol.nc').rename({'time_counter':'time'}).psol


corr_ctrlghg1980 = xr.Dataset({"u_adj": xr.zeros_like(ds_ctrlghg1980.slp)})
corr_ctrlghg1980['v_adj'] =  xr.zeros_like(ds_ctrlghg1980.slp)
corr_ctrlghg1980['R'] =  xr.zeros_like(ds_ctrlghg1980.slp)
corr_ctrlghg1980['R_corr'] =  xr.zeros_like(ds_ctrlghg1980.slp)



corr_ctrlghg2004 = xr.Dataset({"u_adj": xr.zeros_like(ds_ctrlghg2004.slp)})
corr_ctrlghg2004['v_adj'] =  xr.zeros_like(ds_ctrlghg2004.slp)
corr_ctrlghg2004['R'] =  xr.zeros_like(ds_ctrlghg2004.slp)
corr_ctrlghg2004['R_corr'] =  xr.zeros_like(ds_ctrlghg2004.slp)



corr_ghgda2004 = xr.Dataset({"u_adj": xr.zeros_like(ds_ghgda2004.slp)})
corr_ghgda2004['v_adj'] =  xr.zeros_like(ds_ghgda2004.slp)
corr_ghgda2004['R'] =  xr.zeros_like(ds_ghgda2004.slp)
corr_ghgda2004['R_corr'] =  xr.zeros_like(ds_ghgda2004.slp)



corr_ghgom2004 = xr.Dataset({"u_adj": xr.zeros_like(ds_ghgom2004.slp)})
corr_ghgom2004['v_adj'] =  xr.zeros_like(ds_ghgom2004.slp)
corr_ghgom2004['R'] =  xr.zeros_like(ds_ghgom2004.slp)
corr_ghgom2004['R_corr'] =  xr.zeros_like(ds_ghgom2004.slp)



######### Y calculo la corrección barotrópica ##########

############## PARA CTRLGHG1980 #################

for i in tqdm(range(100)):
    
    ds = ds_ctrlghg1980.isel(time = np.arange(90*i,90*i+90,1))
    q_int = -vertical_integral(ds.ovap, 1, ds.psol, g = 9.80665, rho_w = 1, dim="presnivs")
    dqint_dt =  time_tendency_centered(q_int, time_dim="time")

    uq = ds.vitu * ds.ovap
    vq = ds.vitv * ds.ovap
    uq_int = -vertical_integral(uq, 1, ds.psol, g = 9.80665, rho_w = 1, dim="presnivs")
    vq_int = -vertical_integral(vq, 1, ds.psol, g = 9.80665, rho_w = 1, dim="presnivs")

    # Calculo el término source    
    source = ds.evap - ds.precip

    R_orig, uq_corr, vq_corr = uv_col_budg_adj(uq_int, vq_int, dqint_dt, source,
                    lat_str="lat", lon_str="lon", time_str="time")

    u_adj = uq_corr / q_int
    v_adj = vq_corr / q_int

    R =  resid_after_col_adj(uq_int - uq_corr, vq_int - vq_corr, dqint_dt, source)

    # Los pongo en xarrays

    corr_ctrlghg1980['u_adj'][90*i:90*i+90,:,:] = u_adj
    corr_ctrlghg1980['v_adj'][90*i:90*i+90,:,:] = v_adj
    corr_ctrlghg1980['R'][90*i:90*i+90,:,:] = R_orig
    corr_ctrlghg1980['R_corr'][90*i:90*i+90,:,:] = R

# Y guardo el fichero 
corr_ctrlghg1980.to_netcdf('/data/pfernand/Postdoc/Forced_simulations/CTRLAER1980mb_19800701_20790930_1D_histday_world_barotropic_wind_correction_psol.nc')


############### PARA CTRLGHG2004 ##############

for i in tqdm(range(100)):
    
    ds = ds_ctrlghg2004.isel(time = np.arange(90*i,90*i+90,1))
    q_int = -vertical_integral(ds.ovap, 1, ds.psol, g = 9.80665, rho_w = 1, dim="presnivs")
    dqint_dt =  time_tendency_centered(q_int, time_dim="time")

    uq = ds.vitu * ds.ovap
    vq = ds.vitv * ds.ovap
    uq_int = -vertical_integral(uq, 1, ds.psol, g = 9.80665, rho_w = 1, dim="presnivs")
    vq_int = -vertical_integral(vq, 1, ds.psol, g = 9.80665, rho_w = 1, dim="presnivs")

    # Calculo el término source
    source = ds.evap - ds.precip

    R_orig, uq_corr, vq_corr = uv_col_budg_adj(uq_int, vq_int, dqint_dt, source,
                    lat_str="lat", lon_str="lon", time_str="time")

    u_adj = uq_corr / q_int
    v_adj = vq_corr / q_int

    R =  resid_after_col_adj(uq_int - uq_corr, vq_int - vq_corr, dqint_dt, source)

    # Los pongo en xarrays

    corr_ctrlghg2004['u_adj'][90*i:90*i+90,:,:] = u_adj
    corr_ctrlghg2004['v_adj'][90*i:90*i+90,:,:] = v_adj
    corr_ctrlghg2004['R'][90*i:90*i+90,:,:] = R_orig
    corr_ctrlghg2004['R_corr'][90*i:90*i+90,:,:] = R

# Y guardo el fichero
corr_ctrlghg2004.to_netcdf('/data/pfernand/Postdoc/Forced_simulations/CTRLAER2004mb_19800701_20790930_1D_histday_world_barotropic_wind_correction_psol.nc')

########### PARA GHGDA2004 ##########

for i in tqdm(range(71)):
    
    ds = ds_ghgda2004.isel(time = np.arange(90*i,90*i+90,1))
    q_int = -vertical_integral(ds.ovap, 1, ds.psol, g = 9.80665, rho_w = 1, dim="presnivs")
    dqint_dt =  time_tendency_centered(q_int, time_dim="time")

    uq = ds.vitu * ds.ovap
    vq = ds.vitv * ds.ovap
    uq_int = -vertical_integral(uq, 1, ds.psol, g = 9.80665, rho_w = 1, dim="presnivs")
    vq_int = -vertical_integral(vq, 1, ds.psol, g = 9.80665, rho_w = 1, dim="presnivs")

    # Calculo el término source
    source = ds.evap - ds.precip

    R_orig, uq_corr, vq_corr = uv_col_budg_adj(uq_int, vq_int, dqint_dt, source,
                    lat_str="lat", lon_str="lon", time_str="time")

    u_adj = uq_corr / q_int
    v_adj = vq_corr / q_int

    R =  resid_after_col_adj(uq_int - uq_corr, vq_int - vq_corr, dqint_dt, source)

    # Los pongo en xarrays

    corr_ghgda2004['u_adj'][90*i:90*i+90,:,:] = u_adj
    corr_ghgda2004['v_adj'][90*i:90*i+90,:,:] = v_adj
    corr_ghgda2004['R'][90*i:90*i+90,:,:] = R_orig
    corr_ghgda2004['R_corr'][90*i:90*i+90,:,:] = R

# Y guardo el fichero 
corr_ghgda2004.to_netcdf('/data/pfernand/Postdoc/Forced_simulations/AERDA2004mb_19800701_20500930_1D_histday_world_barotropic_wind_correction_psol.nc')

############### PARA GHGOM2004 ##########

for i in tqdm(range(100)):
    
    ds = ds_ghgom2004.isel(time = np.arange(90*i,90*i+90,1))
    q_int = -vertical_integral(ds.ovap, 1, ds.psol, g = 9.80665, rho_w = 1, dim="presnivs")
    dqint_dt =  time_tendency_centered(q_int, time_dim="time")

    uq = ds.vitu * ds.ovap
    vq = ds.vitv * ds.ovap
    uq_int = -vertical_integral(uq, 1, ds.psol, g = 9.80665, rho_w = 1, dim="presnivs")
    vq_int = -vertical_integral(vq, 1, ds.psol, g = 9.80665, rho_w = 1, dim="presnivs")

    # Calculo el término source
    source = ds.evap - ds.precip

    R_orig, uq_corr, vq_corr = uv_col_budg_adj(uq_int, vq_int, dqint_dt, source,
                    lat_str="lat", lon_str="lon", time_str="time")

    u_adj = uq_corr / q_int
    v_adj = vq_corr / q_int

    R =  resid_after_col_adj(uq_int - uq_corr, vq_int - vq_corr, dqint_dt, source)


    # Los pongo en xarrays 
    corr_ghgom2004['u_adj'][90*i:90*i+90,:,:] = u_adj
    corr_ghgom2004['v_adj'][90*i:90*i+90,:,:] = v_adj
    corr_ghgom2004['R'][90*i:90*i+90,:,:] = R_orig
    corr_ghgom2004['R_corr'][90*i:90*i+90,:,:] = R

# Y guardo el fichero 
corr_ghgom2004.to_netcdf('/data/pfernand/Postdoc/Forced_simulations/AEROM2004mb_19800701_20790930_1D_histday_world_barotropic_wind_correction_psol.nc')



############### Y LO MISMO PARA LOS GHG ##################



############# CARGO LOS FICHEROS #################### 

ds_ctrlghg1980 = xr.open_dataset('/thredds/tgcc/work/fernandb/Hard_links/CTRLGHG1980mb/DA/CTRLGHG1980mb_19800701_20790930_1D_histday_world.nc').rename({'time_counter':'time'})

ds_ctrlghg2004 = xr.open_dataset('/thredds/tgcc/work/fernandb/Hard_links/CTRLGHG2004mb/DA/CTRLGHG2004mb_19800701_20790930_1D_histday_world.nc').rename({'time_counter':'time'})

ds_ghgda2004 = xr.open_dataset('/thredds/tgcc/work/fernandb/Hard_links/GHGDA2004mb/DA/GHGDA2004mb_19800701_20790930_1D_histday_world.nc').rename({'time_counter':'time'})

ds_ghgom2004 = xr.open_dataset('/thredds/tgcc/work/fernandb/Hard_links/GHGOM2004mb/DA/GHGOM2004mb_19800701_20790930_1D_histday_world.nc').rename({'time_counter':'time'})


##### Les meto la presión de superficie como variable ########

ds_ctrlghg1980['psol'] = xr.open_dataset('/thredds/tgcc/store/fernandb/CTRLGHG1980mb/DA/CTRLGHG1980mb_19800701_20790930_1D_histday_psol.nc').rename({'time_counter':'time'}).psol

ds_ctrlghg2004['psol'] = xr.open_dataset('/thredds/tgcc/store/fernandb/CTRLGHG2004mb/DA/CTRLGHG2004mb_19800701_20790930_1D_histday_psol.nc').rename({'time_counter':'time'}).psol

ds_ghgda2004['psol'] = xr.open_dataset('/thredds/tgcc/store/fernandb/GHGDA2004mb/DA/GHGDA2004mb_19800701_20790930_1D_histday_psol.nc').rename({'time_counter':'time'}).psol

ds_ghgom2004['psol'] = xr.open_dataset('/thredds/tgcc/store/fernandb/GHGOM2004mb/DA/GHGOM2004mb_19800701_20790930_1D_histday_psol.nc').rename({'time_counter':'time'}).psol


corr_ctrlghg1980 = xr.Dataset({"u_adj": xr.zeros_like(ds_ctrlghg1980.slp)})
corr_ctrlghg1980['v_adj'] =  xr.zeros_like(ds_ctrlghg1980.slp)
corr_ctrlghg1980['R'] =  xr.zeros_like(ds_ctrlghg1980.slp)
corr_ctrlghg1980['R_corr'] =  xr.zeros_like(ds_ctrlghg1980.slp)



corr_ctrlghg2004 = xr.Dataset({"u_adj": xr.zeros_like(ds_ctrlghg2004.slp)})
corr_ctrlghg2004['v_adj'] =  xr.zeros_like(ds_ctrlghg2004.slp)
corr_ctrlghg2004['R'] =  xr.zeros_like(ds_ctrlghg2004.slp)
corr_ctrlghg2004['R_corr'] =  xr.zeros_like(ds_ctrlghg2004.slp)



corr_ghgda2004 = xr.Dataset({"u_adj": xr.zeros_like(ds_ghgda2004.slp)})
corr_ghgda2004['v_adj'] =  xr.zeros_like(ds_ghgda2004.slp)
corr_ghgda2004['R'] =  xr.zeros_like(ds_ghgda2004.slp)
corr_ghgda2004['R_corr'] =  xr.zeros_like(ds_ghgda2004.slp)



corr_ghgom2004 = xr.Dataset({"u_adj": xr.zeros_like(ds_ghgom2004.slp)})
corr_ghgom2004['v_adj'] =  xr.zeros_like(ds_ghgom2004.slp)
corr_ghgom2004['R'] =  xr.zeros_like(ds_ghgom2004.slp)
corr_ghgom2004['R_corr'] =  xr.zeros_like(ds_ghgom2004.slp)



######### Y calculo la corrección barotrópica ##########

############## PARA CTRLGHG1980 #################

for i in tqdm(range(100)):
    
    ds = ds_ctrlghg1980.isel(time = np.arange(90*i,90*i+90,1))
    q_int = -vertical_integral(ds.ovap, 1, ds.psol, g = 9.80665, rho_w = 1, dim="presnivs")
    dqint_dt =  time_tendency_centered(q_int, time_dim="time")

    uq = ds.vitu * ds.ovap
    vq = ds.vitv * ds.ovap
    uq_int = -vertical_integral(uq, 1, ds.psol, g = 9.80665, rho_w = 1, dim="presnivs")
    vq_int = -vertical_integral(vq, 1, ds.psol, g = 9.80665, rho_w = 1, dim="presnivs")

    # Calculo el término source    
    source = ds.evap - ds.precip

    R_orig, uq_corr, vq_corr = uv_col_budg_adj(uq_int, vq_int, dqint_dt, source,
                    lat_str="lat", lon_str="lon", time_str="time")

    u_adj = uq_corr / q_int
    v_adj = vq_corr / q_int

    R =  resid_after_col_adj(uq_int - uq_corr, vq_int - vq_corr, dqint_dt, source)

    # Los pongo en xarrays

    corr_ctrlghg1980['u_adj'][90*i:90*i+90,:,:] = u_adj
    corr_ctrlghg1980['v_adj'][90*i:90*i+90,:,:] = v_adj
    corr_ctrlghg1980['R'][90*i:90*i+90,:,:] = R_orig
    corr_ctrlghg1980['R_corr'][90*i:90*i+90,:,:] = R

# Y guardo el fichero 
corr_ctrlghg1980.to_netcdf('/data/pfernand/Postdoc/Forced_simulations/CTRLGHG1980mb_19800701_20790930_1D_histday_world_barotropic_wind_correction_psol.nc')


############### PARA CTRLGHG2004 ##############

for i in tqdm(range(100)):
    
    ds = ds_ctrlghg2004.isel(time = np.arange(90*i,90*i+90,1))
    q_int = -vertical_integral(ds.ovap, 1, ds.psol, g = 9.80665, rho_w = 1, dim="presnivs")
    dqint_dt =  time_tendency_centered(q_int, time_dim="time")

    uq = ds.vitu * ds.ovap
    vq = ds.vitv * ds.ovap
    uq_int = -vertical_integral(uq, 1, ds.psol, g = 9.80665, rho_w = 1, dim="presnivs")
    vq_int = -vertical_integral(vq, 1, ds.psol, g = 9.80665, rho_w = 1, dim="presnivs")

    # Calculo el término source
    source = ds.evap - ds.precip

    R_orig, uq_corr, vq_corr = uv_col_budg_adj(uq_int, vq_int, dqint_dt, source,
                    lat_str="lat", lon_str="lon", time_str="time")

    u_adj = uq_corr / q_int
    v_adj = vq_corr / q_int

    R =  resid_after_col_adj(uq_int - uq_corr, vq_int - vq_corr, dqint_dt, source)

    # Los pongo en xarrays

    corr_ctrlghg2004['u_adj'][90*i:90*i+90,:,:] = u_adj
    corr_ctrlghg2004['v_adj'][90*i:90*i+90,:,:] = v_adj
    corr_ctrlghg2004['R'][90*i:90*i+90,:,:] = R_orig
    corr_ctrlghg2004['R_corr'][90*i:90*i+90,:,:] = R

# Y guardo el fichero
corr_ctrlghg2004.to_netcdf('/data/pfernand/Postdoc/Forced_simulations/CTRLGHG2004mb_19800701_20790930_1D_histday_world_barotropic_wind_correction_psol.nc')

########### PARA GHGDA2004 ##########

for i in tqdm(range(100)):
    
    ds = ds_ghgda2004.isel(time = np.arange(90*i,90*i+90,1))
    q_int = -vertical_integral(ds.ovap, 1, ds.psol, g = 9.80665, rho_w = 1, dim="presnivs")
    dqint_dt =  time_tendency_centered(q_int, time_dim="time")

    uq = ds.vitu * ds.ovap
    vq = ds.vitv * ds.ovap
    uq_int = -vertical_integral(uq, 1, ds.psol, g = 9.80665, rho_w = 1, dim="presnivs")
    vq_int = -vertical_integral(vq, 1, ds.psol, g = 9.80665, rho_w = 1, dim="presnivs")

    # Calculo el término source
    source = ds.evap - ds.precip

    R_orig, uq_corr, vq_corr = uv_col_budg_adj(uq_int, vq_int, dqint_dt, source,
                    lat_str="lat", lon_str="lon", time_str="time")

    u_adj = uq_corr / q_int
    v_adj = vq_corr / q_int

    R =  resid_after_col_adj(uq_int - uq_corr, vq_int - vq_corr, dqint_dt, source)

    # Los pongo en xarrays

    corr_ghgda2004['u_adj'][90*i:90*i+90,:,:] = u_adj
    corr_ghgda2004['v_adj'][90*i:90*i+90,:,:] = v_adj
    corr_ghgda2004['R'][90*i:90*i+90,:,:] = R_orig
    corr_ghgda2004['R_corr'][90*i:90*i+90,:,:] = R

# Y guardo el fichero 
corr_ghgda2004.to_netcdf('/data/pfernand/Postdoc/Forced_simulations/GHGDA2004mb_19800701_20500930_1D_histday_world_barotropic_wind_correction_psol.nc')

############### PARA GHGOM2004 ##########

for i in tqdm(range(100)):
    
    ds = ds_ghgom2004.isel(time = np.arange(90*i,90*i+90,1))
    q_int = -vertical_integral(ds.ovap, 1, ds.psol, g = 9.80665, rho_w = 1, dim="presnivs")
    dqint_dt =  time_tendency_centered(q_int, time_dim="time")

    uq = ds.vitu * ds.ovap
    vq = ds.vitv * ds.ovap
    uq_int = -vertical_integral(uq, 1, ds.psol, g = 9.80665, rho_w = 1, dim="presnivs")
    vq_int = -vertical_integral(vq, 1, ds.psol, g = 9.80665, rho_w = 1, dim="presnivs")

    # Calculo el término source
    source = ds.evap - ds.precip

    R_orig, uq_corr, vq_corr = uv_col_budg_adj(uq_int, vq_int, dqint_dt, source,
                    lat_str="lat", lon_str="lon", time_str="time")

    u_adj = uq_corr / q_int
    v_adj = vq_corr / q_int

    R =  resid_after_col_adj(uq_int - uq_corr, vq_int - vq_corr, dqint_dt, source)


    # Los pongo en xarrays 
    corr_ghgom2004['u_adj'][90*i:90*i+90,:,:] = u_adj
    corr_ghgom2004['v_adj'][90*i:90*i+90,:,:] = v_adj
    corr_ghgom2004['R'][90*i:90*i+90,:,:] = R_orig
    corr_ghgom2004['R_corr'][90*i:90*i+90,:,:] = R

# Y guardo el fichero 
corr_ghgom2004.to_netcdf('/data/pfernand/Postdoc/Forced_simulations/GHGOM2004mb_19800701_20790930_1D_histday_world_barotropic_wind_correction_psol.nc')


