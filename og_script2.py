import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import sklearn.metrics.pairwise as skmp
from scipy.stats import ttest_1samp, ks_2samp
import scipy.stats
import sys, os
from datetime import timedelta
from pathlib import Path
import matplotlib.animation as animation
import matplotlib.colors as colors
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.path as mpath
from scipy.stats import chi2

path_data = "D:/data/CMIP6_instanton_analysis/"
path_results = "C:/Users/rnoyelle/Dropbox/PC (2)/Desktop/CMIP6_instanton_analysis/"
name_model = "_day_IPSL-CM6A-LR_piControl_r1i1p1f1_gr_"
start_date = "18500101"
end_date = "38491231"


############### FUNCTIONS ############

# Selection functions

def extract_time_series_observable(min_lat, min_lon, max_lat, max_lon): 
    
    result = xr.open_dataset(path_data+"tas"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
    result = result.sel(time=(result['time.month']>=5) & (result['time.month']<=9),lat=slice(min_lat,max_lat),lon=slice(min_lon,max_lon))['tas'].mean(dim=['lat','lon'])
    
    return result

def select_field(v, j, c_n):
    if v=="slp":
        data = xr.open_dataset(path_data+"psl"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'psl'
    elif v=="t2m":
        data = xr.open_dataset(path_data+"tas"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'tas'
    elif v=="t850":
        data = xr.open_dataset(path_data+"ta850"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'ta'
    elif v=="v250":
        data = xr.open_dataset(path_data+"va250"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'va'
    elif v=="u250":
        data = xr.open_dataset(path_data+"ua250"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'ua'
    elif v=="v500":
        data = xr.open_dataset(path_data+"va500"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'va'
    elif v=="u500":
        data = xr.open_dataset(path_data+"ua500"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'ua'
    elif v=="z500":
        data = xr.open_dataset(path_data+"z500"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'zg'
    elif v=="mrsos":
        data = xr.open_dataset(path_data+"mrsos"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'mrsos'
                
    dates_to_select = c_n.time.values + timedelta(days=j)
    result = data.sel(time=dates_to_select[0])[v_bis]
    for d in dates_to_select[1:]:
        if ((v=="t850") & (d.year<1870)):
            continue
        result = xr.concat([result, data.sel(time=str(d))[v_bis]], dim='time')
                
    if v=="slp":
        result = result.transpose("time","lat","lon")/100
    elif v=="t2m":
        result = result.transpose("time","lat","lon") - 273.15
    elif v=="t850":
        result = result.squeeze("plev").transpose("time","lat","lon") - 273.15
    elif v in ["v250","u250","v500","u500"]:
        result = result.squeeze("plev").transpose("time","lat","lon") 
    elif v=="z500":
        result = result.squeeze("plev").transpose("time","lat","lon")/9.81
    elif v=="mrsos":
        result = result.transpose("time","lat","lon")
    
    return result

def select_and_arange(data, v, v_bis, d, j_begin, j_end):
    selection = data.sel(time=slice( str(d+timedelta(days=j_begin)), str(d+timedelta(days=j_end)) ))[v_bis]
    if v=="slp":
        selection = selection.transpose("time","lat","lon")/100
    elif v=="t2m":
        selection = selection.transpose("time","lat","lon") - 273.15
    elif v=="t850":
        selection = selection.squeeze("plev").transpose("time","lat","lon") - 273.15
    elif v in ["v250","u250","v500","u500"]:
        selection = selection.squeeze("plev").transpose("time","lat","lon") 
    elif v=="z500":
        selection = selection.squeeze("plev").transpose("time","lat","lon")
    elif v=="mrsos":
        selection = selection.transpose("time","lat","lon")
    return selection


def select_field_group(v, j_list, level_tab, c_n_list):
    if v=="slp":
        data = xr.open_dataset(path_data+"psl"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'psl'
    elif v=="t2m":
        data = xr.open_dataset(path_data+"tas"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'tas'
    elif v=="t850":
        data = xr.open_dataset(path_data+"ta850"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'ta'
    elif v=="v250":
        data = xr.open_dataset(path_data+"va250"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'va'
    elif v=="u250":
        data = xr.open_dataset(path_data+"ua250"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'ua'
    elif v=="v500":
        data = xr.open_dataset(path_data+"va500"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'va'
    elif v=="u500":
        data = xr.open_dataset(path_data+"ua500"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'ua'
    elif v=="z500":
        data = xr.open_dataset(path_data+"z500"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'zg'
    elif v=="mrsos":
        data = xr.open_dataset(path_data+"mrsos"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'mrsos'
        
    result_list = []
        
    for idlevel, level in enumerate(level_tab):
        dates_to_select = c_n_list[idlevel].time.sortby('time', ascending=False).values
        result = []
        # Initialization
        selection = select_and_arange(data, v, v_bis, dates_to_select[0], j_list[0], j_list[-1])
        for idj,j in enumerate(j_list):
            result.append(selection[idj,:,:])
        # Continue selecting dates
        for d in dates_to_select[1:]:
            selection = select_and_arange(data, v, v_bis, d, j_list[0], j_list[-1])
            if selection.shape[0] == 0:
                continue
            for idj,j in enumerate(j_list):
                result[idj] = xr.concat([result[idj], selection[idj,:,:]], dim='time')
                
        result_list.append(result)
    
    return result_list
    


def select_field_group_rolling(v, j_list, level_tab, c_n_list):
    if v=="slp":
        data = xr.open_dataset(path_data+"psl"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'psl'
    elif v=="t2m":
        data = xr.open_dataset(path_data+"tas"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'tas'
    elif v=="t850":
        data = xr.open_dataset(path_data+"ta850"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'ta'
    elif v=="v250":
        data = xr.open_dataset(path_data+"va250"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'va'
    elif v=="u250":
        data = xr.open_dataset(path_data+"ua250"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'ua'
    elif v=="v500":
        data = xr.open_dataset(path_data+"va500"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'va'
    elif v=="u500":
        data = xr.open_dataset(path_data+"ua500"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'ua'
    elif v=="z500":
        data = xr.open_dataset(path_data+"z500"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'zg'
    elif v=="mrsos":
        data = xr.open_dataset(path_data+"mrsos"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
        v_bis = 'mrsos'
        
    result_list = []
        
    for idlevel, level in enumerate(level_tab):
        dates_to_select = c_n_list[idlevel].time.sortby('time', ascending=False).values
        result = []
        # Initialization
        selection = select_and_arange(data, v, v_bis, dates_to_select[0], j_list[0], j_list[-1])
        result.append(selection)
        # Continue selecting dates
        for d in dates_to_select[1:]:
            selection = select_and_arange(data, v, v_bis, d, j_list[0], j_list[-1])
            if selection.shape[0] == 0:
                continue
            result.append(selection)
                
        result_list.append(result)
    
    return result_list


def select_climato(v, r=1):
    if v=="slp":
        climato_mean = xr.open_dataset(path_data+"psl"+name_model+start_date+"-"+end_date+"_mean.nc", use_cftime=True)['psl'][0,:,:]/100
        if r==1:
            climato_var = xr.open_dataset(path_data+"psl"+name_model+start_date+"-"+end_date+"_variance.nc", use_cftime=True)['psl'][0,:,:]/10000
        else:
            climato_var = xr.open_dataset(path_data+"psl"+name_model+start_date+"-"+end_date+"_variance_r"+str(r)+".nc", use_cftime=True)['psl'][0,:,:]/10000
    elif v=="t2m":
        climato_mean = xr.open_dataset(path_data+"tas"+name_model+start_date+"-"+end_date+"_mean.nc", use_cftime=True)['tas'][0,:,:]-273.15
        if r==1:
            climato_var = xr.open_dataset(path_data+"tas"+name_model+start_date+"-"+end_date+"_variance.nc", use_cftime=True)['tas'][0,:,:]
        else:
            climato_var = xr.open_dataset(path_data+"tas"+name_model+start_date+"-"+end_date+"_variance_r"+str(r)+".nc", use_cftime=True)['tas'][0,:,:]
    elif v=="t850":
        climato_mean = xr.open_dataset(path_data+"ta850"+name_model+start_date+"-"+end_date+"_mean.nc", use_cftime=True)['ta'][0,0,:,:]-273.15
        if r==1:
            climato_var = xr.open_dataset(path_data+"ta850"+name_model+start_date+"-"+end_date+"_variance.nc", use_cftime=True)['ta'][0,0,:,:]
        else:
            climato_var = xr.open_dataset(path_data+"ta850"+name_model+start_date+"-"+end_date+"_variance_r"+str(r)+".nc", use_cftime=True)['ta'][0,0,:,:]
    elif v=="v250":
        climato_mean = xr.open_dataset(path_data+"va250"+name_model+start_date+"-"+end_date+"_mean.nc", use_cftime=True)['va'][0,0,:,:]
        if r==1:
            climato_var = xr.open_dataset(path_data+"va250"+name_model+start_date+"-"+end_date+"_variance.nc", use_cftime=True)['va'][0,0,:,:]
        else:
            climato_var = xr.open_dataset(path_data+"va250"+name_model+start_date+"-"+end_date+"_variance_r"+str(r)+".nc", use_cftime=True)['va'][0,0,:,:]
    elif v=="u250":
        climato_mean = xr.open_dataset(path_data+"ua250"+name_model+start_date+"-"+end_date+"_mean.nc", use_cftime=True)['ua'][0,0,:,:]
        if r==1:
            climato_var = xr.open_dataset(path_data+"ua250"+name_model+start_date+"-"+end_date+"_variance.nc", use_cftime=True)['ua'][0,0,:,:]
        else:
            climato_var = xr.open_dataset(path_data+"ua250"+name_model+start_date+"-"+end_date+"_variance_r"+str(r)+".nc", use_cftime=True)['ua'][0,0,:,:]
    elif v=="v500":
        climato_mean = xr.open_dataset(path_data+"va500"+name_model+start_date+"-"+end_date+"_mean.nc", use_cftime=True)['va'][0,0,:,:]
        if r==1:
            climato_var = xr.open_dataset(path_data+"va500"+name_model+start_date+"-"+end_date+"_variance.nc", use_cftime=True)['va'][0,0,:,:]
        else:
            climato_var = xr.open_dataset(path_data+"va500"+name_model+start_date+"-"+end_date+"_variance_r"+str(r)+".nc", use_cftime=True)['va'][0,0,:,:]
    elif v=="u500":
        climato_mean = xr.open_dataset(path_data+"ua500"+name_model+start_date+"-"+end_date+"_mean.nc", use_cftime=True)['ua'][0,0,:,:]
        if r==1:
            climato_var = xr.open_dataset(path_data+"ua500"+name_model+start_date+"-"+end_date+"_variance.nc", use_cftime=True)['ua'][0,0,:,:]
        else:
            climato_var = xr.open_dataset(path_data+"ua500"+name_model+start_date+"-"+end_date+"_variance_r"+str(r)+".nc", use_cftime=True)['ua'][0,0,:,:]
    elif v=="z500":
        climato_mean = xr.open_dataset(path_data+"z500"+name_model+start_date+"-"+end_date+"_mean.nc", use_cftime=True)['zg'][0,0,:,:]
        if r==1:
            climato_var = xr.open_dataset(path_data+"z500"+name_model+start_date+"-"+end_date+"_variance.nc", use_cftime=True)['zg'][0,0,:,:]
        else:
            climato_var = xr.open_dataset(path_data+"z500"+name_model+start_date+"-"+end_date+"_variance_r"+str(r)+".nc", use_cftime=True)['zg'][0,0,:,:]
    elif v=="mrsos":
        climato_mean = xr.open_dataset(path_data+"mrsos"+name_model+start_date+"-"+end_date+"_mean.nc", use_cftime=True)['mrsos'][0,:,:]
        if r==1:
            climato_var = xr.open_dataset(path_data+"mrsos"+name_model+start_date+"-"+end_date+"_variance.nc", use_cftime=True)['mrsos'][0,:,:]
        else:
            climato_var = xr.open_dataset(path_data+"mrsos"+name_model+start_date+"-"+end_date+"_variance_r"+str(r)+".nc", use_cftime=True)['mrsos'][0,:,:]
    return climato_mean, climato_var
            

# Autocorrelation series

def compute_autocorrelation_series(series_obs_rolling, days=31):
    series_obs_rolling_deseasonalized = series_obs_rolling.groupby('time.dayofyear') - series_obs_rolling.groupby('time.dayofyear').mean()
    auto_corr_tab = np.zeros(days)
    for i in range(days):
        auto_corr_tab[i] = xr.corr(series_obs_rolling_deseasonalized,series_obs_rolling_deseasonalized.roll(time=i))
    return auto_corr_tab


# Closest neighbors functions

def find_closest_neighbors(level_obs, series_obs, nb_closest, calendar_spacing):
    
    series_obs_closest = xr.DataArray([],coords=dict(time=[]))
    
    for serie_year in series_obs.groupby('time.year'):
        temp = serie_year[1].sortby((serie_year[1]-level_obs)**2, ascending=False)
        calendar_temp = temp.time.dt.dayofyear
        i = temp.size - 1
        while i>0:
            tab = (np.abs(calendar_temp - calendar_temp[i]) >= calendar_spacing) | (calendar_temp == calendar_temp[i])
            i += - (calendar_temp.size - np.sum(tab) + 1)
            calendar_temp = calendar_temp[tab]
            temp = temp[tab]
        series_obs_closest = xr.concat([series_obs_closest,temp], dim='time')
        
    return series_obs_closest.sortby((series_obs_closest-level_obs)**2)[:nb_closest]

def compute_closest_days_observable(series_obs, rol_days, nb_closest, level_tab):
    
    # Rolling mean
    series_obs_rolling = series_obs.rolling(time=rol_days, center=True).mean()
    
    # Autocorrelation 
    auto_corr_tab = compute_autocorrelation_series(series_obs_rolling, days=31) # autocorrelation for months 5-6-7-8-9 
    
    # Closest neighbors
    series_obs_rolling = series_obs_rolling.sel(time=(series_obs['time.month']>=6) & (series_obs['time.month']<=8))
    level_obs = [series_obs_rolling.quantile(q=q).values for q in level_tab]
    result = []
    for l in level_obs:
        if rol_days <= 20 :
            result.append(find_closest_neighbors(l, series_obs_rolling, nb_closest, 15))
        else:
            result.append(find_closest_neighbors(l, series_obs_rolling, nb_closest, 30))

    #return auto_corr_tab, m, s, result
    return  auto_corr_tab, level_obs, result

def extract_coordinate_north_atlantic(f):
    result = f.copy()
    result.coords['lon'] = (result.coords['lon'] + 180) % 360 - 180
    result = result.sel(lon=result.lon[(result.lon >= ((280+180)%360-180)) & (result.lon <= ((50+180)%360-180))])
    result = result.sortby(result.lon)
    return result.sel(lat=slice(22.5,70),lon=slice(-80,50))

#%%
############### LOADING DATA ###############

# Observable selection
        # madrid/ 38,39 - 354.75,356.25
        # paris/ 49,50 - 1.25,3.75
        # uppsala/ 59,60 - 13.75,16.25
        # wce/ 46,53.5 - 0, 25
name_directory_results = ["madrid/","paris/","uppsala/","wce/"]
colors_c = ['gold','darkorange', 'red', 'black']    
min_lat_tab = [38,49,59,46]
max_lat_tab = [39,50,60,53.5]
min_lon_tab = [354.75,1.25,13.75,0]
max_lon_tab = [356.25,3.75,16.25,25]
    
print("Extracting observable time series")
series_obs = []
for i in range(len(name_directory_results)):
    series_obs.append(extract_time_series_observable(min_lat_tab[i], min_lon_tab[i], max_lat_tab[i], max_lon_tab[i]))
    
#% Parameters
nb_closest = 50 # number of closest neighbors of the observable
rolling_periods_tab = np.array([1,5,15]) # number of rolling days
level_tab = np.array([0.75,0.95,0.99,0.999]) # quantiles
j_list = range(-15,16) # days considered with respect to central date (closest observable)
    
# Load dates closest neighbors 
closest_neighbors_list = []
for i in range(len(name_directory_results)):
    temp1 = []
    for idr,r in enumerate(rolling_periods_tab):
        temp2 = []
        for l in level_tab:
            temp2.append(xr.load_dataarray(path_results+name_directory_results[i]+"data_closest/closest_neighbors_obs_r"+str(r)+"_q"+str(l)+".nc", use_cftime=True))
        temp1.append(temp2)
    closest_neighbors_list.append(temp1)
    

#%% 
########### FIGURES ########
# Figure 1: histograms with extension of closest neighbors

dec = [0,0.01,0.02,0.03]
names = ["S", "W", "N", "WCE"]
panels = [['(a)','(b)','(c)'],['(d)','(e)','(f)'],['(g)','(h)','(i)'],['(j)','(k)','(l)']]

plt.rc('font',family='serif',size=20) 
fig, axes = plt.subplots(nrows=len(name_directory_results), ncols=len(rolling_periods_tab), figsize=(20,10))
plt.tight_layout()

for i in range(len(closest_neighbors_list)):
    for idr, r in enumerate(rolling_periods_tab):
        series_obs_rolling = series_obs[i].rolling(time=r, center=True).mean().sel(time=(series_obs[i]['time.month']>=6) & (series_obs[i]['time.month']<=8))
    
        ax = axes[i,idr]
        
        ax.hist(series_obs_rolling-273.15, bins=100, histtype='step', density=True, color='black')
        
        quantile_tab = [series_obs_rolling.quantile(q=q).values for q in level_tab]
        
        for idj,j in enumerate(level_tab):
            ax.plot([quantile_tab[idj]-273.15, quantile_tab[idj]-273.15],[0,0.2], color=colors_c[idj], label=r"$\alpha = $"+str(j))
            ax.plot([closest_neighbors_list[i][idr][idj].min()-273.15,closest_neighbors_list[i][idr][idj].max()-273.15],[0.05+dec[idj],0.05+dec[idj]],color=colors_c[idj],linewidth=2)

        ax.set_title(panels[i][idr]+" "+names[i]+" r="+str(r))
        
        if i==3:
            ax.set_xlabel("Temperature [°C]")
        else:
            ax.axes.xaxis.set_ticklabels([])
        if idr==0:
            ax.set_ylabel("Density")
        else:
            ax.axes.yaxis.set_ticklabels([])
            
        if (i==3) and (idr==2):
            ax.legend(loc='lower right', prop={'size': 13})
            
        ax.set_xlim(10,30)
        ax.set_ylim(0,0.2)

plt.subplots_adjust(top=0.964,bottom=0.072,left=0.047,right=0.989,hspace=0.22,wspace=0.072)
plt.show()
            

#%% Figure 3,4,5 + annexe : r=5, champ moyen t2m (anomalie) + z500 contours + variances maps, all quantiles, NA + NH en annexe

plt.rc('font',family='serif',size=20) 

v = "t2m"
loc = 1
idr = 2
r = rolling_periods_tab[idr]
panels = ['(a)','(b)','(c)','(d)']

j_list_adapted = range(-r//2+1,r//2+1)
field_levels = select_field_group_rolling(v, j_list_adapted, level_tab, closest_neighbors_list[loc][idr])
field_levels_z500 = select_field_group_rolling("z500", j_list_adapted, level_tab, closest_neighbors_list[loc][idr])
     
#%
       
###### NORTH ATLANTIC #####
    
# Load climatology 
climato_mean, climato_var = select_climato(v, r=r) 
climato_mean = extract_coordinate_north_atlantic(climato_mean)
climato_var = extract_coordinate_north_atlantic(climato_var)

climato_mean_z500, climato_var_z500 = select_climato("z500", r=r) 
climato_mean_z500 = extract_coordinate_north_atlantic(climato_mean_z500)
climato_var_z500 = extract_coordinate_north_atlantic(climato_var_z500)

# Adapt fields
field_levels_prepared = []
for f in field_levels:
    temp = f[0].mean("time")
    for i in f[1:]:
        temp = xr.concat([temp, i.mean("time")], dim='time')
    field_levels_prepared.append(extract_coordinate_north_atlantic(temp))
    
field_levels_prepared_z500 = []
for f in field_levels_z500:
    temp = f[0].mean("time")
    for i in f[1:]:
        temp = xr.concat([temp, i.mean("time")], dim='time')
    field_levels_prepared_z500.append(extract_coordinate_north_atlantic(temp))
    
yticks = [25, 45, 65]
xticks = [0, 30, 60, -30, -60]
xticklabels = ['0°', '30°E', '60°E', '30°W', '60°W']
yticklabels = ['25°N', '45°N', '65°N']
    
# Plotting mean anomalies

fig = plt.figure(figsize=(27,10))
gs = fig.add_gridspec((len(level_tab)-1)//2+1,2)
        
lon = field_levels_prepared[0]['lon']
lat = field_levels_prepared[0]['lat']
    
levels = np.arange(-7,8,1)

for idlevel, level in enumerate(level_tab):

    ax = fig.add_subplot(gs[idlevel//2,idlevel%2], projection=ccrs.PlateCarree())
    cl = ax.contour(lon, lat, (field_levels_prepared_z500[idlevel].mean('time').values-climato_mean_z500.values), extend='both', levels=np.arange(-100,120,20), colors='black')
    ax.clabel(cl, cl.levels, inline=True, fontsize=15) 
    cp = ax.contourf(lon, lat, (field_levels_prepared[idlevel].mean('time').values-climato_mean.values), extend='both', levels=levels, cmap='RdBu_r')
    #plt.colorbar(cp, label="[°C]")
    ax.coastlines('50m', color='0', linewidth=.4)    
    ax.gridlines(draw_labels = False, xlocs=xticks, ylocs=yticks, color='.7', alpha=0.4, linewidth=.3)
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    if idlevel//2 == 1:
        ax.set_xticklabels(xticklabels)
    else:
        ax.set_xticklabels("")
    if idlevel%2 == 0:
        ax.set_yticklabels(yticklabels)
    else:
        ax.set_yticklabels("")
    
    ax.set_title(panels[idlevel] + r" $\alpha = $"+str(level))
    
        
    # Box for the observable
    ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(min_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    ax.plot([(max_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],min_lat_tab[loc]], linewidth=2, color='lime')
    ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[max_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
     
cbar_ax = fig.add_axes([0.935, 0.05, 0.02, 0.9])
cbar=fig.colorbar(cp, label="Anomaly of T2M [°C]", spacing='proportional', cax=cbar_ax)
plt.subplots_adjust(top=0.985,bottom=0.015,left=0.035,right=0.925,hspace=0.0,wspace=0.01)
plt.show()
    

#% Plotting variance t2m
fig = plt.figure(figsize=(27,10))
gs = fig.add_gridspec((len(level_tab)-1)//2+1,2)
        
lon = field_levels_prepared[0]['lon']
lat = field_levels_prepared[0]['lat'] 

for idlevel, level in enumerate(level_tab):

    ax = fig.add_subplot(gs[idlevel//2,idlevel%2], projection=ccrs.PlateCarree())
    
    cp = ax.contour(lon, lat, field_levels_prepared[idlevel].mean('time').values, extend='both', levels=np.arange(10,32,4), colors='black')
    ax.clabel(cp, cp.levels, inline=True, fontsize=15)
    
    #cl = ax.contourf(lon, lat, field_levels_prepared[idlevel].var('time').values/climato_var.values*100, extend='max', levels=np.arange(0,70,10), cmap='Blues', alpha=0.8)
    chi2_test = field_levels_prepared[idlevel].var('time').values/climato_var.values*(field_levels_prepared[idlevel].shape[0]-1) < chi2.ppf(0.05,(field_levels_prepared[idlevel].shape[0]-1))
    temp = field_levels_prepared[idlevel].var('time').values/climato_var.values*100
    temp[~chi2_test] = np.nan
    cl = ax.contourf(lon, lat, temp, extend='max', levels=np.arange(0,70,10), cmap='viridis', alpha=0.8)

    ax.coastlines('50m', color='0', linewidth=.4)    
    ax.gridlines(draw_labels = False, xlocs=xticks, ylocs=yticks, color='.7', alpha=0.4, linewidth=.3)
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    if idlevel//2 == 1:
        ax.set_xticklabels(xticklabels)
    else:
        ax.set_xticklabels("")
    if idlevel%2 == 0:
        ax.set_yticklabels(yticklabels)
    else:
        ax.set_yticklabels("")
        
    ax.set_title(panels[idlevel] + r" $\alpha = $"+str(level))
        
    # Box for the observable
    ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(min_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    ax.plot([(max_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],min_lat_tab[loc]], linewidth=2, color='lime')
    ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[max_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    
cbar_ax = fig.add_axes([0.935, 0.05, 0.02, 0.9])
cbar=fig.colorbar(cl, label=r"$\hat{V}$ [%]", spacing='proportional', cax=cbar_ax)
plt.subplots_adjust(top=0.985,bottom=0.015,left=0.035,right=0.925,hspace=0.0,wspace=0.01)
plt.show()

#% Plotting variance z500
fig = plt.figure(figsize=(27,10))
gs = fig.add_gridspec((len(level_tab)-1)//2+1,2)
        
lon = field_levels_prepared[0]['lon']
lat = field_levels_prepared[0]['lat'] 

for idlevel, level in enumerate(level_tab):

    ax = fig.add_subplot(gs[idlevel//2,idlevel%2], projection=ccrs.PlateCarree())
    cp = ax.contour(lon, lat, field_levels_prepared_z500[idlevel].mean('time').values, extend='both', levels=np.arange(5400,5950,50), colors='black')
    ax.clabel(cp, cp.levels, inline=True, fontsize=15 ) 

    #cl = ax.contourf(lon, lat, field_levels_prepared_z500[idlevel].var('time').values/climato_var_z500.values*100, extend='max', levels=np.arange(0,70,10), cmap='viridis', alpha=0.8)
    chi2_test = field_levels_prepared_z500[idlevel].var('time').values/climato_var_z500.values*(field_levels_prepared_z500[idlevel].shape[0] - 1) < chi2.ppf(0.05,field_levels_prepared_z500[idlevel].shape[0] - 1)
    temp = field_levels_prepared_z500[idlevel].var('time').values/climato_var_z500.values*100
    temp[~chi2_test] = np.nan
    cl = ax.contourf(lon, lat, temp, extend='max', levels=np.arange(0,70,10), cmap='viridis', alpha=0.8)
    
    ax.coastlines('50m', color='0', linewidth=.4)    
    ax.gridlines(draw_labels = False, xlocs=xticks, ylocs=yticks, color='.7', alpha=0.4, linewidth=.3)
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    if idlevel//2 == 1:
        ax.set_xticklabels(xticklabels)
    else:
        ax.set_xticklabels("")
    if idlevel%2 == 0:
        ax.set_yticklabels(yticklabels)
    else:
        ax.set_yticklabels("")
        
    ax.set_title(panels[idlevel] + r" $\alpha = $"+str(level))
        
    # Box for the observable
    ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(min_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    ax.plot([(max_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],min_lat_tab[loc]], linewidth=2, color='lime')
    ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[max_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    
cbar_ax = fig.add_axes([0.935, 0.05, 0.02, 0.9])
cbar=fig.colorbar(cl, label=r"$\hat{V}$ [%]", spacing='proportional', cax=cbar_ax)
plt.subplots_adjust(top=0.985,bottom=0.015,left=0.035,right=0.925,hspace=0.0,wspace=0.01)
plt.show()

#%


##### NORTH HEMISPHERE ####
        
# Load climatology 
climato_mean, climato_var = select_climato(v, r=r) 
lat = climato_mean['lat']
climato_mean, lon = add_cyclic_point(climato_mean, coord=climato_mean['lon'])
climato_var = add_cyclic_point(climato_var, coord=climato_var['lon'])[0]

climato_mean_z500, climato_var_z500 = select_climato("z500", r=r) 
climato_mean_z500, lon = add_cyclic_point(climato_mean_z500, coord=climato_mean_z500['lon'])
climato_var_z500 = add_cyclic_point(climato_var_z500, coord=climato_var_z500['lon'])[0]

#% Adapt fields
field_levels_prepared = []
for f in field_levels:
    temp = f[0].mean("time")
    for i in f[1:]:
        temp = xr.concat([temp, i.mean("time")], dim='time')
    field_levels_prepared.append(add_cyclic_point(temp, coord=temp['lon'])[0])
    
field_levels_prepared_z500 = []
for f in field_levels_z500:
    temp = f[0].mean("time")
    for i in f[1:]:
        temp = xr.concat([temp, i.mean("time")], dim='time')
    field_levels_prepared_z500.append(add_cyclic_point(temp, coord=temp['lon'])[0])
    

yticks = [25, 45, 65]
xticks = [0, 30, 60, 90, 120, 150, 180, -30, -60, -90, -120, -150]


# Plotting mean anomalies
fig = plt.figure(figsize=(17,100))
gs = fig.add_gridspec((len(level_tab)-1)//2+1,2)
    
levels = np.arange(-7,8,1)  

for idlevel, level in enumerate(level_tab):

    ax = fig.add_subplot(gs[idlevel//2,idlevel%2], projection=ccrs.NorthPolarStereo(central_longitude=0))
    cl = ax.contour(lon, lat, np.mean(field_levels_prepared_z500[idlevel],axis=0)-climato_mean_z500, extend='both', levels=np.arange(-100,120,20), colors='black', transform=ccrs.PlateCarree())
    ax.clabel(cl, cl.levels, fontsize=10,  inline_spacing = 50) 
    cp = ax.contourf(lon, lat, np.mean(field_levels_prepared[idlevel]-climato_mean, axis=0), extend='both', levels=levels, cmap='RdBu_r', transform=ccrs.PlateCarree())
    #plt.colorbar(cp, label="[°C]")
    ax.coastlines('50m', color='0', linewidth=.4)    
    ax.gridlines(draw_labels = False, xlocs=xticks, ylocs=yticks, color='.7', alpha=0.4, linewidth=.3)
    
    ax.set_extent([-180, 180, lat[0], lat[-1]], ccrs.PlateCarree())
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.set_title(panels[idlevel] + r" $\alpha = $"+str(level))
        
    # Box for the observable
    ax.plot([min_lon_tab[loc],min_lon_tab[loc]],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    ax.plot([max_lon_tab[loc],max_lon_tab[loc]],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    ax.plot([min_lon_tab[loc],max_lon_tab[loc]],[min_lat_tab[loc],min_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    ax.plot([min_lon_tab[loc],max_lon_tab[loc]],[max_lat_tab[loc],max_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())

cbar_ax = fig.add_axes([0.90, 0.05, 0.03, 0.9])
cbar=fig.colorbar(cp, label="Anomaly of T2M [°C]", spacing='proportional', cax=cbar_ax)
plt.subplots_adjust(top=0.96,bottom=0.01,left=0.0,right=0.93,hspace=0.095,wspace=0.0)
plt.show()
    
    
    
    
#% Plotting variance t2m

fig = plt.figure(figsize=(17,100))
plt.subplots_adjust(top=0.965,bottom=0.005,left=0.06,right=0.955,hspace=0.07,wspace=0.095)
gs = fig.add_gridspec((len(level_tab)-1)//2+1,2)
    
for idlevel, level in enumerate(level_tab):

    ax = fig.add_subplot(gs[idlevel//2,idlevel%2], projection=ccrs.NorthPolarStereo(central_longitude=0))
    
    cp = ax.contour(lon, lat, np.mean(field_levels_prepared[idlevel], axis=0), extend='both', levels=np.arange(10,32,4), colors='black', transform=ccrs.PlateCarree())
    ax.clabel(cp, cp.levels, fontsize=10,  inline_spacing = 150 )
    
    #cl = ax.contourf(lon, lat, np.var(field_levels_prepared[idlevel],axis=0)/climato_var*100, extend='max', levels=np.arange(0,70,10), cmap='viridis', alpha=0.8, transform=ccrs.PlateCarree())    
    chi2_test = np.var(field_levels_prepared[idlevel],axis=0)/climato_var*(field_levels_prepared[idlevel].shape[0] - 1) < chi2.ppf(0.05,field_levels_prepared[idlevel].shape[0] - 1)
    temp = np.var(field_levels_prepared[idlevel],axis=0)/climato_var*100
    temp[~chi2_test] = np.nan
    cl = ax.contourf(lon, lat, temp, extend='max', levels=np.arange(0,70,10), cmap='viridis', alpha=0.8, transform=ccrs.PlateCarree())    
     
    ax.coastlines('50m', color='0', linewidth=.4)    
    ax.gridlines(draw_labels = False, xlocs=xticks, ylocs=yticks, color='.7', alpha=0.4, linewidth=.3)
    ax.set_extent([-180, 180, lat[0], lat[-1]], ccrs.PlateCarree())
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.set_title(panels[idlevel] + r" $\alpha = $"+str(level))
        
    # Box for the observable
    ax.plot([min_lon_tab[loc],min_lon_tab[loc]],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    ax.plot([max_lon_tab[loc],max_lon_tab[loc]],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    ax.plot([min_lon_tab[loc],max_lon_tab[loc]],[min_lat_tab[loc],min_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    ax.plot([min_lon_tab[loc],max_lon_tab[loc]],[max_lat_tab[loc],max_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())

cbar_ax = fig.add_axes([0.90, 0.05, 0.03, 0.9])
cbar=fig.colorbar(cl, label=r"$\hat{V}$ [%]", spacing='proportional', cax=cbar_ax)
plt.subplots_adjust(top=0.96,bottom=0.01,left=0.0,right=0.93,hspace=0.095,wspace=0.0)
plt.show()    

#% Plotting variance z500

fig = plt.figure(figsize=(17,100))
plt.subplots_adjust(top=0.965,bottom=0.005,left=0.06,right=0.955,hspace=0.07,wspace=0.095)
gs = fig.add_gridspec((len(level_tab)-1)//2+1,2)
    
for idlevel, level in enumerate(level_tab):

    ax = fig.add_subplot(gs[idlevel//2,idlevel%2], projection=ccrs.NorthPolarStereo(central_longitude=0))
    
    cp = ax.contour(lon, lat, np.mean(field_levels_prepared_z500[idlevel], axis=0), extend='both', levels=np.arange(5400,5950,50), colors='black', transform=ccrs.PlateCarree())
    ax.clabel(cp, cp.levels, fontsize=10, inline_spacing = 150 ) 
    
    #cl = ax.contourf(lon, lat, np.var(field_levels_prepared_z500[idlevel],axis=0)/climato_var_z500*100, extend='max', levels=np.arange(0,70,10), cmap='viridis', alpha=0.8, transform=ccrs.PlateCarree())
    chi2_test = np.var(field_levels_prepared_z500[idlevel],axis=0)/climato_var_z500*(field_levels_prepared_z500[idlevel].shape[0] - 1) < chi2.ppf(0.05,field_levels_prepared_z500[idlevel].shape[0] - 1)
    temp = np.var(field_levels_prepared_z500[idlevel],axis=0)/climato_var_z500*100
    temp[~chi2_test] = np.nan
    cl = ax.contourf(lon, lat, temp, extend='max', levels=np.arange(0,70,10), cmap='viridis', alpha=0.8, transform=ccrs.PlateCarree())
    
    ax.coastlines('50m', color='0', linewidth=.4)    
    ax.gridlines(draw_labels = False, xlocs=xticks, ylocs=yticks, color='.7', alpha=0.4, linewidth=.3)
    ax.set_extent([-180, 180, lat[0], lat[-1]], ccrs.PlateCarree())
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.set_title(panels[idlevel] + r" $\alpha = $"+str(level))
        
    # Box for the observable
    ax.plot([min_lon_tab[loc],min_lon_tab[loc]],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    ax.plot([max_lon_tab[loc],max_lon_tab[loc]],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    ax.plot([min_lon_tab[loc],max_lon_tab[loc]],[min_lat_tab[loc],min_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    ax.plot([min_lon_tab[loc],max_lon_tab[loc]],[max_lat_tab[loc],max_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())

cbar_ax = fig.add_axes([0.90, 0.05, 0.03, 0.9])
cbar=fig.colorbar(cl, label=r"$\hat{V}$ [%]", spacing='proportional', cax=cbar_ax)
plt.subplots_adjust(top=0.96,bottom=0.01,left=0.0,right=0.93,hspace=0.095,wspace=0.0)
plt.show()    


#%% Figure 6: r=5, mrsos, slp, t850, v250, champ moyen + variance contour mais uniquement q=0.999, NA + NH en annexe

plt.rc('font',family='serif',size=20) 

v_tab = ["mrsos","slp","t850","v250"]
loc = 1
idr = 1
r = rolling_periods_tab[idr]
panels = ['(a)','(b)','(c)','(d)']
color_maps=['BrBG',"PuOr_r","RdBu_r","PuOr_r"]

j_list_adapted = range(-r//2+1,r//2+1)

yticks = [25, 45, 65]
xticks = [0, 30, 60, -30, -60]
xticklabels = ['0°', '30°E', '60°E', '30°W', '60°W']
yticklabels = ['25°N', '45°N', '65°N']

##### NORTH ATLANTIC #####

fig = plt.figure(figsize=(35,10))
gs = fig.add_gridspec((len(v_tab)-1)//2+1,2)

for idv,v in enumerate(v_tab):
    print(v)
    field_levels = select_field_group_rolling(v, j_list_adapted, level_tab, closest_neighbors_list[loc][idr])
    climato_mean, climato_var = select_climato(v, r=r) 
    climato_mean = extract_coordinate_north_atlantic(climato_mean)
    climato_var = extract_coordinate_north_atlantic(climato_var)
    
    # Adapt fields
    field_levels_prepared = []
    for f in field_levels:
        temp = f[0].mean("time")
        for i in f[1:]:
            temp = xr.concat([temp, i.mean("time")], dim='time')
        field_levels_prepared.append(extract_coordinate_north_atlantic(temp))
        
    lon = field_levels_prepared[0]['lon']
    lat = field_levels_prepared[0]['lat']
    
    if v=="slp": # anomaly
        levels = np.arange(-5,6,1)
        unit = "Anomaly of SLP [hPa]"
    elif v=="t850": # no anomaly
        levels = [-7.5,-5.5,-3.5, -1.5, -0.75, 0, 0.75, 1.5, 3.5, 5.5, 7.5]
        unit = "Anomaly of T850 [°C]"
    elif v=="v250": # no anomaly
        levels = np.arange(-15,18,3)
        unit = "V250 [m/s]"
    elif v=="mrsos":
        levels = np.arange(-5,6,1)
        unit = "Anomaly of SM [kg/m²]"
    
    # Mean
    ax = fig.add_subplot(gs[idv//2,idv%2], projection=ccrs.PlateCarree())
    if v=="mrsos":
        cp = ax.contourf(lon, lat, (field_levels_prepared[3].mean('time').values-climato_mean.values), extend='both', levels=levels, cmap=color_maps[idv])
    elif v=="slp" or v=="t850":
        cp = ax.contourf(lon, lat, (field_levels_prepared[3].mean('time').values-climato_mean.values), extend='both', levels=levels, cmap=color_maps[idv])
    else: 
        cp = ax.contourf(lon, lat, field_levels_prepared[3].mean('time').values, extend='both', levels=levels, cmap=color_maps[idv])
    plt.colorbar(cp, label=unit)
    
    mask = field_levels_prepared[3].var('time').values/climato_var.values*100 > 70
    cl = ax.contourf(lon, lat, mask, levels=2, hatches=[None,"//","//"], colors="none")
        
    #ax.clabel(cl, cl.levels, inline=True, fontsize=15 ) 
    #plt.colorbar(cp, label="Normalized variance [%]")
    ax.coastlines('50m', color='0', linewidth=.4)    
    ax.gridlines(draw_labels = False, xlocs=xticks, ylocs=yticks, color='.7', alpha=0.4, linewidth=.3)
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    if idv//2 == 1:
        ax.set_xticklabels(xticklabels)
    else:
        ax.set_xticklabels("")
    if idv%2 == 0:
        ax.set_yticklabels(yticklabels)
    else:
        ax.set_yticklabels("")
        
    ax.set_title(panels[idv])
    
    # Box for the observable
    ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(min_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    ax.plot([(max_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],min_lat_tab[loc]], linewidth=2, color='lime')
    ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[max_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    
    # Variance
    #ax = fig.add_subplot(gs[idv,1], projection=ccrs.PlateCarree())
    #cp = ax.contourf(lon, lat, field_levels_prepared[3].var('time').values/climato_var.values*100, extend='both', levels=np.arange(5,75,5), cmap='coolwarm')
    #plt.colorbar(cp, label="Normalized variance [%]")
    #ax.coastlines()
    #ax.set_title(panels[idv*2+1])
    
    # Box for the observable
    #ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(min_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    #ax.plot([(max_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    #ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],min_lat_tab[loc]], linewidth=2, color='lime')
    #ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[max_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    
plt.subplots_adjust(top=0.985,bottom=0.015,left=0.035,right=0.925,hspace=0.0,wspace=0.01)
plt.show()


#### NORTH HEMISPHERE ####

yticks = [25, 45, 65]
xticks = [0, 30, 60, 90, 120, 150, 180, -30, -60, -90, -120, -150]

fig = plt.figure(figsize=(17,100))
gs = fig.add_gridspec((len(v_tab)-1)//2+1,2)

for idv,v in enumerate(v_tab):
    print(v)
    field_levels = select_field_group_rolling(v, j_list_adapted, level_tab, closest_neighbors_list[loc][idr])
    climato_mean, climato_var = select_climato(v, r=r) 
    lat = climato_mean['lat']
    climato_mean, lon = add_cyclic_point(climato_mean, coord=climato_mean['lon'])
    climato_var = add_cyclic_point(climato_var, coord=climato_var['lon'])[0]
    
    # Adapt fields
    field_levels_prepared = []
    for f in field_levels:
        temp = f[0].mean("time")
        for i in f[1:]:
            temp = xr.concat([temp, i.mean("time")], dim='time')
        field_levels_prepared.append(add_cyclic_point(temp, coord=temp['lon'])[0])
    
    if v=="slp": # anomaly
        levels = np.arange(-5,6,1)
        unit = "Anomaly of SLP [hPa]"
    elif v=="t850": # no anomaly
        levels = [-7.5,-5.5,-3.5, -1.5, -0.75, 0, 0.75, 1.5, 3.5, 5.5, 7.5]
        unit = "Avomaly of T850 [°C]"
    elif v=="v250": # no anomaly
        levels = np.arange(-15,18,3)
        unit = "V250 [m/s]"
    elif v=="mrsos":
        levels = np.arange(-5,6,1)
        unit = "Anomaly of SM [kg/m²]"
    
    # Mean
    ax = fig.add_subplot(gs[idv//2,idv%2], projection=ccrs.NorthPolarStereo(central_longitude=0))
    if v=="slp" or v=="mrsos" or v=="t850":
        cp = ax.contourf(lon, lat, np.mean(field_levels_prepared[3]-climato_mean, axis=0), extend='both', levels=levels, cmap=color_maps[idv], transform=ccrs.PlateCarree())
    else: 
        cp = ax.contourf(lon, lat, np.mean(field_levels_prepared[3], axis=0), extend='both', levels=levels, cmap=color_maps[idv], transform=ccrs.PlateCarree())
    plt.colorbar(cp, label=unit)
    #if v=="mrsos":
    #    cl = ax.contour(lon, lat, np.var(field_levels_prepared[3],axis=0)/climato_var*100, extend='both', levels=np.arange(0,70,20), cmap='viridis', linewidths=1.5, transform=ccrs.PlateCarree())
    #else:
    #    cl = ax.contour(lon, lat, np.var(field_levels_prepared[3],axis=0)/climato_var*100, extend='both', levels=np.arange(0,70,10), cmap='viridis', linewidths=1.5, transform=ccrs.PlateCarree())
    #ax.clabel(cl, cl.levels, inline=True, fontsize=12 ) 
    
    mask = np.var(field_levels_prepared[3],axis=0)/climato_var*100 > 70
    cl = ax.contourf(lon, lat, mask, levels=2, hatches=[None,"///","///"], colors="none", transform=ccrs.PlateCarree())
    
    ax.coastlines('50m', color='0', linewidth=.4)    
    ax.gridlines(draw_labels = False, xlocs=xticks, ylocs=yticks, color='.7', alpha=0.4, linewidth=.3)
    
    ax.set_extent([-180, 180, lat[0], lat[-1]], ccrs.PlateCarree())
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.set_title(panels[idv])

    # Box for the observable
    ax.plot([min_lon_tab[loc],min_lon_tab[loc]],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    ax.plot([max_lon_tab[loc],max_lon_tab[loc]],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    ax.plot([min_lon_tab[loc],max_lon_tab[loc]],[min_lat_tab[loc],min_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    ax.plot([min_lon_tab[loc],max_lon_tab[loc]],[max_lat_tab[loc],max_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    
    # Variance
    #ax = fig.add_subplot(gs[idv,1], projection=ccrs.NorthPolarStereo(central_longitude=0))
    #cp = ax.contourf(lon, lat, np.var(field_levels_prepared[3],axis=0)/climato_var*100, extend='both', levels=np.arange(5,75,5), cmap='coolwarm', transform=ccrs.PlateCarree())
    #plt.colorbar(cp, label="Normalized variance [%]")
    #ax.coastlines()
    #ax.set_title(panels[idv*2+1])

    # Box for the observable
    #ax.plot([min_lon_tab[loc],min_lon_tab[loc]],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    #ax.plot([max_lon_tab[loc],max_lon_tab[loc]],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    #ax.plot([min_lon_tab[loc],max_lon_tab[loc]],[min_lat_tab[loc],min_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    #ax.plot([min_lon_tab[loc],max_lon_tab[loc]],[max_lat_tab[loc],max_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    
plt.subplots_adjust(top=0.965,bottom=0.005,left=0.06,right=0.955,hspace=0.07,wspace=0.095)
plt.show()


#%% Figure 7: Variance evolution for all variables (idem pour les autres endroits -> à mettre en annexe) 

plt.rc('font',family='serif',size=20) 

v_tab = ["t2m","z500","mrsos","slp","t850","v250"]
names_v_tab = ["T2M","Z500","SM","SLP","T850","V250"]
loc = 0
panels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)','(m)','(n)','(o)','(p)','(q)','(r)']

fig = plt.figure(figsize=(20,30))
plt.tight_layout()
gs = fig.add_gridspec(len(v_tab),len(rolling_periods_tab))


for idv, v in enumerate(v_tab):
    print(v)
    
    # Load climatology 
    _, climato_var = select_climato(v) 
    climato_var_NA = extract_coordinate_north_atlantic(climato_var)


    for idr, r in enumerate(rolling_periods_tab):
        
        ax = fig.add_subplot(gs[idv,idr])
        ax.set_title(panels[idv*3+idr]+" "+names_v_tab[idv]+" r="+str(r)+ " days", size=18)
        ax.plot([0,0],[0,2*100],color='black',linestyle='dashed',alpha=0.3)
        ax.plot([10,10],[0,2*100],color='black',linestyle='dashed',alpha=0.3)
        ax.plot([-10,-10],[0,2*100],color='black',linestyle='dashed',alpha=0.3)
        ax.plot([-15,15],[0.75*100,0.75*100],color='black',linestyle='dashed',alpha=0.3)
                
        for idl,level in enumerate(level_tab):
            
            # NORTH ATLANTIC
            mean_variance_NA = []
            for idj,j in enumerate(j_list):
                temp = xr.load_dataarray(path_results+name_directory_results[loc]+"data_closest/"+v+"_r"+str(r)+"_q"+str(level)+"_day"+str(j)+"_variance.nc", use_cftime=True)
                temp = extract_coordinate_north_atlantic(temp)/climato_var_NA #%%
                weights = np.cos(np.deg2rad(temp.lat))
                weights.name = "weights"                
                mean_variance_NA.append(temp.weighted(weights).mean(("lon","lat")).values*100)
                
                if ttest_1samp((temp.values*np.array([weights for i in temp.lon]).T).flatten()/np.sum(np.array([weights for i in temp.lon])), popmean=1, alternative='less')[1] > 0.05:
                    print("v", v)
                    print("r", r)
                    print("level", level)
                    print('j', j)
            
            ax.plot(j_list, mean_variance_NA, color=colors_c[idl],label=r"$\alpha = $"+str(level)+" NA")
            ax.scatter(j_list, mean_variance_NA, s=4, color=colors_c[idl])
            
            # NORTH HEMISPHERE
            mean_variance_NH = []
            for idj,j in enumerate(j_list):
                temp = xr.load_dataarray(path_results+name_directory_results[loc]+"data_closest/"+v+"_r"+str(r)+"_q"+str(level)+"_day"+str(j)+"_variance.nc", use_cftime=True)/climato_var
                weights = np.cos(np.deg2rad(temp.lat))
                weights.name = "weights"                
                mean_variance_NH.append(temp.weighted(weights).mean(("lon","lat")).values*100)
                
                if ttest_1samp((temp.values*np.array([weights for i in temp.lon]).T).flatten()/np.sum(np.array([weights for i in temp.lon])), popmean=1, alternative='less')[1] > 0.05:
                    print('NH')
                    print("v", v)
                    print("r", r)
                    print("level", level)
                    print('j', j)
                
            ax.plot(j_list, mean_variance_NH, color=colors_c[idl],label=r"$\alpha = $"+str(level)+" NH",linestyle='dashed')
            ax.scatter(j_list, mean_variance_NH, s=4, color=colors_c[idl])
        
        if idv==5:
            ax.set_xlabel("Relative days")
        else:
            ax.axes.xaxis.set_ticklabels([])
        if idr==0:
            ax.set_ylabel(r"$<\tilde{V}>$")
        else:
            ax.axes.yaxis.set_ticklabels([])
        
        ax.set_xlim(j_list[0],j_list[-1])
                
        if v=="t2m":
            ax.set_ylim(0.4*100,1.1*100)
        elif v=="z500":
            ax.set_ylim(0.45*100,1.*100)
        elif v=="slp":
            ax.set_ylim(0.6*100,1.*100)
        elif v=="t850":
            ax.set_ylim(0.50*100,1.1*100)
        elif v=="v250":
            ax.set_ylim(0.60*100,1.1*100)
        elif v=="mrsos":
            ax.set_ylim(0.7*100,1.1*100)
            
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower left',ncol=4,frameon=False, bbox_to_anchor=[0.06, 0])
plt.subplots_adjust(top=0.975,bottom=0.130,left=0.090,right=0.995,hspace=0.200,wspace=0.035)
plt.show()


#%% Figures 9,10,11: montrer résultats pour q=0.999 et r=5 uniquement (pour slp, z500, t2m, t850, v250 pour chaque lieu), NA + NH



plt.rc('font',family='serif',size=20) 

v_tab = ["t2m","z500","mrsos","slp","t850","v250"]
loc = 3
idr = 1
r = rolling_periods_tab[idr]
panels = ['(a)','(b)','(c)','(d)','(e)','(f)']
color_maps=["RdBu_r","PuOr_r",'BrBG',"PuOr_r","RdBu_r","PuOr_r"]

j_list_adapted = range(-r//2+1,r//2+1)

yticks = [25, 45, 65]
xticks = [0, 30, 60, -30, -60]
xticklabels = ['0°', '30°E', '60°E', '30°W', '60°W']
yticklabels = ['25°N', '45°N', '65°N']

##### NORTH ATLANTIC #####

fig = plt.figure(figsize=(30,10))
gs = fig.add_gridspec((len(v_tab)-1)//2+1,2)

for idv,v in enumerate(v_tab):
    print(v)
    field_levels = select_field_group_rolling(v, j_list_adapted, level_tab, closest_neighbors_list[loc][idr])
    climato_mean, climato_var = select_climato(v, r=r) 
    climato_mean = extract_coordinate_north_atlantic(climato_mean)
    climato_var = extract_coordinate_north_atlantic(climato_var)
    
    # Adapt fields
    field_levels_prepared = []
    for f in field_levels:
        temp = f[0].mean("time")
        for i in f[1:]:
            temp = xr.concat([temp, i.mean("time")], dim='time')
        field_levels_prepared.append(extract_coordinate_north_atlantic(temp))
        
    lon = field_levels_prepared[0]['lon']
    lat = field_levels_prepared[0]['lat']
    
    if v=="slp": # anomaly
        levels = np.arange(-5,6,1)
        unit = "Anomaly of SLP [hPa]"
    elif v=="t850": # no anomaly
        levels = [-7.5,-5.5,-3.5, -1.5, -0.75, 0, 0.75, 1.5, 3.5, 5.5, 7.5]
        unit = "Anomaly of T850 [°C]"
    elif v=="v250": # no anomaly
        levels = np.arange(-15,18,3)
        unit = "V250 [m/s]"
    elif v=="mrsos":
        levels = np.arange(-5,6,1)
        unit = "Anomaly of SM [kg/m²]"
    elif v=="t2m":
        levels = np.arange(-7,8,1)
        unit = "Anomaly of T2M [°C]"
    elif v=="z500":
        levels = np.arange(-100,120,20)
        unit = 'Anomaly of Z500 [m]'
    
    # Mean
    ax = fig.add_subplot(gs[idv//2,idv%2], projection=ccrs.PlateCarree())
    if v=="slp" or v=="mrsos" or v=="t2m" or v=="z500" or v=="t850":
        cp = ax.contourf(lon, lat, (field_levels_prepared[3].mean('time').values-climato_mean.values), extend='both', levels=levels, cmap=color_maps[idv])
    else: 
        cp = ax.contourf(lon, lat, field_levels_prepared[3].mean('time').values, extend='both', levels=levels, cmap=color_maps[idv])
    plt.colorbar(cp, label=unit)
    #if v=="mrsos":
     #   cl = ax.contour(lon, lat, field_levels_prepared[3].var('time').values/climato_var.values*100, extend='both', levels=np.arange(0,70,20), cmap='viridis', linewidths=1.5)
    #else:
    #    cl = ax.contour(lon, lat, field_levels_prepared[3].var('time').values/climato_var.values*100, extend='both', levels=np.arange(0,70,10), cmap='viridis', linewidths=1.5)
    #ax.clabel(cl, cl.levels, inline=True, fontsize=15 ) 
    #plt.colorbar(cp, label="Normalized variance [%]")
    
    mask = field_levels_prepared[3].var('time').values/climato_var.values*100 > 70
    cl = ax.contourf(lon, lat, mask, levels=2, hatches=[None,"//","//"], colors="none")
    
    ax.coastlines('50m', color='0', linewidth=.4)    
    ax.gridlines(draw_labels = False, xlocs=xticks, ylocs=yticks, color='.7', alpha=0.4, linewidth=.3)
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    if idv//2 == 2:
        ax.set_xticklabels(xticklabels)
    else:
        ax.set_xticklabels("")
    if idv%2 == 0:
        ax.set_yticklabels(yticklabels)
    else:
        ax.set_yticklabels("")
        
    ax.set_title(panels[idv])
    
    # Box for the observable
    ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(min_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    ax.plot([(max_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],min_lat_tab[loc]], linewidth=2, color='lime')
    ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[max_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    
    # Variance
    #ax = fig.add_subplot(gs[idv,1], projection=ccrs.PlateCarree())
    #cp = ax.contourf(lon, lat, field_levels_prepared[3].var('time').values/climato_var.values*100, extend='both', levels=np.arange(5,75,5), cmap='coolwarm')
    #plt.colorbar(cp, label="Normalized variance [%]")
    #ax.coastlines()
    #ax.set_title(panels[idv*2+1])
    
    # Box for the observable
    #ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(min_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    #ax.plot([(max_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    #ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],min_lat_tab[loc]], linewidth=2, color='lime')
    #ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[max_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    
plt.subplots_adjust(top=0.965,bottom=0.030,left=0.035,right=1.,hspace=0.115,wspace=0.0)
plt.show()


#### NORTH HEMISPHERE ####

yticks = [25, 45, 65]
xticks = [0, 30, 60, 90, 120, 150, 180, -30, -60, -90, -120, -150]

fig = plt.figure(figsize=(17,100))
gs = fig.add_gridspec((len(v_tab)-1)//2+1,2)

for idv,v in enumerate(v_tab):
    print(v)
    field_levels = select_field_group_rolling(v, j_list_adapted, level_tab, closest_neighbors_list[loc][idr])
    climato_mean, climato_var = select_climato(v, r=r) 
    lat = climato_mean['lat']
    climato_mean, lon = add_cyclic_point(climato_mean, coord=climato_mean['lon'])
    climato_var = add_cyclic_point(climato_var, coord=climato_var['lon'])[0]
    
    # Adapt fields
    field_levels_prepared = []
    for f in field_levels:
        temp = f[0].mean("time")
        for i in f[1:]:
            temp = xr.concat([temp, i.mean("time")], dim='time')
        field_levels_prepared.append(add_cyclic_point(temp, coord=temp['lon'])[0])
    
    if v=="slp": # anomaly
        levels = np.arange(-5,6,1)
        unit = "Anomaly of SLP [hPa]"
    elif v=="t850": # no anomaly
        levels = [-7.5,-5.5,-3.5, -1.5, -0.75, 0, 0.75, 1.5, 3.5, 5.5, 7.5]
        unit = "Anomaly of T850 [°C]"
    elif v=="v250": # no anomaly
        levels = np.arange(-15,18,3)
        unit = "V250 [m/s]"
    elif v=="mrsos":
        levels = np.arange(-5,6,1)
        unit = "Anomaly of SM [kg/m²]"
    elif v=="t2m":
        levels = np.arange(-7,8,1)
        unit = "Anomaly of T2M [°C]"
    elif v=="z500":
        levels = np.arange(-100,120,20)
        unit = 'Anomaly of Z500 [m]'
    
    # Mean
    ax = fig.add_subplot(gs[idv//2,idv%2], projection=ccrs.NorthPolarStereo(central_longitude=0))
    if v=="slp" or v=="mrsos" or v=="t2m" or v=="z500" or v=="t850":
        cp = ax.contourf(lon, lat, np.mean(field_levels_prepared[3]-climato_mean, axis=0), extend='both', levels=levels, cmap=color_maps[idv], transform=ccrs.PlateCarree())
    else: 
        cp = ax.contourf(lon, lat, np.mean(field_levels_prepared[3], axis=0), extend='both', levels=levels, cmap=color_maps[idv], transform=ccrs.PlateCarree())
    plt.colorbar(cp, label=unit)
    #if v=='mrsos':
    #    cl = ax.contour(lon, lat, np.var(field_levels_prepared[3],axis=0)/climato_var*100, extend='both', levels=np.arange(0,70,20), cmap='viridis', linewidths=1.5, transform=ccrs.PlateCarree())
    #else:
    #    cl = ax.contour(lon, lat, np.var(field_levels_prepared[3],axis=0)/climato_var*100, extend='both', levels=np.arange(0,70,10), cmap='viridis', linewidths=1.5, transform=ccrs.PlateCarree())
    #ax.clabel(cl, cl.levels, inline=True, fontsize=12 ) 
    
    mask = np.var(field_levels_prepared[3],axis=0)/climato_var*100 > 70
    cl = ax.contourf(lon, lat, mask, levels=2, hatches=[None,"///","///"], colors="none", transform=ccrs.PlateCarree())
    
    ax.coastlines('50m', color='0', linewidth=.4)    
    ax.gridlines(draw_labels = False, xlocs=xticks, ylocs=yticks, color='.7', alpha=0.4, linewidth=.3)
    
    ax.set_extent([-180, 180, lat[0], lat[-1]], ccrs.PlateCarree())
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.set_title(panels[idv])

    # Box for the observable
    ax.plot([min_lon_tab[loc],min_lon_tab[loc]],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    ax.plot([max_lon_tab[loc],max_lon_tab[loc]],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    ax.plot([min_lon_tab[loc],max_lon_tab[loc]],[min_lat_tab[loc],min_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    ax.plot([min_lon_tab[loc],max_lon_tab[loc]],[max_lat_tab[loc],max_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    
    # Variance
    #ax = fig.add_subplot(gs[idv,1], projection=ccrs.NorthPolarStereo(central_longitude=0))
    #cp = ax.contourf(lon, lat, np.var(field_levels_prepared[3],axis=0)/climato_var*100, extend='both', levels=np.arange(5,75,5), cmap='coolwarm', transform=ccrs.PlateCarree())
    #plt.colorbar(cp, label="Normalized variance [%]")
    #ax.coastlines()
    #ax.set_title(panels[idv*2+1])

    # Box for the observable
    #ax.plot([min_lon_tab[loc],min_lon_tab[loc]],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    #ax.plot([max_lon_tab[loc],max_lon_tab[loc]],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    #ax.plot([min_lon_tab[loc],max_lon_tab[loc]],[min_lat_tab[loc],min_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    #ax.plot([min_lon_tab[loc],max_lon_tab[loc]],[max_lat_tab[loc],max_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
    
plt.subplots_adjust(top=0.970,bottom=0.005,left=0.035,right=0.930,hspace=0.095,wspace=0.095)
plt.show()

#%% Standard deviation of calendar days

loc = 1

for idr, r in enumerate(rolling_periods_tab):
    print("Considering r=", r)
    for idl,level in enumerate(level_tab):
        print("For level ", level, " std=", np.std(closest_neighbors_list[loc][idr][idl].time.dt.dayofyear))
        print("For level ", level, " m=", np.mean(closest_neighbors_list[loc][idr][idl].time.dt.dayofyear))


#%% Figure for modelling animation presentation: observable at Paris


fig = plt.figure(figsize=(27,10))
plt.subplots_adjust(top=0.955,bottom=0.01,left=0.0,right=1.0,hspace=0.17,wspace=0.0)
gs = fig.add_gridspec(1,1)
loc=1
climato_mean, _ = select_climato("z500") 
climato_mean = extract_coordinate_north_atlantic(climato_mean)
        
lon = climato_mean['lon']
lat = climato_mean['lat']

ax = fig.add_subplot(gs[0,0], projection=ccrs.PlateCarree())
cl = ax.contour(lon, lat, climato_mean.values, extend='both', levels=np.arange(-100,120,20), colors='black', alpha=0.5)  
ax.coastlines()
        
# Box for the observable
ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(min_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
ax.plot([(max_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],min_lat_tab[loc]], linewidth=2, color='lime')
ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[max_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
     
plt.subplots_adjust(top=0.985,bottom=0.015,left=0.005,right=0.925,hspace=0.0,wspace=0.01)
plt.show()

#%%

fig = plt.figure(figsize=(27,10))
plt.subplots_adjust(top=0.955,bottom=0.01,left=0.0,right=1.0,hspace=0.17,wspace=0.0)
gs = fig.add_gridspec(1,1)
loc=1
climato_mean, _ = select_climato("z500") 
climato_mean = extract_coordinate_north_atlantic(climato_mean)
        
lon = climato_mean['lon']
lat = climato_mean['lat']

ax = fig.add_subplot(gs[0,0], projection=ccrs.PlateCarree())
cl = ax.contour(lon, lat, climato_mean.values, extend='both', levels=np.arange(-100,120,20), colors='black', alpha=0.5)  
ax.coastlines()
        
# Box for the observable
ax.plot([(6.25 +2.5 + 180) % 360 - 180,(6.25 +2.5 + 180) % 360 - 180],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
ax.plot([(8.75 +2.5 + 180) % 360 - 180,(8.75 +2.5 + 180) % 360 - 180],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
ax.plot([(6.25 +2.5 + 180) % 360 - 180,(8.75 +2.5 + 180) % 360 - 180],[min_lat_tab[loc],min_lat_tab[loc]], linewidth=2, color='lime')
ax.plot([(6.25 +2.5 + 180) % 360 - 180,(8.75 +2.5 + 180) % 360 - 180],[max_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
     
plt.subplots_adjust(top=0.985,bottom=0.015,left=0.005,right=0.925,hspace=0.0,wspace=0.01)
plt.show()

#%% Figure for modelling animation presentation: variance for Paris

plt.rc('font',family='serif',size=20) 

v_tab = ["t2m","z500","mrsos","slp","t850","v250"]
names_v_tab = ["T2M","Z500","SM","SLP","T850","V250"]
loc = 1
panels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)','(m)','(n)','(o)','(p)','(q)','(r)']

fig = plt.figure(figsize=(20,30))
plt.tight_layout()
gs = fig.add_gridspec(2,3)


for idv, v in enumerate(v_tab):
    print(v)
    
    # Load climatology 
    _, climato_var = select_climato(v) 
    climato_var_NA = extract_coordinate_north_atlantic(climato_var)

    idr = 1
    r = rolling_periods_tab[idr]
        
    ax = fig.add_subplot(gs[idv//3,idv%3])
                
    for idl,level in enumerate(level_tab):
            
        # NORTH ATLANTIC
        mean_variance_NA = []
        for idj,j in enumerate(j_list):
            temp = xr.load_dataarray(path_results+name_directory_results[loc]+"data_closest/"+v+"_r"+str(r)+"_q"+str(level)+"_day"+str(j)+"_variance.nc", use_cftime=True)
            temp = extract_coordinate_north_atlantic(temp)/climato_var_NA #%%
            weights = np.cos(np.deg2rad(temp.lat))
            weights.name = "weights"                
            mean_variance_NA.append(temp.weighted(weights).mean(("lon","lat")).values)
            
        ax.plot(j_list, np.array(mean_variance_NA)*100, color=colors_c[idl],label=r"$\alpha = $"+str(level))
        ax.scatter(j_list, np.array(mean_variance_NA)*100, s=4, color=colors_c[idl])

    
    ax.set_title(panels[idv]+" "+names_v_tab[idv]+" r="+str(r)+ " days", size=18)
    ax.plot([0,0],[0,2*100],color='black',linestyle='dashed',alpha=0.4)
    ax.plot([10,10],[0,2*100],color='black',linestyle='dashed',alpha=0.4)
    ax.plot([-10,-10],[0,2*100],color='black',linestyle='dashed',alpha=0.4)
    ax.plot([-15,15],[0.75*100,0.75*100],color='black',linestyle='dashed',alpha=0.4)
    
    if idv>2:
        ax.set_xlabel("Relative days")
    else:
        ax.set_xticklabels("")
    if idv==0 or idv==3:
        ax.set_ylabel(r"$<\tilde{V}>$")
    else:
        ax.set_yticklabels("")
        
    ax.set_xlim(j_list[0],j_list[-1])
                
    ax.set_ylim(0.4*100,1.1*100)
            
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower left',ncol=4,frameon=False, bbox_to_anchor=[0.2, 0])
plt.subplots_adjust(top=0.965,bottom=0.140,left=0.055,right=0.990,hspace=0.305,wspace=0.200)
plt.show()



#%% Investigation cut-off

plt.rc('font',family='serif',size=20) 

v = "t2m"
loc = 1
idr = 1
r = rolling_periods_tab[idr]

j_list_adapted = range(-1//2+1,1//2+1)
field_levels_z500 = select_field_group_rolling("z500", j_list_adapted, level_tab, closest_neighbors_list[loc][idr])

#%%

level = 3

def find_isolated_minimum(f, min_lon, max_lon, min_lat, max_lat):
    # Extract regions
    temp = f.copy()
    temp = temp.sel(lon=temp.lon[(temp.lon >= min_lon-40) & (temp.lon <= max_lon)])
    temp = temp.sel(lat=temp.lat[(temp.lat >= min_lat-20) & (temp.lat <= max_lat+5)])
    tem = temp.values
    
    # Result
    result = []
    nlat, nlon = tem.shape
    for i in range(1,nlat-1):
        for j in range(1,nlon-1):
            if tem[i,j] < min([tem[i+1,j+1],tem[i+1,j],tem[i+1,j-1],tem[i,j+1],tem[i,j-1],tem[i-1,j+1],tem[i-1,j],tem[i-1,j-1]]):
                result.append((temp.lat[i],temp.lon[j]))
    return np.array(result)
    
       
###### NORTH ATLANTIC #####

# Adapt fields
field_levels_prepared_z500 = []
for f in field_levels_z500:
    temp = f[0].mean("time")
    for i in f[1:]:
        temp = xr.concat([temp, i.mean("time")], dim='time')
    field_levels_prepared_z500.append(extract_coordinate_north_atlantic(temp))
    
yticks = [25, 45, 65]
xticks = [0, 30, 60, -30, -60]
xticklabels = ['0°', '30°E', '60°E', '30°W', '60°W']
yticklabels = ['25°N', '45°N', '65°N']
    
        
lon = field_levels_prepared_z500[0]['lon']
lat = field_levels_prepared_z500[0]['lat']

levels = np.arange(-7,8,1)
number_cut_offs = 0
    
#
for idf,f in enumerate(field_levels_prepared_z500[level]):
    c = find_isolated_minimum(f, (min_lon_tab[loc] + 180) % 360 - 180, (max_lon_tab[loc] + 180) % 360 - 180, min_lat_tab[loc], max_lat_tab[loc])
    if c.size>0:
        number_cut_offs += 1
    
    fig = plt.figure(figsize=(13.5,5))
    gs = fig.add_gridspec(1,1)
    
    ax = fig.add_subplot(gs[0,0], projection=ccrs.PlateCarree())
    cl = ax.contour(lon, lat, (f.values), extend='both', levels=np.arange(5500,6000,25), colors='black')
    ax.contourf(lon, lat, (f.values), extend='both', levels=np.arange(5500,6000,25), cmap='RdBu_r')
    #if c.size>0:
    #    ax.scatter(c[:,1],c[:,0], s=100, color='lime')
    #    ax.set_title("True "+str(len(c)))
    ax.clabel(cl, cl.levels, inline=True, fontsize=15) 
        #cp = ax.contourf(lon, lat, (field_levels_prepared[idlevel].mean('time').values-climato_mean.values), extend='both', levels=levels, cmap='RdBu_r')
        #plt.colorbar(cp, label="[°C]")
    ax.coastlines('50m', color='0', linewidth=.4)    
    ax.gridlines(draw_labels = False, xlocs=xticks, ylocs=yticks, color='.7', alpha=0.4, linewidth=.3)
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
            
    # Box for the observable
    ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(min_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    ax.plot([(max_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],min_lat_tab[loc]], linewidth=2, color='lime')
    ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[max_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
     
    plt.subplots_adjust(top=0.985,bottom=0.015,left=0.035,right=0.925,hspace=0.0,wspace=0.01)
    plt.show()
    
print(number_cut_offs, len(field_levels_prepared_z500[level]), number_cut_offs/len(field_levels_prepared_z500[level])*100)



#%% cut-off using circulation

plt.rc('font',family='serif',size=20) 

loc = 3
idr = 2
r = rolling_periods_tab[idr]

j_list_adapted = range(-1//2+1,1//2+1)
field_levels_t2m = select_field_group_rolling("t2m", j_list_adapted, level_tab, closest_neighbors_list[loc][idr])
field_levels_u500 = select_field_group_rolling("u500", j_list_adapted, level_tab, closest_neighbors_list[loc][idr])
field_levels_v500 = select_field_group_rolling("v500", j_list_adapted, level_tab, closest_neighbors_list[loc][idr])
field_levels_u250 = select_field_group_rolling("u250", j_list_adapted, level_tab, closest_neighbors_list[loc][idr])
field_levels_v250 = select_field_group_rolling("v250", j_list_adapted, level_tab, closest_neighbors_list[loc][idr])

# Load climatology 
climato_mean_t2m, _ = select_climato("t2m") 
climato_mean_t2m = extract_coordinate_north_atlantic(climato_mean_t2m)

#%%

level = 3

# Adapt fields
field_levels_prepared_t2m = []
for f in field_levels_t2m:
    temp = f[0].mean("time")
    for i in f[1:]:
        temp = xr.concat([temp, i.mean("time")], dim='time')
    field_levels_prepared_t2m.append(extract_coordinate_north_atlantic(temp))
    
field_levels_prepared_u500 = []
for f in field_levels_u500:
    temp = f[0].mean("time")
    for i in f[1:]:
        temp = xr.concat([temp, i.mean("time")], dim='time')
    field_levels_prepared_u500.append(temp)
    
field_levels_prepared_v500 = []
for f in field_levels_v500:
    temp = f[0].mean("time")
    for i in f[1:]:
        temp = xr.concat([temp, i.mean("time")], dim='time')
    field_levels_prepared_v500.append(temp)
    
field_levels_prepared_u250 = []
for f in field_levels_u250:
    temp = f[0].mean("time")
    for i in f[1:]:
        temp = xr.concat([temp, i.mean("time")], dim='time')
    field_levels_prepared_u250.append(temp)
    
field_levels_prepared_v250 = []
for f in field_levels_v250:
    temp = f[0].mean("time")
    for i in f[1:]:
        temp = xr.concat([temp, i.mean("time")], dim='time')
    field_levels_prepared_v250.append(temp)

    
yticks = [25, 45, 65]
xticks = [0, 30, 60, -30, -60]
xticklabels = ['0°', '30°E', '60°E', '30°W', '60°W']
yticklabels = ['25°N', '45°N', '65°N']
    
        
lon = field_levels_prepared_t2m[0]['lon']
lat = field_levels_prepared_t2m[0]['lat']

levels = np.arange(-7,8,1)

field_t2m = field_levels_prepared_t2m[level].sortby('time')
field_u500 = field_levels_prepared_u500[level].sortby('time')
field_v500 = field_levels_prepared_v500[level].sortby('time')
field_u250 = field_levels_prepared_u250[level].sortby('time')
field_v250 = field_levels_prepared_v250[level].sortby('time')

#
for idf,f in enumerate(field_levels_prepared_t2m[level]):

    # North-Atlantic
    fig = plt.figure(figsize=(18,9))
    plt.subplots_adjust(top=0.985,bottom=0.01,left=0.005,right=0.98,hspace=0.12,wspace=0.05)    
    gs = fig.add_gridspec(1,1)
        
    temp_u = field_u500[idf]
    temp_u.coords['lon'] = (temp_u.coords['lon'] + 180) % 360 - 180
    temp_u = temp_u.sortby(temp_u.lon)
    temp_u = temp_u.sel(lat=slice(22.5,80),lon=slice(-100,70))
    temp_v = field_v500[idf]
    temp_v.coords['lon'] = (temp_v.coords['lon'] + 180) % 360 - 180
    temp_v = temp_v.sortby(temp_v.lon)
    temp_v = temp_v.sel(lat=slice(22.5,80),lon=slice(-100,70))
    
    temp_u250 = field_u250[idf]
    temp_u250.coords['lon'] = (temp_u250.coords['lon'] + 180) % 360 - 180
    temp_u250 = temp_u250.sortby(temp_u250.lon)
    temp_u250 = temp_u250.sel(lat=slice(22.5,80),lon=slice(-100,70))
    temp_v250 = field_v250[idf]
    temp_v250.coords['lon'] = (temp_v250.coords['lon'] + 180) % 360 - 180
    temp_v250 = temp_v250.sortby(temp_v250.lon)
    temp_v250 = temp_v250.sel(lat=slice(22.5,80),lon=slice(-100,70))
    
    lon = temp_u['lon']
    lat = temp_u['lat']
    
    ax = fig.add_subplot(gs[0,0], projection=ccrs.PlateCarree())
    
    strm = ax.streamplot(lon, lat, temp_u.values, temp_v.values, density=2., arrowstyle ='->', arrowsize=2, broken_streamlines=False, transform=ccrs.PlateCarree(), linewidth=1, color=np.sqrt(temp_u.values**2+temp_v.values**2), cmap=plt.cm.get_cmap('Blues', 10), norm=colors.Normalize(vmin=0,vmax=20))        
    
    cf = ax.contourf(f['lon'], f['lat'], (f.values-climato_mean_t2m.values), extend='both', levels=np.arange(-12,13,1), cmap='coolwarm', transform=ccrs.PlateCarree())
    plt.colorbar(cf, label="STD", fraction=0.046, pad=0.02)
    
    mask_jet = np.sqrt(temp_u250.values**2+temp_v250.values**2) > 25
    ax.contourf(temp_u250['lon'], temp_u250['lat'], mask_jet, levels=2, hatches=[None,"//","//"], colors="none")

    ax.coastlines()
    ax.set_ylim(30,70)
    ax.set_xlim(-80,50)
    
            
    # Box for the observable
    ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(min_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    ax.plot([(max_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[min_lat_tab[loc],min_lat_tab[loc]], linewidth=2, color='lime')
    ax.plot([(min_lon_tab[loc] + 180) % 360 - 180,(max_lon_tab[loc] + 180) % 360 - 180],[max_lat_tab[loc],max_lat_tab[loc]], linewidth=2, color='lime')
    
        
    plt.show()
    

#%% Investigation cut-off bi-modality Z500

level = 3
loc = 1
idr = 1
r = rolling_periods_tab[idr]
j_list_adapted = range(-r//2+1,r//2+1)

field_levels_z500 = select_field_group_rolling("z500", j_list_adapted, level_tab, closest_neighbors_list[loc][idr])[level]

z500_tab1 = []
z500_tab2 = []

for f in field_levels_z500:
    var = f.sel(lat=slice(44,45),lon=slice(0,1)).mean()
    z500_tab1.append(var.values)
    var = f.sel(lat=slice(44,45),lon=slice(344,346)).mean()
    z500_tab2.append(var.values)
    
fig = plt.figure()
plt.scatter(z500_tab1,z500_tab1)
plt.scatter(z500_tab2,z500_tab2)
plt.show()


#%% Making videos

import cv2

def prepare_fields(field_levels):
    field_levels_prepared = []
    for f in field_levels:
        temp = []
        for g in f:
            temp.append(add_cyclic_point(g, coord=g['lon'])[0])
        field_levels_prepared.append(temp)
    return field_levels_prepared

plt.rc('font',family='serif',size=20) 

v_tab = ["t2m","z500","mrsos","slp","t850","v250"]
panels = ['(a)','(b)','(c)','(d)','(e)','(f)']
color_maps=["RdBu_r","PuOr_r",'BrBG',"PuOr_r","RdBu_r","PuOr_r"]

observables_name = ["S","W","N","WCE"]

yticks = [25, 45, 65]
xticks = [0, 30, 60, 90, 120, 150, 180, -30, -60, -90, -120, -150]

j_list = range(-10,11)

for loc in range(4):
    print("loc", loc)

    for idr in range(3):
        print("r", rolling_periods_tab[idr])
        
        r = rolling_periods_tab[idr]
        idlevel = 3
        
        field_levels_var = [prepare_fields(select_field_group(v, j_list, level_tab, closest_neighbors_list[loc][idr]))[idlevel] for v in v_tab]
        
        plt.ioff()
        for idj,j in enumerate(j_list):    
            plt.rc('font',family='serif',size=20) 
            fig = plt.figure(figsize=(15,15))
            gs = fig.add_gridspec((len(v_tab)-1)//2+1,2)
        
            for idv,v in enumerate(v_tab):
                field = field_levels_var[idv][idj]
                climato_mean, climato_var = select_climato(v, r=1) 
                lat = climato_mean['lat']
                climato_mean, lon = add_cyclic_point(climato_mean, coord=climato_mean['lon'])
                climato_var = add_cyclic_point(climato_var, coord=climato_var['lon'])[0]
                
                if v=="slp": # anomaly
                    levels = np.arange(-5,6,1)
                    unit = "Anomaly of SLP [hPa]"
                elif v=="t850": # no anomaly
                    levels = [-7.5,-5.5,-3.5, -1.5, -0.75, 0, 0.75, 1.5, 3.5, 5.5, 7.5]
                    unit = "Anomaly of T850 [°C]"
                elif v=="v250": # no anomaly
                    levels = np.arange(-15,18,3)
                    unit = "V250 [m/s]"
                elif v=="mrsos":
                    levels = np.arange(-5,6,1)
                    unit = "Anomaly of SM [kg/m²]"
                elif v=="t2m":
                    levels = np.arange(-7,8,1)
                    unit = "Anomaly of T2M [°C]"
                elif v=="z500":
                    levels = np.arange(-100,120,20)
                    unit = 'Anomaly of Z500 [m]'
                
                # Mean
                ax = fig.add_subplot(gs[idv//2,idv%2], projection=ccrs.NorthPolarStereo(central_longitude=0))
                if v=="slp" or v=="mrsos" or v=="t2m" or v=="z500" or v=="t850":
                    cp = ax.contourf(lon, lat, np.mean(field - climato_mean, axis=0), extend='both', levels=levels, cmap=color_maps[idv], transform=ccrs.PlateCarree())
                else: 
                    cp = ax.contourf(lon, lat, np.mean(field, axis=0), extend='both', levels=levels, cmap=color_maps[idv], transform=ccrs.PlateCarree())
                plt.colorbar(cp, label=unit)
                
                mask = np.var(field,axis=0)/climato_var*100 > 70
                cl = ax.contourf(lon, lat, mask, levels=2, hatches=[None,"///","///"], colors="none", transform=ccrs.PlateCarree())
                
                ax.coastlines('50m', color='0', linewidth=.4)    
                ax.gridlines(draw_labels = False, xlocs=xticks, ylocs=yticks, color='.7', alpha=0.4, linewidth=.3)
                
                ax.set_extent([-180, 180, lat[0], lat[-1]], ccrs.PlateCarree())
                # Compute a circle in axes coordinates, which we can use as a boundary
                # for the map. We can pan/zoom as much as we like - the boundary will be
                # permanently circular.
                theta = np.linspace(0, 2*np.pi, 100)
                center, radius = [0.5, 0.5], 0.5
                verts = np.vstack([np.sin(theta), np.cos(theta)]).T
                circle = mpath.Path(verts * radius + center)
                ax.set_boundary(circle, transform=ax.transAxes)
            
                ax.set_title(panels[idv])
            
                # Box for the observable
                ax.plot([min_lon_tab[loc],min_lon_tab[loc]],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
                ax.plot([max_lon_tab[loc],max_lon_tab[loc]],[min_lat_tab[loc],max_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
                ax.plot([min_lon_tab[loc],max_lon_tab[loc]],[min_lat_tab[loc],min_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
                ax.plot([min_lon_tab[loc],max_lon_tab[loc]],[max_lat_tab[loc],max_lat_tab[loc]], linewidth=3, color='lime', transform=ccrs.PlateCarree())
            
            fig.text(0.025, 0.95, r"$t=$" + str(j), bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                
            plt.subplots_adjust(top=0.970,bottom=0.005,left=0.035,right=0.930,hspace=0.095,wspace=0.095)
            plt.savefig(path_results + "Figures/videos/" + "loc" + str(loc) + "_r" + str(r) + "_j" + str(j) + ".png")
            plt.close()
        plt.ion()
        
        
        frameSize = (1500, 1500)
        out = cv2.VideoWriter(path_results + "Figures/videos/observable_" + observables_name[loc] + "_rolling_mean_" + str(r) + "_days.avi",cv2.VideoWriter_fourcc(*'DIVX'), 1, frameSize)
        
        for idj,j in enumerate(j_list):
            print(j)
            img = cv2.imread(path_results + "Figures/videos/" + "loc" + str(loc) + "_r" + str(r) + "_j" + str(j) + ".png")
            os.remove(path_results + "Figures/videos/" + "loc" + str(loc) + "_r" + str(r) + "_j" + str(j) + ".png")
            out.write(img)
        
        out.release()
    
    
    
    
    
    
    
    
    
    
    
    
    










    


