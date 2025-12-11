
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from scipy.stats import ttest_ind, ks_2samp
import scipy.stats
import sys, os
from datetime import timedelta
from pathlib import Path
import matplotlib.animation as animation
import matplotlib.colors as colors


path_data = "./synthetic_data.nc"
path_results = "./outputs/"
name_model = "_day_IPSL-CM6A-LR_piControl_r1i1p1f1_gr_"
start_date = "18500101"
end_date = "38491231"

plt.ioff()
#plt.ion()

############### FUNCTIONS ############

# Selection functions

def extract_time_series_observable(min_lat, min_lon, max_lat, max_lon): 
    
    # result = xr.open_dataset(path_data+"tas"+name_model+start_date+"-"+end_date+".nc", use_cftime=True)
    # result = result.sel(time=(result['time.month']>=5) & (result['time.month']<=9),lat=slice(min_lat,max_lat),lon=slice(min_lon,max_lon))['tas'].mean(dim=['lat','lon'])
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    return xr.open_dataset(path_data, decode_times=time_coder)["t2m"]

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


def select_climato(v):
    if v=="slp":
        climato_mean = xr.open_dataset(path_data+"psl"+name_model+start_date+"-"+end_date+"_mean.nc", use_cftime=True)['psl'][0,:,:]/100
        climato_var = xr.open_dataset(path_data+"psl"+name_model+start_date+"-"+end_date+"_variance.nc", use_cftime=True)['psl'][0,:,:]/10000
    elif v=="t2m":
        climato_mean = xr.open_dataset(path_data+"tas"+name_model+start_date+"-"+end_date+"_mean.nc", use_cftime=True)['tas'][0,:,:]-273.15
        climato_var = xr.open_dataset(path_data+"tas"+name_model+start_date+"-"+end_date+"_variance.nc", use_cftime=True)['tas'][0,:,:]
    elif v=="t850":
        climato_mean = xr.open_dataset(path_data+"ta850"+name_model+start_date+"-"+end_date+"_mean.nc", use_cftime=True)['ta'][0,0,:,:]-273.15
        climato_var = xr.open_dataset(path_data+"ta850"+name_model+start_date+"-"+end_date+"_variance.nc", use_cftime=True)['ta'][0,0,:,:]
    elif v=="v250":
        climato_mean = xr.open_dataset(path_data+"va250"+name_model+start_date+"-"+end_date+"_mean.nc", use_cftime=True)['va'][0,0,:,:]
        climato_var = xr.open_dataset(path_data+"va250"+name_model+start_date+"-"+end_date+"_variance.nc", use_cftime=True)['va'][0,0,:,:]
    elif v=="u250":
        climato_mean = xr.open_dataset(path_data+"ua250"+name_model+start_date+"-"+end_date+"_mean.nc", use_cftime=True)['ua'][0,0,:,:]
        climato_var = xr.open_dataset(path_data+"ua250"+name_model+start_date+"-"+end_date+"_variance.nc", use_cftime=True)['ua'][0,0,:,:]
    elif v=="v500":
        climato_mean = xr.open_dataset(path_data+"va500"+name_model+start_date+"-"+end_date+"_mean.nc", use_cftime=True)['va'][0,0,:,:]
        climato_var = xr.open_dataset(path_data+"va500"+name_model+start_date+"-"+end_date+"_variance.nc", use_cftime=True)['va'][0,0,:,:]
    elif v=="u500":
        climato_mean = xr.open_dataset(path_data+"ua500"+name_model+start_date+"-"+end_date+"_mean.nc", use_cftime=True)['ua'][0,0,:,:]
        climato_var = xr.open_dataset(path_data+"ua500"+name_model+start_date+"-"+end_date+"_variance.nc", use_cftime=True)['ua'][0,0,:,:]
    elif v=="z500":
        climato_mean = xr.open_dataset(path_data+"z500"+name_model+start_date+"-"+end_date+"_mean.nc", use_cftime=True)['zg'][0,0,:,:]
        climato_var = xr.open_dataset(path_data+"z500"+name_model+start_date+"-"+end_date+"_variance.nc", use_cftime=True)['zg'][0,0,:,:]
    if v=="mrsos":
        climato_mean = xr.open_dataset(path_data+"mrsos"+name_model+start_date+"-"+end_date+"_mean.nc", use_cftime=True)['mrsos'][0,:,:]
        climato_var = xr.open_dataset(path_data+"mrsos"+name_model+start_date+"-"+end_date+"_variance.nc", use_cftime=True)['mrsos'][0,:,:]
    return climato_mean, climato_var
            

# Autocorrelation series

def compute_autocorrelation_series(series_obs_rolling, days=31):
    series_obs_rolling_deseasonalized = series_obs_rolling.groupby('time.dayofyear') - series_obs_rolling.groupby('time.dayofyear').mean()
    auto_corr_tab = np.zeros(days)
    for i in range(days):
        auto_corr_tab[i] = xr.corr(series_obs_rolling_deseasonalized, series_obs_rolling_deseasonalized.roll(time=i))
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


# Saving functions

def save_dates(temp, r, name_directory_results, level_tab):
    Path(path_results+name_directory_results+"data_closest/").mkdir(parents=True, exist_ok=True)
    for idi,i in enumerate(temp):
        i.to_netcdf(path_results+name_directory_results+"data_closest/closest_neighbors_obs_r"+str(r)+"_q"+str(level_tab[idi])+".nc")


def save_fields(m, var, v, r, j, level, name_directory_results):
    Path(path_results+name_directory_results+"data_closest/").mkdir(parents=True, exist_ok=True)
    m.to_netcdf(path_results+name_directory_results+"data_closest/"+v+"_r"+str(r)+"_q"+str(level)+"_day"+str(j)+"_mean.nc")
    var.to_netcdf(path_results+name_directory_results+"data_closest/"+v+"_r"+str(r)+"_q"+str(level)+"_day"+str(j)+"_variance.nc")
    

# Significance functions


    
# Plotting functions

def plot_autocorrelation(auto_corr_series, rolling_periods_tab, name_directory_results):
    fig = plt.figure(figsize=(20,10))
    plt.tight_layout()
    
    for idi, i in enumerate(rolling_periods_tab):
        plt.plot(np.arange(0,31,1),auto_corr_series[idi],label="rolling period = "+str(rolling_periods_tab[idi]))
        
    plt.legend()
    plt.xlabel("Days")
    plt.ylabel("Autocorrelation")
    plt.xlim(0,30)
    plt.ylim(0,1)
    
    plt.savefig(path_results+name_directory_results+"autocorrelation.png")
    plt.close()


def plot_histograms(closest_neighbors_list, quantile_tab, rolling_periods_tab, series_obs, level_tab, name_directory_results):
    colors = ['midnightblue','blue','magenta', 'green', 'orange', 'red']
    dec = [0,0.01,0.02,0.03,0.04,0.05]
    
    fig = plt.figure(figsize=(20,10))
    plt.tight_layout()
    gs = fig.add_gridspec((len(closest_neighbors_list)-1)//3+1,3)
    
    for i in range(len(closest_neighbors_list)):
        series_obs_rolling = series_obs.rolling(time=rolling_periods_tab[i], center=True).mean().sel(time=(series_obs['time.month']>=6) & (series_obs['time.month']<=8))

        ax = fig.add_subplot(gs[i//3,i%3])
        
        ax.hist(series_obs_rolling-273.15, bins=100, histtype='step', density=True)
        
        for idj,j in enumerate(level_tab):
            ax.plot([quantile_tab[i,idj]-273.15, quantile_tab[i,idj]-273.15],[0,0.15],color=colors[idj],label="q = "+str(j))
            ax.plot([closest_neighbors_list[i][idj].min()-273.15,closest_neighbors_list[i][idj].max()-273.15],[0.05+dec[idj],0.05+dec[idj]],color=colors[idj],linewidth=2)
            
        
        ax.set_title("r = "+str(rolling_periods_tab[i])+" days")
        ax.set_xlabel("Temperature [°C]")
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_xlim(10,35)
        ax.set_ylim(0,0.25)
        
    plt.savefig(path_results+name_directory_results+"histograms.png")
    plt.close()


def plot_histograms_dates(closest_neighbors_list, rolling_periods_tab, level_tab, name_directory_results):
    colors = ['midnightblue','blue','magenta', 'green', 'orange', 'red']
    
    fig = plt.figure(figsize=(20,10))
    plt.tight_layout()
    gs = fig.add_gridspec((len(closest_neighbors_list)-1)//3+1,3)
    
    for i in range(len(closest_neighbors_list)):

        ax = fig.add_subplot(gs[i//3,i%3])
                
        for idj,j in enumerate(level_tab):
            if idj%2 == 0:
                temp = np.histogram(closest_neighbors_list[i][idj].time.dt.dayofyear, bins=np.arange(152,248,5)-0.5)[0]
                ax.plot(np.arange(152,243,5), temp, color=colors[idj],label="q = "+str(j))
            #ax.hist(closest_neighbors_list[i][idj].time.dt.dayofyear, bins=np.arange(152,245,5)-0.5, color=colors[idj],label="q = "+str(j))
        
        ax.set_title("r = "+str(rolling_periods_tab[i])+" days")
        ax.set_xlabel("Calendar days")
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_xlim(152,242)
        ax.set_ylim(0,15)
        
    plt.savefig(path_results+name_directory_results+"histograms_dates.png")
    plt.close()

    
def extract_coordinate_north_atlantic(f):
    result = f.copy()
    result.coords['lon'] = (result.coords['lon'] + 180) % 360 - 180
    result = result.sel(lon=result.lon[(result.lon >= ((280+180)%360-180)) & (result.lon <= ((50+180)%360-180))])
    result = result.sortby(result.lon)
    return result.sel(lat=slice(22.5,70),lon=slice(-80,50))


def plot_north_atlantic(field_levels, level_tab, v, r, j, idj, name_directory_results, min_lon, min_lat, max_lon, max_lat):
    
    # Create directory for results if it does not exists
    Path(path_results+name_directory_results+v+"_r"+str(r)+"/").mkdir(parents=True, exist_ok=True)
    
    # Load climatology 
    climato_mean, climato_var = select_climato(v) 
    climato_mean = extract_coordinate_north_atlantic(climato_mean)
    climato_var = extract_coordinate_north_atlantic(climato_var)

    # Adapt fields
    field_levels_prepared = []
    for f in field_levels:
        field_levels_prepared.append(extract_coordinate_north_atlantic(f[idj]))
    
    # Plotting mean
    fig = plt.figure(figsize=(20,10))
    plt.subplots_adjust(top=0.955,bottom=0.01,left=0.0,right=1.0,hspace=0.17,wspace=0.0)
    gs = fig.add_gridspec((len(level_tab)-1)//2+1,2)
    
        # Welch's t-test
    
    lon = field_levels_prepared[0]['lon']
    lat = field_levels_prepared[0]['lat']
    
    if v=="slp":
        levels = np.arange(995,1026,1)
        unit ="hPa"
    elif v=="t2m":
        levels = np.arange(10,32,2)
        unit = "°C"
    elif v=="t850":
        levels = np.arange(0,22,2)
        unit = "°C"
    elif v=='v250':
        levels = np.arange(-16,18,2)
        unit = "m/s"
    elif v=='u250':
        levels = np.arange(-20,30,4)
        unit = "m/s"
    elif v=='v500':
        levels = np.arange(-10,17,2)
        unit = "m/s"
    elif v=='u500':
        levels = np.arange(-15,16,2)
        unit = "m/s"
    elif v=="z500":
        levels = np.arange(5400,5925,25)
        unit = 'm'
    elif v=="mrsos":
        levels = np.arange(5,25,1)
        unit = "kg/m²"

    for idlevel, level in enumerate(level_tab):

        ax = fig.add_subplot(gs[idlevel//2,idlevel%2], projection=ccrs.PlateCarree())
        cp = ax.contourf(lon, lat, field_levels_prepared[idlevel].mean('time').values, extend='both', levels=levels, cmap='coolwarm')
        plt.colorbar(cp, label="["+unit+"]")
        ax.coastlines()
        ax.set_title(str(v)+" r="+str(r)+" q="+str(level)+" j="+str(j))
        
        # Box for the observable
        ax.plot([(min_lon + 180) % 360 - 180,(min_lon + 180) % 360 - 180],[min_lat,max_lat], linewidth=2, color='lime')
        ax.plot([(max_lon + 180) % 360 - 180,(max_lon + 180) % 360 - 180],[min_lat,max_lat], linewidth=2, color='lime')
        ax.plot([(min_lon + 180) % 360 - 180,(max_lon + 180) % 360 - 180],[min_lat,min_lat], linewidth=2, color='lime')
        ax.plot([(min_lon + 180) % 360 - 180,(max_lon + 180) % 360 - 180],[max_lat,max_lat], linewidth=2, color='lime')

    plt.savefig(path_results+name_directory_results+v+"_r"+str(r)+"/north-atlantic_j"+str(j)+"_mean.png")
    plt.close()


    # Plotting mean anomalies
    fig = plt.figure(figsize=(20,10))
    plt.subplots_adjust(top=0.955,bottom=0.01,left=0.0,right=1.0,hspace=0.17,wspace=0.0)
    gs = fig.add_gridspec((len(level_tab)-1)//2+1,2)
    
        # Welch's t-test
    
    lon = field_levels_prepared[0]['lon']
    lat = field_levels_prepared[0]['lat']
    
    levels = np.arange(-1,1.1,0.1)

    for idlevel, level in enumerate(level_tab):

        ax = fig.add_subplot(gs[idlevel//2,idlevel%2], projection=ccrs.PlateCarree())
        cp = ax.contourf(lon, lat, (field_levels_prepared[idlevel].mean('time').values-climato_mean.values)/np.sqrt(climato_var.values), extend='both', levels=levels, cmap='coolwarm')
        plt.colorbar(cp, label="[STD]")
        ax.coastlines()
        ax.set_title(str(v)+" r="+str(r)+" q="+str(level)+" j="+str(j))
        
        # Box for the observable
        ax.plot([(min_lon + 180) % 360 - 180,(min_lon + 180) % 360 - 180],[min_lat,max_lat], linewidth=2, color='lime')
        ax.plot([(max_lon + 180) % 360 - 180,(max_lon + 180) % 360 - 180],[min_lat,max_lat], linewidth=2, color='lime')
        ax.plot([(min_lon + 180) % 360 - 180,(max_lon + 180) % 360 - 180],[min_lat,min_lat], linewidth=2, color='lime')
        ax.plot([(min_lon + 180) % 360 - 180,(max_lon + 180) % 360 - 180],[max_lat,max_lat], linewidth=2, color='lime')

    plt.savefig(path_results+name_directory_results+v+"_r"+str(r)+"/north-atlantic_j"+str(j)+"_mean_anomaly.png")
    plt.close()
    
    # Plotting variance
    fig = plt.figure(figsize=(20,10))
    plt.subplots_adjust(top=0.955,bottom=0.01,left=0.0,right=1.0,hspace=0.17,wspace=0.0)
    gs = fig.add_gridspec((len(level_tab)-1)//2+1,2)
    
        # F test
    
    lon = field_levels_prepared[0]['lon']
    lat = field_levels_prepared[0]['lat'] 

    for idlevel, level in enumerate(level_tab):

        ax = fig.add_subplot(gs[idlevel//2,idlevel%2], projection=ccrs.PlateCarree())
        cp = ax.contourf(lon, lat, field_levels_prepared[idlevel].var('time').values/climato_var.values*100, extend='both', levels=np.arange(30,130,10), cmap='coolwarm')
        plt.colorbar(cp, label="%")
        ax.coastlines()
        ax.set_title(str(v)+" r="+str(r)+" q="+str(level)+" j="+str(j))
        
        # Box for the observable
        ax.plot([(min_lon + 180) % 360 - 180,(min_lon + 180) % 360 - 180],[min_lat,max_lat], linewidth=2, color='lime')
        ax.plot([(max_lon + 180) % 360 - 180,(max_lon + 180) % 360 - 180],[min_lat,max_lat], linewidth=2, color='lime')
        ax.plot([(min_lon + 180) % 360 - 180,(max_lon + 180) % 360 - 180],[min_lat,min_lat], linewidth=2, color='lime')
        ax.plot([(min_lon + 180) % 360 - 180,(max_lon + 180) % 360 - 180],[max_lat,max_lat], linewidth=2, color='lime')
        
    
    plt.savefig(path_results+name_directory_results+v+"_r"+str(r)+"/north-atlantic_j"+str(j)+"_variance.png")
    plt.close()
    
    
    
def plot_northern_hemisphere(field_levels, level_tab, v, r, j, idj, name_directory_results, min_lon, min_lat, max_lon, max_lat):
    
    # Create directory for results if it does not exists
    Path(path_results+name_directory_results+v+"_r"+str(r)+"/").mkdir(parents=True, exist_ok=True)
    
    # Load climatology 
    climato_mean, climato_var = select_climato(v) 
    lat = climato_mean['lat']
    climato_mean, lon = add_cyclic_point(climato_mean, coord=climato_mean['lon'])
    climato_var = add_cyclic_point(climato_var, coord=climato_var['lon'])[0]

    # Adapt fields
    field_levels_prepared = []
    for f in field_levels:
        field_levels_prepared.append(add_cyclic_point(f[idj].copy(), coord=f[idj]['lon'])[0])
    
    # Plotting mean
    fig = plt.figure(figsize=(20,10))
    plt.subplots_adjust(top=0.965,bottom=0.005,left=0.06,right=0.955,hspace=0.07,wspace=0.095)
    gs = fig.add_gridspec((len(level_tab)-1)//3+1,3)
    
        # Welch's t-test
    
    
    if v=="slp":
        levels = np.arange(995,1026,1)
        unit = "hPa"
    elif v=="t2m":
        levels = np.arange(10,32,2)
        unit = "°C"
    elif v=="t850":
        levels = np.arange(0,22,2)
        unit = "°C"
    elif v=='v250':
        levels = np.arange(-16,18,2)
        unit = "m/s"
    elif v=='u250':
        levels = np.arange(-20,30,4)
        unit = "m/s"
    elif v=='v500':
        levels = np.arange(-10,17,2)
        unit = "m/s"
    elif v=='u500':
        levels = np.arange(-15,16,2)
        unit = "m/s"
    elif v=="z500":
        levels = np.arange(5400,5925,25)
        unit = 'm'
    elif v=="mrsos":
        levels = np.arange(5,25,1)
        unit = "kg/m²"

    for idlevel, level in enumerate(level_tab):

        ax = fig.add_subplot(gs[idlevel//3,idlevel%3], projection=ccrs.NorthPolarStereo(central_longitude=0))
        cp = ax.contourf(lon, lat, np.mean(field_levels_prepared[idlevel], axis=0), extend='both', levels=levels, cmap='coolwarm', transform=ccrs.PlateCarree())
        plt.colorbar(cp, label="["+unit+"]")
        ax.coastlines()
        ax.set_title(str(v)+" r="+str(r)+" q="+str(level)+" j="+str(j))
        
        # Box for the observable
        ax.plot([min_lon,min_lon],[min_lat,max_lat], linewidth=3, color='lime', transform=ccrs.PlateCarree())
        ax.plot([max_lon,max_lon],[min_lat,max_lat], linewidth=3, color='lime', transform=ccrs.PlateCarree())
        ax.plot([min_lon,max_lon],[min_lat,min_lat], linewidth=3, color='lime', transform=ccrs.PlateCarree())
        ax.plot([min_lon,max_lon],[max_lat,max_lat], linewidth=3, color='lime', transform=ccrs.PlateCarree())

    plt.savefig(path_results+name_directory_results+v+"_r"+str(r)+"/north-hemisphere_j"+str(j)+"_mean.png")
    plt.close()
    
    # Plotting mean anomalies
    fig = plt.figure(figsize=(20,10))
    plt.subplots_adjust(top=0.965,bottom=0.005,left=0.06,right=0.955,hspace=0.07,wspace=0.095)
    gs = fig.add_gridspec((len(level_tab)-1)//3+1,3)
    
        # Welch's t-test
    
    
    levels = np.arange(-1,1.1,0.1)

    for idlevel, level in enumerate(level_tab):

        ax = fig.add_subplot(gs[idlevel//3,idlevel%3], projection=ccrs.NorthPolarStereo(central_longitude=0))
        cp = ax.contourf(lon, lat, (np.mean(field_levels_prepared[idlevel],axis=0)-climato_mean)/np.sqrt(climato_var), extend='both', levels=levels, cmap='coolwarm', transform=ccrs.PlateCarree())
        plt.colorbar(cp, label="[STD]")
        ax.coastlines()
        ax.set_title(str(v)+" r="+str(r)+" q="+str(level)+" j="+str(j))
        
        # Box for the observable
        ax.plot([min_lon,min_lon],[min_lat,max_lat], linewidth=3, color='lime', transform=ccrs.PlateCarree())
        ax.plot([max_lon,max_lon],[min_lat,max_lat], linewidth=3, color='lime', transform=ccrs.PlateCarree())
        ax.plot([min_lon,max_lon],[min_lat,min_lat], linewidth=3, color='lime', transform=ccrs.PlateCarree())
        ax.plot([min_lon,max_lon],[max_lat,max_lat], linewidth=3, color='lime', transform=ccrs.PlateCarree())

    plt.savefig(path_results+name_directory_results+v+"_r"+str(r)+"/north-hemisphere_j"+str(j)+"_mean_anomaly.png")
    plt.close()
    
    # Plotting variance
    fig = plt.figure(figsize=(20,10))
    plt.subplots_adjust(top=0.965,bottom=0.005,left=0.06,right=0.955,hspace=0.07,wspace=0.095)
    gs = fig.add_gridspec((len(level_tab)-1)//3+1,3)
    
        # F test
    

    for idlevel, level in enumerate(level_tab):

        ax = fig.add_subplot(gs[idlevel//3,idlevel%3], projection=ccrs.NorthPolarStereo(central_longitude=0))
        cp = ax.contourf(lon, lat, np.var(field_levels_prepared[idlevel],axis=0)/climato_var*100, extend='both', levels=np.arange(30,130,10), cmap='coolwarm', transform=ccrs.PlateCarree())
        plt.colorbar(cp, label="%")
        ax.coastlines()
        ax.set_title(str(v)+" r="+str(r)+" q="+str(level)+" j="+str(j))
        
        # Box for the observable
        ax.plot([min_lon,min_lon],[min_lat,max_lat], linewidth=3, color='lime', transform=ccrs.PlateCarree())
        ax.plot([max_lon,max_lon],[min_lat,max_lat], linewidth=3, color='lime', transform=ccrs.PlateCarree())
        ax.plot([min_lon,max_lon],[min_lat,min_lat], linewidth=3, color='lime', transform=ccrs.PlateCarree())
        ax.plot([min_lon,max_lon],[max_lat,max_lat], linewidth=3, color='lime', transform=ccrs.PlateCarree())
        
    
    plt.savefig(path_results+name_directory_results+v+"_r"+str(r)+"/north-hemisphere_j"+str(j)+"_variance.png")
    plt.close()
    

def create_gifs(v, r, j_list, name_directory_results):
    
    # North Atlantic
    # Mean
    fig, ax = plt.subplots(figsize=(20,10))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off')
    ims = []
    for idj,j in enumerate(j_list):
        im = ax.imshow(plt.imread(path_results+name_directory_results+v+"_r"+str(r)+"/north-atlantic_j"+str(j)+"_mean.png"), animated = True)
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=1000)
    ani.save(path_results+name_directory_results+v+"_r"+str(r)+"/gif_north-atlantic_mean.gif")
    plt.close()
    
    # Mean anomaly
    fig, ax = plt.subplots(figsize=(20,10))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off')
    ims = []
    for idj,j in enumerate(j_list):
        im = ax.imshow(plt.imread(path_results+name_directory_results+v+"_r"+str(r)+"/north-atlantic_j"+str(j)+"_mean_anomaly.png"), animated = True)
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=1000)
    ani.save(path_results+name_directory_results+v+"_r"+str(r)+"/gif_north-atlantic_mean_anomaly.gif")
    plt.close()
    
    # Variance
    fig, ax = plt.subplots(figsize=(20,10))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off')
    ims = []
    for idj,j in enumerate(j_list):
        im = ax.imshow(plt.imread(path_results+name_directory_results+v+"_r"+str(r)+"/north-atlantic_j"+str(j)+"_variance.png"), animated = True)
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=1000)
    ani.save(path_results+name_directory_results+v+"_r"+str(r)+"/gif_north-atlantic_variance.gif")
    plt.close()
    
    # Northern Hemisphere
    # Mean
    fig, ax = plt.subplots(figsize=(20,10))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off')
    ims = []
    for idj,j in enumerate(j_list):
        im = ax.imshow(plt.imread(path_results+name_directory_results+v+"_r"+str(r)+"/north-hemisphere_j"+str(j)+"_mean.png"), animated = True)
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=1000)
    ani.save(path_results+name_directory_results+v+"_r"+str(r)+"/gif_north-hemisphere_mean.gif")
    plt.close()
    
    # Mean anomaly
    fig, ax = plt.subplots(figsize=(20,10))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off')
    ims = []
    for idj,j in enumerate(j_list):
        im = ax.imshow(plt.imread(path_results+name_directory_results+v+"_r"+str(r)+"/north-hemisphere_j"+str(j)+"_mean_anomaly.png"), animated = True)
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=1000)
    ani.save(path_results+name_directory_results+v+"_r"+str(r)+"/gif_north-hemisphere_mean_anomaly.gif")
    plt.close()
    
    # Variance
    fig, ax = plt.subplots(figsize=(20,10))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off')
    ims = []
    for idj,j in enumerate(j_list):
        im = ax.imshow(plt.imread(path_results+name_directory_results+v+"_r"+str(r)+"/north-hemisphere_j"+str(j)+"_variance.png"), animated = True)
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=1000)
    ani.save(path_results+name_directory_results+v+"_r"+str(r)+"/gif_north-hemisphere_variance.gif")
    plt.close()
    
    
def variance_evolution(v, j_list, rolling_periods_tab, level_tab, name_directory_results):
    colors = ['midnightblue','blue','magenta', 'green', 'orange', 'red']
    
    # Load climatology 
    _, climato_var = select_climato(v) 
    climato_var_NA = extract_coordinate_north_atlantic(climato_var)
    
    # North-Atlantic
    fig = plt.figure(figsize=(20,10))
    plt.tight_layout()
    gs = fig.add_gridspec((len(rolling_periods_tab)-1)//3+1,3)
    
    for idr, r in enumerate(rolling_periods_tab):
        
        ax = fig.add_subplot(gs[idr//3,idr%3])
                
        for idl,level in enumerate(level_tab):
            
            mean_variance = []
            
            for idj,j in enumerate(j_list):
                temp = xr.load_dataarray(path_results+name_directory_results+"data_closest/"+v+"_r"+str(r)+"_q"+str(level)+"_day"+str(j)+"_variance.nc", use_cftime=True)
                temp = extract_coordinate_north_atlantic(temp)/climato_var_NA #%%
                weights = np.cos(np.deg2rad(temp.lat))
                weights.name = "weights"                
                mean_variance.append(temp.weighted(weights).mean(("lon","lat")).values)
                
            ax.plot(j_list, mean_variance, color=colors[idl],label="q = "+str(level))
            ax.scatter(j_list, mean_variance, s=4, color=colors[idl])

        ax.set_title("r = "+str(r)+" days")
        ax.set_xlabel("Days relative to maximum")
        ax.set_ylabel("North-Atlantic normalized variance")
        ax.legend()
        ax.set_xlim(j_list[0],j_list[-1])
        ax.set_ylim(0.3,1.2)
        
    plt.savefig(path_results+name_directory_results+v+"_NA_variance_evolution.png")
    plt.close()
    
    # North-Hemisphere
    fig = plt.figure(figsize=(20,10))
    plt.tight_layout()
    gs = fig.add_gridspec((len(rolling_periods_tab)-1)//3+1,3)
    
    for idr, r in enumerate(rolling_periods_tab):
        
        ax = fig.add_subplot(gs[idr//3,idr%3])
                
        for idl,level in enumerate(level_tab):
            
            mean_variance = []
            
            for idj,j in enumerate(j_list):
                temp = xr.load_dataarray(path_results+name_directory_results+"data_closest/"+v+"_r"+str(r)+"_q"+str(level)+"_day"+str(j)+"_variance.nc", use_cftime=True)/climato_var
                weights = np.cos(np.deg2rad(temp.lat))
                weights.name = "weights"                
                mean_variance.append(temp.weighted(weights).mean(("lon","lat")).values)
                
            ax.plot(j_list, mean_variance, color=colors[idl],label="q = "+str(level))
            ax.scatter(j_list, mean_variance, s=4, color=colors[idl])

        ax.set_title("r = "+str(r)+" days")
        ax.set_xlabel("Days relative to maximum")
        ax.set_ylabel("North-Hemisphere normalized variance")
        ax.legend()
        ax.set_xlim(j_list[0],j_list[-1])
        ax.set_ylim(0.3,1.2)
        
    plt.savefig(path_results+name_directory_results+v+"_NH_variance_evolution.png")
    plt.close()


def plot_north_atlantic_rolling(field_levels, level_tab, v, r, name_directory_results, min_lon, min_lat, max_lon, max_lat):
    
    # North Atlantic
    
    # Load climatology 
    climato_mean, climato_var = select_climato(v) 
    climato_mean = extract_coordinate_north_atlantic(climato_mean)
    climato_var = extract_coordinate_north_atlantic(climato_var)

    # Adapt fields
    field_levels_prepared = []
    for f in field_levels:
        temp = f[0].mean("time")
        for i in f[1:]:
            temp = xr.concat([temp, i.mean("time")], dim='time')
        field_levels_prepared.append(extract_coordinate_north_atlantic(temp))
    
    # Plotting mean
    fig = plt.figure(figsize=(20,10))
    plt.subplots_adjust(top=0.955,bottom=0.01,left=0.0,right=1.0,hspace=0.17,wspace=0.0)
    gs = fig.add_gridspec((len(level_tab)-1)//2+1,2)
    
        # Welch's t-test
    
    lon = field_levels_prepared[0]['lon']
    lat = field_levels_prepared[0]['lat']
    
    if v=="slp":
        levels = np.arange(995,1026,1)
        unit ="hPa"
    elif v=="t2m":
        levels = np.arange(10,32,2)
        unit = "°C"
    elif v=="t850":
        levels = np.arange(0,22,2)
        unit = "°C"
    elif v=='v250':
        levels = np.arange(-16,18,2)
        unit = "m/s"
    elif v=='u250':
        levels = np.arange(-20,30,4)
        unit = "m/s"
    elif v=='v500':
        levels = np.arange(-10,17,2)
        unit = "m/s"
    elif v=='u500':
        levels = np.arange(-15,16,2)
        unit = "m/s"
    elif v=="z500":
        levels = np.arange(5400,5925,25)
        unit = 'm'
    elif v=="mrsos":
        levels = np.arange(5,25,1)
        unit = "kg/m²"

    for idlevel, level in enumerate(level_tab):

        ax = fig.add_subplot(gs[idlevel//2,idlevel%2], projection=ccrs.PlateCarree())
        cp = ax.contourf(lon, lat, field_levels_prepared[idlevel].mean('time').values, extend='both', levels=levels, cmap='coolwarm')
        plt.colorbar(cp, label="["+unit+"]")
        ax.coastlines()
        ax.set_title(str(v)+" r="+str(r)+" q="+str(level))
        
        # Box for the observable
        ax.plot([(min_lon + 180) % 360 - 180,(min_lon + 180) % 360 - 180],[min_lat,max_lat], linewidth=2, color='lime')
        ax.plot([(max_lon + 180) % 360 - 180,(max_lon + 180) % 360 - 180],[min_lat,max_lat], linewidth=2, color='lime')
        ax.plot([(min_lon + 180) % 360 - 180,(max_lon + 180) % 360 - 180],[min_lat,min_lat], linewidth=2, color='lime')
        ax.plot([(min_lon + 180) % 360 - 180,(max_lon + 180) % 360 - 180],[max_lat,max_lat], linewidth=2, color='lime')

    plt.savefig(path_results+name_directory_results+v+"_r"+str(r)+"/full_north-atlantic_mean.png")
    plt.close()


    # Plotting mean anomalies
    fig = plt.figure(figsize=(20,10))
    plt.subplots_adjust(top=0.955,bottom=0.01,left=0.0,right=1.0,hspace=0.17,wspace=0.0)
    gs = fig.add_gridspec((len(level_tab)-1)//2+1,2)
    
        # Welch's t-test
    
    lon = field_levels_prepared[0]['lon']
    lat = field_levels_prepared[0]['lat']
    
    levels = np.arange(-1,1.1,0.1)

    for idlevel, level in enumerate(level_tab):

        ax = fig.add_subplot(gs[idlevel//2,idlevel%2], projection=ccrs.PlateCarree())
        cp = ax.contourf(lon, lat, (field_levels_prepared[idlevel].mean('time').values-climato_mean.values)/np.sqrt(climato_var.values), extend='both', levels=levels, cmap='coolwarm')
        plt.colorbar(cp, label="[STD]")
        ax.coastlines()
        ax.set_title(str(v)+" r="+str(r)+" q="+str(level))
        
        # Box for the observable
        ax.plot([(min_lon + 180) % 360 - 180,(min_lon + 180) % 360 - 180],[min_lat,max_lat], linewidth=2, color='lime')
        ax.plot([(max_lon + 180) % 360 - 180,(max_lon + 180) % 360 - 180],[min_lat,max_lat], linewidth=2, color='lime')
        ax.plot([(min_lon + 180) % 360 - 180,(max_lon + 180) % 360 - 180],[min_lat,min_lat], linewidth=2, color='lime')
        ax.plot([(min_lon + 180) % 360 - 180,(max_lon + 180) % 360 - 180],[max_lat,max_lat], linewidth=2, color='lime')

    plt.savefig(path_results+name_directory_results+v+"_r"+str(r)+"/full_north-atlantic_mean_anomaly.png")
    plt.close()
    
    # Plotting variance
    fig = plt.figure(figsize=(20,10))
    plt.subplots_adjust(top=0.955,bottom=0.01,left=0.0,right=1.0,hspace=0.17,wspace=0.0)
    gs = fig.add_gridspec((len(level_tab)-1)//2+1,2)
    
        # F test
    
    lon = field_levels_prepared[0]['lon']
    lat = field_levels_prepared[0]['lat'] 

    for idlevel, level in enumerate(level_tab):

        ax = fig.add_subplot(gs[idlevel//2,idlevel%2], projection=ccrs.PlateCarree())
        cp = ax.contourf(lon, lat, field_levels_prepared[idlevel].var('time').values/climato_var.values*100, extend='both', levels=np.arange(5,33,2), cmap='coolwarm')
        plt.colorbar(cp, label="%")
        ax.coastlines()
        ax.set_title(str(v)+" r="+str(r)+" q="+str(level))
        
        # Box for the observable
        ax.plot([(min_lon + 180) % 360 - 180,(min_lon + 180) % 360 - 180],[min_lat,max_lat], linewidth=2, color='lime')
        ax.plot([(max_lon + 180) % 360 - 180,(max_lon + 180) % 360 - 180],[min_lat,max_lat], linewidth=2, color='lime')
        ax.plot([(min_lon + 180) % 360 - 180,(max_lon + 180) % 360 - 180],[min_lat,min_lat], linewidth=2, color='lime')
        ax.plot([(min_lon + 180) % 360 - 180,(max_lon + 180) % 360 - 180],[max_lat,max_lat], linewidth=2, color='lime')
        
    
    plt.savefig(path_results+name_directory_results+v+"_r"+str(r)+"/full_north-atlantic_variance.png")
    plt.close()
    
    
def plot_northern_hemisphere_rolling(field_levels, level_tab, v, r, name_directory_results, min_lon, min_lat, max_lon, max_lat):
    
    # Create directory for results if it does not exists
    Path(path_results+name_directory_results+v+"_r"+str(r)+"/").mkdir(parents=True, exist_ok=True)
    
    # Load climatology 
    climato_mean, climato_var = select_climato(v) 
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
    
    # Plotting mean
    fig = plt.figure(figsize=(20,10))
    plt.subplots_adjust(top=0.965,bottom=0.005,left=0.06,right=0.955,hspace=0.07,wspace=0.095)
    gs = fig.add_gridspec((len(level_tab)-1)//3+1,3)
    
        # Welch's t-test
    
    
    if v=="slp":
        levels = np.arange(995,1026,1)
        unit = "hPa"
    elif v=="t2m":
        levels = np.arange(10,32,2)
        unit = "°C"
    elif v=="t850":
        levels = np.arange(0,22,2)
        unit = "°C"
    elif v=='v250':
        levels = np.arange(-16,18,2)
        unit = "m/s"
    elif v=='u250':
        levels = np.arange(-20,30,4)
        unit = "m/s"
    elif v=='v500':
        levels = np.arange(-10,17,2)
        unit = "m/s"
    elif v=='u500':
        levels = np.arange(-15,16,2)
        unit = "m/s"
    elif v=="z500":
        levels = np.arange(5400,5925,25)
        unit = 'm'
    elif v=="mrsos":
        levels = np.arange(5,25,1)
        unit = "kg/m²"

    for idlevel, level in enumerate(level_tab):

        ax = fig.add_subplot(gs[idlevel//3,idlevel%3], projection=ccrs.NorthPolarStereo(central_longitude=0))
        cp = ax.contourf(lon, lat, np.mean(field_levels_prepared[idlevel], axis=0), extend='both', levels=levels, cmap='coolwarm', transform=ccrs.PlateCarree())
        plt.colorbar(cp, label="["+unit+"]")
        ax.coastlines()
        ax.set_title(str(v)+" r="+str(r)+" q="+str(level))
        
        # Box for the observable
        ax.plot([min_lon,min_lon],[min_lat,max_lat], linewidth=3, color='lime', transform=ccrs.PlateCarree())
        ax.plot([max_lon,max_lon],[min_lat,max_lat], linewidth=3, color='lime', transform=ccrs.PlateCarree())
        ax.plot([min_lon,max_lon],[min_lat,min_lat], linewidth=3, color='lime', transform=ccrs.PlateCarree())
        ax.plot([min_lon,max_lon],[max_lat,max_lat], linewidth=3, color='lime', transform=ccrs.PlateCarree())

    plt.savefig(path_results+name_directory_results+v+"_r"+str(r)+"/full_north-hemisphere_mean.png")
    plt.close()
    
    # Plotting mean anomalies
    fig = plt.figure(figsize=(20,10))
    plt.subplots_adjust(top=0.965,bottom=0.005,left=0.06,right=0.955,hspace=0.07,wspace=0.095)
    gs = fig.add_gridspec((len(level_tab)-1)//3+1,3)
    
        # Welch's t-test
    
    
    levels = np.arange(-1,1.1,0.1)

    for idlevel, level in enumerate(level_tab):

        ax = fig.add_subplot(gs[idlevel//3,idlevel%3], projection=ccrs.NorthPolarStereo(central_longitude=0))
        cp = ax.contourf(lon, lat, (np.mean(field_levels_prepared[idlevel],axis=0)-climato_mean)/np.sqrt(climato_var), extend='both', levels=levels, cmap='coolwarm', transform=ccrs.PlateCarree())
        plt.colorbar(cp, label="[STD]")
        ax.coastlines()
        ax.set_title(str(v)+" r="+str(r)+" q="+str(level))
        
        # Box for the observable
        ax.plot([min_lon,min_lon],[min_lat,max_lat], linewidth=3, color='lime', transform=ccrs.PlateCarree())
        ax.plot([max_lon,max_lon],[min_lat,max_lat], linewidth=3, color='lime', transform=ccrs.PlateCarree())
        ax.plot([min_lon,max_lon],[min_lat,min_lat], linewidth=3, color='lime', transform=ccrs.PlateCarree())
        ax.plot([min_lon,max_lon],[max_lat,max_lat], linewidth=3, color='lime', transform=ccrs.PlateCarree())

    plt.savefig(path_results+name_directory_results+v+"_r"+str(r)+"/full_north-hemisphere_mean_anomaly.png")
    plt.close()
    
    # Plotting variance
    fig = plt.figure(figsize=(20,10))
    plt.subplots_adjust(top=0.965,bottom=0.005,left=0.06,right=0.955,hspace=0.07,wspace=0.095)
    gs = fig.add_gridspec((len(level_tab)-1)//3+1,3)
    
        # F test
    

    for idlevel, level in enumerate(level_tab):

        ax = fig.add_subplot(gs[idlevel//3,idlevel%3], projection=ccrs.NorthPolarStereo(central_longitude=0))
        cp = ax.contourf(lon, lat, np.var(field_levels_prepared[idlevel],axis=0)/climato_var*100, extend='both', levels=np.arange(5,33,2), cmap='coolwarm', transform=ccrs.PlateCarree())
        plt.colorbar(cp, label="%")
        ax.coastlines()
        ax.set_title(str(v)+" r="+str(r)+" q="+str(level))
        
        # Box for the observable
        ax.plot([min_lon,min_lon],[min_lat,max_lat], linewidth=3, color='lime', transform=ccrs.PlateCarree())
        ax.plot([max_lon,max_lon],[min_lat,max_lat], linewidth=3, color='lime', transform=ccrs.PlateCarree())
        ax.plot([min_lon,max_lon],[min_lat,min_lat], linewidth=3, color='lime', transform=ccrs.PlateCarree())
        ax.plot([min_lon,max_lon],[max_lat,max_lat], linewidth=3, color='lime', transform=ccrs.PlateCarree())
        
    
    plt.savefig(path_results+name_directory_results+v+"_r"+str(r)+"/full_north-hemisphere_variance.png")
    plt.close()
    

def plot_north_atlantic_circulation(level_tab, height, r, j, name_directory_results, min_lon, min_lat, max_lon, max_lat):
    
    # North-Atlantic
    fig = plt.figure(figsize=(20,10))
    plt.subplots_adjust(top=0.985,bottom=0.01,left=0.005,right=0.98,hspace=0.12,wspace=0.05)    
    gs = fig.add_gridspec((len(level_tab)-1)//2+1,2)
    
    climato_mean_t2m, climato_var_t2m = select_climato("t2m") 
    climato_mean_t2m = extract_coordinate_north_atlantic(climato_mean_t2m)
    climato_var_t2m = extract_coordinate_north_atlantic(climato_var_t2m)    

    for idlevel, level in enumerate(level_tab):

        temp_u = xr.load_dataarray(path_results+name_directory_results+"data_closest/u"+height+"_r"+str(r)+"_q"+str(level)+"_day"+str(j)+"_mean.nc", use_cftime=True)
        temp_u.coords['lon'] = (temp_u.coords['lon'] + 180) % 360 - 180
        temp_u = temp_u.sortby(temp_u.lon)
        temp_u = temp_u.sel(lat=slice(22.5,80),lon=slice(-100,70))
        temp_v = xr.load_dataarray(path_results+name_directory_results+"data_closest/v"+height+"_r"+str(r)+"_q"+str(level)+"_day"+str(j)+"_mean.nc", use_cftime=True)
        temp_v.coords['lon'] = (temp_v.coords['lon'] + 180) % 360 - 180
        temp_v = temp_v.sortby(temp_v.lon)
        temp_v = temp_v.sel(lat=slice(22.5,80),lon=slice(-100,70))
        temp_t2m = xr.load_dataarray(path_results+name_directory_results+"data_closest/t2m_r"+str(r)+"_q"+str(level)+"_day"+str(j)+"_mean.nc", use_cftime=True)
        temp_t2m = extract_coordinate_north_atlantic(temp_t2m)
    
        lon = temp_u['lon']
        lat = temp_u['lat']
        
        if height=='250':
            vmax = 25
        elif height=='500':
            vmax = 20
    
        ax = fig.add_subplot(gs[idlevel//2,idlevel%2], projection=ccrs.PlateCarree())
        strm = ax.streamplot(lon, lat, temp_u.values, temp_v.values, density=1., arrowstyle ='->', arrowsize=2, broken_streamlines=False, transform=ccrs.PlateCarree(), linewidth=3, color=np.sqrt(temp_u.values**2+temp_v.values**2), cmap=plt.cm.get_cmap('Blues', 10), norm=colors.Normalize(vmin=0,vmax=vmax))        
        plt.colorbar(strm.lines, label="m/s", ax=ax, extend='max', fraction=0.046, pad=0.05)
        cf = ax.contourf(temp_t2m['lon'], temp_t2m['lat'], (temp_t2m.values-climato_mean_t2m.values)/np.sqrt(climato_var_t2m), extend='both', levels=np.arange(-1.5,2.1,0.1), cmap='coolwarm', transform=ccrs.PlateCarree())
        plt.colorbar(cf, label="STD", fraction=0.046, pad=0.02)
        ax.coastlines()
        ax.set_title("circulation "+height+"hPa r="+str(r)+" q="+str(level)+" j="+str(j))
        ax.set_ylim(30,70)
        ax.set_xlim(-80,50)
    
            
        # Box for the observable
        ax.plot([(min_lon + 180) % 360 - 180,(min_lon + 180) % 360 - 180],[min_lat,max_lat], linewidth=3, color='lime')
        ax.plot([(max_lon + 180) % 360 - 180,(max_lon + 180) % 360 - 180],[min_lat,max_lat], linewidth=3, color='lime')
        ax.plot([(min_lon + 180) % 360 - 180,(max_lon + 180) % 360 - 180],[min_lat,min_lat], linewidth=3, color='lime')
        ax.plot([(min_lon + 180) % 360 - 180,(max_lon + 180) % 360 - 180],[max_lat,max_lat], linewidth=3, color='lime')

        
    plt.savefig(path_results+name_directory_results+"v"+height+"_r"+str(r)+"/north-atlantic_j"+str(j)+"_circulation.png")
    plt.close()
    
def create_gifs_circulation(height, r, j_list, name_directory_results):
    
    # North Atlantic
    # Mean
    fig, ax = plt.subplots(figsize=(20,10))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off')
    ims = []
    for idj,j in enumerate(j_list):
        im = ax.imshow(plt.imread(path_results+name_directory_results+"v"+height+"_r"+str(r)+"/north-atlantic_j"+str(j)+"_circulation.png"), animated = True)
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=1000)
    ani.save(path_results+name_directory_results+"v"+height+"_r"+str(r)+"/gif_north-atlantic_circulation.gif")
    plt.close()
        
### LAUNCHING THE CODE ###############

if __name__ == "__main__":
    
    # analysis
    # wce: all
    
    #for 4xCO2 do the same with the 10 members: 2550-2750

    #Hyp: no detrend and no deseasonalization
    
    # attention au time_shift de 500 ans pour la température (seule variable où c'est le cas apparemment)
    # il manque 500 and pour zg

    # Observable selection
        # madrid/ 38,39 - 354.75,356.25
        # paris/ 49,50 - 1.25,3.75
        # uppsala/ 59,60 - 13.75,16.25
        # wce/ 46,53.5 - 0, 25
    name_directory_results = "/paris/" 
    Path(path_results+name_directory_results).mkdir(parents=True, exist_ok=True)
    min_lat, max_lat = 49,50 # both included
    min_lon, max_lon = 1.25,3.75 # both included
    
    print("Extracting observable time series")
    series_obs = extract_time_series_observable(min_lat, min_lon, max_lat, max_lon)
    
    #% Parameters
    nb_closest = 50 # number of closest neighbors of the observable
    rolling_periods_tab = np.array([1,3,5,11,15,31]) # number of rolling days
    level_tab = np.array([0.5,0.75,0.95,0.99,0.999,0.9999]) # quantiles
    j_list = range(-15,16) # days considered with respect to central date (closest observable)
    var_tab = ["t2m"] # ["slp","t2m","v250","u250","v500","u500","t850", "z500","mrsos"] # variables to consider, look at others, missing data for z500, hur-hus -> relative and specific humidity
    #available for 2000 years:
        #wap: omega:dp/dt
        #mrso: total soil moisture content
        #mrsos: moisture content of soil layer
        #pr: precipitation
        #prc: convective precipitation
        #rlut: TOA outgoing longwave radiation
        #snc: snow area fraction
        #huss: near-surface specific humidity
        #hursmin: min near surface relative humidity
        #hurs: near surface relative-humidity 

    auto_corr_series = np.zeros((rolling_periods_tab.size,31)) # 31 is for days where autocorrelation is computed
    quantile_tab = np.zeros((rolling_periods_tab.size,level_tab.size))
    closest_neighbors_list = []

    for idr,r in enumerate(rolling_periods_tab):
        print("Computing closest neighbors of the observable for r =",r)
        auto_corr_series[idr,:], quantile_tab[idr,:], temp = compute_closest_days_observable(series_obs, r, nb_closest, level_tab)
        closest_neighbors_list.append(temp)
        save_dates(temp, r, name_directory_results, level_tab)
    
#%   
    # Plotting autocorrelation
    plot_autocorrelation(auto_corr_series, rolling_periods_tab, name_directory_results)
    
    # Plotting histograms of observable
    plot_histograms(closest_neighbors_list, quantile_tab, rolling_periods_tab, series_obs, level_tab, name_directory_results)
    
    # Plotting histograms od calendar dates
    plot_histograms_dates(closest_neighbors_list, rolling_periods_tab, level_tab, name_directory_results)
    
    # Load dates closest neighbors 
    closest_neighbors_list = []
    for idr,r in enumerate(rolling_periods_tab):
        temp = []
        for l in level_tab:
            temp.append(xr.load_dataarray(path_results+name_directory_results+"data_closest/closest_neighbors_obs_r"+str(r)+"_q"+str(l)+".nc", use_cftime=True))
        closest_neighbors_list.append(temp)
            

    min_lat_tab = [38,49,59,46]
    max_lat_tab = [39,50,60,53.5]
    min_lon_tab = [354.75,1.25,13.75,0]
    max_lon_tab = [356.25,3.75,16.25,25]
    name_directory_results_tab = ['madrid/','paris/','uppsala/','wce/']

    for i in range(1, 2):
        
        name_directory_results = name_directory_results_tab[i]
        Path(path_results+name_directory_results).mkdir(parents=True, exist_ok=True)
        min_lat, max_lat = min_lat_tab[i],max_lat_tab[i] # both included
        min_lon, max_lon = min_lon_tab[i],max_lon_tab[i] # both included
        
        print("Extracting observable time series")
        series_obs = extract_time_series_observable(min_lat, min_lon, max_lat, max_lon)
        
        #% Parameters
        nb_closest = 50 # number of closest neighbors of the observable
        rolling_periods_tab = np.array([1,3,5,11,15,31]) # number of rolling days
        level_tab = np.array([0.5,0.75,0.95,0.99,0.999,0.9999]) # quantiles
        j_list = range(-15,16) # days considered with respect to central date (closest observable)
        var_tab = ["t2m"] # ["slp","t2m","v250","u250","v500","u500","t850", "z500","mrsos"] # variables to consider, look at others, missing data for z500, hur-hus -> relative and specific humidity
        
        # Load dates closest neighbors 
        closest_neighbors_list = []
        for idr,r in enumerate(rolling_periods_tab):
            temp = []
            for l in level_tab:
                temp.append(xr.load_dataarray(path_results+name_directory_results+"data_closest/closest_neighbors_obs_r"+str(r)+"_q"+str(l)+".nc", use_cftime=True))
            closest_neighbors_list.append(temp)
    
        for v in var_tab:
            print("Considering ", v)
            for idr,r in enumerate(rolling_periods_tab):
                print("r =", r)
                field_levels = select_field_group(v, j_list, level_tab, closest_neighbors_list[idr])
                
                for idj,j in enumerate(j_list):
                    print("j =", j)
                        
                    for idlevel, level in enumerate(level_tab):
                        f = field_levels[idlevel][idj]
                        # Mean and variance over the members
                        m = f.mean('time')
                        var = f.var('time')
                            
                        # Save fields
                        save_fields(m, var, v, r, j, level, name_directory_results)
                        
                    plot_north_atlantic(field_levels, level_tab, v, r, j, idj, name_directory_results, min_lon, min_lat, max_lon, max_lat)
                    
                    plot_northern_hemisphere(field_levels, level_tab, v, r, j, idj, name_directory_results, min_lon, min_lat, max_lon, max_lat)
                    
    
                    
    #% Do the GIFs
    
        for v in var_tab:
            print("Considering ", v)
            for idr,r in enumerate(rolling_periods_tab):
                print("r =", r)            
                # Create GIF
                create_gifs(v, r, j_list, name_directory_results)
                
    #% Variance normalized spatial mean
            
        for v in var_tab:
            print("Considering ", v)
            variance_evolution(v, j_list, rolling_periods_tab, level_tab, name_directory_results)
                
    
    #% Mean and variance over rolling time
    
        for v in var_tab:
            print("Considering ", v)
            for idr,r in enumerate(rolling_periods_tab):
                print("r =", r)
                j_list_adapted = range(-r//2+1,r//2+1)
                
                field_levels = select_field_group_rolling(v, j_list_adapted, level_tab, closest_neighbors_list[idr])
                        
                plot_north_atlantic_rolling(field_levels, level_tab, v, r, name_directory_results, min_lon, min_lat, max_lon, max_lat)
                    
                plot_northern_hemisphere_rolling(field_levels, level_tab, v, r, name_directory_results, min_lon, min_lat, max_lon, max_lat)
            
#% Circulation, streamfunction
    
    for height in ["250","500"]:
        print("Considering ", height)
        for idr,r in enumerate(rolling_periods_tab):
            print("r =", r)                
            for idj,j in enumerate(j_list):
                print("j =", j)
                
                plot_north_atlantic_circulation(level_tab, height, r, j, name_directory_results, min_lon, min_lat, max_lon, max_lat)
                    
            create_gifs_circulation(height, r, j_list, name_directory_results)
            
            

      
                    
                    