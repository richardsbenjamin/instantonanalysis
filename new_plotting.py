

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