import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colors, cm
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Polygon
import rasterio

def setup_map(ax=None, projection=ccrs.PlateCarree()):
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': projection})
    else:
        fig = ax.figure

    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=0.5)

    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none'
    )
    ax.add_feature(states_provinces, edgecolor='black', linewidth=0.5)

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        alpha=0.5,
        linestyle='--'
    )
    gl.top_labels = False
    gl.right_labels = False
    return fig, ax


def add_grid(ax, lon, lat, dx=1.0, dy=1.0, **kwargs):
    for x, y in zip(lon, lat):
        rect = Rectangle(
            (x - dx / 2, y - dy / 2), dx, dy,
            transform=ccrs.PlateCarree(),
            **kwargs
        )
        ax.add_patch(rect)


def plot_era5_and_buoys(df, buoys_lat, buoys_lon, region, dx=0.5, dy=0.5):
    """Plot ERA5 grid, CNN region, and buoy positions."""
    latlon = df.loc[df.index.get_level_values('valid_time').min()]
    lat = latlon.index.get_level_values('latitude')
    lon = latlon.index.get_level_values('longitude')

    fig, ax = setup_map()
    ax.scatter(lon, lat, transform=ccrs.PlateCarree(), s=20)
    add_grid(ax, lon, lat, dx=dx, dy=dy, edgecolor="black", facecolor="none", linewidth=0.5)

    # CNN region
    region_lats, region_lons = region.exterior.xy
    ax.plot(region_lons, region_lats, transform=ccrs.PlateCarree(), color="red", linewidth=2)

    # Buoys
    true_buoys_lat, true_buoys_lon = [], []
    for b_lat, b_lon in zip(buoys_lat, buoys_lon):
        true_buoys_lat.append(lat[np.abs(lat - b_lat).argmin()])
        true_buoys_lon.append(lon[np.abs(lon - b_lon).argmin()])

    ax.scatter(true_buoys_lon, true_buoys_lat,
               transform=ccrs.PlateCarree(),
               s=200, color='orange', edgecolors='black', linewidths=0.5)

    plt.show()


def plot_tiled_map(df, buoys_lat, buoys_lon, region, dx=0.5, dy=0.5):
    """Plot ERA5 grid as colored tiles and buoys as distinct colored cells."""
    latlon = df.loc[df.index.get_level_values('valid_time').min()]
    lat = latlon.index.get_level_values('latitude')
    lon = latlon.index.get_level_values('longitude')

    fig, ax = setup_map()

    # Paint all grid cells
    for x, y in zip(lon, lat):
        rect = Rectangle(
            (x - dx / 2, y - dy / 2), dx, dy,
            transform=ccrs.PlateCarree(),
            facecolor='lightblue',
            edgecolor='black',
            linewidth=0.3,
            alpha=0.9
        )
        ax.add_patch(rect)

    # Highlight buoys
    for b_lat, b_lon in zip(buoys_lat, buoys_lon):
        tile_lat = lat[np.abs(lat - b_lat).argmin()]
        tile_lon = lon[np.abs(lon - b_lon).argmin()]
        rect = Rectangle(
            (tile_lon - dx / 2, tile_lat - dy / 2), dx, dy,
            transform=ccrs.PlateCarree(),
            facecolor='orange',
            edgecolor='red',
            linewidth=0.8
        )
        ax.add_patch(rect)

    # Optional: add red CNN region box for reference
    region_lats, region_lons = region.exterior.xy
    ax.plot(region_lons, region_lats, transform=ccrs.PlateCarree(), color="red", linewidth=2)

    plt.show()


def plot_dataset(df):
    buoys_lat = (-28.4, -27.4, -24.3)
    buoys_lon = (-46.8, -42.8, -41.8)
    region = Polygon([
        (-29.15, -49.5),
        (-29.15, -41.0),
        (-22.15, -41.0),
        (-22.15, -41.5),
        (-22.65, -41.5),
        (-22.65, -42.0),
        (-23.15, -42.0),
        (-23.15, -44.5),
        (-23.65, -44.5),
        (-23.65, -46.0),
        (-24.15, -46.0),
        (-24.15, -47.0),
        (-24.65, -47.0),
        (-24.65, -48.0),
        (-25.15, -48.0),
        (-25.15, -48.5),
        (-28.15, -48.5),
        (-28.65, -48.5),
        (-28.65, -49.5)
    ])

    # Option 1: Original approach
    #plot_era5_and_buoys(df, buoys_lat, buoys_lon, region)

    # Option 2: Tiled map
    plot_tiled_map(df, buoys_lat, buoys_lon, region)


def plot_dataset_bath(df, bath):
    buoys_lat = (-28.4, -27.4, -24.4)
    buoys_lon = (-46.8, -42.8, -41.8)

    region = Polygon([
        (-29.15, -49.5),
        (-29.15, -41.0),
        (-22.15, -41.0),
        (-22.15, -41.5),
        (-22.65, -41.5),
        (-22.65, -42.0),
        (-23.15, -42.0),
        (-23.15, -44.5),
        (-23.65, -44.5),
        (-23.65, -46.0),
        (-24.15, -46.0),
        (-24.15, -47.0),
        (-24.65, -47.0),
        (-24.65, -48.0),
        (-25.15, -48.0),
        (-25.15, -48.5),
        (-28.15, -48.5),
        (-28.65, -48.5),
        (-28.65, -49.5)
    ])

    bathy, bounds = bath

    latlon = df.loc[df.index.get_level_values('valid_time').min()]
    lat = latlon.index.get_level_values('latitude')
    lon = latlon.index.get_level_values('longitude')

    fig, ax = setup_map()
    img = ax.imshow(
     np.ma.masked_where(bathy > 0, bathy),
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin='upper',
        transform=ccrs.PlateCarree(),
        cmap=plt.get_cmap('GnBu_r', 10),
        vmin=-5000,
        vmax=0
    )
    cbar = plt.colorbar(img, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Bathymetry (m)')
    dx = 0.5
    dy = 0.5
    for x, y in zip(lon, lat):
        rect = Rectangle(
            (x - dx / 2, y - dy / 2),
            dx,
            dy,
            transform=ccrs.PlateCarree(),
            facecolor='none',
            edgecolor='black',
            linewidth=0.8
        )
        ax.add_patch(rect)
    for b_lat, b_lon in zip(buoys_lat, buoys_lon):
        tile_lat = lat[np.abs(lat - b_lat).argmin()]
        tile_lon = lon[np.abs(lon - b_lon).argmin()]
        rect = Rectangle(
            (tile_lon - dx / 2, tile_lat - dy / 2),
            dx,
            dy,
            transform=ccrs.PlateCarree(),
            facecolor='orange',
            edgecolor='red',
            linewidth=1.2
        )
        ax.plot(
            tile_lon,
            tile_lat,
            marker='*',
            color='black',
            markersize=20,
            transform=ccrs.PlateCarree(),
            zorder=10
        )
        ax.add_patch(rect)

    region_lats, region_lons = region.exterior.xy
    ax.plot(
        region_lons,
        region_lats,
        transform=ccrs.PlateCarree(),
        color='red',
        linewidth=2
    )
    ax.set_extent(
        [-50, -40.5, -29.5, -22],
        crs=ccrs.PlateCarree()
    )
    for artist in ax.artists:
        artist.remove()
    ax.set_xticks(np.arange(-50, -40, 1), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-29, -21, 1), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{abs(x):.0f}°W"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{abs(x):.0f}°S"))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.show()


def study_autocorrelation(df):
    # Calculations
    df = df.reset_index()
    time = df['valid_time'].unique().to_numpy()
    coords = df[df['valid_time']==pd.to_datetime('2000-01-01 00:00:00')][['latitude','longitude']].to_numpy()
    data = df[['mpts','shts','mdts']].to_numpy().reshape(time.shape[0], coords.shape[0], 3)
    data = np.stack([data[:,:,0],data[:,:,1]*np.cos(data[:,:,2]), data[:,:,1]*np.sin(data[:,:,2])], axis=2)
    max_dt = 72 # hours

    autocorrs = np.zeros((coords.shape[0],max_dt,data.shape[2]))
    for lag in range(1,max_dt+1):
        data_lag_body = data[lag:,:,:]
        data_lag_bottom = np.zeros([lag,data.shape[1],data.shape[2]])*np.nan
        data_lag = np.concat([data_lag_body, data_lag_bottom], axis=0)
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                corr = np.corrcoef(data[~np.isnan(data_lag[:,i,j]),i,j], data_lag[~np.isnan(data_lag[:,i,j]),i,j])[0,1]
                autocorrs[i,lag-1,j] = corr

    # Map plotting
    norm = colors.Normalize(vmin=0.7, vmax=0.9)
    cmap = cm.viridis
    plot_lag = 12  # hours
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    for i, ax in enumerate(axes):
        setup_map(ax=ax)
        for x, y, v in zip(coords[:,1], coords[:,0], autocorrs[:,plot_lag-1,i]):
            rect = Rectangle(
                (x - 0.5 / 2, y - 0.5 / 2), 0.5, 0.5,
                transform=ccrs.PlateCarree(),
                facecolor=cmap(norm(v)),
                edgecolor='black',
                linewidth=0.3,
                alpha=0.9)
            ax.add_patch(rect)
        ax.autoscale()
        ax.set_title(f"{'Tm' if i==0 else 'Hscos' if i==1 else 'Hssin'}")
    fig.suptitle(f'Autocorrelation at lag {plot_lag}h')
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical')#, fraction=0.025, pad=0.02)
    cbar.set_label('Value')

    # Mean plot
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Tm", "Hscos", "Hssin"]
    for i, label in enumerate(labels):
        ax.plot(np.arange(1, max_dt+1),
                np.mean(autocorrs[:,:,i],axis=0),
                marker='o',
                label=label)
    ax.set_title('Mean autocorrelation vs lag')
    ax.set_xlabel('Lag [hours]')
    ax.set_ylabel('Autocorrelation')
    ax.grid(True)
    ax.legend()
    
    plt.show()


def study_spatial_correlation(df, savefig=True):
    """
    Spatial correlation maps using 3 fixed buoy locations.

    Final map per variable = mean correlation across 3 buoys.

    Buoys shown as red tiles snapped to nearest grid cell.
    """

    buoys_lat = np.array([-28.4, -27.4, -24.3])
    buoys_lon = np.array([-46.8, -42.8, -41.8])

    df = df.reset_index()

    time = df['valid_time'].unique().to_numpy()

    coords = df[df['valid_time'] == time[0]][
        ['latitude', 'longitude']
    ].to_numpy()

    data = df[['mpts', 'shts', 'mdts']].to_numpy().reshape(
        time.shape[0],
        coords.shape[0],
        3
    )

    Tm = data[:, :, 0]
    Hs = data[:, :, 1]
    Dir = np.degrees(data[:, :, 2] % (2 * np.pi))

    vars_data = [Tm, Hs, Dir]
    var_labels = ["Tm correlation", "Hs correlation", "Dir correlation"]

    cmap = cm.plasma
    dx = 0.5
    dy = 0.5

    true_buoys_lat = []
    true_buoys_lon = []
    ref_indices = []

    for blat, blon in zip(buoys_lat, buoys_lon):
        dist2 = (coords[:, 0] - blat) ** 2 + (coords[:, 1] - blon) ** 2
        idx = np.argmin(dist2)

        ref_indices.append(idx)
        true_buoys_lat.append(coords[idx, 0])
        true_buoys_lon.append(coords[idx, 1])

    true_buoys_lat = np.array(true_buoys_lat)
    true_buoys_lon = np.array(true_buoys_lon)

    for values, title in zip(vars_data, var_labels):

        corr_accum = np.zeros((len(ref_indices), coords.shape[0]))

        for b, ref_idx in enumerate(ref_indices):

            ref_series = values[:, ref_idx]

            for i in range(coords.shape[0]):

                target = values[:, i]

                mask = ~np.isnan(ref_series) & ~np.isnan(target)

                if np.sum(mask) > 2:
                    corr_accum[b, i] = np.corrcoef(
                        ref_series[mask],
                        target[mask]
                    )[0, 1]
                else:
                    corr_accum[b, i] = np.nan

        corrs = np.nanmean(corr_accum, axis=0)

        fig, ax = plt.subplots(
            figsize=(8, 6),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )

        setup_map(ax=ax)

        vmin = np.nanpercentile(corrs, 10)
        vmax = np.nanpercentile(corrs, 90)

        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        for x, y, v in zip(coords[:, 1], coords[:, 0], corrs):
            ax.add_patch(
                Rectangle(
                    (x - dx / 2, y - dy / 2),
                    dx,
                    dy,
                    transform=ccrs.PlateCarree(),
                    facecolor=cmap(norm(v)),
                    edgecolor='black',
                    linewidth=0.3,
                    alpha=0.9
                )
            )

        # buoys as red tiles (snapped grid cells)
        for yb, xb in zip(true_buoys_lat, true_buoys_lon):
            ax.add_patch(
                Rectangle(
                    (xb - dx / 2, yb - dy / 2),
                    dx,
                    dy,
                    transform=ccrs.PlateCarree(),
                    facecolor='red',
                    edgecolor='darkred',
                    linewidth=1.5,
                    alpha=1.0,
                    zorder=10
                )
            )

        ax.autoscale()
        ax.set_title(f"Mean correlation (3 buoys)\n{title}")

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=ax,
            orientation='vertical',
            shrink=0.9,
            pad=0.02
        )
        cbar.set_label("Correlation")

        for artist in ax.artists:
            artist.remove()

        ax.set_xticks(np.arange(-50, -40, 1), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-29, -21, 1), crs=ccrs.PlateCarree())

        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, pos: f"{abs(x):.0f}°W")
        )
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, pos: f"{abs(y):.0f}°S")
        )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        ax.set_extent(
            [-49.5, -41, -29.14, -22],
            crs=ccrs.PlateCarree()
        )

        if savefig:
            fname = f"{title}_mean3buoys.png".replace(" ", "_")
            plt.savefig(fname, dpi=300, bbox_inches='tight')

        # plt.show()


def study_TmHsDir(df, bins_tm=30, bins_hs=30, n_dirs=16):
    """
    Generates:
    - Tm histogram with 50% and 90% central interval amplitudes
    - Hs histogram with 50% and 90% central interval amplitudes
    - Directional rose (%) with 50% and 90% angular interval amplitudes
    - Hs-weighted directional rose (%) with 50% and 90% angular interval amplitudes

    Direction assumed in radians (0–2π).
    Returns dictionary with all interval limits and amplitudes.
    """

    Tm = df["mpts"].values
    Hs = df["shts"].values
    Dir = df["mdts"].values % (2 * np.pi)

    results = {}

    # ============================================================
    # Linear central intervals
    # ============================================================
    def central_interval(x, p):
        low = np.percentile(x, (100 - p) / 2)
        high = np.percentile(x, 100 - (100 - p) / 2)
        return low, high, high - low

    Tm_50 = central_interval(Tm, 50)
    Tm_90 = central_interval(Tm, 90)
    Hs_50 = central_interval(Hs, 50)
    Hs_90 = central_interval(Hs, 90)

    results["Tm_50"] = Tm_50
    results["Tm_90"] = Tm_90
    results["Hs_50"] = Hs_50
    results["Hs_90"] = Hs_90

    # ============================================================
    # Circular minimal interval (unweighted)
    # ============================================================
    def circular_min_interval(theta, p):
        theta = np.sort(theta)
        n = len(theta)
        k = int(np.floor(p / 100 * n))

        theta_ext = np.concatenate([theta, theta + 2*np.pi])

        min_width = 2*np.pi
        best_start = 0

        for i in range(n):
            width = theta_ext[i + k] - theta_ext[i]
            if width < min_width:
                min_width = width
                best_start = theta_ext[i]

        start = best_start % (2*np.pi)
        end = (best_start + min_width) % (2*np.pi)

        return start, end, min_width

    # ============================================================
    # Circular minimal interval (weighted)
    # ============================================================
    def circular_min_interval_weighted(theta, weights, p):
        idx = np.argsort(theta)
        theta = theta[idx]
        weights = weights[idx]

        theta_ext = np.concatenate([theta, theta + 2*np.pi])
        weights_ext = np.concatenate([weights, weights])

        total_weight = weights.sum()
        target = p / 100 * total_weight

        n = len(theta)
        min_width = 2*np.pi
        best_start = 0

        j = 0
        cumulative = 0

        for i in range(n):
            while cumulative < target and j < i + n:
                cumulative += weights_ext[j]
                j += 1

            width = theta_ext[j - 1] - theta_ext[i]
            if width < min_width:
                min_width = width
                best_start = theta_ext[i]

            cumulative -= weights_ext[i]

        start = best_start % (2*np.pi)
        end = (best_start + min_width) % (2*np.pi)

        return start, end, min_width

    Dir_50 = circular_min_interval(Dir, 50)
    Dir_90 = circular_min_interval(Dir, 90)

    DirW_50 = circular_min_interval_weighted(Dir, Hs, 50)
    DirW_90 = circular_min_interval_weighted(Dir, Hs, 90)

    results["Dir_50"] = Dir_50
    results["Dir_90"] = Dir_90
    results["DirW_50"] = DirW_50
    results["DirW_90"] = DirW_90

    # ============================================================
    # Histograms Tm & Hs
    # ============================================================
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].hist(Tm, bins=bins_tm, density=True, edgecolor="k")
    ax[0].axvspan(Tm_50[0], Tm_50[1], alpha=0.25)
    ax[0].axvspan(Tm_90[0], Tm_90[1], alpha=0.15)
    ax[0].set_title(
        f"Tm Histogram\n50%={Tm_50[2]:.2f}s | 90%={Tm_90[2]:.2f}s"
    )
    ax[0].set_xlabel("Tm (s)")
    ax[0].set_ylabel("Density")

    ax[1].hist(Hs, bins=bins_hs, density=True, edgecolor="k")
    ax[1].axvspan(Hs_50[0], Hs_50[1], alpha=0.25)
    ax[1].axvspan(Hs_90[0], Hs_90[1], alpha=0.15)
    ax[1].set_title(
        f"Hs Histogram\n50%={Hs_50[2]:.2f}m | 90%={Hs_90[2]:.2f}m"
    )
    ax[1].set_xlabel("Hs (m)")
    ax[1].set_ylabel("Density")

    plt.tight_layout()
    plt.show()

    # ============================================================
    # Directional roses
    # ============================================================
    dir_bins = np.linspace(0, 2*np.pi, n_dirs + 1)
    dir_centers = (dir_bins[:-1] + dir_bins[1:]) / 2
    width = (2*np.pi) / n_dirs

    hist_dir, _ = np.histogram(Dir, bins=dir_bins)
    hist_dir_pct = 100 * hist_dir / hist_dir.sum()

    hist_hs_dir, _ = np.histogram(Dir, bins=dir_bins, weights=Hs)
    hist_hs_dir_pct = 100 * hist_hs_dir / hist_hs_dir.sum()

    fig = plt.figure(figsize=(12, 5))

    # ------------------------------------------------------------
    # Unweighted rose
    # ------------------------------------------------------------
    ax1 = fig.add_subplot(121, polar=True)
    ax1.bar(dir_centers, hist_dir_pct, width=width, edgecolor="k")

    ax1.bar(
        (Dir_50[0] + Dir_50[2]/2) % (2*np.pi),
        max(hist_dir_pct),
        width=Dir_50[2],
        alpha=0.15
    )
    ax1.bar(
        (Dir_90[0] + Dir_90[2]/2) % (2*np.pi),
        max(hist_dir_pct),
        width=Dir_90[2],
        alpha=0.08
    )

    ax1.set_theta_zero_location("N")
    ax1.set_theta_direction(-1)
    ax1.set_title(
        f"Dir Rose\n"
        f"50%={np.degrees(Dir_50[2]):.1f}° | "
        f"90%={np.degrees(Dir_90[2]):.1f}°"
    )

    # ------------------------------------------------------------
    # Weighted rose
    # ------------------------------------------------------------
    ax2 = fig.add_subplot(122, polar=True)
    ax2.bar(dir_centers, hist_hs_dir_pct, width=width, edgecolor="k")

    ax2.bar(
        (DirW_50[0] + DirW_50[2]/2) % (2*np.pi),
        max(hist_hs_dir_pct),
        width=DirW_50[2],
        alpha=0.15
    )
    ax2.bar(
        (DirW_90[0] + DirW_90[2]/2) % (2*np.pi),
        max(hist_hs_dir_pct),
        width=DirW_90[2],
        alpha=0.08
    )

    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    ax2.set_title(
        f"Hs-Dir Rose\n"
        f"50%={np.degrees(DirW_50[2]):.1f}° | "
        f"90%={np.degrees(DirW_90[2]):.1f}°"
    )

    plt.tight_layout()
    plt.show()

    return results


def study_std_heatmap(df):
    """
    Computes and plots the spatial heatmap of the temporal standard deviation
    for each tile and each variable:
        - Tm
        - Hs
        - Dir (circular standard deviation, degrees)

    One figure per variable.
    Colorbar on the right side.
    Latitude/Longitude labels included.

    Assumes:
        mdts in radians (0–2π)
        Regular 0.5° grid
        setup_map(ax) already defined
    """

    # -----------------------------
    # Data organization
    # -----------------------------
    df = df.reset_index()
    time = df['valid_time'].unique().to_numpy()
    coords = df[df['valid_time'] == time[0]][
        ['latitude', 'longitude']
    ].to_numpy()
    data = df[['mpts', 'shts', 'mdts']].to_numpy().reshape(
        time.shape[0],
        coords.shape[0],
        3
    )
    Tm = data[:, :, 0]
    Hs = data[:, :, 1]
    Dir = data[:, :, 2] % (2 * np.pi)

    # -----------------------------
    # Standard deviations
    # -----------------------------
    Tm_std = np.nanstd(Tm, axis=0)
    Hs_std = np.nanstd(Hs, axis=0)
    # Circular std
    sin_mean = np.nanmean(np.sin(Dir), axis=0)
    cos_mean = np.nanmean(np.cos(Dir), axis=0)
    R = np.sqrt(sin_mean**2 + cos_mean**2)
    Dir_std = np.sqrt(-2 * np.log(R))
    Dir_std = np.degrees(Dir_std)
    stds = [Tm_std, Hs_std, Dir_std]
    labels = [
        "Tm std (s)",
        "Hs std (m)",
        "Dir std (deg)"
    ]

    cmap = cm.plasma
    # -----------------------------
    # One figure per variable
    # -----------------------------
    for values, title in zip(stds, labels):
        fig, ax = plt.subplots(
            figsize=(8, 6),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )
        setup_map(ax=ax)

        # Color scale
        vmin = np.nanpercentile(values, 10)
        vmax = np.nanpercentile(values, 90)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        # Plot tiles
        for x, y, v in zip(coords[:, 1], coords[:, 0], values):
            rect = Rectangle(
                (x - 0.25, y - 0.25),
                0.5,
                0.5,
                transform=ccrs.PlateCarree(),
                facecolor=cmap(norm(v)),
                edgecolor='black',
                linewidth=0.3,
                alpha=0.9
            )
            ax.add_patch(rect)

        ax.autoscale()
        ax.set_title(f"{title}\n")

        # Colorbar on right
        sm = plt.cm.ScalarMappable(
            norm=norm,
            cmap=cmap
        )
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=ax,
            orientation='vertical',
            shrink=0.9,
            # fraction=0.045,
            pad=0.02
        )
        cbar.set_label(f"Standard deviation {title[-3:] if title[-3]=='(' else title[-5:]}")

        for artist in ax.artists:
            artist.remove()
        ax.set_xticks(np.arange(-50, -40, 1), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-29, -21, 1), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{abs(x):.0f}°W"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{abs(x):.0f}°S"))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        ax.set_extent(
            [-49.5, -41, -29.14, -22],
            crs=ccrs.PlateCarree()
        )

        plt.savefig(f"{title}.png", dpi=300)


def yearly_std_dots(df):
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
    })

    df = df.reset_index()
    df['year'] = df['valid_time'].dt.year

    years = np.sort(df['year'].unique())

    tm_std, hs_std, dir_std = [], [], []

    for y in years:
        d = df[df['year'] == y]

        Tm = d['mpts'].to_numpy()
        Hs = d['shts'].to_numpy()
        Dir = d['mdts'].to_numpy() % (2 * np.pi)

        tm_std.append(np.nanstd(Tm))
        hs_std.append(np.nanstd(Hs))

        sin_mean = np.nanmean(np.sin(Dir))
        cos_mean = np.nanmean(np.cos(Dir))
        R = np.sqrt(sin_mean**2 + cos_mean**2)
        dir_std.append(np.degrees(np.sqrt(-2 * np.log(R))))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Left axis
    ax1.plot(years, tm_std, 'o-', label='Tm std (s)')
    ax1.plot(years, hs_std, 's-', label='Hs std (m)')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Standard deviation (Tm: s, Hs: m)')
    ax1.set_xticks(years)
    ax1.grid(True)

    # Right axis
    ax2 = ax1.twinx()
    ax2.plot(years, dir_std, '^-', color='tab:green', label='Dir std (deg)')
    ax2.set_ylabel('Direction standard deviation (deg)', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.set_ylim(35, 50)

    # Combined legend
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')

    fig.tight_layout()
    plt.show()


import seaborn as sns
def yearly_distributions(df, years=None, prefix="yearly"):

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
    })

    df = df.reset_index()
    df["year"] = df["valid_time"].dt.year

    if years is None:
        years = sorted(df["year"].unique())

    palette = sns.color_palette("tab10", len(years))

    # ---------------- Tm ----------------

    fig, ax = plt.subplots(figsize=(6, 5))

    for color, year in zip(palette, years):

        d = df[df.year == year]

        Tm = d["mpts"].dropna().to_numpy()

        ax_std = np.std(Tm)

        sns.kdeplot(
            Tm,
            ax=ax,
            color=color,
            linewidth=2,
            label=f"{year} (std={ax_std:.2f} s)"
        )

    ax.set_xlabel("Tm (s)")
    ax.set_ylabel("Density")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(f"{prefix}_tm.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Hs ----------------

    fig, ax = plt.subplots(figsize=(6, 5))

    for color, year in zip(palette, years):

        d = df[df.year == year]

        Hs = d["shts"].dropna().to_numpy()

        ax_std = np.std(Hs)

        sns.kdeplot(
            Hs,
            ax=ax,
            color=color,
            linewidth=2,
            label=f"{year} (std={ax_std:.2f} m)"
        )

    ax.set_xlabel("Hs (m)")
    ax.set_ylabel("Density")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(f"{prefix}_hs.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Direction ----------------

    fig, ax = plt.subplots(
        figsize=(6, 6),
        subplot_kw={"projection": "polar"}
    )

    bins = np.linspace(0, 2*np.pi, 37)

    for color, year in zip(palette, years):

        d = df[df.year == year]

        Dir = d["mdts"].dropna().to_numpy() % (2*np.pi)

        sin_mean = np.mean(np.sin(Dir))
        cos_mean = np.mean(np.cos(Dir))
        R = np.sqrt(sin_mean**2 + cos_mean**2)
        dir_std = np.degrees(np.sqrt(-2*np.log(R)))

        ax.hist(
            Dir,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=color,
            label=f"{year} (std={dir_std:.1f}°)"
        )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1.0)
    )

    fig.tight_layout()
    fig.savefig(f"{prefix}_dir.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_masks(df,name):
    # name = 'piml'
    mask = np.load('data/masks.npz')[name]

    df = df.reset_index()

    coords = df[
        df['valid_time'] == pd.to_datetime('2000-01-01 00:00:00')
    ][['latitude', 'longitude']].to_numpy()

    fig, ax = plt.subplots(
        figsize=(8, 6),
        subplot_kw={'projection': ccrs.PlateCarree()})
    setup_map(ax=ax)

    for i in range(len(mask)):
        x = coords[i, 1]
        y = coords[i, 0]
        rect = Rectangle(
            (x - 0.25, y - 0.25),
            0.5,
            0.5,
            transform=ccrs.PlateCarree(),
            facecolor='lightgreen' if mask[i] else 'lightcoral',
            edgecolor='black',
            linewidth=0.3,
            alpha=0.6)
        ax.add_patch(rect)

    ax.autoscale()
    ax.set_title(f"Mask: {name}\n")
    for artist in ax.artists:
        artist.remove()
    ax.set_xticks(
        np.arange(-50, -40, 1),
        crs=ccrs.PlateCarree())
    ax.set_yticks(
        np.arange(-29, -21, 1),
        crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: f"{abs(x):.0f}°W"))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: f"{abs(x):.0f}°S"))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_extent(
        [-49.5, -41, -29.14, -22],
        crs=ccrs.PlateCarree())
    plt.savefig(f"mask_{name}.png",dpi=300)


if __name__ == "__main__":
    #df = pd.read_pickle('data/f58a282669b2b695c159f2d5f79df76a.pickle')
    df = pd.read_pickle('data/dataframe.pickle')
    with rasterio.open('data/gebco_2026_n-20.0_s-30.0_w-50.0_e-40.0_geotiff.tif') as src:
        bath = [src.read(1), src.bounds]
    # plot_dataset(df)
    # plot_dataset_bath(df,bath)
    # study_autocorrelation(df)
    # study_spatial_correlation(df)
    # study_TmHsDir(df)
    study_std_heatmap(df)
    # yearly_std_dots(df)
    # yearly_distributions(df, years=[2000, 2003, 2006, 2010])
    # plot_masks(df,'piml')