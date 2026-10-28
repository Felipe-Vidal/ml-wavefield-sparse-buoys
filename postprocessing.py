import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colors, cm
import matplotlib.ticker as mticker
import cartopy.crs as ccrs

from dataset_aux import setup_map

def index_agreement(s, o):
    """
        index of agreement

        Willmott (1981, 1982)
        input:
        s: simulated
        o: observed
    output:
        ia: index of agreement
    """
    ia = 1 - (np.sum((o - s) ** 2)) / (
        np.sum((np.abs(s - np.mean(o)) + np.abs(o - np.mean(o))) ** 2)
    )
    return ia

def index_agreement_angle(s, o):
    """
        index_agreement adapted to angular (IN DEGREES!) data
    """
    def angular_dif(a, b):
        return (a - b + 180) % 360 - 180

    def circular_mean(a):
        b = np.deg2rad(a)
        mean_sin = np.mean(np.sin(b))
        mean_cos = np.mean(np.cos(b))
        return np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360

    ia = 1 - (np.sum(angular_dif(s,o) ** 2)) / (
        np.sum((np.abs(angular_dif(s,circular_mean(o))) + np.abs(angular_dif(o,circular_mean(o)))) ** 2)
    )
    return ia

def load_model_log(model_name, stats):
    log = np.load(f"logs/log-{model_name}.npz")

    scale = np.array([1,
        stats["Tm_std"],
        stats["Hscos_std"],
        stats["Hssin_std"]]) ** 2

    train = log['train_loss'] * scale
    val   = log['val_loss'] * scale

    print(f"{model_name};{str(np.cumsum(log['elapsed_time'])[-1]).replace('.', ',')}")
    return log['epoch'], np.cumsum(log['elapsed_time']), train, val

def load_model_adv_log(model_lt, stats, mode, folder):
    adv_log_raw = np.load(f"logs/{folder}/adv_log-{model_lt}h.npz")['arr_0']
    t = adv_log_raw[:,0]
    adv_log = adv_log_raw[:,1:]

    mean = np.array([
        stats["latitude_mean"],
        stats["longitude_mean"],
        stats["Tm_mean"],
        stats["Hscos_mean"],
        stats["Hssin_mean"]])
    mean = np.concatenate([mean,mean[2:]])
    std = np.array([
        stats["latitude_std"],
        stats["longitude_std"],
        stats["Tm_std"],
        stats["Hscos_std"],
        stats["Hssin_std"]])
    std = np.concatenate([std,std[2:]])
    adv_log = adv_log * std + mean # denormalize
    data = np.vstack([adv_log[:,0].round(2),adv_log[:,1].round(2),
                      adv_log[:,2],(adv_log[:,3]**2+adv_log[:,4]**2)**0.5,np.atan2(adv_log[:,4],adv_log[:,3])*180/np.pi,
                      adv_log[:,5].round(2),(adv_log[:,6]**2+adv_log[:,7]**2)**0.5,np.atan2(adv_log[:,7],adv_log[:,6])*180/np.pi]).T # Tm, Hs, Dir

    if mode == 'ts':
        return np.hstack([t.reshape(-1,1),data])
    errors = np.vstack([data[:,0],data[:,1],(data[:,2]-data[:,5])**2,(data[:,3]-data[:,6])**2,((data[:,4]-data[:,7]+180)%360-180)**2]).T # Tm error**2, Hs error**2, Dir error**2
    if mode == 'map':
        return errors
    if mode == 'only_rmse':
        return np.sqrt(np.mean(errors[:,2:], axis=0))
    if mode == 'only_ioa':
        return np.vstack([index_agreement(data[:, 2],data[:, 5]),
                          index_agreement(data[:, 3], data[:, 6]),
                          index_agreement_angle(data[:, 4], data[:, 7])])
    else:
        return None

def mapping(data):
    # unique coordinates and inverse indices
    coords, inv = np.unique(data[:, :2], axis=0, return_inverse=True)

    # accumulate sums and counts
    sums = np.zeros((coords.shape[0], data.shape[1] - 2))
    counts = np.zeros(coords.shape[0], dtype=int)

    np.add.at(sums, inv, data[:, 2:])
    np.add.at(counts, inv, 1)

    # compute means
    means = sums / counts[:, None]

    # build output
    mapped_data = np.zeros((coords.shape[0], data.shape[1]))
    mapped_data[:, :2] = coords
    mapped_data[:, 2:] = means**0.5

    return mapped_data

def plot_train_vs_val_per_model(epochs_list, trains, vals, models, loss_idx, loss_info):
    title, ylabel = loss_info[loss_idx]

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True)

    for ax, epochs, train, val, model in zip(axes, epochs_list, trains, vals, models):
        ax.plot(epochs, train[:, loss_idx], label="Train")
        ax.plot(epochs, val[:, loss_idx], label="Validation")
        ax.set_title(model.upper())
        ax.set_xlabel("Epochs")
        ax.grid(True)
        ax.legend()

    axes[0].set_ylabel(ylabel)
    #plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_validation_losses_by_time(times, vals, models):
    losses = [
        (1, "Tm loss", "Tm loss [s²]"),
        (2, "Hscos loss", "Hscos loss [m²]"),
        (3, "Hssin loss", "Hssin loss [m²]")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (idx, title, ylabel) in zip(axes, losses):
        for t, val, model in zip(times, vals, models):
            ax.plot(t, val[:, idx], label=model.upper())

        ax.set_title(title)
        ax.set_xlabel("Elapsed time [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

def curves(models,view):
    loss_idx = 0 # 0=general, 1=Tm, 2=Hscos, 3=Hssin

    loss_info = {
        0: ("General loss", "Mean loss"),
        1: ("Tm loss", "Tm loss [s²]"),
        2: ("Hscos loss", "Hscos loss [m²]"),
        3: ("Hssin loss", "Hssin loss [m²]")}

    stats = np.load("data/statistics.npz")

    epochs_list = []
    times_list = []
    trains = []
    vals = []

    for model in models:
        epochs, time, train, val = load_model_log(model, stats)
        epochs_list.append(epochs)
        times_list.append(time)
        trains.append(train)
        vals.append(val)
        # print(f"{model};{round(val[1,-1],4)};{round(val[2,-1],4)};{round(val[3,-1],4)};{int(time[-1])}")

    if view == "per_model":
        plot_train_vs_val_per_model(
            epochs_list,
            trains,
            vals,
            models,
            loss_idx,
            loss_info)

    elif view == "val_losses":
        plot_validation_losses_by_time(
            times_list,
            vals,
            models)

def heatmap(models, lead_time, folder):

    labels = [
        "MLP-TE",
        "LSTM-MLP-TE",
        "LSTM-CNN-TE",
    ]

    if isinstance(models, str):
        models = [models]

    if len(models) != 3:
        raise ValueError("Use exactly 3 models")

    vars_names = [
        ("RMSE Tm (s)", "tm"),
        ("RMSE Hs (m)", "hs"),
        ("RMSE dir (deg)", "dir"),
    ]

    cmap = cm.viridis

    model_ids = [f"{m}-{lead_time}" for m in models]

    stats = np.load("data/statistics.npz")

    all_data = {
        m_id: load_model_adv_log(m_id, stats, "map", folder)
        for m_id in model_ids
    }

    map_data = {
        m_id: mapping(data)
        for m_id, data in all_data.items()
    }

    norms = [
        colors.Normalize(
            *np.percentile(
                np.concatenate(
                    [map_data[m_id][:, i + 2] for m_id in model_ids]
                ),
                [1, 99],
            )
        )
        for i in range(3)
    ]

    for c, (var_name, fname) in enumerate(vars_names):

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(4.5, 8),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )

        for r, (ax, model, m_id, label) in enumerate(
            zip(axes, models, model_ids, labels)
        ):

            setup_map(ax)

            ax.set_extent(
                [-49.5, -41, -29.14, -22],
                crs=ccrs.PlateCarree()
            )

            xticks = [-48, -46, -44, -42]
            yticks = [-28, -26, -24, -22]

            for artist in ax.artists:
                artist.remove()

            ax.set_xticks(xticks, crs=ccrs.PlateCarree())
            ax.set_yticks(yticks, crs=ccrs.PlateCarree())

            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(
                    lambda x, pos: f"{abs(x):.0f}°W"
                )
            )

            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(
                    lambda y, pos: f"{abs(y):.0f}°S"
                )
            )

            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_ylabel("Latitude")

            if r == 0:
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position("top")
                ax.set_xlabel("Longitude")
            else:
                ax.set_xticklabels([])

            data = map_data[m_id]

            rmse = np.sqrt(
                np.mean(all_data[m_id][:, 2:5], axis=0)
            )
            # print(
            #     f"{model};{str(rmse[0]).replace('.', ',')};{str(rmse[1]).replace('.', ',')};{str(rmse[2]).replace('.', ',')}")

            for x, y, v in data[:, [1, 0, c + 2]]:

                ax.add_patch(
                    Rectangle(
                        (x - 0.25, y - 0.25),
                        0.5,
                        0.5,
                        transform=ccrs.PlateCarree(),
                        facecolor=cmap(norms[c](v)),
                        edgecolor="black",
                        linewidth=0.2,
                    )
                )

            ax.text(
                0.03,
                0.97,
                f"RMSE={rmse[c]:.3f}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                bbox=dict(
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                ),
            )

            ax.text(
                -0.05,
                0.5,
                label,
                transform=ax.transAxes,
                rotation=90,
                va="center",
                ha="right",
                fontsize=9,
                fontweight="bold",
            )

        sm = cm.ScalarMappable(
            norm=norms[c],
            cmap=cmap,
        )

        cbar = fig.colorbar(
            sm,
            ax=axes,
            orientation="horizontal",
            # fraction=0.035,
            # pad=0.04,
            # aspect=35,
            shrink=0.5,
        )

        cbar.set_label(var_name)

        fig.subplots_adjust(
            # left=0.12,
            # right=0.88,
            # top=0.96,
            bottom=0.25,
            hspace=0.02,
        )

        plt.savefig(
            f"heatmap_{fname}_{lead_time}h.png",
            bbox_inches="tight",
            # pad_inches=0.02,
            dpi=300,
        )

        plt.close()

def rmse_horz(models, max_lead_time, folder):

    labels = [
        "MLP-TE",
        "LSTM-MLP-TE",
        "LSTM-CNN-TE",
    ]

    if isinstance(models, str):
        models = [models]

    if len(models) != 3:
        raise ValueError("Use exactly 3 models")

    lead_times = np.arange(max_lead_time + 1)

    stats = np.load("data/statistics.npz")

    model_ids = [
        f"{m}-{lt}"
        for m in models
        for lt in lead_times
    ]

    all_data = {
        m_id: load_model_adv_log(m_id, stats, "only_rmse", folder)
        for m_id in model_ids
    }

    data = {model: [] for model in models}

    for model in models:
        for key in all_data:
            if key.startswith(model + "-"):
                data[model].append(all_data[key])

    for model in models:
        data[model] = np.stack(data[model], axis=0)

    plots = [
        (0, "RMSE Tm (s)", "tm"),
        (1, "RMSE Hs (m)", "hs"),
        (2, "RMSE dir (deg)", "dir"),
    ]

    for jdx, ylabel, fname in plots:

        fig, ax = plt.subplots(figsize=(5, 4))

        for model, label in zip(models, labels):

            ax.plot(
                lead_times,
                data[model][:, jdx],
                label=label,
            )

        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 4))

        ax.set_xlabel("Forecast time (h)")
        ax.set_ylabel(ylabel)

        ax.grid(True)

        ax.legend(fontsize=8)

        fig.tight_layout()

        plt.savefig(
            f"rmse_horz_{fname}.png",
            # bbox_inches="tight",
            dpi=300,
        )

        plt.close()

def ioa_horz(models, max_lead_time, folder):

    labels = [
        "MLP-TE",
        "LSTM-MLP-TE",
        "LSTM-CNN-TE",
    ]

    if isinstance(models, str):
        models = [models]

    if len(models) != 3:
        raise ValueError("Use exactly 3 models")

    lead_times = np.arange(max_lead_time + 1)

    stats = np.load("data/statistics.npz")

    model_ids = [
        f"{m}-{lt}"
        for m in models
        for lt in lead_times
    ]

    all_data = {
        m_id: load_model_adv_log(m_id, stats, "only_ioa", folder)
        for m_id in model_ids
    }

    data = {model: [] for model in models}

    for model in models:
        for key in all_data:
            if key.startswith(model + "-"):
                data[model].append(all_data[key])

    for model in models:
        data[model] = np.stack(data[model], axis=0)

    plots = [
        (0, "IOA Tm", "tm"),
        (1, "IOA Hs", "hs"),
        (2, "IOA dir", "dir"),
    ]

    for jdx, ylabel, fname in plots:

        fig, ax = plt.subplots(figsize=(5, 4))

        for model, label in zip(models, labels):

            ax.plot(
                lead_times,
                data[model][:, jdx],
                label=label,
            )

        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 4))

        ax.set_xlabel("Forecast time (h)")
        ax.set_ylabel(ylabel)

        ax.grid(True)

        ax.legend(fontsize=8)

        fig.tight_layout()

        plt.savefig(
            f"ioa_horz_{fname}.png",
            # bbox_inches="tight",
            dpi=300
        )

        plt.close()

def time_series(models, dates, points, max_lead_time, folder):

    labels = [
        "MLP-TE",
        "LSTM-MLP-TE",
        "LSTM-CNN-TE",
    ]

    if isinstance(models, str):
        models = [models]

    if len(models) != 3:
        raise ValueError("Use exactly 3 models")

    lead_times = np.arange(max_lead_time + 1)

    stats = np.load("data/statistics.npz")

    model_ids = [
        f"{m}-{lt}"
        for m in models
        for lt in lead_times
    ]

    all_data = {
        m_id: load_model_adv_log(m_id, stats, "ts", folder)
        for m_id in model_ids
    }

    data = {model: [] for model in models}

    for model in models:
        for key in all_data:
            if key.startswith(model + "-"):
                data[model].append(all_data[key])

    for model in models:
        data[model] = np.stack(data[model], axis=2)

    plots = [
        (0, "Tm (s)", "tm"),
        (1, "Hs (m)", "hs"),
        (2, "Dir (deg)", "dir"),
    ]

    buoys = np.array([
        [-28.39, -46.76],
        [-27.39, -42.76],
        [-24.39, -41.76],
    ])

    truth_times = np.arange(-12, max_lead_time + 1)

    context_times = np.arange(-12, 1, 6)

    def wrap_direction(ts, center):
        return ((ts - center + 180) % 360) - 180 + center

    for date in dates:

        hour = (
            pd.to_datetime(date)
            - pd.Timestamp("2000-01-01 00:00:00")
        ) / pd.Timedelta(hours=1)

        for point in points:

            for jdx, ylabel, fname in plots:

                fig, ax = plt.subplots(figsize=(5, 4))

                truth_ts = None

                for model, label in zip(models, labels):

                    idx = np.where(
                        (
                            data[model][:, :3, 0]
                            == [hour, point[0], point[1]]
                        ).all(axis=1)
                    )[0][0]

                    if model == models[0]:

                        truth_ts = data[model][
                            np.arange(
                                idx - 12 * 168,
                                idx + 168 * max_lead_time + 1,
                                168,
                            ),
                            -3:,
                            0,
                        ].copy()

                        if jdx == 2:

                            center = truth_ts[12, jdx]

                            truth_ts[:, jdx] = wrap_direction(
                                truth_ts[:, jdx],
                                center,
                            )

                        ax.plot(
                            truth_times,
                            truth_ts[:, jdx],
                            c="black",
                            label="Ground truth",
                            zorder=1,
                        )

                        if np.any(np.all(point == buoys, axis=1)):

                            ax.scatter(
                                context_times,
                                truth_ts[0:13:6, jdx],
                                s=20,
                                c="red",
                                label="Context",
                                zorder=2,
                            )

                        ax.axvline(
                            x=0,
                            color="gray",
                            linestyle="--",
                            linewidth=1,
                        )

                    forecast_ts = data[model][
                        idx,
                        -6:-3,
                        :
                    ].copy()

                    if jdx == 2:

                        center = truth_ts[12, jdx]

                        forecast_ts[jdx, :] = wrap_direction(
                            forecast_ts[jdx, :],
                            center,
                        )

                    ax.plot(
                        lead_times,
                        forecast_ts[jdx, :],
                        label=label,
                    )

                ax.set_xlim(-12, 24)

                ax.set_xticks(np.arange(-12, 25, 6))

                ax.set_xlabel("Time (h)")

                ax.set_ylabel(ylabel)

                if jdx == 2:
                    ax.set_ylim(center - 180, center + 180)

                ax.grid(True)

                ax.legend(fontsize=8)

                # fig.suptitle(
                #     f"Date: {date}; "
                #     f"Coordinates (deg): "
                #     f"{point[0]} (lat), {point[1]} (lon)"
                # )

                fig.tight_layout()

                plt.savefig(
                    f"time_series_{fname}_{str(date).replace(':', '-')}_"
                    f"{point[0]:.2f}_{point[1]:.2f}.png",
                    bbox_inches="tight",
                )

                plt.close()

def nowcasting_time_series(models, dates, points, days, folder):

    labels = ["MLP-TE", "LSTM-MLP-TE", "LSTM-CNN-TE"]

    if isinstance(models, str):
        models = [models]

    if len(models) != 3:
        raise ValueError("Use exactly 3 models")

    stats = np.load("data/statistics.npz")

    model_ids = [f"{m}-0" for m in models]

    all_data = {
        m_id: load_model_adv_log(m_id, stats, "ts", folder)
        for m_id in model_ids
    }

    data = {model: all_data[f"{model}-0"] for model in models}

    plots = [
        (0, "Tm (s)", "tm"),
        (1, "Hs (m)", "hs"),
        (2, "Dir (deg)", "dir"),
    ]

    for date in dates:

        start_hour = (
            pd.to_datetime(date)
            - pd.Timestamp("2000-01-01 00:00:00")
        ) / pd.Timedelta(hours=1)

        for point in points:

            indices = []

            for d in range(days * 24 + 1):

                current_hour = start_hour + d

                idx = np.where(
                    (
                        data[models[0]][:, :3]
                        == [current_hour, point[0], point[1]]
                    ).all(axis=1)
                )[0][0]

                indices.append(idx)

            time_axis = pd.date_range(
                start=pd.to_datetime(date),
                periods=days * 24 + 1,
                freq="1h"
            )

            xticks = pd.date_range(
                start="2010-01-01",
                end="2011-01-01",
                freq="MS"
            )

            for jdx, ylabel, fname in plots:

                fig, ax = plt.subplots(figsize=(9, 3), dpi=300)

                truth = []
                forecasts = {model: [] for model in models}

                for idx in indices:

                    gt = data[models[0]][idx, -3:][jdx]
                    truth.append(gt)

                    for model in models:
                        pred = data[model][idx, -6:-3][jdx]
                        forecasts[model].append(pred)

                truth = np.array(truth)

                if jdx == 2:
                    truth[truth<0] = truth[truth<0] + 360
                    all_vals = [truth]
                    for m in models:
                        arr = np.array(forecasts[m])
                        arr[arr<0] = arr[arr<0] + 360
                        forecasts[m] = arr
                        all_vals.append(forecasts[m])
                    all_vals = np.concatenate(all_vals)

                ax.plot(
                    time_axis,
                    truth,
                    c="black",
                    label="Ground truth",
                    linewidth=2,
                    alpha=1
                )

                for model, label in zip(models, labels):
                    ax.plot(
                        time_axis,
                        forecasts[model],
                        label=label,
                        linewidth=1,
                        alpha=0.9
                    )

                ax.set_xlabel("Date")
                ax.set_ylabel(ylabel)

                if jdx == 1:
                    ax.set_ylim(0, 5)

                if jdx == 0:
                    ax.set_ylim(5, 15)

                ax.set_xlim(time_axis[0], time_axis[-1])

                ax.set_xticks(xticks)
                ax.set_xticklabels([d.strftime("%Y-%m") for d in xticks], rotation=45)

                ax.grid(True)
                ax.legend(fontsize=8)

                fig.suptitle(
                    f"Start date: {date}; "
                    f"Coordinates (deg): {point[0]} (lat), {point[1]} (lon)"
                )

                fig.tight_layout()

                plt.savefig(
                    f"nowcasting_{fname}_"
                    f"{str(date).replace(':', '-')}"
                    f"_{point[0]:.2f}_{point[1]:.2f}.png",
                    bbox_inches="tight",
                )

                plt.close()

def nowcasting_time_series2(models, dates, days, folder):

    labels = ["MLP-TE", "LSTM-MLP-TE", "LSTM-CNN-TE"]

    points = [
        (-27.39, -43.76),  # Offshore
        (-24.39, -46.26),  # Nearshore
    ]

    column_titles = [
        "Selected offshore location (27.4°S, 43.8°W)",
        "Selected nearshore location (24.4°S, 46.3°W)",
    ]

    if isinstance(models, str):
        models = [models]

    if len(models) != 3:
        raise ValueError("Use exactly 3 models")

    stats = np.load("data/statistics.npz")

    model_ids = [f"{m}-0" for m in models]

    all_data = {
        m_id: load_model_adv_log(m_id, stats, "ts", folder)
        for m_id in model_ids
    }

    data = {model: all_data[f"{model}-0"] for model in models}

    plots = [
        (0, "Tm (s)"),
        (1, "Hs (m)"),
        (2, "Dir (deg)"),
    ]

    for date in dates:

        start_hour = (
            pd.to_datetime(date)
            - pd.Timestamp("2000-01-01 00:00:00")
        ) / pd.Timedelta(hours=1)

        time_axis = pd.date_range(
            start=pd.to_datetime(date),
            periods=days * 24 + 1,
            freq="1h",
        )

        xmin = pd.Timestamp("2010-01-01")
        xmax = pd.Timestamp("2011-01-01")

        xticks = pd.date_range(
            start=xmin,
            end=xmax,
            freq="MS",
        )

        fig, axes = plt.subplots(
            3,
            2,
            figsize=(14, 7),   # 2:1 aspect ratio
            dpi=300,
            sharex=True,
        )

        legend_handles = None
        legend_labels = None

        for col, point in enumerate(points):

            indices = []

            for d in range(days * 24 + 1):

                current_hour = start_hour + d

                idx = np.where(
                    (
                        data[models[0]][:, :3]
                        == [current_hour, point[0], point[1]]
                    ).all(axis=1)
                )[0][0]

                indices.append(idx)

            axes[0, col].set_title(
                column_titles[col],
                fontsize=12,
                fontweight="bold",
            )

            for row, (jdx, ylabel) in enumerate(plots):

                ax = axes[row, col]

                truth = []
                forecasts = {model: [] for model in models}

                for idx in indices:

                    truth.append(data[models[0]][idx, -3:][jdx])

                    for model in models:
                        forecasts[model].append(
                            data[model][idx, -6:-3][jdx]
                        )

                truth = np.asarray(truth)

                if jdx == 2:

                    truth = truth.copy()
                    truth[truth < 0] += 360

                    for model in models:
                        arr = np.asarray(forecasts[model])
                        arr[arr < 0] += 360
                        forecasts[model] = arr

                ax.plot(
                    time_axis,
                    truth,
                    color="black",
                    linewidth=2,
                    label="Ground truth",
                )

                for model, label in zip(models, labels):
                    ax.plot(
                        time_axis,
                        forecasts[model],
                        linewidth=1.2,
                        label=label,
                    )

                ax.set_ylabel(ylabel)

                if jdx == 0:
                    ax.set_ylim(5, 15)

                elif jdx == 1:
                    ax.set_ylim(0, 5)

                ax.grid(True)
                ax.set_xlim(xmin, xmax)

                if row == 2:
                    ax.set_xlabel("Date")
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(
                        [d.strftime("%Y-%m") for d in xticks],
                        rotation=45,
                        ha="right",
                    )

                if row == 2 and col == 0:
                    legend_handles, legend_labels = ax.get_legend_handles_labels()

                if ax.get_legend() is not None:
                    ax.get_legend().remove()

        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=4,
            fontsize=12,
            frameon=False,
            handlelength=3,
            columnspacing=2.0,
        )

        fig.tight_layout(rect=[0, 0.04, 1, 0.97])

        plt.savefig(
            f"nowcasting_{str(date).replace(':', '-')}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

def nowcasting_time_series3(models, dates, days, folder):

    labels = ["MLP-TE", "LSTM-MLP-TE", "LSTM-CNN-TE"]

    point = (-24.39, -46.26)  # Nearshore

    if isinstance(models, str):
        models = [models]

    if len(models) != 3:
        raise ValueError("Use exactly 3 models")

    stats = np.load("data/statistics.npz")

    model_ids = [f"{m}-0" for m in models]

    all_data = {
        m_id: load_model_adv_log(m_id, stats, "ts", folder)
        for m_id in model_ids
    }

    data = {model: all_data[f"{model}-0"] for model in models}

    plots = [
        (0, "Tm (s)"),
        (1, "Hs (m)"),
        (2, "Dir (deg)"),
    ]

    for date in dates:

        start_hour = (
            pd.to_datetime(date)
            - pd.Timestamp("2000-01-01 00:00:00")
        ) / pd.Timedelta(hours=1)

        time_axis = pd.date_range(
            start=pd.to_datetime(date),
            periods=days * 24 + 1,
            freq="1h",
        )

        xmin = pd.Timestamp("2010-01-01")
        xmax = pd.Timestamp("2011-01-01")

        xticks = pd.date_range(
            start=xmin,
            end=xmax,
            freq="MS",
        )

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(7, 5),
            dpi=300,
            sharex=True,
        )

        indices = []

        for d in range(days * 24 + 1):

            current_hour = start_hour + d

            idx = np.where(
                (
                    data[models[0]][:, :3]
                    == [current_hour, point[0], point[1]]
                ).all(axis=1)
            )[0][0]

            indices.append(idx)

        legend_handles = None
        legend_labels = None

        for row, (jdx, ylabel) in enumerate(plots):

            ax = axes[row]

            truth = []
            forecasts = {model: [] for model in models}

            for idx in indices:

                truth.append(data[models[0]][idx, -3:][jdx])

                for model in models:
                    forecasts[model].append(
                        data[model][idx, -6:-3][jdx]
                    )

            truth = np.asarray(truth)

            if jdx == 2:

                truth = truth.copy()
                truth[truth < 0] += 360

                for model in models:
                    arr = np.asarray(forecasts[model])
                    arr[arr < 0] += 360
                    forecasts[model] = arr

            ax.plot(
                time_axis,
                truth,
                color="black",
                linewidth=2,
                label="Ground truth",
            )

            for model, label in zip(models, labels):
                ax.plot(
                    time_axis,
                    forecasts[model],
                    linewidth=1.2,
                    label=label,
                )

            ax.set_ylabel(ylabel)

            if jdx == 0:
                ax.set_ylim(5, 13)
                ax.set_yticks([5, 7, 9, 11, 13])

            elif jdx == 1:
                ax.set_ylim(0, 4)
                ax.set_yticks([0, 1, 2, 3, 4])

            ax.grid(True)
            ax.set_xlim(xmin, xmax)

            if row == 2:
                ax.set_xlabel("Date")
                ax.set_xticks(xticks)
                ax.set_xticklabels(
                    [d.strftime("%Y-%m") for d in xticks],
                    rotation=45,
                    ha="right",
                )

            if row == 0:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

            if ax.get_legend() is not None:
                ax.get_legend().remove()

        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.03),  # closer to panels
            ncol=4,
            fontsize=8,
            frameon=False,
            handlelength=2.5,
            columnspacing=1.2,
        )

        fig.tight_layout(rect=[0, 0.05, 1, 1])

        plt.savefig(
            f"nowcasting_single_{str(date).replace(':', '-')}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

def nowcasting_time_series4(models, dates, days, folder):

    label = "LSTM-CNN-TE"

    points = [
        (-27.39, -43.76),  # Offshore
        (-24.39, -46.26),  # Nearshore
    ]

    column_titles = [
        "Selected offshore location (27.4°S, 43.8°W)",
        "Selected nearshore location (24.4°S, 46.3°W)",
    ]

    if isinstance(models, str):
        models = [models]

    if len(models) != 3:
        raise ValueError("Use exactly 3 models")

    # Use only the third model
    model = models[-1]

    stats = np.load("data/statistics.npz")

    data = load_model_adv_log(f"{model}-0", stats, "ts", folder)

    plots = [
        (0, "Tm (s)"),
        (1, "Hs (m)"),
        (2, "Dir (deg)"),
    ]

    for date in dates:

        start_hour = (
            pd.to_datetime(date)
            - pd.Timestamp("2000-01-01 00:00:00")
        ) / pd.Timedelta(hours=1)

        time_axis = pd.date_range(
            start=pd.to_datetime(date),
            periods=days * 24 + 1,
            freq="1h",
        )

        xmin = pd.Timestamp("2010-01-01")
        xmax = pd.Timestamp("2011-01-01")

        xticks = pd.date_range(
            start=xmin,
            end=xmax,
            freq="MS",
        )

        fig, axes = plt.subplots(
            3,
            2,
            figsize=(14, 7),
            dpi=300,
            sharex=True,
        )

        legend_handles = None
        legend_labels = None

        for col, point in enumerate(points):

            indices = []

            for d in range(days * 24 + 1):

                current_hour = start_hour + d

                idx = np.where(
                    (
                        data[:, :3]
                        == [current_hour, point[0], point[1]]
                    ).all(axis=1)
                )[0][0]

                indices.append(idx)

            axes[0, col].set_title(
                column_titles[col],
                fontsize=12,
                fontweight="bold",
            )

            for row, (jdx, ylabel) in enumerate(plots):

                ax = axes[row, col]

                truth = []
                forecast = []

                for idx in indices:

                    truth.append(data[idx, -3:][jdx])
                    forecast.append(data[idx, -6:-3][jdx])

                truth = np.asarray(truth)
                forecast = np.asarray(forecast)

                if jdx == 2:
                    truth = truth.copy()
                    truth[truth < 0] += 360

                    forecast = forecast.copy()
                    forecast[forecast < 0] += 360

                ax.plot(
                    time_axis,
                    truth,
                    color="black",
                    linewidth=2,
                    label="Ground truth",
                )

                ax.plot(
                    time_axis,
                    forecast,
                    linewidth=1.5,
                    label=label,
                )

                ax.set_ylabel(ylabel)

                if jdx == 0:
                    ax.set_ylim(5, 15)
                elif jdx == 1:
                    ax.set_ylim(0, 5)

                ax.grid(True)
                ax.set_xlim(xmin, xmax)

                if row == 2:
                    ax.set_xlabel("Date")
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(
                        [d.strftime("%Y-%m") for d in xticks],
                        rotation=45,
                        ha="right",
                    )

                if row == 2 and col == 0:
                    legend_handles, legend_labels = (
                        ax.get_legend_handles_labels()
                    )

                if ax.get_legend() is not None:
                    ax.get_legend().remove()

        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=2,
            fontsize=12,
            frameon=False,
            handlelength=3,
            columnspacing=2.0,
        )

        fig.tight_layout(rect=[0, 0.04, 1, 0.97])

        plt.savefig(
            f"nowcasting_{str(date).replace(':', '-')}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

def parity_plot(models, folder):
    labels = ["MLP-TE", "LSTM-MLP-TE", "LSTM-CNN-TE"]
    markers = ["o", "D", "^"]
    alphas = [0.75, 0.5, 0.25]

    if isinstance(models, str):
        models = [models]

    if len(models) != 3:
        raise ValueError("Use exactly 3 models")

    stats = np.load("data/statistics.npz")
    model_ids = [f"{m}-0" for m in models]

    all_data = {
        m_id: load_model_adv_log(m_id, stats, "ts", folder)
        for m_id in model_ids
    }

    data = {m: all_data[f"{m}-0"] for m in models}

    vars_info = [
        (0, "Tm (s)", "tm"),
        (1, "Hs (m)", "hs"),
        (2, "Dir (deg)", "dir"),
    ]

    for jdx, var_label, fname in vars_info:

        fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

        for i, (model, label) in enumerate(zip(models, labels)):

            gt = data[model][:, -3 + jdx]
            pred = data[model][:, -6 + jdx]

            if jdx == 2:
                gt = gt + 180
                pred = pred + 180

            ax.scatter(
                pred,
                gt,
                s=50,
                alpha=alphas[i],
                marker=markers[i],
                edgecolors="none",
                label=label,
                rasterized=True,
            )

        if jdx == 0:
            lim = [5, 15]
            ticks = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            var_label = 'Tm, s'
        elif jdx == 1:
            lim = [0, 5]
            ticks = [0, 1, 2, 3, 4, 5]
            var_label = 'Hs, m'
        else:
            lim = [0, 360]
            ticks = [0,60,120,180,240,300,360]
            var_label = 'dir, deg'

        ax.plot(lim, lim, "k--", linewidth=1)

        ax.set_xlabel(f"Reconstruct ({var_label})")
        ax.set_ylabel(f"Ground Truth ({var_label})")

        ax.set_xlim(lim)
        ax.set_xticks(ticks)
        ax.set_ylim(lim)
        ax.set_yticks(ticks)

        ax.legend(loc="upper left")
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(f"parity_{fname}.png", bbox_inches="tight")
        plt.close()

def print_rmse_ioa(models, folder):
    stats = np.load("data/statistics.npz")

    model_ids = [f"{m}-0" for m in models]

    all_data = {
        m_id: [load_model_adv_log(m_id, stats, "only_rmse", folder),
               load_model_adv_log(m_id, stats, "only_ioa", folder)]
        for m_id in model_ids
    }

    for m_id in model_ids:
        print(f"{m_id};"
              f"{str(all_data[m_id][0][0]).replace('.', ',')};"
              f"{str(all_data[m_id][0][1]).replace('.', ',')};"
              f"{str(all_data[m_id][0][2]).replace('.', ',')};"
              f"{str(all_data[m_id][1][0][0]).replace('.', ',')};"
              f"{str(all_data[m_id][1][1][0]).replace('.', ',')};"
              f"{str(all_data[m_id][1][2][0]).replace('.', ',')}")

def time_series2(models, dates, points, max_lead_time, folder):
    """Figure-18 grid: 3 rows (Tm, Hs, Dir) x (len(dates) * len(points)) cols.

    Each cell shows a forecast window from -12 h to +24 h relative to the
    reference time (x = 0, dashed vertical line): the 12 h prior context of
    ground truth (red dots at buoy points) followed by the nowcasting
    reference, and the model forecast lines out to max_lead_time hours.
    Mirrors time_series but consolidated into a single shared-scale figure.
    """

    def _point_labels(points):
        """Return [(point, name, lat_str, lon_str)] classifying offshore/nearshore.

        The nearshore reference (Santos Port) is (-24.39, -46.26); the offshore
        virtual buoy is (-27.39, -42.76). Any other point is classed offshore.
        """
        nearshore = (-24.39, -46.26)
        out = []
        for pt in points:
            if tuple(pt) == nearshore:
                out.append((tuple(pt), "nearshore", "24.4", "46.3"))
            else:
                out.append((tuple(pt), "offshore", "27.4", "42.8"))
        return out

    def _default_season_titles(dates, points):
        """Return [(title, first_col, last_col)] for centralized season banners.

        With exactly two dates and two points, the first date spans the first two
        columns ("RANDOM SUMMER DAY FORECAST") and the second date the last two
        ("RANDOM WINTER DAY FORECAST"). Otherwise no banner is emitted.
        """
        if len(dates) == 2 and len(points) == 2:
            return [("RANDOM SUMMER DAY FORECAST", 0, 1),
                    ("RANDOM WINTER DAY FORECAST", 2, 3)]
        return []

    labels = ["MLP-TE", "LSTM-MLP-TE", "LSTM-CNN-TE"]

    buoys = np.array([
        [-28.39, -46.76],
        [-27.39, -42.76],
        [-24.39, -41.76],
    ])

    if isinstance(models, str):
        models = [models]

    if len(models) != 3:
        raise ValueError("Use exactly 3 models")

    stats = np.load("data/statistics.npz")

    lead_times = np.arange(max_lead_time + 1)

    model_ids = [
        f"{m}-{lt}"
        for m in models
        for lt in lead_times
    ]

    all_data = {
        m_id: load_model_adv_log(m_id, stats, "ts", folder)
        for m_id in model_ids
    }

    data = {model: [] for model in models}

    for model in models:
        for key in all_data:
            if key.startswith(model + "-"):
                data[model].append(all_data[key])

    for model in models:
        data[model] = np.stack(data[model], axis=2)

    plots = [
        (0, "Tm (s)"),
        (1, "Hs (m)"),
        (2, "Dir (deg)"),
    ]

    truth_times = np.arange(-12, max_lead_time + 1)
    context_times = np.arange(-12, 1, 6)

    def wrap_direction(ts, center):
        return ((ts - center + 180) % 360) - 180 + center

    columns = [
        (season_date, point)
        for season_date in dates
        for point in points
    ]

    n_cols = len(columns)

    fig, axes = plt.subplots(
        3,
        n_cols,
        figsize=(4 * n_cols, 9),
        dpi=300,
        sharex="col",
        sharey="row",
    )

    season_titles = _default_season_titles(dates, points)

    point_labeler = {pt: f"Selected {name} location\n({lat}°S, {lon}°W)"
                     for pt, name, lat, lon in _point_labels(points)}

    legend_handles = None
    legend_labels = None

    row_lo = [None, None, None]
    row_hi = [None, None, None]

    for col, (season_date, point) in enumerate(columns):

        hour = (
            pd.to_datetime(season_date)
            - pd.Timestamp("2000-01-01 00:00:00")
        ) / pd.Timedelta(hours=1)

        point_is_buoy = np.any(np.all(point == buoys, axis=1))

        axes[0, col].set_title(
            point_labeler[tuple(point)],
            fontsize=11,
            fontweight="bold",
            pad=16,
        )

        for row, (jdx, ylabel) in enumerate(plots):

            ax = axes[row, col]

            idx = np.where(
                (
                    data[models[0]][:, :3, 0]
                    == [hour, point[0], point[1]]
                ).all(axis=1)
            )[0][0]

            rows = np.arange(
                idx - 12 * 168,
                idx + 168 * max_lead_time + 1,
                168,
            )
            truth_ts = data[models[0]][
                rows,
                -3:,
                0,
            ].copy()

            if jdx == 2:
                center = truth_ts[12, jdx]
                truth_ts[:, jdx] = wrap_direction(
                    truth_ts[:, jdx],
                    center,
                )

            truth_series = truth_ts[:, jdx]

            ax.plot(
                truth_times,
                truth_series,
                color="black",
                linewidth=2,
                label="Ground truth",
                zorder=1,
            )

            if point_is_buoy:

                ax.scatter(
                    context_times,
                    truth_series[0:13:6],
                    s=20,
                    c="red",
                    label="Context",
                    zorder=2,
                )

            ax.axvline(
                x=0,
                color="gray",
                linestyle="--",
                linewidth=1,
            )

            for model, label in zip(models, labels):

                forecast = data[model][idx, -6:-3, :][jdx]

                if jdx == 2:
                    forecast = wrap_direction(
                        forecast,
                        center,
                    )

                ax.plot(
                    lead_times,
                    forecast,
                    linewidth=1.2,
                    label=label,
                )

            ax.set_xlim(-12, 24)
            ax.set_xticks(np.arange(-12, 25, 6))

            if row == 2:
                ax.set_xlabel("Time (h)")

            if col == 0:
                ax.set_ylabel(ylabel)

            ax.grid(True)

            values = np.concatenate([truth_series]
                                    + [data[m][idx, -6:-3, :][jdx] for m in models])
            if jdx == 2:
                values = wrap_direction(values, center)
            if len(values):
                row_lo[jdx] = values.min() if row_lo[jdx] is None else min(row_lo[jdx], values.min())
                row_hi[jdx] = values.max() if row_hi[jdx] is None else max(row_hi[jdx], values.max())

            if row == 0 and col == 0:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

            if ax.get_legend() is not None:
                ax.get_legend().remove()

    tick_spacing = {0: 0.5, 1: 0.5, 2: 30.0}

    for row, (lo, hi) in enumerate(zip(row_lo, row_hi)):
        if lo is None or hi is None:
            continue
        step = tick_spacing[row]
        lo = np.floor(lo / step) * step
        hi = np.ceil(hi / step) * step
        for col in range(n_cols):
            axes[row, col].set_ylim(lo, hi)
            axes[row, col].set_yticks(
                np.arange(lo, hi + step, step)
            )

    for row in range(3):
        for col in range(n_cols):
            letter = chr(ord("A") + row * n_cols + col)
            axes[row, col].annotate(
                f"({letter})",
                xy=(0.015, 0.98), xycoords=axes[row, col].transAxes,
                ha="left", va="top",
                fontsize=13, fontweight="bold", color="red",
            )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=len(legend_handles),
        fontsize=11,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.8,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.97])

    for (season_title, first_col, last_col) in season_titles:
        x = axes[0, first_col].get_position().x0
        x2 = axes[0, last_col].get_position().x1
        fig.text((x + x2) / 2, 1.0, season_title,
                 ha="center", va="top", fontsize=13, fontweight="bold")

    plt.savefig(
        "ts_forecast2.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

if __name__ == "__main__":
    points = np.array([[-27.39, -42.76],
                       [-24.39, -46.26]
                       ])
    dates = ['2010-01-01 00:00:00', '2010-07-01 00:00:00']
    models = [
                "mlp_0_timeenc",
                "lstm_mlp_0_timeenc",
                "lstm_cnn_0_timeenc",
             ]
    folder = 'advs2'

    # curves(models, "val_losses")
    # curves(models, "per_model")
    # heatmap(models, 0, folder)
    # heatmap(models, 6)
    # heatmap(models, 12)
    # heatmap(models, 18)
    # heatmap(models, 24)
    # rmse_horz(models, 24, folder)
    # ioa_horz(models, 24, folder)
    # time_series(models, dates, points, 24, folder)
    # time_series2(models, dates, points, 24, folder)
    # nowcasting_time_series(models,['2010-01-01 00:00:00'], points, 364, folder)
    # nowcasting_time_series2(models, ['2010-01-01 00:00:00'], 364, folder)
    # nowcasting_time_series3(models, ['2010-01-01 00:00:00'], 364, folder)
    nowcasting_time_series4(models, ['2010-01-01 00:00:00'], 364, folder)
    # parity_plot(models)
    # print_rmse_ioa(models, folder)