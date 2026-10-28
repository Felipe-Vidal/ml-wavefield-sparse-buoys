import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
import os

def reduce_df(df_file):
    # Load the dataset
    df = pd.read_pickle('data/f58a282669b2b695c159f2d5f79df76a.pickle')
    lat = df.index.get_level_values('latitude')
    lon = df.index.get_level_values('longitude')
    # Define polygon by (lat,lon) coordinates
    poly = Polygon([
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
        (-28.65, -49.5)])
    # Create mask using shapely
    mask = np.array([poly.contains(Point(lat, lon)) for lat, lon in zip(lat, lon)])
    df2 = df[mask]
    # Save the dataframe
    df2.to_pickle(df_file)

def get_stats(df):
    stats = {}
    for coord in ['latitude','longitude']:
        stats[coord+'_mean'] = df.index.get_level_values(coord).to_numpy().mean()
        stats[coord+'_std']  = df.index.get_level_values(coord).to_numpy().std()
    for col in df.columns:
        stats[col+'_mean'] = df[col].mean()
        stats[col+'_std']  = df[col].std()
    np.savez_compressed('data/statistics.npz', **stats)
    return stats

def buoys_data(df,coords):
    buoys = pd.DataFrame((df.index.get_level_values('valid_time').unique()-pd.Timestamp('2000-01-01 00:00:00'))/pd.Timedelta(hours=1))
    for i in range(len(coords)):
        epsilon = 1e-2
        buoy = coords[i]
        dph = df.iloc[np.logical_and(df.index.get_level_values('latitude') < buoy[0] + epsilon, df.index.get_level_values('latitude') > buoy[0] - epsilon)]
        dph = dph.iloc[np.logical_and(dph.index.get_level_values('longitude') < buoy[1] + epsilon, dph.index.get_level_values('longitude') > buoy[1] - epsilon)]
        buoys[['Tm'+str(i),'Hscos'+str(i),'Hssin'+str(i)]]=np.column_stack([dph['Tm'].to_numpy(), dph['Hscos'].to_numpy(), dph['Hssin'].to_numpy()])

    return buoys

def build_buoys_dataset(df, buoys=(0, 1, 2), max_lag=12, step=6):
    """
    Create dataset for multiple buoys with lagged features and save to .npz.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame.
    buoys : tuple
        List/tuple of buoy IDs to include (e.g. (0,1,2)).
    max_lag : int
        Maximum lag in hours.
    step : int
        Lag step in hours (e.g. 6h, 12h...).
    outdir : str
        Directory where the .npz file will be saved.
    name : str
        Prefix name for the output file.

    Returns
    -------
    pd.DataFrame
        Consolidated DataFrame with valid_time, buoy columns and lagged versions.
    """
    result = df[['valid_time']].copy()
    for buoy in buoys:
        cols = [f'Tm{buoy}', f'Hscos{buoy}', f'Hssin{buoy}']
        temp = df[cols].copy()
        # add lagged columns for each buoy
        for lag in range(step, max_lag + 1, step):
            for c in cols:
                temp[f'{c}_lag{lag}h'] = df[c].shift(lag)
        result = pd.concat([result, temp], axis=1)
    result = result.dropna().reset_index(drop=True)

    return result

def prep_data(df, max_lag):
    df = df.reset_index()
    df['valid_time'] = (df['valid_time']-pd.Timestamp('2000-01-01 00:00:00'))/pd.Timedelta(hours=1)
    return df[df['valid_time']>=max_lag].reset_index(drop=True)

def split_train_test(df):
    test_start = (pd.to_datetime('2010-01-01 00:00:00')-pd.Timestamp('2000-01-01 00:00:00'))/pd.Timedelta(hours=1)
    df_train = df[df['valid_time'] < test_start]
    df_test = df[df['valid_time'] >= test_start]
    return df_train, df_test

def cnn_label(df):
    df = df.copy()

    # Unique coordinates
    lats = np.sort(df['latitude'].unique())
    lons = np.sort(df['longitude'].unique())

    # Full MultiIndex (lat, lon)
    full_cols = pd.MultiIndex.from_product([lats, lons], names=['latitude','longitude'])

    # Pivot helper
    def pivot_var(var):
        p = df.pivot_table(
            index='valid_time',
            columns=['latitude','longitude'],
            values=var,
            aggfunc='first')
        p = p.reindex(columns=full_cols)
        return p.fillna(-10)

    tm = pivot_var('Tm')
    hscos = pivot_var('Hscos')
    hssin = pivot_var('Hssin')

    # Convert to arrays
    tm_a = tm.to_numpy()
    hscos_a = hscos.to_numpy()
    hssin_a = hssin.to_numpy()

    # Number of coordinate pairs
    n_pairs = len(lats) * len(lons)

    # Interleave Tm, Hscos, Hssin for each coordinate
    features = np.empty((tm_a.shape[0], 3 * n_pairs))

    for i in range(n_pairs):
        features[:, 3*i]     = tm_a[:, i]
        features[:, 3*i + 1] = hscos_a[:, i]
        features[:, 3*i + 2] = hssin_a[:, i]

    # Prepend valid_time
    t = tm.index.to_numpy().reshape(-1, 1)
    label = np.hstack([t, features])

    # Return as DataFrame
    label = pd.DataFrame(label)
    cols = ['valid_time']
    for lat in range(lats.size):
        for lon in range(lons.size):
            cols.extend([f'Tm_{lat},{lon}', f'Hscos_{lat},{lon}', f'Hssin_{lat},{lon}'])
    label.columns = cols

    return label, full_cols.to_frame(index=False)

def time_encoding_explicit(t):
    periods = np.array([#24,         # daily
                        #24*29.50,   # lunar cycle
                        #24*91.25,   # season
                        24*365.25]) # annual
    x = 2 * np.pi * t[:,None]/periods[None,:]
    cos = np.cos(x)
    sin = np.sin(x)
    return np.concatenate([cos, sin], axis=1)

def encode_time_in_df(df):
    """Replace first column (valid_time) with time encoding."""
    t = df.iloc[:, 0].to_numpy()
    encoded = time_encoding_explicit(t)
    # df_encoded = df.iloc[:, 1:].reset_index(drop=True)
    df_encoded = df.copy()

    for i in range(encoded.shape[1]):
        df_encoded.insert(i+1, f'time_feat_{i}', encoded[:, i])
    
    return df_encoded

def to_cnn(arr, row_lengths, n_cols=17):
    out = []
    idx = 0
    for L in row_lengths:
        row = arr[idx:idx+L]
        idx += L
        pad = [False] * (n_cols - L)
        out.extend(pad + list(row))
    return np.array(out, dtype=bool).reshape(14,17).T

def custom_masks():
    row_lengths = [
        17,
        15,15,15,15,15,15,15,
        14,
        12,
        10,
        7,
        2,
        1]
    piml = np.array([False,False,False, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                                 False, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                                 False, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                                 False, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                                 False, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                                 False, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                                 False, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                                 False,False, True, True, True, True, True, True, True, True, True, True, True, True, True,
                                       False,False,False, True, True, True, True, True, True, True, True, True, True, True,
                                                   False,False,False, True, True, True, True, True, True, True, True, True,
                                                               False,False,False,False, True, True, True, True, True, True,
                                                                                 False,False,False,False,False,False, True,
                                                                                                               False,False,
                                                                                                                     False])
    half1 = np.array([ True,False, True,False, True,False, True,False, True,False, True,False, True,False, True,False, True,
                                  False, True,False, True,False, True,False, True,False, True,False, True,False, True,False,
                                   True,False, True,False, True,False, True,False, True,False, True,False, True,False, True,
                                  False, True,False, True,False, True,False, True,False, True,False, True,False, True,False,
                                   True,False, True,False, True,False, True,False, True,False, True,False, True,False, True,
                                  False, True,False, True,False, True,False, True,False, True,False, True,False, True,False,
                                   True,False, True,False, True,False, True,False, True,False, True,False, True,False, True,
                                  False, True,False, True,False, True,False, True,False, True,False, True,False, True,False,
                                        False, True,False, True,False, True,False, True,False, True,False, True,False, True,
                                                     True,False, True,False, True,False, True,False, True,False, True,False,
                                                                False, True,False, True,False, True,False, True,False, True,
                                                                                  False, True,False, True,False, True,False,
                                                                                                                False, True,
                                                                                                                      False])
    half2 = np.array([False, True,False, True,False, True,False, True,False, True,False, True,False, True,False, True,False,
                                   True,False, True,False, True,False, True,False, True,False, True,False, True,False, True,
                                  False, True,False, True,False, True,False, True,False, True,False, True,False, True,False,
                                   True,False, True,False, True,False, True,False, True,False, True,False, True,False, True,
                                  False, True,False, True,False, True,False, True,False, True,False, True,False, True,False,
                                   True,False, True,False, True,False, True,False, True,False, True,False, True,False, True,
                                  False, True,False, True,False, True,False, True,False, True,False, True,False, True,False,
                                   True,False, True,False, True,False, True,False, True,False, True,False, True,False, True,
                                         True,False, True,False, True,False, True,False, True,False, True,False, True,False,
                                                    False, True,False, True,False, True,False, True,False, True,False, True,
                                                                 True,False, True,False, True,False, True,False, True,False,
                                                                                   True,False, True,False, True,False, True,
                                                                                                                 True,False,
                                                                                                                       True])
    four1 = np.array([ True,False, True,False, True,False, True,False, True,False, True,False, True,False, True,False, True,
                                  False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
                                   True,False, True,False, True,False, True,False, True,False, True,False, True,False, True,
                                  False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
                                   True,False, True,False, True,False, True,False, True,False, True,False, True,False, True,
                                  False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
                                   True,False, True,False, True,False, True,False, True,False, True,False, True,False, True,
                                  False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
                                        False, True,False, True,False, True,False, True,False, True,False, True,False, True,
                                                    False,False,False,False,False,False,False,False,False,False,False,False,
                                                                False, True,False, True,False, True,False, True,False, True,
                                                                                  False,False,False,False,False,False,False,
                                                                                                                False, True,
                                                                                                                      False])
    four2 = np.array([ True,False, True,False, True,False, True,False, True,False, True,False, True,False, True,False, True,
                                  False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
                                  False, True,False, True,False, True,False, True,False, True,False, True,False, True,False,
                                  False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
                                   True,False, True,False, True,False, True,False, True,False, True,False, True,False, True,
                                  False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
                                  False, True,False, True,False, True,False, True,False, True,False, True,False, True,False,
                                  False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
                                        False, True,False, True,False, True,False, True,False, True,False, True,False, True,
                                                    False,False,False,False,False,False,False,False,False,False,False,False,
                                                                 True,False, True,False, True,False, True,False, True,False,
                                                                                  False,False,False,False,False,False,False,
                                                                                                                False, True,
                                                                                                                      False])
    coast = np.array([ True, True, True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
                                   True,False,False, True,False,False,False,False,False,False,False,False,False,False,False,
                                   True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
                                   True,False,False,False,False,False,False,False,False,False,False, True,False,False,False,
                                   True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
                                   True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
                                   True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
                                   True, True,False,False,False,False,False,False,False,False,False,False,False,False,False,
                                         True, True, True,False,False,False,False,False,False,False,False,False,False,False,
                                                     True, True, True,False,False,False,False,False,False,False, True,False,
                                                                 True, True, True, True,False,False,False,False,False,False,
                                                                                   True, True, True, True, True, True,False,
                                                                                                                 True, True,
                                                                                                                       True])


    np.savez_compressed('data/masks.npz',
                        piml=piml.astype(np.bool),
                        half1=half1.astype(np.bool),
                        half2=half2.astype(np.bool),
                        four1=four1.astype(np.bool),
                        four2=four2.astype(np.bool),
                        coast=coast.astype(np.bool),
                        piml_cnn=to_cnn(piml, row_lengths).astype(np.bool),
                        half1_cnn=to_cnn(half1, row_lengths).astype(np.bool),
                        half2_cnn=to_cnn(half2, row_lengths).astype(np.bool),
                        four1_cnn=to_cnn(four1, row_lengths).astype(np.bool),
                        four2_cnn=to_cnn(four2, row_lengths).astype(np.bool),
                        coast_cnn=to_cnn(coast,row_lengths).astype(np.bool))

if __name__ == '__main__':
    masks = True
    if masks:
        custom_masks()
    else:
        # dataframe
        df_file = "data/dataframe.pickle"
        if not os.path.exists(df_file):
            reduce_df(df_file)
        df = pd.read_pickle(df_file)

        # transforming (dir,Tm,Hs) -> (Tm,Hs*cos(dir),Hs*sin(dir))
        df = df.rename(columns={"mpts": "Tm"})
        df["Hscos"] = df["shts"] * np.cos(df["mdts"])
        df["Hssin"] = df["shts"] * np.sin(df["mdts"])
        df = df.drop(columns=["mdts","shts"])

        # z-norm
        stats = get_stats(df)
        for coord in ['latitude','longitude']:
            idx = df.index
            levels = list(idx.levels)
            codes = list(idx.codes)
            level_vals = (idx.get_level_values(coord) - stats[f"{coord}_mean"]) / stats[f"{coord}_std"]
            codes1, levels1 = pd.factorize(level_vals)
            codes[idx.names.index(coord)] = codes1
            levels[idx.names.index(coord)] = pd.Index(levels1)
            df.index = pd.MultiIndex(codes=codes, levels=levels, names=idx.names)
        for col in ['Tm','Hscos','Hssin']:
            df[col] = (df[col]-stats[col+'_mean'])/stats[col+'_std']

        # buoys
        buoys_deg = np.array([(-28.39, -46.76), (-27.39, -42.76), (-24.39, -41.76)])
        buoys_coords = np.matmul(buoys_deg-np.tile([stats["latitude_mean"],stats["longitude_mean"]],(3, 1)),
                                    [(1/stats["latitude_std"],0),(0,1/stats["longitude_std"])])
        buoys_t = buoys_data(df,buoys_coords)
        max_lag = 12 # hours
        step = 6 # horas
        buoys_input = build_buoys_dataset(buoys_t, buoys=(0,1,2), max_lag=max_lag, step=step)

        # mlp/lstm label
        label = prep_data(df,max_lag)

        # cnn label
        label_cnn, cnn_coords = cnn_label(label)

        # -----

        # split train/test
        buoys_train, buoys_test = split_train_test(buoys_input)
        label_train, label_test = split_train_test(label)
        cnn_train,   cnn_test   = split_train_test(label_cnn)

        # saving without time encoding
        np.savez_compressed('data/train.npz',
                            buoys_coords=buoys_coords.astype(np.float32),
                            buoys=buoys_train.to_numpy().astype(np.float32),
                            label=label_train.to_numpy().astype(np.float32),
                            label_cnn = cnn_train.to_numpy().astype(np.float32),
                            cnn_coords = cnn_coords.to_numpy().astype(np.float32))
        np.savez_compressed('data/test.npz',
                            buoys_coords=buoys_coords.astype(np.float32),
                            buoys=buoys_test.to_numpy().astype(np.float32),
                            label=label_test.to_numpy().astype(np.float32),
                            label_cnn = cnn_test.to_numpy().astype(np.float32),
                            cnn_coords = cnn_coords.to_numpy().astype(np.float32))

        # -----

        # time encoding
        buoys_train, buoys_test = encode_time_in_df(buoys_train), encode_time_in_df(buoys_test)
        label_train, label_test = encode_time_in_df(label_train), encode_time_in_df(label_test)
        cnn_train,   cnn_test   = encode_time_in_df(cnn_train), encode_time_in_df(cnn_test)

        # saving with time encoding
        np.savez_compressed('data/train_timeenc.npz',
                            buoys_coords=buoys_coords.astype(np.float32),
                            buoys=buoys_train.to_numpy().astype(np.float32),
                            label=label_train.to_numpy().astype(np.float32),
                            label_cnn = cnn_train.to_numpy().astype(np.float32),
                            cnn_coords = cnn_coords.to_numpy().astype(np.float32))
        np.savez_compressed('data/test_timeenc.npz',
                            buoys_coords=buoys_coords.astype(np.float32),
                            buoys=buoys_test.to_numpy().astype(np.float32),
                            label=label_test.to_numpy().astype(np.float32),
                            label_cnn = cnn_test.to_numpy().astype(np.float32),
                            cnn_coords = cnn_coords.to_numpy().astype(np.float32))