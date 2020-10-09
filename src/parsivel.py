import glob
from netCDF4 import Dataset, num2date
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def calc_nd(sr_event, size, dsize=.125, dt=60):
    area = (180 * (30 - (size / 2))) * 10 ** -6
    ND = 0
    for speed in sr_event.index:
        if pd.notnull(sr_event.loc[speed]):
            ND += sr_event.loc[speed] / (speed * area * dt * dsize)
    return ND


def paleta():
    cdict = {'red': ((0., 1, 1),
                     (0.05, 1, 1),
                     (0.11, 0, 0),
                     (0.66, 1, 1),
                     (0.89, 1, 1),
                     (1, 0.5, 0.5)),
             'green': ((0., 1, 1),
                       (0.05, 1, 1),
                       (0.11, 0, 0),
                       (0.375, 1, 1),
                       (0.64, 1, 1),
                       (0.91, 0, 0),
                       (1, 0, 0)),
             'blue': ((0., 1, 1),
                      (0.05, 1, 1),
                      (0.11, 1, 1),
                      (0.34, 1, 1),
                      (0.65, 0, 0),
                      (1, 0, 0))}

    return matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)


def parsivel_2(file):
    nc = Dataset(file, 'r')
    sizes = [round(x, 4) for x in nc.variables['particle_size'][:]]
    d_sizes = [round(x, 4) for x in nc.variables['class_size_width'][:]]
    dt_sizes = dict(zip(sizes, d_sizes))
    vels = [round(x, 4) for x in nc.variables['raw_fall_velocity'][:]]
    lerm_vel = [round(x, 4) for x in nc.variables['fall_velocity_calculated'][:]]
    list_size = collections.OrderedDict(sorted(dt_sizes.items()))

    idx_header = pd.MultiIndex.from_product([sizes, vels], names=['Size', 'Speed'])
    idx_header_lerm = pd.MultiIndex.from_product([sizes, lerm_vel], names=['Size', 'Speed'])
    times = []
    data = []
    # Tokay filter for rainy minutes
    events = [i for i, e in enumerate(nc.variables['number_detected_particles']) if (e > 10) and
              (nc.variables['precip_rate'][i] > 0.1)]
    for i in events:
        dates = nc.variables['time'][i]
        time = num2date(dates, nc.variables['time'].units)
        times.append(time)
        dsd = np.ndarray.flatten(nc.variables['raw_spectrum'][i][:])
        data.append(dsd)

    dsd_data_final = pd.DataFrame(data=data)
    dsd_data_final.columns = idx_header
    dsd_data_final.index = times
    df_lermithe = dsd_data_final
    df_lermithe.columns = idx_header_lerm

    idx_events = dsd_data_final.index

    # # df_drisd_vel = dsd_data_final.pivot(index='Speed', columns='Size', values=0)
    #
    # plt.pcolormesh(df_drisd_vel)
    # plt.colorbar()
    # plt.show()

    for event_ext in idx_events:
        sr_nd = pd.Series(index=sorted(list_size))
        for size in list_size:
            df_size = dsd_data_final.xs(size, level=0, axis=1)
            sr_event = df_size.loc[event_ext]
            dsize = list_size[size]

            nd = calc_nd(sr_event=sr_event, size=size, dsize=dsize)
            sr_nd.loc[size] = nd
        fig, ax = plt.subplots()
        ax.scatter(x=sr_nd.index, y=sr_nd, marker='*')
        ax.set_yscale('log')
        plt.ylim(10 ** -1, 10 ** 4)
        plt.show()
        plt.close('all')



    print(nc)


def main():
    file = glob.glob('../data/*.cdf')
    parsivel_2(file[0])


if __name__ == '__main__':
    main()
