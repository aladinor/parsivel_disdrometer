import glob
from netCDF4 import Dataset, num2date
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calc_nd(sr_event, size, dsize=.125, dt=60):
    area = (180 * (30 - (size / 2))) * 10 ** -6
    ND = 0
    for speed in sr_event.index:
        if pd.notnull(sr_event.loc[speed]):
            ND += sr_event.loc[speed] / (speed * area * dt * dsize)
    return ND


def parsivel_2(file):
    nc = Dataset(file, 'r')
    sizes = [round(x, 4) for x in nc.variables['particle_size'][:]]
    d_sizes = [round(x, 4) for x in nc.variables['class_size_width'][:]]
    dt_sizes = dict(zip(sizes, d_sizes))
    vels = [round(x, 4) for x in nc.variables['raw_fall_velocity'][:]]

    list_size = collections.OrderedDict(sorted(dt_sizes.items()))

    idx_header = pd.MultiIndex.from_product([sizes, vels], names=['Size', 'Speed'])
    times = []
    data = []
    events = [i for i, e in enumerate(nc.variables['number_detected_particles']) if e >= 1000]
    for i in events:
        dates = nc.variables['time'][i]
        time = num2date(dates, nc.variables['time'].units)
        times.append(time)
        dsd = np.ndarray.flatten(nc.variables['raw_spectrum'][i][:])
        data.append(dsd)

    dsd_data_final = pd.DataFrame(data=data)
    dsd_data_final.columns = idx_header
    dsd_data_final.index = times

    num_gotas = 10
    sr_sum_drisd = dsd_data_final.sum(axis=1)
    idx_events = sr_sum_drisd[sr_sum_drisd > num_gotas].index

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
