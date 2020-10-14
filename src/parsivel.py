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


def lermithe(D):
    return 9.24*(1.-np.exp(-6.8 * (D/10.) ** 2 - 4.88 * (D/10.)))


def lina_filter(sizes, vels, tresh=2):
    filtered = []
    for idx, i in enumerate(sizes):
        lina_vel_up = lermithe(np.array(i)) + tresh
        lina_vel_down = lermithe(np.array(i)) - tresh
        fil = np.ma.masked_where(vels >= lina_vel_up, vels)
        fil = np.ma.masked_where(fil <= lina_vel_down, fil)
        filtered.append(np.ma.getmaskarray(fil))
    return np.array(filtered)


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
    times = []
    data_spec = []
    # Tokay filter for rainy minutes
    events = [i for i, e in enumerate(nc.variables['number_detected_particles']) if (e > 100) and
              (nc.variables['precip_rate'][i] > 0.5)]
    spectra_filter = lina_filter(sizes=sizes, vels=vels, tresh=2)

    for i in events:
        dates = nc.variables['time'][i]
        time = num2date(dates, nc.variables['time'].units)
        times.append(time)
        spectrum = np.array(nc.variables['raw_spectrum'][i][:])
        spectrum = np.ma.masked_array(spectrum, mask=spectra_filter.T, fill_value=0)
        spectrum = spectrum.filled()
        # plt.pcolormesh(sizes, vels, spectrum, cmap=paleta())
        # plt.plot(sizes, lermithe(np.array(sizes)))
        # plt.plot(sizes, lermithe(np.array(sizes)) + 2.1, 'r')
        # plt.plot(sizes, lermithe(np.array(sizes)) - 2.1, 'r')
        # plt.plot(sizes, lermithe(np.array(sizes)) + lermithe(np.array(sizes)) * 0.4, 'c')
        # plt.plot(sizes, lermithe(np.array(sizes)) - lermithe(np.array(sizes)) * 0.4, 'c')
        # plt.ylim([0, 10])
        # plt.xlim([0, 10])
        # plt.show()
        spectrum = np.ndarray.flatten(spectrum)
        data_spec.append(spectrum)

    dsd_data_final = pd.DataFrame(data=data_spec, columns=pd.MultiIndex.from_product([sizes, vels],
                                                                                     names=['Size', 'Speed']),
                                  index=times)

    for event_ext in dsd_data_final.index:
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


def main():
    file = glob.glob('../data/*.cdf')
    parsivel_2(file[0])


if __name__ == '__main__':
    main()
