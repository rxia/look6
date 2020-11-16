import h5py
import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal
from scipy import stats
import matplotlib.pyplot as plt

# h5_path='D:/Dropbox/analysis/sgrating/sgrating_prelim_data.h5'
h5_path='D:/Dropbox/analysis/sgrating/hb_20201008.h5'


def h5_read(f):
    cur_dict = dict()
    for key in f.keys():
        if isinstance(f[key], h5py.Group):
            cur_dict[key] = h5_read(f[key])
        else:
            data_content = np.array(f[key])
            if np.ndim(data_content)==2 and np.min(data_content.shape)==1:
                data_content = np.squeeze(data_content)
            if np.ndim(data_content)>=2:
                data_content = np.transpose(data_content)
            cur_dict[key] = data_content

    return cur_dict


with h5py.File(h5_path, 'r') as f:
    data = h5_read(f)

time_win_plot = [-300, 700]
ts_plot = np.arange(time_win_plot[0], time_win_plot[1], 1)


def realign(times, time_win=time_win_plot, data_spk=data['spk'], ts=data['ts']):
    new_data = []
    for i in range(data_spk.shape[0]):
        data_in_range = data_spk[i, (ts >= times[i]+time_win[0]) & (ts < times[i]+time_win[1])]
        if ts[0] > (times[i]+time_win[0]):
            data_in_range = np.concatenate((np.ones((int(ts[0]-(times[i]+time_win[0])-1), data_in_range.shape[1]))*np.nan, data_in_range))
        if ts[-1] < (times[i]+time_win[1]):
            data_in_range = np.concatenate((data_in_range, np.ones((int(times[i]+time_win[1]-ts[-1]-1), data_in_range.shape[1]))*np.nan))

        new_data.append(data_in_range)
    return np.stack(new_data)


def select_visual_unit(data, ts, time_win_1, time_win_2):
    selected = []
    for u in range(data.shape[2]):
        fr_1 = data[:, (ts >= time_win_1[0]) & (ts < time_win_1[1]), u].mean(axis=1)
        fr_2 = data[:, (ts >= time_win_2[0]) & (ts < time_win_2[1]), u].mean(axis=1)
        _, p_u = sp.stats.ttest_ind(fr_1, fr_2)
        cohen_d = (fr_2 - fr_1).mean() / (fr_2 - fr_1).std()
        selected.append((p_u < 0.01) & (cohen_d > 1))
        print([cohen_d, p_u, selected[u]])
    selected = np.stack(selected)
    return selected


data['trial_info'] = pd.DataFrame.from_dict(data['trial_info'])
data['spk'] = data['spk'] * 1000

##
# Plot PSTH for the whole trial

def plot_varied_times(ax, times):
    left = np.mean(times) - np.std(times)
    right = np.mean(times) + np.std(times)
    plt.fill_betweenx(ax.get_ylim(), left, right, alpha=0.4)


plt.figure()
ax = plt.gca()
plt.plot(data['ts'], data['spk'].mean(axis=(0, 2)))
plot_varied_times(ax, data['trial_info']['st1_on'])
plot_varied_times(ax, data['trial_info']['st1_acquired'])
plot_varied_times(ax, data['trial_info']['st2_acquired'])
##
# PSTH realigned to different events

def SmoothTrace(data, sk_std=None, fs=1.0, ts=None, axis=1):
    if ts is None:     # use fs to determine ts
        ts = np.arange(0, data.shape[axis])*(1.0/fs)
    else:              # use ts to determine fs
        fs = 1.0/np.mean(np.diff(ts))
    if sk_std is not None:  # condition for using smoothness kernel
        ts_interval = 1.0/fs  # get sampling interval
        kernel_std = sk_std / ts_interval  # std in frames
        kernel_len = int(np.ceil(kernel_std) * 3 * 2 + 1)  # num of frames, 3*std on each side, an odd number
        smooth_kernel = sp.signal.gaussian(kernel_len, kernel_std)
        smooth_kernel = smooth_kernel / smooth_kernel.sum()  # normalized smooth kernel
        # convolution using fftconvolve(), which is faster than convolve()
        data_smooth = sp.ndimage.convolve1d(data, smooth_kernel, mode='reflect', axis=axis)
    else:
        data_smooth = data

    return data_smooth


to_plot = ['st1_on', 'st1_acquired', 'st2_on', 'st2_acquired']
plt.figure()
selected_units = select_visual_unit(realign(data['trial_info']['st1_on']), ts_plot, [-300,0], [100,400])
for i in range(4):
    plt.subplot(2, 2, i+1)
    spk_realigned = realign(data['trial_info'][to_plot[i]], data_spk=data['spk'][:,:,selected_units])
    plt.plot(ts_plot, SmoothTrace(np.nanmean(spk_realigned, axis=0), sk_std=5, ts=ts_plot, axis=0))
    plt.title(to_plot[i])
    plt.axvline(0)
##
# Orientation/sfreq selectivity
time_win_plot = [-300, 700]
ts_plot = np.arange(time_win_plot[0], time_win_plot[1], 1)
st1_units = select_visual_unit(realign(data['trial_info']['st1_on']), ts_plot, [-300,0], [100,400])
fovea_units = select_visual_unit(realign(data['trial_info']['st1_acquired']), ts_plot, [-300,0], [100,400])


def analyse_tuning(epoch, time_win_calculate=[50, 550], selected_units=st1_units, plot=False):
    spk_realigned = realign(data['trial_info'][epoch], time_win_plot, data_spk=data['spk'][:,:,selected_units])
    fr_all = np.nanmean(spk_realigned[:, (ts_plot>=time_win_calculate[0]) & (ts_plot<time_win_calculate[1])], axis=1)

    orientations = np.unique(data['trial_info']['st1_orientation'])
    sfreqs = np.unique(data['trial_info']['st1_sfreq'])
    fr_ij = np.zeros((fr_all.shape[1], len(orientations), len(sfreqs)))
    data_i = []
    for i in range(len(orientations)):
        for j in range(len(sfreqs)):
            trials = (data['trial_info']['st1_orientation']==orientations[i]) & (data['trial_info']['st1_sfreq']==sfreqs[j])
            fr_ij[:, i, j] = fr_all[trials].mean(axis=0)
            if j == 3:
                data_i.append(fr_all[trials])
    if plot:
        plt.figure()
        for u in range(fr_ij.shape[0]):
            plt.subplot(4, 7, u+1)
            plt.pcolormesh(fr_ij[u])
            plt.ylabel('orientation')
            plt.xlabel('spatial freq')

    F_all = []
    for u in range(fr_ij.shape[0]):
        F, P = sp.stats.f_oneway(data_i[0][:,u], data_i[1][:,u], data_i[2][:,u], data_i[3][:,u])
        print([F, P])
        # F, P = sp.stats.f_oneway(np.concatenate([data_i[0][:,u],data_i[2][:,u]]), np.concatenate([data_i[1][:,u],data_i[3][:,u]]))
        # F = np.std(np.concatenate(data_i)[:,u])/np.sum([np.std(data_i[i][:,u]) for i in range(len(data_i))])
        F_all.append(F)

    return np.log(np.stack(F_all))


_, h_ax = plt.subplots(2, 2, sharex=True, sharey=True)

plt.axes(h_ax[0, 0])
plt.plot(analyse_tuning('st1_on'), analyse_tuning('st2_acquired'), '+')
plt.plot(h_ax[0, 0].get_xlim(), h_ax[0, 0].get_xlim(), 'k--')
h_ax[0, 0].set_aspect('equal')
plt.axis('square')
plt.xlabel('F value (Before foveal view)')
plt.ylabel('F value (After foveal view)')

plt.axes(h_ax[0, 1])
plt.plot(analyse_tuning('st1_on', [50,300]), analyse_tuning('st1_on', [300,550]), '+')
plt.plot(h_ax[0, 1].get_xlim(), h_ax[0, 1].get_xlim(), 'k--')
h_ax[0, 1].set_aspect('equal')
plt.axis('square')
plt.xlabel('F value (Before foveal view, early)')
plt.ylabel('F value (Before foveal view, late)')

plt.axes(h_ax[1, 0])
plt.plot(analyse_tuning('st1_on', selected_units=fovea_units), analyse_tuning('st2_acquired', selected_units=fovea_units), '+')
plt.plot(h_ax[1, 0].get_xlim(), h_ax[1, 0].get_xlim(), 'k--')
h_ax[1, 0].set_aspect('equal')
plt.axis('square')
plt.xlabel('F value (Before foveal view)')
plt.ylabel('F value (After foveal view)')

plt.axes(h_ax[1, 1])
plt.plot(analyse_tuning('st1_on', [50,300], selected_units=fovea_units), analyse_tuning('st1_on', [300,550], selected_units=fovea_units), '+')
plt.plot(h_ax[1, 1].get_xlim(), h_ax[1, 1].get_xlim(), 'k--')
h_ax[1, 1].set_aspect('equal')
plt.axis('square')
plt.xlabel('F value (Before foveal view, early)')
plt.ylabel('F value (Before foveal view, late)')