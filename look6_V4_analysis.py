import h5py
import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from PyNeuroAna import *


def h5_read(f, keys=None, excluded_keys=[]):
    cur_dict = dict()
    if keys is None:
        keys = f.keys()
    for key in keys:
        if key not in excluded_keys:
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


def load_data_1_session(h5_path, session, do_not_load=['lfp']):
    with h5py.File(h5_path, 'r') as f:
        data = h5_read(f[session], excluded_keys=do_not_load)
        data['spk'] = data['spk'].astype(np.float) * 1000
    return data


# h5_path='D:/Dropbox/analysis/donatas_data/look6_V4_data.h5'
h5_path='D:/Analysis/look6/data/V4_delay.h5'
with h5py.File(h5_path, 'r') as f:
    sessions = list(f.keys())
data = load_data_1_session(h5_path, sessions[0])
time_win_plot = [-600, 700]
ts_plot = np.arange(time_win_plot[0], time_win_plot[1], 1)


# with h5py.File(h5_path, 'r+') as f:
#     sessions = list(f.keys())
#     del f['hb_20180216_1']['lfp']


##
def realign(times, data_to_align, ts=data['ts'], time_win=time_win_plot, selected_trials=None):
    new_data = []
    if selected_trials is None:
        selected_trials = np.arange(data_to_align.shape[0])
    if np.issubdtype(selected_trials.dtype, np.bool_):
        selected_trials = np.flatnonzero(selected_trials)
    for i in selected_trials:
        data_in_range = data_to_align[i, (ts >= times[i] + time_win[0]) & (ts < times[i] + time_win[1])]
        if ts[0] > (times[i] + time_win[0]):
            data_in_range = np.concatenate((np.ones(
                (int(ts[0] - (times[i] + time_win[0])), data_in_range.shape[1])) * np.nan, data_in_range))
        if ts[-1] < (times[i] + time_win[1]):
            data_in_range = np.concatenate((data_in_range, np.ones(
                (int(times[i] + time_win[1] - ts[-1] - 1), data_in_range.shape[1])) * np.nan))
        if data_in_range.shape[0] > time_win[1] - time_win[0]:
            data_in_range = data_in_range[:(time_win[1] - time_win[0])]
        new_data.append(data_in_range)
    # return new_data
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

# data['st1_units'] = select_visual_unit(realign(data['trial_info']['st1_on'], data['spk']), ts_plot, [-50, 50], [50, 150])
# data['fovea_units'] = select_visual_unit(realign(data['trial_info']['st1_acquired'], data['spk']), ts_plot, [-50, 50], [50, 150])
# data['peripheral_units'] = data['st1_units'] * (~data['fovea_units'])

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


def get_fr_by_conds(data, epoch, time_win_calculate, data_type='spk', selected_units=None, selected_trials=None, avg_trials=True, conds=['memory_location','same_memory_st1']):
    if selected_units is None:
        selected_units = np.ones(data[data_type].shape[2])>0
    else:
        selected_units = data[selected_units]

    if selected_trials is None:
        selected_trials = np.arange(data[data_type].shape[0])

    if data_type == 'spk' or data_type == 'lfp':
        ts = data['ts']
        data_realigned = realign(data['trial_info'][epoch]-data['trial_info']['first_display'], data_to_align=data[data_type][:,:,selected_units], ts=ts, time_win=time_win_calculate, selected_trials=selected_trials)
    else:
        ts = data['ts_{}'.format(data_type)]
        data_realigned = data[data_type][:, (ts>=time_win_calculate[0])&(ts<time_win_calculate[1])]
    fr_all = np.nanmean(data_realigned, axis=1)

    # fr_all = fr_all/fr_all.mean(axis=0)[None,:]

    cond1 = data['trial_info'][conds[0]][selected_trials] + 0
    cond1_unique = np.unique(cond1)
    if len(conds)==2:
        cond2 = data['trial_info'][conds[1]][selected_trials] + 0
        cond2_unique = np.unique(cond2)
    fr_by_conds = []
    if avg_trials:
        for i in range(len(cond1_unique)):
            data_i = []
            trials = cond1==cond1_unique[i]
            if len(conds) == 1:
                fr_by_conds.append(np.nanmean(fr_all[trials], axis=0))
            elif len(conds) == 2:
                for j in range(len(cond2_unique)):
                    trials = (cond2==cond2_unique[j]) & (cond1==cond1_unique[i])
                    data_i.append(np.nanmean(fr_all[trials], axis=0))
                fr_by_conds.append(np.stack(data_i, axis=0))
        fr_by_conds = np.stack(fr_by_conds, axis=0)
    else:
        for i in range(len(cond1_unique)):
            data_i = []
            trials = cond1==cond1_unique[i]
            if len(conds) == 1:
                selected_fr = fr_all[trials]
                data_i = selected_fr[(np.isnan(selected_fr).sum(axis=1))==0, :]
            elif len(conds) == 2:
                for j in range(len(cond2_unique)):
                    trials = (cond2==cond2_unique[j]) & (cond1==cond1_unique[i])
                    selected_fr = fr_all[trials]
                    data_i.append(selected_fr[(np.isnan(selected_fr).sum(axis=1))==0, :])
            fr_by_conds.append(data_i)


    return fr_by_conds


##
to_plot = ['memory_on', 'target_on']
h_fig, h_ax = plt.subplots(1, len(to_plot), sharey=True)
h_ax = h_ax.flatten()
selected_units = np.arange(data['spk'].shape[2])
for i in range(len(to_plot)):
    plt.axes(h_ax[i])
    times = data['trial_info'][to_plot[i]]-data['trial_info']['first_display']
    spk_realigned = realign(times, data_to_align=data['spk'][:,:,selected_units])
    plt.plot(ts_plot, SmoothTrace(np.nanmean(spk_realigned, axis=0), sk_std=5, ts=ts_plot, axis=0))
    plt.plot(ts_plot, SmoothTrace(np.nanmean(spk_realigned, axis=(0,2)), sk_std=5, ts=ts_plot, axis=0), color='black', linewidth=4)
    plt.title(to_plot[i])
    plt.axvline(0)
    plt.xlabel('Time (ms)')
    plt.ylabel('Firing rate (Hz)')


## PCA state space

# conds = ['memory_location','same_memory_st1']
conds = ['memory_location', 'esetup_background_texture_line_angle']
selected_trials = np.flatnonzero(np.isfinite(data['trial_info']['response_maintained']) & (data['trial_info']['esetup_background_texture_on']==1))
data_pca_1 = get_fr_by_conds(data, 'memory_on', [0,500], selected_trials=selected_trials, conds=[conds[0]])
data_pca_2 = get_fr_by_conds(data, 'texture_on_1', [0,500], selected_trials=selected_trials, conds=[conds[1]])
pca_1 = PCA(n_components=1, whiten=True)
pca_1.fit(data_pca_1)
pca_2 = PCA(n_components=1, whiten=True)
pca_2.fit(data_pca_2)

step_size = 10
window_size = 200
n_steps = 50
step_centers = np.arange(n_steps)*step_size
step_centers = (np.arange(n_steps)-10)*step_size
epoch = 'memory_on'
colors = cm.get_cmap('rainbow')

h_fig, h_ax = plt.subplots(2, 2, figsize=[15,15])
plt.axes(h_ax[0, 0])
ts_psth = np.arange(step_centers[0]-window_size/2, step_centers[-1]+window_size/2)
spk_realigned = realign(data['trial_info'][epoch]-data['trial_info']['first_display'], data_to_align=data['spk'][:,:,selected_units], time_win=[step_centers[0]-window_size/2, step_centers[-1]+window_size/2])
plt.plot(ts_psth, SmoothTrace(np.nanmean(spk_realigned, axis=0), sk_std=5, ts=ts_plot, axis=0))
plt.plot(ts_psth, SmoothTrace(np.nanmean(spk_realigned, axis=(0,2)), sk_std=5, ts=ts_plot, axis=0), color='black', linewidth=4)
vline1 = plt.axvline(step_centers[0], color='y', linewidth=4)
# plt.axes(h_ax[0, 1])
# ts_psth = np.arange(step_centers[0]-window_size/2, step_centers[-1]+window_size/2)
# spk_realigned = realign(data['trial_info']['target_on']-data['trial_info']['first_display'], data_to_align=data['spk'][:,:,selected_units], time_win=[step_centers[0]-window_size/2, step_centers[-1]+window_size/2])
# plt.plot(ts_psth, SmoothTrace(np.nanmean(spk_realigned, axis=0), sk_std=5, ts=ts_plot, axis=0))
# plt.plot(ts_psth, SmoothTrace(np.nanmean(spk_realigned, axis=(0,2)), sk_std=5, ts=ts_plot, axis=0), color='black', linewidth=4)
# vline2 = plt.axvline(step_centers[0], color='y', linewidth=4)

for t in range(n_steps):
    plt.axes(h_ax[0, 0])
    vline1.set_xdata([step_centers[t], step_centers[t]])
    # plt.axes(h_ax[0, 1])
    # vline2.set_xdata([step_centers[t], step_centers[t]])

    plt.axes(h_ax[1, 0])
    plt.cla()
    data_to_plot = get_fr_by_conds(data, epoch, [step_centers[t]-window_size/2, step_centers[t]+window_size/2], selected_trials=selected_trials, avg_trials=False, conds=conds)
    for i in range(len(data_to_plot)):
        for j in range(len(data_to_plot[0])):
            transformed_1 = pca_1.transform(data_to_plot[i][j])
            transformed_2 = pca_2.transform(data_to_plot[i][j])
            # plt.scatter(transformed_1, transformed_2, color=colors((i*len(data_to_plot[0])+j)/len(data_to_plot)/len(data_to_plot[0])))
            plt.scatter(transformed_1, transformed_2, color=colors(j/len(data_to_plot[0])))
    plt.xlim([-15, 15])
    plt.xlabel(conds[0])
    plt.ylim([-40, 40])
    plt.ylabel(conds[1])

    plt.axes(h_ax[1, 1])
    plt.cla()
    data_to_plot = get_fr_by_conds(data, epoch,
                                   [step_centers[t] - window_size / 2, step_centers[t] + window_size / 2],
                                   selected_trials=selected_trials, avg_trials=False, conds=conds)
    for i in range(len(data_to_plot)):
        for j in range(len(data_to_plot[0])):
            transformed_1 = pca_1.transform(data_to_plot[i][j])
            transformed_2 = pca_2.transform(data_to_plot[i][j])
            plt.scatter(transformed_1, transformed_2, color=colors(i/len(data_to_plot)))
            # plt.scatter(transformed_1, transformed_2, color=colors(j / len(data_to_plot[0])))
    plt.xlim([-15, 15])
    plt.xlabel(conds[0])
    plt.ylim([-40, 40])
    plt.ylabel(conds[1])

    # plt.axes(h_ax[1, 1])
    # plt.cla()
    # data_to_plot = get_fr_by_conds(data, 'target_on', [step_centers[t]-window_size/2, step_centers[t]+window_size/2], avg_trials=False, conds=conds)
    # for i in range(len(data_to_plot)):
    #     for j in range(len(data_to_plot[0])):
    #         transformed_1 = pca_1.transform(data_to_plot[i][j])
    #         transformed_2 = pca_2.transform(data_to_plot[i][j])
    #         plt.scatter(transformed_1, transformed_2, color=colors((i*len(data_to_plot[0])+j)/len(data_to_plot)/len(data_to_plot[0])))
    # plt.xlim([-15, 15])
    # plt.xlabel('Cue location')
    # plt.ylim([-15, 15])
    # plt.xlabel('Task rule')

    plt.draw()
    plt.pause(0.2)

## Decoding
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
step_size = 128
window_size = 256
n_steps = 14
step_centers = np.arange(n_steps) * step_size - 572


def get_X_y(data, data_type, time_win, selected_trials, cond):
    fr_by_conds = get_fr_by_conds(data, 'memory_on', time_win, data_type=data_type, selected_trials=selected_trials,
                                  avg_trials=False, conds=[cond])
    X = np.concatenate(fr_by_conds)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    y = np.concatenate([np.ones(x.shape[0]) * i for i, x in enumerate(fr_by_conds)])
    return X, y


def select_by_label(a, b):
    selected = []
    for i in range(a.shape[0]):
        selected.append(a[i, int(b[i])])
    return np.stack(selected)


def plot_shaded_err(y, x=step_centers, label=None, color=None):
    y_mean = y.mean(axis=1)
    y_se = y.std(axis=1)/np.sqrt(y.shape[1])
    if color is not None:
        plt.plot(x, y_mean, label=label, color=color)
        plt.fill_between(x, y_mean - y_se, y_mean + y_se, color=color, alpha=0.5)
    else:
        plt.plot(x, y_mean, label=label)
        plt.fill_between(x, y_mean - y_se, y_mean + y_se, alpha=0.5)
    return


def decoding(cond, data_type='spk', selected_trials=None, selected_trials2=None, model='svc', ax1=None, ax2=None, color=None, label=None):
    score_t = []
    psth = []
    if selected_trials is None:
        selected_trials = np.flatnonzero(data['trial_info']['correct_trials'])
    if np.issubdtype(selected_trials.dtype, np.bool_):
        selected_trials = np.flatnonzero(selected_trials)
    for t in range(n_steps):
        current_t_win = [step_centers[t] - window_size / 2, step_centers[t] + window_size / 2]
        X, y = get_X_y(data, data_type, current_t_win, selected_trials, cond)
        if ax2 is not None:
            psth_t = get_fr_by_conds(data, 'memory_on', current_t_win, data_type=data_type, selected_trials=selected_trials,
                                          avg_trials=True, conds=[cond])
            psth.append(np.nanmean(psth_t, axis=1))
        if selected_trials2 is None:
            if model=='svc_prob':
                clf = svm.SVC(probability=True, gamma='scale')
                clf.fit(X, y)
                score_t.append(select_by_label(clf.predict_proba(X), y))
            else:
                if model=='svc':
                    clf = svm.SVC(gamma='scale')
                elif model=='svr':
                    clf = svm.SVR(gamma='scale')
                scores = cross_val_score(clf, X, y, cv=5)
                score_t.append(np.mean(scores))
        else:
            X_test, y_test = get_X_y(data, current_t_win, selected_trials2, cond)
            if model=='svc_prob':
                clf = svm.SVC(probability=True, gamma='scale')
                clf.fit(X, y)
                score_t.append(select_by_label(clf.predict_proba(X_test), y_test))
            else:
                if model=='svc':
                    clf = svm.SVC(gamma='scale')
                elif model=='svr':
                    clf = svm.SVR(gamma='scale')
                clf.fit(X, y)
                scores = clf.score(X_test, y_test)
                score_t.append(np.mean(scores))
    score_t = np.stack(score_t)
    if label is None:
        label = cond
    if ax1 is not None:
        plt.axes(ax1)
        if len(score_t.shape) == 1:
            plt.plot(step_centers, score_t, color=color)
        else:
            plot_shaded_err(score_t, label=label, color=color)
    if ax2 is not None:
        psth = np.stack(psth)
        plt.axes(ax2)
        for i in range(psth.shape[1]):
            plt.plot(step_centers, psth[:, i], '--', color=color)
    return score_t


separate_look_avoid = True
decode_all_textures = True
model = 'svc'
colors = ['purple', 'orange', 'limegreen', 'royalblue']

##

for session in sessions:
    print(session)
    data = load_data_1_session(h5_path, session)

    if not decode_all_textures:
        selected_trials = (data['trial_info']['texture1_on'] == 1) & (data['trial_info']['correct_trials'] == 1)
        fr_by_conds = get_fr_by_conds(data, 'texture_on_1', [0,800], selected_trials=selected_trials, avg_trials=True, conds=['texture1_angle'])
        population_preferred_ind1, population_preferred_ind2 = np.argsort(fr_by_conds.mean(axis=1))[[-1, -2]]
        population_preferred_angle1, population_preferred_angle2 = np.unique(data['trial_info']['texture1_angle'])[[population_preferred_ind1, population_preferred_ind2]]
        population_antipreferred_angle = np.abs(90-population_preferred_angle1)
        to_decode = [population_preferred_angle1, population_antipreferred_angle]


    if separate_look_avoid:
        h_fig, h_ax = plt.subplots(1, 2, sharey=True, figsize=[18,8])
        plt.axes(h_ax[0])
        plt.title('look')
        plt.plot(step_centers, decoding('task'), c=colors[0], label='task rule')
        selected_trials = (data['trial_info']['task'] == 1) & (data['trial_info']['correct_trials'] > 0)
        plt.plot(step_centers, decoding('memory_location', selected_trials), c=colors[1], label='cue location')
        try:
            if decode_all_textures:
                selected_trials = (data['trial_info']['texture1_on']==1) & data['trial_info']['cue_in_RF'] & (data['trial_info']['task'] == 1) & (data['trial_info']['correct_trials'] > 0)
                plt.plot(step_centers, decoding('texture1_angle', selected_trials, model=model), '-', c=colors[2])
                selected_trials = (data['trial_info']['texture1_on']==1) & data['trial_info']['cue_opposite_RF'] & (data['trial_info']['task'] == 1) & (data['trial_info']['correct_trials'] > 0)
                plt.plot(step_centers, decoding('texture1_angle', selected_trials, model=model), '--', c=colors[2])
                selected_trials = (data['trial_info']['texture2_on']==1) & data['trial_info']['cue_in_RF'] & (data['trial_info']['task'] == 1) & (data['trial_info']['correct_trials'] > 0)
                plt.plot(step_centers, decoding('texture2_angle', selected_trials, model=model), '-', c=colors[3], label='texture (cue inside RF)')
                selected_trials = (data['trial_info']['texture2_on']==1) & data['trial_info']['cue_opposite_RF'] & (data['trial_info']['task'] == 1) & (data['trial_info']['correct_trials'] > 0)
                plt.plot(step_centers, decoding('texture2_angle', selected_trials, model=model), '--', c=colors[3], label='texture (cue outside RF)')
            else:
                selected_trials = (data['trial_info']['texture1_on']==1) & data['trial_info']['cue_in_RF'] & (data['trial_info']['task'] == 1) & (data['trial_info']['texture1_angle']==to_decode[0]) | (data['trial_info']['texture1_angle']==to_decode[1]) & (data['trial_info']['correct_trials'] > 0)
                plt.plot(step_centers, decoding('texture1_angle', selected_trials, model=model), '-', c=colors[2])
                selected_trials = (data['trial_info']['texture1_on']==1) & data['trial_info']['cue_opposite_RF'] & (data['trial_info']['task'] == 1) & (data['trial_info']['texture1_angle']==to_decode[0]) | (data['trial_info']['texture1_angle']==to_decode[1]) & (data['trial_info']['correct_trials'] > 0)
                plt.plot(step_centers, decoding('texture1_angle', selected_trials, model=model), '--', c=colors[2])
                selected_trials = (data['trial_info']['texture2_on']==1) & data['trial_info']['cue_in_RF'] & (data['trial_info']['task'] == 1) & (data['trial_info']['texture2_angle']==to_decode[0]) | (data['trial_info']['texture2_angle']==to_decode[1]) & (data['trial_info']['correct_trials'] > 0)
                plt.plot(step_centers, decoding('texture2_angle', selected_trials, model=model), '-', c=colors[3], label='texture (cue inside RF)')
                selected_trials = (data['trial_info']['texture2_on']==1) & data['trial_info']['cue_opposite_RF'] & (data['trial_info']['task'] == 1) & (data['trial_info']['texture2_angle']==to_decode[0]) | (data['trial_info']['texture2_angle']==to_decode[1]) & (data['trial_info']['correct_trials'] > 0)
                plt.plot(step_centers, decoding('texture2_angle', selected_trials, model=model), '--', c=colors[3], label='texture (cue opposite RF)')
        except Exception:
            print('too few data')

        plt.axes(h_ax[1])
        plt.title('avoid')
        plt.plot(step_centers, decoding('task'), label='task rule', c=colors[0])
        selected_trials = (data['trial_info']['task'] == 2) & (data['trial_info']['correct_trials'] > 0)
        plt.plot(step_centers, decoding('memory_location', selected_trials), c=colors[1], label='cue location')
        try:
            if decode_all_textures:
                selected_trials = (data['trial_info']['texture1_on']==1) & data['trial_info']['cue_in_RF'] & (data['trial_info']['task'] == 2) & (data['trial_info']['correct_trials'] > 0)
                plt.plot(step_centers, decoding('texture1_angle', selected_trials, model=model), '-', c=colors[2])
                selected_trials = (data['trial_info']['texture1_on']==1) & data['trial_info']['cue_opposite_RF'] & (data['trial_info']['task'] == 2) & (data['trial_info']['correct_trials'] > 0)
                plt.plot(step_centers, decoding('texture1_angle', selected_trials, model=model), '--',c=colors[2])
                selected_trials = (data['trial_info']['texture2_on']==1) & data['trial_info']['cue_in_RF'] & (data['trial_info']['task'] == 2) & (data['trial_info']['correct_trials'] > 0)
                plt.plot(step_centers, decoding('texture2_angle', selected_trials, model=model), '-', c=colors[3], label='texture (cue inside RF)')
                selected_trials = (data['trial_info']['texture2_on']==1) & data['trial_info']['cue_opposite_RF'] & (data['trial_info']['task'] == 2) & (data['trial_info']['correct_trials'] > 0)
                plt.plot(step_centers, decoding('texture2_angle', selected_trials, model=model), '--', c=colors[3], label='texture (cue outside RF)')
            else:
                selected_trials = (data['trial_info']['texture1_on']==1) & data['trial_info']['cue_in_RF'] & (data['trial_info']['task'] == 2) & (data['trial_info']['texture1_angle']==to_decode[0]) | (data['trial_info']['texture1_angle']==to_decode[1]) & (data['trial_info']['correct_trials'] > 0)
                plt.plot(step_centers, decoding('texture1_angle', selected_trials, model=model), '-', c=colors[2])
                selected_trials = (data['trial_info']['texture1_on']==1) & data['trial_info']['cue_opposite_RF'] & (data['trial_info']['task'] == 2) & (data['trial_info']['texture1_angle']==to_decode[0]) | (data['trial_info']['texture1_angle']==to_decode[1]) & (data['trial_info']['correct_trials'] > 0)
                plt.plot(step_centers, decoding('texture1_angle', selected_trials, model=model), '--', c=colors[2])
                selected_trials = (data['trial_info']['texture2_on']==1) & data['trial_info']['cue_in_RF'] & (data['trial_info']['task'] == 2) & (data['trial_info']['texture2_angle']==to_decode[0]) | (data['trial_info']['texture2_angle']==to_decode[1]) & (data['trial_info']['correct_trials'] > 0)
                plt.plot(step_centers, decoding('texture2_angle', selected_trials, model=model), '-', c=colors[3], label='texture (cue inside RF)')
                selected_trials = (data['trial_info']['texture2_on']==1) & data['trial_info']['cue_opposite_RF'] & (data['trial_info']['task'] == 2) & (data['trial_info']['texture2_angle']==to_decode[0]) | (data['trial_info']['texture2_angle']==to_decode[1]) & (data['trial_info']['correct_trials'] > 0)
                plt.plot(step_centers, decoding('texture2_angle', selected_trials, model=model), '--', c=colors[3], label='texture (cue opposite RF)')
        except Exception:
            print('too few data')
        plt.legend()

    else:
        h_fig = plt.figure()
        plt.plot(step_centers, decoding('task'), c=colors[0], label='task rule')
        plt.plot(step_centers, decoding('memory_location'), c=colors[1], label='cue location')
        try:
            if decode_all_textures:
                selected_trials = (data['trial_info']['texture1_on']==1) & data['trial_info']['cue_in_RF'] & (data['trial_info']['correct_trials'] > 0)
                plt.plot(step_centers, decoding('texture1_angle', selected_trials, model=model), '-', c=colors[2])
                selected_trials = (data['trial_info']['texture1_on']==1) & data['trial_info']['cue_opposite_RF'] & (data['trial_info']['correct_trials'] > 0)
                plt.plot(step_centers, decoding('texture1_angle', selected_trials, model=model), '--', c=colors[2])
                selected_trials = (data['trial_info']['texture2_on']==1) & data['trial_info']['cue_in_RF'] & (data['trial_info']['correct_trials'] > 0)
                plt.plot(step_centers, decoding('texture2_angle', selected_trials, model=model), '-', c=colors[3], label='texture (cue inside RF)')
                selected_trials = (data['trial_info']['texture2_on']==1) & data['trial_info']['cue_opposite_RF'] & (data['trial_info']['correct_trials'] > 0)
                plt.plot(step_centers, decoding('texture2_angle', selected_trials, model=model), '--', c=colors[3], label='texture (cue outside RF)')
            else:
                selected_trials = (data['trial_info']['texture1_on']==1) & data['trial_info']['cue_in_RF'] & (data['trial_info']['texture1_angle']==to_decode[0]) | (data['trial_info']['texture1_angle']==to_decode[1]) & (data['trial_info']['correct_trials'] > 0)
                plt.plot(step_centers, decoding('texture1_angle', selected_trials, model=model), '-', c=colors[2])
                selected_trials = (data['trial_info']['texture1_on']==1) & data['trial_info']['cue_opposite_RF'] & (data['trial_info']['texture1_angle']==to_decode[0]) | (data['trial_info']['texture1_angle']==to_decode[1]) & (data['trial_info']['correct_trials'] > 0)
                plt.plot(step_centers, decoding('texture1_angle', selected_trials, model=model), '--', c=colors[2])
                selected_trials = (data['trial_info']['texture2_on']==1) & data['trial_info']['cue_in_RF'] & (data['trial_info']['texture2_angle']==to_decode[0]) | (data['trial_info']['texture2_angle']==to_decode[1]) & (data['trial_info']['correct_trials'] > 0)
                plt.plot(step_centers, decoding('texture2_angle', selected_trials, model=model), '-', c=colors[3], label='texture (cue inside RF)')
                selected_trials = (data['trial_info']['texture2_on']==1) & data['trial_info']['cue_opposite_RF'] & (data['trial_info']['texture2_angle']==to_decode[0]) | (data['trial_info']['texture2_angle']==to_decode[1]) & (data['trial_info']['correct_trials'] > 0)
                plt.plot(step_centers, decoding('texture2_angle', selected_trials, model=model), '--', c=colors[3], label='texture (cue opposite RF)')
            plt.title('avoid')
        except Exception:
            print('too few data')
        plt.legend()
    plt.savefig('decoding_{}.png'.format(session))
    plt.close(h_fig)


## Single trial

equal_correct_wrong_trials = True

for session in sessions:
    print(session)
    data = load_data_1_session(h5_path, session)
    selected_trials = (data['trial_info']['wrong_trials'] > 0) | (data['trial_info']['correct_trials'] > 0)
    h_fig = plt.figure()
    wrong_trial_labels = np.flatnonzero(data['trial_info']['wrong_trials'][selected_trials] > 0)
    correct_trial_labels = np.flatnonzero(data['trial_info']['correct_trials'][selected_trials] > 0)
    if equal_correct_wrong_trials:
        if len(correct_trial_labels)>len(wrong_trial_labels):
            correct_trial_labels = np.random.choice(correct_trial_labels, len(wrong_trial_labels), replace=False)
        else:
            correct_trial_labels = np.random.choice(wrong_trial_labels, len(correct_trial_labels), replace=False)
    temp = decoding('task', selected_trials=selected_trials, model='svc_prob')

    plot_shaded_err(temp[:, correct_trial_labels], label='correct')
    plot_shaded_err(temp[:, wrong_trial_labels], label='error')
    plt.ylim([0.5, 1])
    plt.legend()
    plt.savefig('decoding_task_{}.png'.format(session))
    plt.close(h_fig)


for session in sessions:
    print(session)
    data = load_data_1_session(h5_path, session)
    h_fig = plt.figure()
    plt.subplot(1, 2, 1)
    selected_trials = ((data['trial_info']['wrong_trials'] > 0) | (data['trial_info']['correct_trials'] > 0)) & (data['trial_info']['task']==1)
    wrong_trial_labels = np.flatnonzero(data['trial_info']['wrong_trials'][selected_trials] > 0)
    correct_trial_labels = np.flatnonzero(data['trial_info']['correct_trials'][selected_trials] > 0)
    if equal_correct_wrong_trials:
        if len(correct_trial_labels)>len(wrong_trial_labels):
            correct_trial_labels = np.random.choice(correct_trial_labels, len(wrong_trial_labels), replace=False)
        else:
            correct_trial_labels = np.random.choice(wrong_trial_labels, len(correct_trial_labels), replace=False)
    temp = decoding('memory_location', selected_trials=selected_trials, model='svc_prob')
    plot_shaded_err(temp[:, correct_trial_labels], label='correct')
    plot_shaded_err(temp[:, wrong_trial_labels], label='error')
    plt.ylim([0.25, 1])
    plt.title('look')
    plt.legend()
    plt.subplot(1, 2, 2)
    selected_trials = ((data['trial_info']['wrong_trials'] > 0) | (data['trial_info']['correct_trials'] > 0)) & (data['trial_info']['task']==2)
    wrong_trial_labels = np.flatnonzero(data['trial_info']['wrong_trials'][selected_trials] > 0)
    correct_trial_labels = np.flatnonzero(data['trial_info']['correct_trials'][selected_trials] > 0)
    if equal_correct_wrong_trials:
        if len(correct_trial_labels)>len(wrong_trial_labels):
            correct_trial_labels = np.random.choice(correct_trial_labels, len(wrong_trial_labels), replace=False)
        else:
            correct_trial_labels = np.random.choice(wrong_trial_labels, len(correct_trial_labels), replace=False)
    temp = decoding('memory_location', selected_trials=selected_trials, model='svc_prob')
    plot_shaded_err(temp[:, correct_trial_labels], label='correct')
    plot_shaded_err(temp[:, wrong_trial_labels], label='error')
    plt.ylim([0.25, 1])
    plt.title('avoid')
    plt.legend()
    plt.savefig('decoding_cue_location_{}.png'.format(session))
    plt.close(h_fig)

##
decode_condition = 'task'
for session in sessions:
    print(session)
    data = load_data_1_session(h5_path, session)
    h_fig = plt.figure()
    ax1 = plt.gca()
    plt.ylim([0.25, 1])
    ax2 = ax1.twinx()
    # plt.subplot(1, 2, 1)
    selected_trials = ((data['trial_info']['wrong_trials'] > 0) | (data['trial_info']['correct_trials'] > 0))
    # selected_trials = ((data['trial_info']['wrong_trials'] > 0) | (data['trial_info']['correct_trials'] > 0)) & (data['trial_info']['task']==1)
    wrong_trial_labels = np.flatnonzero((data['trial_info']['wrong_trials'] > 0) & selected_trials)
    correct_trial_labels = np.flatnonzero((data['trial_info']['correct_trials'] > 0) & selected_trials)
    if equal_correct_wrong_trials:
        if len(correct_trial_labels)>len(wrong_trial_labels):
            correct_trial_labels = np.random.choice(correct_trial_labels, len(wrong_trial_labels), replace=False)
        else:
            correct_trial_labels = np.random.choice(wrong_trial_labels, len(correct_trial_labels), replace=False)
    temp = decoding(decode_condition, selected_trials=correct_trial_labels, model='svc', ax1=ax1, ax2=ax2, color='salmon', label='correct')
    temp = decoding(decode_condition, selected_trials=wrong_trial_labels, model='svc', ax1=ax1, ax2=ax2, color='skyblue', label='error')
    # plt.title('look')

    # plt.subplot(1, 2, 2)
    # selected_trials = ((data['trial_info']['wrong_trials'] > 0) | (data['trial_info']['correct_trials'] > 0)) & (data['trial_info']['task']==2)
    # wrong_trial_labels =  np.flatnonzero((data['trial_info']['wrong_trials'] > 0) & selected_trials)
    # correct_trial_labels =  np.flatnonzero((data['trial_info']['correct_trials'] > 0) & selected_trials)
    # temp = decoding(decode_condition, selected_trials=correct_trial_labels, model='svc')
    # plt.plot(step_centers, temp, label='correct')
    # temp = decoding(decode_condition, selected_trials=wrong_trial_labels, model='svc')
    # plt.plot(step_centers, temp, label='error')
    # plt.ylim([0.25, 1])
    # plt.title('avoid')
    # plt.legend()
    plt.savefig('decoding_{}_{}.png'.format(decode_condition, session))
    plt.close(h_fig)

## Simulation
from sklearn.svm import SVC, SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

label_index = np.arange(6)
possible_labels = label_index/(len(label_index))*np.pi
N_neurons = 16
neurons_preferred = np.random.rand(N_neurons)*np.pi/4 + np.pi/2
neurons_bandwidth = np.random.rand(N_neurons)*np.pi/4
neurons_amp = np.random.rand(N_neurons)

def gaussian_curve(x, mu, sigma):
    return np.exp(-(x-mu)**2/(2*sigma**2))

def neural_data_generator(condition, mu, sigma, amp, noise_amp=1):
    output = np.sum([gaussian_curve(condition + period, mu, sigma) for period in np.pi * np.array([-3, -2, -1, 0, 1, 2, 3])], axis=0)
    output = amp*output + noise_amp*np.random.rand(*condition.shape)
    return output

# plt.plot(np.arange(60)/60*np.pi, neural_data_generator(np.arange(60)/60*np.pi, np.pi/2, np.pi/4, 1, noise_amp=0.2))

N_trials = 1000
y_ind = np.random.choice(label_index, N_trials)
y_angle = possible_labels[y_ind]
response = []
for i in range(N_neurons):
    response.append(neural_data_generator(y_angle, neurons_preferred[i], neurons_bandwidth[i], neurons_amp[i]))
X = np.stack(response, axis=1)

##
regr = make_pipeline(StandardScaler(), SVC())
print(cross_val_score(regr, X, y_ind, cv=5))

regr = make_pipeline(StandardScaler(), SVR())
print(cross_val_score(regr, X, y_angle, cv=5))

## Behavior

plt.figure()
for i,session in enumerate(sessions):
    print(session)
    data = load_data_1_session(h5_path, session, do_not_load=['spk', 'lfp'])
    data['trial_info']['memory_location'] = np.angle(data['trial_info']['esetup_memory_coord'][:,0]+data['trial_info']['esetup_memory_coord'][:,1]*1j) * 180 / np.pi
    unique_conds = np.unique(data['trial_info']['memory_location'])
    unique_conds = np.append(unique_conds, unique_conds[0])
    plt.subplot(3, 5, i+1, polar=True)
    N = []
    for i, c in enumerate(unique_conds):
        trials = data['trial_info']['memory_location']==c
        N.append(np.sum(data['trial_info']['wrong_trials'][trials]))
    plt.polar(unique_conds/180*np.pi, N, label='error')

    N = []
    for i, c in enumerate(unique_conds):
        trials = data['trial_info']['memory_location']==c
        N.append(np.sum(data['trial_info']['early_abort_trials'][trials]))
    plt.polar(unique_conds/180*np.pi, N, label='early abort')

    N = []
    for i, c in enumerate(unique_conds):
        trials = data['trial_info']['memory_location']==c
        N.append(np.sum(data['trial_info']['late_abort_trials'][trials]))
    plt.polar(unique_conds/180*np.pi, N, label='late abort')

    N = []
    for i, c in enumerate(unique_conds):
        trials = data['trial_info']['memory_location']==c
        N.append(np.sum(trials)/3)
    plt.polar(unique_conds/180*np.pi, N, label='all trials')

    plt.legend()


## LFP
alpha_range = [3, 13]
beta_range = [13, 30]
gamma_range = [30, 50]
# for session in sessions:
for session in ['hb_20180216_1', 'hb_20180219', 'hb_20180221']:
    data = load_data_1_session(h5_path, session, do_not_load=[])
    std_lfp = np.nanstd(data['lfp'], axis=(1, 2))
    selected_trials = (data['trial_info']['correct_trials'] > 0) & (np.abs(std_lfp-np.nanmean(std_lfp)) < 2*np.nanstd(std_lfp))
    data_realigned = realign(times=data['trial_info']['memory_on'] - data['trial_info']['first_display'],
                             data_to_align=data['lfp'], ts=data['ts'], time_win=[-700, 1300],
                             selected_trials=selected_trials)
    spcg, spcg_t, spcg_f = ComputeSpectrogram(data_realigned, fs=1000.0, t_ini=-0.7, t_bin=window_size/1000, t_step=step_size/1000, f_lim=[3,100])
    spcg = np.swapaxes(spcg, 2, 3)
    data['spcg_alpha'] = np.nanmean(np.log(spcg[:, (spcg_f>=alpha_range[0])&(spcg_f<alpha_range[1])]), axis=1)
    data['ts_spcg_alpha'] = spcg_t * 1000
    data['spcg_beta'] = np.nanmean(np.log(spcg[:, (spcg_f>=beta_range[0])&(spcg_f<beta_range[1])]), axis=1)
    data['ts_spcg_beta'] = spcg_t * 1000
    data['spcg_gamma'] = np.nanmean(np.log(spcg[:, (spcg_f>=gamma_range[0])&(spcg_f<gamma_range[1])]), axis=1)
    data['ts_spcg_gamma'] = spcg_t * 1000
    h_fig, h_ax = plt.subplots(1,2)
    decoding('task', data_type='spcg_beta', selected_trials=selected_trials, model='svc', ax1=h_ax[0], ax2=h_ax[1])
    plt.savefig('spcg_beta_decoding_task_{}.png'.format(session))
    plt.close(h_fig)

    h_fig, h_ax = plt.subplots(1,2)
    decoding('task', data_type='spcg_alpha', selected_trials=selected_trials, model='svc', ax1=h_ax[0], ax2=h_ax[1])
    plt.savefig('spcg_alpha_decoding_task_{}.png'.format(session))
    plt.close(h_fig)

    h_fig, h_ax = plt.subplots(1,2)
    decoding('task', data_type='spcg_gamma', selected_trials=selected_trials, model='svc', ax1=h_ax[0], ax2=h_ax[1])
    plt.savefig('spcg_gamma_decoding_task_{}.png'.format(session))
    plt.close(h_fig)

    # selected_trials_1 = np.flatnonzero((data['trial_info']['task'][selected_trials] == 1) & (np.abs(std_lfp-np.nanmean(std_lfp)) < 2*np.nanstd(std_lfp)))
    # selected_trials_2 = np.flatnonzero((data['trial_info']['task'][selected_trials] == 2) & (np.abs(std_lfp-np.nanmean(std_lfp)) < 2*np.nanstd(std_lfp)))
    # # plt.figure()
    # # for i in range(24):
    # #     plt.subplot(4,6,i+1)
    # #     selected_trials_plot = np.flatnonzero(data['trial_info']['task'][selected_trials]==2)
    # #     plt.pcolormesh(spcg_t, spcg_f, np.log(np.nanmean(spcg[selected_trials_plot,:,i,:], axis=0)))
    # h_fig = plt.figure(figsize=[18, 6])
    # plt.subplot(1,3,1)
    # plt.pcolormesh(spcg_t, spcg_f, np.log(np.nanmean(spcg[selected_trials_1], axis=(0,2))))
    # plt.title('N={}'.format(len(selected_trials_1)))
    # plt.subplot(1,3,2)
    # plt.pcolormesh(spcg_t, spcg_f, np.log(np.nanmean(spcg[selected_trials_2], axis=(0,2))))
    # plt.title('N={}'.format(len(selected_trials_2)))
    # plt.subplot(1,3,3)
    # difference = np.log(np.nanmean(spcg[selected_trials_1], axis=(0,2)))-np.log(np.nanmean(spcg[selected_trials_2], axis=(0,2)))
    # max_abs_difference = np.nanmax(np.abs(difference))
    # plt.pcolormesh(spcg_t, spcg_f, difference, cmap='coolwarm')
    # plt.clim([-max_abs_difference, max_abs_difference])
    # plt.savefig('spcg_task_{}.png'.format(session))
    # plt.close(h_fig)