import h5py
import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from PyNeuroAna import *
from PyNeuroPlot import *
import matplotlib as mpl
mpl.style.use('default')


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
    print(data['spk'].shape[2])
    if data['spk'].shape[2]>33:
        data['spk'] = data['spk'][:, :, 32:]
        data['unit_info']['visual_cell'] = data['unit_info']['visual_cell'][32:]
        data['unit_info']['memory_cell'] = data['unit_info']['memory_cell'][32:]
        data['unit_info']['motor_cell'] = data['unit_info']['motor_cell'][32:]
        print(data['spk'].shape[2])
    data['spk'] = data['spk'].astype(np.float) * 1000
    data['trial_info']['target_on'] = np.zeros(data['spk'].shape[0])
    if 'inactivation' not in data['trial_info'].keys():
        data['trial_info']['inactivation'] = np.ones(data['spk'].shape[0])
    return data

h5_path='D:/Analysis/look6/data/V4_delay.h5'
# h5_path='D:/Analysis/look6/data/V4_delay_inactivation.h5'
with h5py.File(h5_path, 'r') as f:
    sessions = list(f.keys())
data = load_data_1_session(h5_path, sessions[0])

# score_look = np.load('score_look.npy')
# score_look_shuffle = np.load('score_look_shuffle.npy')
# score_avoid = np.load('score_avoid.npy')
# score_avoid_shuffle = np.load('score_avoid_shuffle.npy')
##
def realign(times, data_to_align, ts=data['ts'], time_win=[data['ts'][0], data['ts'][-1]], selected_trials=None):
    new_data = []
    if selected_trials is None:
        selected_trials = np.arange(data_to_align.shape[0])
    if np.issubdtype(selected_trials.dtype, np.bool_):
        selected_trials = np.flatnonzero(selected_trials)
    for i in selected_trials:
        data_in_range = data_to_align[i, (ts >= times[i] + time_win[0]) & (ts <= times[i] + time_win[1])]
        if ts[0] > (times[i] + time_win[0]):
            data_in_range = np.concatenate((np.ones(
                (int(ts[0] - (times[i] + time_win[0])), data_in_range.shape[1])) * np.nan, data_in_range))
        if ts[-1] < (times[i] + time_win[1]):
            data_in_range = np.concatenate((data_in_range, np.ones(
                (int(times[i] + time_win[1] - ts[-1] - 1), data_in_range.shape[1])) * np.nan))
        if data_in_range.shape[0] > time_win[1] - time_win[0] + 1:
            data_in_range = data_in_range[:int(time_win[1] - time_win[0])]
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


def get_fr_by_conds(data, epoch, time_win_calculate, data_type='spk', selected_units=None, selected_trials=None, avg_trials=True, conds=['cue_location'], return_trial_index=False):
    if selected_units is None:
        selected_units = np.ones(data['spk'].shape[2])>0

    if selected_trials is None:
        selected_trials = np.arange(data[data_type].shape[0])

    if data_type == 'spk' or data_type == 'lfp':
        ts = data['ts']
        data_realigned = realign(data['trial_info'][epoch], data_to_align=data[data_type][:,:,selected_units], ts=ts, time_win=time_win_calculate, selected_trials=selected_trials)
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
    fr_by_conds, trial_index = [], []
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
                trial_index_i = selected_trials[trials][(np.isnan(selected_fr).sum(axis=1))==0]
            elif len(conds) == 2:
                trial_index_i = []
                for j in range(len(cond2_unique)):
                    trials = (cond2==cond2_unique[j]) & (cond1==cond1_unique[i])
                    selected_fr = fr_all[trials]
                    data_i.append(selected_fr[(np.isnan(selected_fr).sum(axis=1))==0, :])
                    trial_index_i.append(selected_trials[trials](np.isnan(selected_fr).sum(axis=1))==0)
            fr_by_conds.append(data_i)
            trial_index.append(trial_index_i)
    if return_trial_index:
        return fr_by_conds, trial_index


    return fr_by_conds


## Decoding
from sklearn import svm
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
step_size = 50
window_size = 300
step_centers = np.arange(-800, 51, step_size) + window_size/2


def get_X_y(data, data_type, time_win, cond, selected_units, selected_trials, selected_trials2=None):
    if selected_trials2 is None:
        fr_by_conds = get_fr_by_conds(data, 'target_on', time_win, data_type=data_type, selected_units=selected_units, selected_trials=selected_trials,
                                      avg_trials=False, conds=[cond])
        # N_samples = [x.shape[0] for x in fr_by_conds]
        # for i in range(len(fr_by_conds)):
        #     N_sample_i = fr_by_conds[i].shape[0]
        #     original_ind_i = np.arange(N_sample_i)
        #     resampled_ind_i = np.random.choice(original_ind_i, np.min(N_samples), replace=False)
        #     fr_by_conds[i] = fr_by_conds[i][resampled_ind_i]

        X = np.concatenate(fr_by_conds)
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        y = np.concatenate([np.ones(x.shape[0]) * i for i, x in enumerate(fr_by_conds)])
        # randomized_ind = np.random.permutation(len(y))
        # X = X[randomized_ind, :]
        # y = y[randomized_ind]
    else:
        fr_by_conds1, trial_index1 = get_fr_by_conds(data, 'target_on', time_win, data_type=data_type, selected_units=selected_units, selected_trials=selected_trials,
                                      avg_trials=False, conds=[cond], return_trial_index=True)
        fr_by_conds2, trial_index2 = get_fr_by_conds(data, 'target_on', time_win, data_type=data_type, selected_units=selected_units, selected_trials=selected_trials2,
                                      avg_trials=False, conds=[cond], return_trial_index=True)
        X_train = np.concatenate(fr_by_conds1)
        # trial_index = np.concatenate(trial_index)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X = scaler.transform(np.concatenate(fr_by_conds2))
        y = np.concatenate([np.ones(x.shape[0]) * i for i, x in enumerate(fr_by_conds2)])
        # X = X[np.in1d(trial_index, selected_trials2),:]
        # y = y[np.in1d(trial_index, selected_trials2)]

    return X, y

    # fr_by_conds = get_fr_by_conds(data, 'target_on', time_win, data_type=data_type, selected_trials=selected_trials,
    #                               avg_trials=False, conds=[cond])
    # X = np.concatenate(fr_by_conds)
    # scaler = StandardScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)
    # y = np.concatenate([np.ones(x.shape[0]) * i for i, x in enumerate(fr_by_conds)])
    # return X, y


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


def decoding(cond, data_type='spk', step_centers=step_centers, window_size=window_size, selected_trials=None, selected_trials2=None, selected_units=None, model='svc', shuffle_train=False, shuffle_test=False):
    score_t = []
    psth = []
    if selected_trials is None:
        selected_trials = np.flatnonzero(data['trial_info']['performance'])
    if np.issubdtype(selected_trials.dtype, np.bool_):
        selected_trials = np.flatnonzero(selected_trials)
    if selected_trials2 is not None and np.issubdtype(selected_trials2.dtype, np.bool_):
        selected_trials2 = np.flatnonzero(selected_trials2)
    n_steps = len(step_centers)
    for t in range(n_steps):
        current_t_win = [step_centers[t] - window_size / 2, step_centers[t] + window_size / 2]
        X, y = get_X_y(data, data_type, current_t_win, cond, selected_units, selected_trials)
        if shuffle_train:
            np.random.shuffle(y)
        if selected_trials2 is None:
            if model=='svc_prob':
                clf = svm.SVC(probability=True, gamma='scale')
                clf.fit(X, y)
                score_t.append(select_by_label(clf.predict_proba(X), y))
            else:
                if model=='svc':
                    clf = svm.SVC(class_weight='balanced', gamma='scale')
                elif model=='svr':
                    clf = svm.SVR(class_weight='balanced', gamma='scale')
                scores = cross_val_score(clf, X, y, cv=LeaveOneOut())
                score_t.append(np.mean(scores))
        else:
            X_test, y_test = get_X_y(data, data_type, current_t_win, cond, selected_units, selected_trials, selected_trials2)
            if shuffle_test:
                np.random.shuffle(y_test)
            if model=='svc_prob':
                clf = svm.SVC(probability=True, gamma='scale')
                clf.fit(X, y)
                score_t.append(select_by_label(clf.predict_proba(X_test), y_test))
            else:
                if model=='svc':
                    clf = svm.SVC(class_weight='balanced', gamma='scale')
                elif model=='svr':
                    clf = svm.SVR(class_weight='balanced', gamma='scale')
                clf.fit(X, y)
                scores = clf.score(X_test, y_test)
                score_t.append(np.mean(scores))
    score_t = np.stack(score_t)
    return score_t


model = 'svc'
colors = ['limegreen', 'red', 'dodgerblue']



## PSTH
ts_plot = data['ts']
for s, session in enumerate(sessions):
    print(session)
    plt.figure(figsize=[10, 10])
    data = load_data_1_session(h5_path, session)
    selected_units = np.arange(data['spk'].shape[2])
    unique_location = np.unique(data['trial_info']['cue_location'])
    for k in range(2):
        for j in range(2):
            plt.subplot(2, 4, k*4 + j + 1)
            for i in range(len(unique_location)):
                selected_trials = (data['trial_info']['inactivation']==k+1) & (data['trial_info']['cue_location']==unique_location[i]) & (data['trial_info']['task'] == j+1) & (data['trial_info']['performance'] == 1)
                if np.sum(selected_trials) == 0:
                    continue
                spk_realigned = realign(data['trial_info']['target_on'], data_to_align=data['spk'][:, :, selected_units], selected_trials=selected_trials)
                plt.plot(ts_plot, SmoothTrace(np.nanmean(spk_realigned, axis=(0, 2)), sk_std=5, ts=ts_plot, axis=0))
            plt.axvline(0, color='black', ls=':')
            plt.xlabel('Time (ms)')
            plt.ylabel('Firing rate (Hz)')
            plt.title('N trials = {}'.format(((data['trial_info']['task'] == j+1) & (data['trial_info']['performance'] == 1)).sum()))

            plt.subplot(2, 4, k*4 + 3)
            selected_trials = (data['trial_info']['inactivation']==k+1) & (data['trial_info']['task'] == j+1) & (data['trial_info']['performance'] == 1)
            # selected_trials = (data['trial_info']['cue_location']==1) | (data['trial_info']['cue_location']==3) & (data['trial_info']['task'] == j+1) & (data['trial_info']['performance'] == 1)
            if np.sum(selected_trials) < 10:
                continue
            if j==0:
                plt.plot(step_centers, decoding('cue_location', selected_trials=selected_trials, model=model), '-', color='black')
            else:
                plt.plot(step_centers, decoding('cue_location', selected_trials=selected_trials, model=model), '--', color='black')
            plt.axhline(0.25, color='black', ls=':')

            plt.subplot(2, 4, k*4 + 4)
            selected_trials = (data['trial_info']['inactivation']==k+1) & (data['trial_info']['performance'] == 1)
            if np.sum(selected_trials) < 10:
                continue
            plt.plot(step_centers, decoding('task', selected_trials=selected_trials, model=model), '-', color='black')
            plt.axhline(0.5, color='black', ls=':')

    plt.suptitle(session)
    plt.savefig('./figures/{}.png'.format(session))
    plt.close()


## Decode cue location for look VS avoid
downsample_trial = False
score_look, score_avoid = [], []
score_look2, score_avoid2 = [], []
score_look_shuffle, score_avoid_shuffle = [], []
score_look_shuffle2, score_avoid_shuffle2 = [], []

time_win = [-300, 0]
N_shuffle = 0
for s, session in enumerate(sessions):
    data = load_data_1_session(h5_path, session)
    selected_units = np.arange(data['spk'].shape[2])
    selected_units = np.flatnonzero(data['unit_info']['visual_cell'] > 0)
    # selected_units = np.flatnonzero(data['unit_info']['memory_cell'] > 0)
    print(selected_units)
    if np.sum(selected_units) < 1:
        print([session, 'not enough units'])
        continue
    unique_location = np.unique(data['trial_info']['cue_location'])
    selected_trials1 = (data['trial_info']['inactivation']==1) & ((data['trial_info']['cue_location']==1) | (data['trial_info']['cue_location']==3)) & (data['trial_info']['task'] == 1) & (data['trial_info']['performance'] == 1)
    selected_trials2 = (data['trial_info']['inactivation']==1) & ((data['trial_info']['cue_location']==1) | (data['trial_info']['cue_location']==3)) & (data['trial_info']['task'] == 2) & (data['trial_info']['performance'] == 1)
    selected_trials1_2 = (data['trial_info']['inactivation']==2) & ((data['trial_info']['cue_location']==1) | (data['trial_info']['cue_location']==3)) & (data['trial_info']['task'] == 1) & (data['trial_info']['performance'] == 1)
    selected_trials2_2 = (data['trial_info']['inactivation']==2) & ((data['trial_info']['cue_location']==1) | (data['trial_info']['cue_location']==3)) & (data['trial_info']['task'] == 2) & (data['trial_info']['performance'] == 1)
    # selected_trials1 = (data['trial_info']['inactivation']==1) & (data['trial_info']['task'] == 1) & (data['trial_info']['performance'] == 1)
    # selected_trials2 = (data['trial_info']['inactivation']==1) & (data['trial_info']['task'] == 2) & (data['trial_info']['performance'] == 1)
    # selected_trials1_2 = (data['trial_info']['inactivation']==2) & (data['trial_info']['task'] == 1) & (data['trial_info']['performance'] == 1)
    # selected_trials2_2 = (data['trial_info']['inactivation']==2) & (data['trial_info']['task'] == 2) & (data['trial_info']['performance'] == 1)
    if np.sum(selected_trials1) < 20 or np.sum(selected_trials2) < 20:
        print([session, 'not enough trials'])
        continue
    if len(np.unique(data['trial_info']['inactivation'])) > 1 and (np.sum(selected_trials1_2) < 20 or np.sum(selected_trials2_2) < 20):
        print([session, 'not enough trials'])
        continue
    if downsample_trial:
        N_trials = np.min([selected_trials1.sum(), selected_trials2.sum()])
        selected_trials1 = np.flatnonzero(selected_trials1)[np.random.choice(selected_trials1.sum(), N_trials, replace=False)]
        selected_trials2 = np.flatnonzero(selected_trials2)[np.random.choice(selected_trials2.sum(), N_trials, replace=False)]
    score_look.append(decoding('cue_location', step_centers=[np.mean(time_win)], window_size=np.diff(time_win), selected_trials=selected_trials1, selected_units=selected_units, model=model))
    score_avoid.append(decoding('cue_location', step_centers=[np.mean(time_win)], window_size=np.diff(time_win), selected_trials=selected_trials2, selected_units=selected_units, model=model))
    if N_shuffle>0:
        score_look_shuffle_i, score_avoid_shuffle_i = [], []
        for i in range(N_shuffle):
            print([session, i])
            score_look_shuffle_i.append(decoding('cue_location', step_centers=[np.mean(time_win)], window_size=np.diff(time_win), selected_trials=selected_trials1, selected_units=selected_units, model=model, shuffle_train=True))
            score_avoid_shuffle_i.append(decoding('cue_location', step_centers=[np.mean(time_win)], window_size=np.diff(time_win), selected_trials=selected_trials2, selected_units=selected_units, model=model, shuffle_train=True))
        score_look_shuffle.append(np.concatenate(score_look_shuffle_i))
        score_avoid_shuffle.append(np.concatenate(score_avoid_shuffle_i))

    if len(np.unique(data['trial_info']['inactivation'])) > 1:
        if downsample_trial:
            N_trials = np.min([selected_trials1_2.sum(), selected_trials2_2.sum()])
            selected_trials1_2 = np.flatnonzero(selected_trials1_2)[np.random.choice(selected_trials1_2.sum(), N_trials, replace=False)]
            selected_trials2_2 = np.flatnonzero(selected_trials2_2)[np.random.choice(selected_trials2_2.sum(), N_trials, replace=False)]
        score_look2.append(decoding('cue_location', step_centers=[np.mean(time_win)], window_size=np.diff(time_win), selected_trials=selected_trials1_2, selected_units=selected_units, model=model))
        score_avoid2.append(decoding('cue_location', step_centers=[np.mean(time_win)], window_size=np.diff(time_win), selected_trials=selected_trials2_2, selected_units=selected_units, model=model))
        print([session, score_look[-1], score_look2[-1], score_avoid[-1], score_avoid2[-1]])
        if N_shuffle>0:
            score_look_shuffle_i, score_avoid_shuffle_i = [], []
            for i in range(N_shuffle):
                print([session, i])
                score_look_shuffle_i.append(decoding('cue_location', step_centers=[np.mean(time_win)], window_size=np.diff(time_win), selected_trials=selected_trials1_2, selected_units=selected_units, model=model, shuffle_train=True))
                score_avoid_shuffle_i.append(decoding('cue_location', step_centers=[np.mean(time_win)], window_size=np.diff(time_win), selected_trials=selected_trials2_2, selected_units=selected_units, model=model, shuffle_train=True))
            score_look_shuffle2.append(np.concatenate(score_look_shuffle_i))
            score_avoid_shuffle2.append(np.concatenate(score_avoid_shuffle_i))

score_look, score_avoid = np.concatenate(score_look), np.concatenate(score_avoid)
nan_removed = (~np.isnan(score_look))&(~np.isnan(score_avoid))
score_look = score_look[nan_removed]
score_avoid = score_avoid[nan_removed]
score_look2, score_avoid2 = np.concatenate(score_look2), np.concatenate(score_avoid2)
nan_removed = (~np.isnan(score_look2))&(~np.isnan(score_avoid2))
score_look2 = score_look2[nan_removed]
score_avoid2 = score_avoid2[nan_removed]


##
_, p12 = sp.stats.ttest_rel(score_look, score_avoid)
_, p1 = sp.stats.ttest_1samp(score_look, 0.5)
_, p2 = sp.stats.ttest_1samp(score_avoid, 0.5)
bin_size = 0.03
bins = np.arange(0, 1, bin_size)
plt.figure()
[score_look_hist, _, _] = plt.hist(score_look, bins, color=colors[0], alpha=0.3, label='Look, p<0.001')
[score_avoid_hist, _, _] = plt.hist(score_avoid, bins, color=colors[1], alpha=0.3, label='Avoid, p<0.001')
plt.xlabel('Decoding accuracy')
plt.ylabel('Session count')
plt.axvline(0.5, ls='--', color='black')
# plt.legend()
# plt.axvline(score_look.mean(), ls='-', color=colors[0])
# plt.axvline(score_avoid.mean(), ls='-', color=colors[1])
ax1 = plt.gca()
ax2 = ax1.twinx()
mu1,sigma1 = curve_fit(norm.cdf, bins[:-1]+bin_size/2, np.cumsum(score_look_hist)/np.sum(score_look_hist), p0=[0,1])[0]
ax2.plot(bins[:-1]+bin_size/2, norm.cdf(bins[:-1]+bin_size/2, mu1, sigma1), color=colors[0], linewidth=4)
mu1,sigma1 = curve_fit(norm.cdf, bins[:-1]+bin_size/2, np.cumsum(score_avoid_hist)/np.sum(score_avoid_hist), p0=[0,1])[0]
ax2.plot(bins[:-1]+bin_size/2, norm.cdf(bins[:-1]+bin_size/2, mu1, sigma1), color=colors[1], linewidth=4)
# ax2.plot(bins[:-1]+bin_size/2, np.cumsum(score_look_hist)/np.sum(score_look_hist), color=colors[0], linewidth=3)
# ax2.plot(bins[:-1]+bin_size/2, np.cumsum(score_avoid_hist)/np.sum(score_avoid_hist), color=colors[1], linewidth=3)
plt.ylim([0, 1.01])
plt.ylabel('Cumulative distribution function')
plt.xlim([0.3, 1])


##
_, p12 = sp.stats.ttest_rel(score_look, score_avoid)
_, p1 = sp.stats.ttest_1samp(score_look, 0.5)
_, p2 = sp.stats.ttest_1samp(score_avoid, 0.5)

plt.figure()
p_look, p_avoid = [], []
for i in range(len(score_look)):
    plt.subplot(4, int(np.ceil(len(score_look)/2)), i+1)
    plt.hist(score_look_shuffle[i], color=colors[0], alpha=0.4)
    plt.axvline(score_look[i], color=colors[0])
    p = np.mean(score_look_shuffle[i]>score_look[i])
    p_look.append(p)
    plt.title('p={}'.format(p))
    plt.subplot(4, int(np.ceil(len(score_look)/2)), i+1+len(score_look))
    plt.hist(score_avoid_shuffle[i], color=colors[1], alpha=0.4)
    plt.axvline(score_avoid[i], color=colors[1])
    p = np.mean(score_avoid_shuffle[i]>score_avoid[i])
    p_avoid.append(p)
    plt.title('p={}'.format(p))
p_look, p_avoid = np.array(p_look), np.array(p_avoid)

plt.figure()
plt.subplot(2,2,1)
plt.hist(score_look, np.arange(0, 1, 0.02), color=colors[0], alpha=0.4, label='Look, p={}'.format(p1))
plt.xlabel('Decoding score (cue location)')
plt.ylabel('Session count')
plt.axvline(0.5, ls='--', color='black')
plt.axvline(score_look.mean(), ls='-', color=colors[0])
plt.xlim([0.3, 1])
plt.title('p = {}'.format(p1))
plt.subplot(2,2,2)
plt.hist(score_avoid, np.arange(0, 1, 0.02), color=colors[1], alpha=0.4, label='Avoid, p={}'.format(p2))
plt.xlabel('Decoding score (cue location)')
plt.ylabel('Session count')
plt.axvline(0.5, ls='--', color='black')
plt.axvline(score_avoid.mean(), ls='-', color=colors[1])
plt.xlim([0.3, 1])
plt.title('p = {}'.format(p2))
plt.subplot(2,2,3)
plt.hist(score_look, np.arange(0, 1, 0.02), color=colors[0], alpha=0.4, label='Look, p<0.001'.format(p1))
plt.hist(score_avoid, np.arange(0, 1, 0.02), color=colors[1], alpha=0.4, label='Avoid, p<0.001'.format(p2))
plt.xlabel('Decoding score (cue location)')
plt.ylabel('Session count')
plt.axvline(0.5, ls='--', color='black')
plt.axvline(score_look.mean(), ls='-', color=colors[0])
plt.axvline(score_avoid.mean(), ls='-', color=colors[1])
plt.xlim([0.3, 1])
plt.legend()
plt.title('p = {}'.format(p12))
plt.tight_layout()
# plt.subplot(2,2,2)
# plt.hist(p_look, np.arange(0, 1, 0.02), color=colors[0], alpha=0.5, label='Look, p<0.05: {}'.format((p_look<0.05).mean()))
# # plt.hist(p_avoid, color=colors[1], alpha=0.5, label='Avoid, p<0.05: {}'.format((p_avoid<0.05).mean()))
# plt.axvline(0.05, ls='--', color='k')
# plt.xlabel('P value')
# plt.ylabel('# sessions')
# plt.legend()
# plt.subplot(2,2,4)
# # plt.hist(p_look, color=colors[0], alpha=0.5, label='Look, p<0.05: {}'.format((p_look<0.05).mean()))
# plt.hist(p_avoid, np.arange(0, 1, 0.02), color=colors[1], alpha=0.5, label='Avoid, p<0.05: {}'.format((p_avoid<0.05).mean()))
# plt.axvline(0.05, ls='--', color='k')
# plt.xlabel('P value')
# plt.ylabel('# sessions')
# plt.legend()


##
bin_size = 0.03
bins = np.arange(0, 1, bin_size)
plt.figure()
h_main, h_top, h_right = scatter_hist(score_look, score_look2, kwargs_scatter={'edgecolors':colors[0], 'marker':'o', 's':50, 'c':[0,0,0,0], 'linewidth':2}, kwargs_hist={'bins':bins, 'color':colors[0], 'alpha':0.3})
scatter_hist(score_avoid, score_avoid2, h_axes=[h_main, h_top, h_right], kwargs_scatter={'edgecolors':colors[1], 'marker':'o', 's':50, 'c':[0,0,0,0], 'linewidth':2}, kwargs_hist={'bins':bins, 'color':colors[1], 'alpha':0.3})
h_main.plot([0, 1], [0, 1], 'k--')
h_main.axvline(0.5, color='black', ls='--')
h_main.axhline(0.5, color='black', ls='--')
h_main.set_ylabel('Decoding accuracy (inactivation)')
h_main.set_xlabel('Decoding accuracy (control)')
h_main.set_xlim([0.3,1])
h_main.set_ylim([0.3,1])
h_top.axvline(0.5, color='black', ls='--')
h_right.axhline(0.5, color='black', ls='--')
h_top2 = h_top.twinx()
mu1,sigma1 = curve_fit(norm.cdf, bins[:-1]+bin_size/2, np.cumsum(np.histogram(score_look, bins)[0])/np.sum(np.histogram(score_look, bins)[0]), p0=[0,1])[0]
h_top2.plot(bins[:-1]+bin_size/2, norm.cdf(bins[:-1]+bin_size/2, mu1, sigma1), color=colors[0], linewidth=3)
mu1,sigma1 = curve_fit(norm.cdf, bins[:-1]+bin_size/2, np.cumsum(np.histogram(score_avoid, bins)[0])/np.sum(np.histogram(score_avoid, bins)[0]), p0=[0,1])[0]
h_top2.plot(bins[:-1]+bin_size/2, norm.cdf(bins[:-1]+bin_size/2, mu1, sigma1), color=colors[1], linewidth=3)
plt.ylim([0, 1.01])
# h_top2.plot(bins[:-1]+bin_size/2, np.cumsum(np.histogram(score_look, bins)[0]), color=colors[0], linewidth=3)
# h_top2.plot(bins[:-1]+bin_size/2, np.cumsum(np.histogram(score_avoid, bins)[0]), color=colors[1], linewidth=3)
h_right2 = h_right.twiny()
mu1,sigma1 = curve_fit(norm.cdf, bins[:-1]+bin_size/2, np.cumsum(np.histogram(score_look2, bins)[0])/np.sum(np.histogram(score_look2, bins)[0]), p0=[0,1])[0]
h_right2.plot(norm.cdf(bins[:-1]+bin_size/2, mu1, sigma1), bins[:-1]+bin_size/2, color=colors[0], linewidth=3)
mu1,sigma1 = curve_fit(norm.cdf, bins[:-1]+bin_size/2, np.cumsum(np.histogram(score_avoid2, bins)[0])/np.sum(np.histogram(score_avoid2, bins)[0]), p0=[0,1])[0]
h_right2.plot(norm.cdf(bins[:-1]+bin_size/2, mu1, sigma1), bins[:-1]+bin_size/2, color=colors[1], linewidth=3)
plt.xlim([0, 1.01])
# h_right2.plot(np.cumsum(np.histogram(score_look2, bins)[0]), bins[:-1]+bin_size/2, color=colors[0], linewidth=3)
# h_right2.plot(np.cumsum(np.histogram(score_avoid2, bins)[0]), bins[:-1]+bin_size/2, color=colors[1], linewidth=3)

##
plt.figure()
p_look, p_avoid, p_look2, p_avoid2 = [], [], [], []
for i in range(len(score_look)):
    plt.subplot(4, len(score_look), i+1)
    plt.hist(score_look_shuffle[i], color=colors[0], alpha=0.5)
    plt.axvline(score_look[i], color=colors[0])
    p = np.mean(score_look_shuffle[i]>score_look[i])
    p_look.append(p)
    plt.title('p={}'.format(p))
    plt.subplot(4, len(score_look), i+1+len(score_look))
    plt.hist(score_avoid_shuffle[i], color=colors[1], alpha=0.5)
    plt.axvline(score_avoid[i], color=colors[1])
    p = np.mean(score_avoid_shuffle[i]>score_avoid[i])
    p_avoid.append(p)
    plt.title('p={}'.format(p))
    plt.subplot(4, len(score_look), i+1+len(score_look)*2)
    plt.hist(score_look_shuffle2[i], color=colors[0], alpha=0.5)
    plt.axvline(score_look2[i], color=colors[0])
    p = np.mean(score_look_shuffle2[i]>score_look2[i])
    p_look2.append(p)
    plt.title('p={}'.format(p))
    plt.subplot(4, len(score_look), i+1+len(score_look)*3)
    plt.hist(score_avoid_shuffle2[i], color=colors[1], alpha=0.5)
    plt.axvline(score_avoid2[i], color=colors[1])
    p = np.mean(score_avoid_shuffle2[i]>score_avoid2[i])
    p_avoid2.append(p)
    plt.title('p={}'.format(p))
plt.figure()
plt.subplot(2,2,1)
# plt.hist(score_look, alpha=0.5)
# plt.axvline(np.mean(score_look), color='darkblue')
# plt.hist(score_look2, alpha=0.5)
# plt.axvline(np.mean(score_look2), color='darkorange')
plt.scatter(score_look, score_look2)
plt.plot([0, 1], [0, 1], 'k--')
plt.axvline(0.5, color='black', ls='--')
plt.axhline(0.5, color='black', ls='--')
plt.ylabel('Inactivation')
plt.xlabel('Control')
_, p = sp.stats.ttest_rel(score_look, score_look2)
# plt.title('p={}'.format(p))
plt.title('Decoding accuracy (Look)')
plt.subplot(2,2,2)
# plt.hist(score_avoid, alpha=0.5)
# plt.axvline(np.mean(score_avoid), color='darkblue')
# plt.hist(score_avoid2, alpha=0.5)
# plt.axvline(np.mean(score_avoid2), color='darkorange')
plt.scatter(score_avoid, score_avoid2)
plt.plot([0, 1], [0, 1], 'k--')
plt.axvline(0.5, color='black', ls='--')
plt.axhline(0.5, color='black', ls='--')
plt.ylabel('Inactivation')
plt.xlabel('Control')
_, p = sp.stats.ttest_rel(score_avoid, score_avoid2)
# plt.title('p={}'.format(p))
plt.title('Decoding accuracy (Avoid)')
plt.subplot(2,2,3)
# plt.hist(p_look, alpha=0.5)
# plt.axvline(np.mean(p_look), color='darkblue')
# plt.hist(p_look2, alpha=0.5)
# plt.axvline(np.mean(p_look2), color='darkorange')
plt.scatter(p_look, p_look2)
plt.plot([0, 1], [0, 1], 'k--')
plt.axvline(0.05, color='black', ls='--')
plt.axhline(0.05, color='black', ls='--')
plt.ylabel('Inactivation')
plt.xlabel('Control')
_, p = sp.stats.ttest_rel(p_look, p_look2)
# plt.title('p={}'.format(p))
plt.title('P value (Look)')
plt.subplot(2,2,4)
# plt.hist(p_avoid, alpha=0.5)
# plt.axvline(np.mean(p_avoid), color='darkblue')
# plt.hist(p_avoid2, alpha=0.5)
# plt.axvline(np.mean(p_avoid2), color='darkorange')
plt.scatter(p_avoid, p_avoid2)
plt.plot([0, 1], [0, 1], 'k--')
plt.axvline(0.05, color='black', ls='--')
plt.axhline(0.05, color='black', ls='--')
plt.ylabel('Inactivation')
plt.xlabel('Control')
_, p = sp.stats.ttest_rel(p_avoid, p_avoid2)
# plt.title('p={}'.format(p))
plt.title('P value (Avoid)')



## Train with look and test with avoid
shuffle_test = True
score_look_avoid, score_look_avoid2 = [], []
if shuffle_test is True:
    score_look_avoid_shuffle, score_look_avoid_shuffle2 = [], []
time_win = [-300, 0]
N_shuffle = 1
for s, session in enumerate(sessions):
    data = load_data_1_session(h5_path, session)
    selected_units = np.arange(data['spk'].shape[2])
    selected_units = np.flatnonzero(data['unit_info']['visual_cell'] > 0)
    # selected_units = np.flatnonzero(data['unit_info']['memory_cell'] > 0)
    print(selected_units)
    if np.sum(selected_units) < 1:
        print([session, 'not enough units'])
        continue
    unique_location = np.unique(data['trial_info']['cue_location'])
    selected_trials1 = (data['trial_info']['inactivation']==1) & ((data['trial_info']['cue_location']==1) | (data['trial_info']['cue_location']==3)) & (data['trial_info']['task'] == 1) & (data['trial_info']['performance'] == 1)
    selected_trials2 = (data['trial_info']['inactivation']==1) & ((data['trial_info']['cue_location']==1) | (data['trial_info']['cue_location']==3)) & (data['trial_info']['task'] == 2) & (data['trial_info']['performance'] == 1)
    selected_trials1_2 = (data['trial_info']['inactivation']==2) & ((data['trial_info']['cue_location']==1) | (data['trial_info']['cue_location']==3)) & (data['trial_info']['task'] == 1) & (data['trial_info']['performance'] == 1)
    selected_trials2_2 = (data['trial_info']['inactivation']==2) & ((data['trial_info']['cue_location']==1) | (data['trial_info']['cue_location']==3)) & (data['trial_info']['task'] == 2) & (data['trial_info']['performance'] == 1)
    # selected_trials1 = (data['trial_info']['inactivation']==1) & (data['trial_info']['task'] == 1) & (data['trial_info']['performance'] == 1)
    # selected_trials2 = (data['trial_info']['inactivation']==1) & (data['trial_info']['task'] == 2) & (data['trial_info']['performance'] == 1)
    # selected_trials1_2 = (data['trial_info']['inactivation']==2) & (data['trial_info']['task'] == 1) & (data['trial_info']['performance'] == 1)
    # selected_trials2_2 = (data['trial_info']['inactivation']==2) & (data['trial_info']['task'] == 2) & (data['trial_info']['performance'] == 1)
    if np.sum(selected_trials1) < 20 or np.sum(selected_trials2) < 20:
        print('not enough trials')
        continue
    if len(np.unique(data['trial_info']['inactivation'])) > 1 and (np.sum(selected_trials1_2) < 20 or np.sum(selected_trials2_2) < 20):
        print('not enough trials')
        continue
    score_look_avoid.append(decoding('cue_location', step_centers=[np.mean(time_win)], window_size=np.diff(time_win), selected_trials=selected_trials1, selected_trials2=selected_trials2, selected_units=selected_units, model=model))
    if shuffle_test is True:
        score_look_avoid_shuffle_i = []
        for i in range(N_shuffle):
            print([session, i])
            score_look_avoid_shuffle_i.append(decoding('cue_location', step_centers=[np.mean(time_win)], window_size=np.diff(time_win), selected_trials=selected_trials1, selected_trials2=selected_trials2, selected_units=selected_units, model=model, shuffle_test=True))
        score_look_avoid_shuffle.append(np.concatenate(score_look_avoid_shuffle_i))

    if len(np.unique(data['trial_info']['inactivation'])) > 1:
        score_look_avoid2.append(decoding('cue_location', step_centers=[np.mean(time_win)], window_size=np.diff(time_win), selected_trials=selected_trials1_2, selected_trials2=selected_trials2_2, selected_units=selected_units, model=model))
        if shuffle_test is True:
            score_look_avoid_shuffle_i = []
            for i in range(N_shuffle):
                score_look_avoid_shuffle_i.append(decoding('cue_location', step_centers=[np.mean(time_win)], window_size=np.diff(time_win), selected_trials=selected_trials1_2, selected_trials2=selected_trials2_2, selected_units=selected_units, model=model, shuffle_test=True))
            score_look_avoid_shuffle2.append(np.concatenate(score_look_avoid_shuffle_i))

score_look_avoid = np.concatenate(score_look_avoid)
score_look_avoid2 = np.concatenate(score_look_avoid2)
# score_look_avoid_shuffle = np.concatenate(score_look_avoid_shuffle)
# print(score_look_avoid_shuffle.mean())
# nan_removed = (~np.isnan(score_look_avoid))&(~np.isnan(score_look_avoid_shuffle))
# score_look_avoid = score_look_avoid[nan_removed]
# score_look_avoid_shuffle = score_look_avoid_shuffle[nan_removed]

##

_, p12 = sp.stats.ttest_rel(score_look, score_avoid)
_, p1 = sp.stats.ttest_1samp(score_look, 0.5)
_, p2 = sp.stats.ttest_1samp(score_avoid, 0.5)
bin_size = 0.03
bins = np.arange(0, 1, bin_size)
plt.figure()
plt.subplot(1,2,1)
[score_look_hist, _, _] = plt.hist(score_look, bins, color=colors[0], alpha=0.3)
[score_avoid_hist, _, _] = plt.hist(score_avoid, bins, color=colors[1], alpha=0.3)
plt.xlabel('Decoding accuracy')
plt.ylabel('Session count')
plt.axvline(0.5, ls='--', color='black')
# plt.legend()
# plt.axvline(score_look.mean(), ls='-', color=colors[0])
# plt.axvline(score_avoid.mean(), ls='-', color=colors[1])
ax1 = plt.gca()
ax2 = ax1.twinx()
mu1,sigma1 = curve_fit(norm.cdf, bins[:-1]+bin_size/2, np.cumsum(score_look_hist)/np.sum(score_look_hist), p0=[0,1])[0]
ax2.plot(bins[:-1]+bin_size/2, norm.cdf(bins[:-1]+bin_size/2, mu1, sigma1), color=colors[0], linewidth=4)
mu1,sigma1 = curve_fit(norm.cdf, bins[:-1]+bin_size/2, np.cumsum(score_avoid_hist)/np.sum(score_avoid_hist), p0=[0,1])[0]
ax2.plot(bins[:-1]+bin_size/2, norm.cdf(bins[:-1]+bin_size/2, mu1, sigma1), color=colors[1], linewidth=4)
# ax2.plot(bins[:-1]+bin_size/2, np.cumsum(score_look_hist)/np.sum(score_look_hist), color=colors[0], linewidth=3)
# ax2.plot(bins[:-1]+bin_size/2, np.cumsum(score_avoid_hist)/np.sum(score_avoid_hist), color=colors[1], linewidth=3)
plt.ylim([0, 1.01])
plt.ylabel('Cumulative distribution function')
plt.xlim([0.3, 1])

plt.subplot(1,2,2)
bin_size = 0.03
bins = np.arange(0, 1, bin_size)
_, p3 = sp.stats.ttest_1samp(score_look_avoid, 0.5)
[score_look_avoid_hist, _, _] = plt.hist(score_look_avoid, bins=bins, alpha=0.3, color=colors[2])
# plt.axvline(score_look_avoid.mean(), ls='-', color=colors[2])
plt.axvline(0.5, ls='--', color='k')
plt.xlabel('Decoding accuracy')
plt.ylabel('Session count')
plt.xlim([0.2, 0.8])
ax1 = plt.gca()
ax2 = ax1.twinx()
mu1,sigma1 = curve_fit(norm.cdf, bins[:-1]+bin_size/2, np.cumsum(score_look_avoid_hist)/np.sum(score_look_avoid_hist), p0=[0,1])[0]
ax2.plot(bins[:-1]+bin_size/2, norm.cdf(bins[:-1]+bin_size/2, mu1, sigma1), color=colors[2], linewidth=4)
plt.ylim([0, 1.01])
# ax2.plot(bins[:-1]+bin_size/2, np.cumsum(score_look_avoid_hist)/np.sum(score_look_avoid_hist), color=colors[2], linewidth=2)
plt.ylabel('Cumulative distribution function')

##
plt.figure()
p_look_avoid = []
for i in range(len(score_look_avoid)):
    plt.subplot(5, 8, i+1)
    plt.hist(score_look_avoid_shuffle[i], alpha=0.5)
    plt.axvline(score_look_avoid[i], color='darkblue')
    p = np.mean(score_look_avoid_shuffle[i]<=score_look_avoid[i])
    p_look_avoid.append(p)
    plt.title('{} p={}'.format(sessions[i], p))
p_look_avoid = np.array(p_look_avoid)

plt.figure()
plt.subplot(1,2,1)
plt.hist(score_look_avoid, alpha=0.5, color=colors[2])
plt.axvline(score_look_avoid.mean(), ls='-', color=colors[2])
plt.axvline(0.5, ls='--', color='k')
plt.xlabel('Decoding accuracy')
plt.ylabel('Session count')
plt.xlim([0.2, 0.8])

plt.tight_layout()
plt.subplot(1,2,2)
plt.hist(p_look_avoid, alpha=0.5)
plt.axvline(0.05, ls='--', color='k')
plt.xlabel('P value')
plt.ylabel('Session count')

_, p = sp.stats.ttest_1samp(score_look_avoid, 0.5)
##
plt.figure()
p_look_avoid, p_look_avoid2 = [], []
for i in range(len(score_look_avoid)):
    plt.subplot(2, len(score_look_avoid), i+1)
    plt.hist(score_look_avoid_shuffle[i], alpha=0.5)
    plt.axvline(score_look_avoid[i], color='darkblue')
    p = np.mean(score_look_avoid_shuffle[i]<score_look_avoid[i])
    p_look_avoid.append(p)
    plt.title('p={}'.format(p))
    plt.subplot(2, len(score_look_avoid), i+1+len(score_look_avoid))
    plt.hist(score_look_avoid_shuffle2[i], color='orange', alpha=0.5)
    plt.axvline(score_look_avoid2[i], color='darkorange')
    p = np.mean(score_look_avoid_shuffle2[i]<score_look_avoid2[i])
    p_look_avoid2.append(p)
    plt.title('p={}'.format(p))

##
bin_size = 0.03
bins = np.arange(0, 1, bin_size)
plt.figure()
h_main, h_top, h_right = scatter_hist(score_look_avoid, score_look_avoid2, kwargs_scatter={'edgecolors':colors[2], 'marker':'o', 's':50, 'c':[0,0,0,0], 'linewidth':2}, kwargs_hist={'bins':bins, 'color':colors[2], 'alpha':0.3})
# plt.scatter(score_look_avoid, score_look_avoid2)
h_main.axvline(0.5, color='black', ls='--')
h_main.axhline(0.5, color='black', ls='--')
h_main.plot([0.2, 0.8], [0.2, 0.8], 'k--')
h_main.set_ylabel('Decoding accuracy (inactivation)')
h_main.set_xlabel('Decoding accuracy (control)')
plt.xlim([0.2, 0.8])
plt.ylim([0.2, 0.8])
h_top.axvline(0.5, color='black', ls='--')
h_right.axhline(0.5, color='black', ls='--')
h_top2 = h_top.twinx()
mu1,sigma1 = curve_fit(norm.cdf, bins[:-1]+bin_size/2, np.cumsum(np.histogram(score_look_avoid, bins)[0])/np.sum(np.histogram(score_look_avoid, bins)[0]), p0=[0,1])[0]
h_top2.plot(bins[:-1]+bin_size/2, norm.cdf(bins[:-1]+bin_size/2, mu1, sigma1), color=colors[2], linewidth=3)
plt.ylim([0, 1.01])
# h_top2.plot(bins[:-1]+bin_size/2, np.cumsum(np.histogram(score_look_avoid, bins)[0]), color=colors[2], linewidth=3)
h_right2 = h_right.twiny()
mu1,sigma1 = curve_fit(norm.cdf, bins[:-1]+bin_size/2, np.cumsum(np.histogram(score_look_avoid2, bins)[0])/np.sum(np.histogram(score_look_avoid2, bins)[0]), p0=[0,1])[0]
h_right2.plot(norm.cdf(bins[:-1]+bin_size/2, mu1, sigma1), bins[:-1]+bin_size/2, color=colors[2], linewidth=3)
plt.xlim([0, 1.01])
# h_right2.plot(np.cumsum(np.histogram(score_look_avoid2, bins)[0]), bins[:-1]+bin_size/2, color=colors[2], linewidth=3)


_, p1 = sp.stats.ttest_1samp(np.concatenate([score_look, score_avoid]), 0.5)
_, p2 = sp.stats.ttest_1samp(np.concatenate([score_look2, score_avoid2]), 0.5)
print([p1, p2])
# _, p12 = sp.stats.ttest_rel(np.concatenate([score_look, score_avoid]), np.concatenate([score_look2, score_avoid2]))
# _, p12 = sp.stats.ttest_rel(score_look, score_look2)
# _, p12 = sp.stats.ttest_rel(score_avoid, score_avoid2)
_, p12 = sp.stats.wilcoxon(np.concatenate([score_look, score_avoid]), np.concatenate([score_look2, score_avoid2]), alternative='greater')
# n_sessions = np.concatenate([score_look, score_avoid]).shape[0]
# n_effective_sessions = np.sum(np.concatenate([score_look, score_avoid]) - np.concatenate([score_look2, score_avoid2]) > 0)
# p12 = sp.stats.binom_test(n_effective_sessions, n_sessions, alternative='greater')
print(p12)

_, p12 = sp.stats.wilcoxon(score_look_avoid, score_look_avoid2, alternative='less')
print(p12)

## Control for time-dependent drift
# Train with 1st half and test with 2nd half
score_look_1_2, score_avoid_1_2, score_look_2_avoid_1 = [], [], []
time_win = [-300, 0]
N_shuffle = 1
for s, session in enumerate(sessions):
    data = load_data_1_session(h5_path, session)
    selected_units = np.arange(data['spk'].shape[2])
    unique_location = np.unique(data['trial_info']['cue_location'])
    selected_trials_look = (data['trial_info']['inactivation']==1) & ((data['trial_info']['cue_location']==1) | (data['trial_info']['cue_location']==3)) & (data['trial_info']['task'] == 1) & (data['trial_info']['performance'] == 1)
    selected_trials_avoid = (data['trial_info']['inactivation']==1) & ((data['trial_info']['cue_location']==1) | (data['trial_info']['cue_location']==3)) & (data['trial_info']['task'] == 2) & (data['trial_info']['performance'] == 1)

    selected_trials1 = np.flatnonzero(selected_trials_look)[:int(selected_trials_look.sum()/2)]
    selected_trials2 = np.flatnonzero(selected_trials_look)[int(selected_trials_look.sum()/2):]
    if len(selected_trials1) < 10 or len(selected_trials2) < 10:
        print('not enough trials')
        continue
    score_look_1_2.append(decoding('cue_location', step_centers=[np.mean(time_win)], window_size=np.diff(time_win), selected_trials=selected_trials1, selected_trials2=selected_trials2, model=model))

    selected_trials1 = np.flatnonzero(selected_trials_avoid)[:int(selected_trials_avoid.sum()/2)]
    selected_trials2 = np.flatnonzero(selected_trials_avoid)[int(selected_trials_avoid.sum()/2):]
    if len(selected_trials1) < 10 or len(selected_trials2) < 10:
        print('not enough trials')
        continue
    score_avoid_1_2.append(decoding('cue_location', step_centers=[np.mean(time_win)], window_size=np.diff(time_win), selected_trials=selected_trials1, selected_trials2=selected_trials2, model=model))

    selected_trials1 = np.flatnonzero(selected_trials_look)[int(selected_trials_look.sum()/2):]
    selected_trials2 = np.flatnonzero(selected_trials_avoid)[:int(selected_trials_look.sum()/2)]
    if len(selected_trials1) < 10 or len(selected_trials2) < 10:
        print('not enough trials')
        continue
    score_look_2_avoid_1.append(decoding('cue_location', step_centers=[np.mean(time_win)], window_size=np.diff(time_win), selected_trials=selected_trials1, selected_trials2=selected_trials2, model=model))


score_look_1_2, score_avoid_1_2, score_look_2_avoid_1 = np.concatenate(score_look_1_2), np.concatenate(score_avoid_1_2), np.concatenate(score_look_2_avoid_1)

##
_, p1 = sp.stats.ttest_1samp(score_look_1_2, 0.5)
_, p2 = sp.stats.ttest_1samp(score_avoid_1_2, 0.5)
_, p3 = sp.stats.ttest_1samp(score_look_2_avoid_1, 0.5)
bin_size = 0.03
bins = np.arange(0, 1, bin_size)
h_fig, h_ax = plt.subplots(1, 3, sharey=True)
plt.axes(h_ax[0])
[score_look_avoid_hist, _, _] = plt.hist(score_look_2_avoid_1, bins, color=colors[2], alpha=0.3)
plt.xlabel('Decoding accuracy')
plt.ylabel('Session count')
plt.axvline(0.5, ls='--', color='black')
ax1 = plt.gca()
ax2 = ax1.twinx()
mu1,sigma1 = curve_fit(norm.cdf, bins[:-1]+bin_size/2, np.cumsum(score_look_avoid_hist)/np.sum(score_look_avoid_hist), p0=[0,1])[0]
ax2.plot(bins[:-1]+bin_size/2, norm.cdf(bins[:-1]+bin_size/2, mu1, sigma1), color=colors[2], linewidth=4)
plt.ylim([0, 1.01])
plt.xlim([0.2, 0.9])
ax2.set_yticks([])

plt.axes(h_ax[1])
[score_look_hist, _, _] = plt.hist(score_look_1_2, bins, color=colors[0], alpha=0.3)
plt.xlabel('Decoding accuracy')
plt.axvline(0.5, ls='--', color='black')
ax1 = plt.gca()
ax2 = ax1.twinx()
mu1,sigma1 = curve_fit(norm.cdf, bins[:-1]+bin_size/2, np.cumsum(score_look_hist)/np.sum(score_look_hist), p0=[0,1])[0]
ax2.plot(bins[:-1]+bin_size/2, norm.cdf(bins[:-1]+bin_size/2, mu1, sigma1), color=colors[0], linewidth=4)
plt.ylim([0, 1.01])
plt.xlim([0.3, 1])
ax2.set_yticks([])

plt.axes(h_ax[2])
[score_avoid_hist, _, _] = plt.hist(score_avoid_1_2, bins, color=colors[1], alpha=0.3)
plt.xlabel('Decoding accuracy')
plt.axvline(0.5, ls='--', color='black')
ax1 = plt.gca()
ax2 = ax1.twinx()
mu1,sigma1 = curve_fit(norm.cdf, bins[:-1]+bin_size/2, np.cumsum(score_avoid_hist)/np.sum(score_avoid_hist), p0=[0,1])[0]
ax2.plot(bins[:-1]+bin_size/2, norm.cdf(bins[:-1]+bin_size/2, mu1, sigma1), color=colors[1], linewidth=4)
plt.ylim([0, 1.01])
plt.ylabel('Cumulative distribution function')
plt.xlim([0.3, 1])