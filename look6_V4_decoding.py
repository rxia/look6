import h5py
import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


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
        data['spk'] = data['spk'].astype(np.float) * 1000  # change the unit from spikes/ms to spikes/sec for later convenience
    return data


h5_path='D:/Dropbox/analysis/donatas_data/look6_V4_data.h5'  # data path

with h5py.File(h5_path, 'r') as f:
    sessions = list(f.keys()) # All session names

data = load_data_1_session(h5_path, sessions[0])  # load one example data session
ts = data['ts']


##
def realign(times, data_to_align, ts=ts, time_win=[-600, 700], selected_trials=None):
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


## Decoding
def get_X_y(data, data_type, time_win, selected_trials, cond, resample='down'):
    fr_by_conds = get_fr_by_conds(data, 'memory_on', time_win, data_type=data_type, selected_trials=selected_trials,
                                  avg_trials=False, conds=[cond])
    if resample=='up':
        N_samples = [x.shape[0] for x in fr_by_conds]
        for i in range(len(fr_by_conds)):
            N_sample_i = fr_by_conds[i].shape[0]
            div_i = divmod(np.max(N_samples), N_sample_i)[0]
            mod_i = divmod(np.max(N_samples), N_sample_i)[1]
            original_ind_i = np.arange(N_sample_i)
            resampled_ind_i = np.concatenate([np.repeat(original_ind_i, div_i), np.random.choice(original_ind_i, mod_i, replace=False)])
            fr_by_conds[i] = fr_by_conds[i][resampled_ind_i]
    elif resample=='down':
        N_samples = [x.shape[0] for x in fr_by_conds]
        for i in range(len(fr_by_conds)):
            N_sample_i = fr_by_conds[i].shape[0]
            original_ind_i = np.arange(N_sample_i)
            resampled_ind_i = np.random.choice(original_ind_i, np.min(N_samples), replace=False)
            fr_by_conds[i] = fr_by_conds[i][resampled_ind_i]

    X = np.concatenate(fr_by_conds)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    y = np.concatenate([np.ones(x.shape[0]) * i for i, x in enumerate(fr_by_conds)])
    randomized_ind = np.random.permutation(len(y))
    X = X[randomized_ind, :]
    y = y[randomized_ind]
    return X, y


def select_by_label(a, b):
    selected = []
    for i in range(a.shape[0]):
        selected.append(a[i, int(b[i])])
    return np.stack(selected)


def plot_shaded_err(y, x, label=None, color=None):
    y_mean = y.mean(axis=1)
    y_se = y.std(axis=1)/np.sqrt(y.shape[1])
    if color is not None:
        plt.plot(x, y_mean, label=label, color=color)
        plt.fill_between(x, y_mean - y_se, y_mean + y_se, color=color, alpha=0.5)
    else:
        plt.plot(x, y_mean, label=label)
        plt.fill_between(x, y_mean - y_se, y_mean + y_se, alpha=0.5)
    return


def decoding(cond, data_type='spk', time_win=None, selected_trials=None, selected_trials2=None, model='svc', shuffle=False, ax1=None, ax2=None, color=None, label=None):
    # cond: the name of condition to decode (a key in the "trial_info" dict)
    # data_type: either "spk" or "lfp". Or add other types of data (e.g. LFP power bands) into the data structure
    # selected_trials: boolean or index, an array of the trials selected to feed into this decoding
    # selected_trials2: used if training and testing with different sets of trials; otherwise will do cross validation
    # model: "svc", "svr", or "svc_prob"
    # ax1: axis to plot the decoding result
    # ax2: axis to plot the PSTH (input data) that used for decoding
    # color: the color of plotting
    # label: the label name of plotting

    score_t = []
    psth = []
    if selected_trials is None:
        selected_trials = np.flatnonzero(data['trial_info']['correct_trials'])
    if np.issubdtype(selected_trials.dtype, np.bool_):
        selected_trials = np.flatnonzero(selected_trials)

    # Decoding with a sliding window
    if time_win is None:
        time_win = [-600, 1100]
        step_size = 128
        window_size = 256
        step_centers = np.arange(time_win[0], time_win[1], step_size)
        n_steps = len(step_centers)
        # n_steps = 14
        # step_centers = np.arange(n_steps) * step_size - 572
    else:
        window_size = time_win[1] - time_win[0]
        step_centers = np.array([np.mean(time_win)])
        n_steps = 1

    for t in range(n_steps):
        current_t_win = [step_centers[t] - window_size / 2, step_centers[t] + window_size / 2]
        X, y = get_X_y(data, data_type, current_t_win, selected_trials, cond)
        if shuffle:
            np.random.shuffle(y)

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
                    clf = svm.SVC(class_weight='balanced', gamma='scale')
                elif model=='svr':
                    clf = svm.SVR(class_weight='balanced', gamma='scale')
                scores = cross_val_score(clf, X, y, cv=5)
                score_t.append(np.mean(scores))
        else:
            X_test, y_test = get_X_y(data, current_t_win, selected_trials2, cond)
            if model=='svc_prob':
                clf = svm.SVC(probability=True, class_weight='balanced', gamma='scale')
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
    if label is None:
        label = cond
    if ax1 is not None:
        plt.axes(ax1)
        if len(score_t.shape) == 1:
            plt.plot(step_centers, score_t, color=color)
        else:
            plot_shaded_err(score_t, step_centers, label=label, color=color)
    if ax2 is not None:
        psth = np.stack(psth)
        plt.axes(ax2)
        for i in range(psth.shape[1]):
            plt.plot(step_centers, psth[:, i], '--', color=color)
    return score_t, step_centers


colors = ['purple', 'orange', 'limegreen', 'royalblue']

## Decode cue location
for session in sessions:
    print(session)
    data = load_data_1_session(h5_path, session)
    h_fig = plt.figure()

    decoding_result, step_centers = decoding('task')
    plt.plot(step_centers, decoding_result, c=colors[0], label='task rule')  # Plot task rule decoding result

    selected_trials = (data['trial_info']['task'] == 1) & (data['trial_info']['correct_trials'] > 0)
    decoding_result, step_centers = decoding('memory_location', selected_trials=selected_trials)
    plt.plot(step_centers, decoding_result, c=colors[1], label='cue location (look)')  # Plot cue loation decoding result using look trials

    selected_trials = (data['trial_info']['task'] == 2) & (data['trial_info']['correct_trials'] > 0)
    decoding_result, step_centers = decoding('memory_location', selected_trials=selected_trials)
    plt.plot(step_centers, decoding_result, c=colors[1], label='cue location (avoid)')  # Plot cur loation decoding result using avoid trials

    plt.legend()
    plt.savefig('decoding_{}.png'.format(session))
    plt.close(h_fig)

## Decode fixation breaks
model = 'svc'
decode_variable = 'broken_trials'
h_fig = plt.figure()
for i, session in enumerate(sessions):
    print(session)
    data = load_data_1_session(h5_path, session)
    if session[:2] == 'aq':
        color = 'cornflowerblue'
    else:
        color = 'salmon'

    broken_trials = ((data['trial_info']['early_abort_trials']>0)|(data['trial_info']['late_abort_trials']>0)) & np.isfinite(data['trial_info']['memory_on'])
    data['trial_info']['broken_trials'] = broken_trials
    correct_trials = data['trial_info']['correct_trials'] > 0
    wrong_trials = data['trial_info']['wrong_trials'] > 0

    if decode_variable=='broken_trials':
        selected_trials = broken_trials | correct_trials | wrong_trials
    elif decode_variable=='correct_trials':
        selected_trials = wrong_trials | correct_trials
    elif decode_variable=='task':
        selected_trials = correct_trials & (data['trial_info']['task'] > 0)
    selected_trials_k = selected_trials & (data['trial_info']['task']==2)
    decoding_result, step_centers = decoding(decode_variable, time_win=[-300, 0], selected_trials=selected_trials)
    decoding_result_shuffle_all = []
    for j in range(100):
        decoding_result_shuffle, step_centers = decoding(decode_variable, time_win=[-300, 0], selected_trials=selected_trials, shuffle=True)
        decoding_result_shuffle_all.append(decoding_result_shuffle)
    decoding_result_shuffle_all = np.stack(decoding_result_shuffle_all)
    plt.plot(i, decoding_result, c=color, marker='o')
    plt.errorbar(i, decoding_result_shuffle_all.mean(axis=0), decoding_result_shuffle_all.std(axis=0), c=color, alpha=0.4, ls='', marker='o')

plt.axhline(0.5, linestyle='--')
plt.savefig('decoding_{}_avoid_all.png'.format(decode_variable), dpi=300)

##
spk_broken = realign(data['trial_info']['memory_on']-data['trial_info']['first_display'], data['spk'], ts=ts, time_win=[-300, 0], selected_trials=broken_trials)
spk_correct = realign(data['trial_info']['memory_on']-data['trial_info']['first_display'], data['spk'], ts=ts, time_win=[-300, 0], selected_trials=correct_trials)
plt.errorbar(np.arange(data['spk'].shape[2]), spk_broken.mean(axis=(0,1)), yerr=spk_broken.std(axis=(0,1))/np.sqrt(data['spk'].shape[1]),marker='o',label='fixation_broken')
plt.errorbar(np.arange(data['spk'].shape[2]), spk_correct.mean(axis=(0,1)), yerr=spk_correct.std(axis=(0,1))/np.sqrt(data['spk'].shape[1]),marker='o',label='correct')
# plt.plot(spk_broken.mean(axis=(0,1)),marker='o',label='fixation_broken')
# plt.plot(spk_correct.mean(axis=(0,1)),marker='o',label='correct')
plt.legend()