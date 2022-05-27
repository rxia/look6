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
import PyNeuroAna as pna


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
def get_X_y(data, data_type, time_win, selected_trials, cond, resample='none'):
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
                clf = svm.SVC(probability=True, kernel='linear', gamma='scale')
                clf.fit(X, y)
                score_t.append(select_by_label(clf.predict_proba(X), y))
            else:
                if model=='svc':
                    clf = svm.SVC(class_weight='balanced', kernel='linear', gamma='scale')
                elif model=='svr':
                    clf = svm.SVR(class_weight='balanced', kernel='linear', gamma='scale')
                scores = cross_val_score(clf, X, y, cv=5)
                # scores = cross_val_score(clf, X, y, cv=np.sum(y==0))
                score_t.append(np.mean(scores))
        else:
            X_test, y_test = get_X_y(data, current_t_win, selected_trials2, cond)
            if model=='svc_prob':
                clf = svm.SVC(probability=True, kernel='linear', class_weight='balanced', gamma='scale')
                clf.fit(X, y)
                score_t.append(select_by_label(clf.predict_proba(X_test), y_test))
            else:
                if model=='svc':
                    clf = svm.SVC(class_weight='balanced', kernel='linear', gamma='scale')
                elif model=='svr':
                    clf = svm.SVR(class_weight='balanced', kernel='linear', gamma='scale')
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

    feature_importance = np.abs(X[y==0].mean(axis=0) - X[y==1].mean(axis=0)) / (X[y==0].std(axis=0) + X[y==1].std(axis=0))
    # clf.fit(X, y)
    # feature_importance = np.abs(clf.coef_)
    return score_t, step_centers, feature_importance


colors = ['purple', 'orange', 'limegreen', 'royalblue']

## Decode task rule (control - first half/second half)
decoding_result_task = np.zeros((len(sessions)))
decoding_result_half = np.zeros((len(sessions)))
decoding_result_shuffle = np.zeros((len(sessions), 100))
corr_feature_importance = []
use_preset_ind = False
preset_inds_half = [[[[ 55, 154], [165, 264]], [[340, 439], [450, 549]]],
                    [[[ 22, 121], [132, 231]], [[351, 450], [461,560]]], #[[406, 505], [516, 615]]],
                    [[[ 19,  58], [ 69, 108]], [[143, 182], [193, 232]], [[260, 299], [310, 349]], [[401, 440], [451, 490]]], #[[508, 547], [558, 597]]],
                    [[[ 73, 172], [183, 282]], [[283, 482], [493, 592]]],
                    [[[ 48, 147], [158, 257]], [[352, 451], [462, 561]]],
                    [[[ 52, 151], [162, 261]], [[402, 501], [512, 611]]],
                    [[[ 59, 148], [159, 248]], [[315, 404], [415, 504]]]]
preset_inds_task = [[[[212, 311], [322, 421]]],
                    [[[146, 245], [256, 355]]],
                    [[[ 80, 119], [130, 169]], [[208, 247], [258, 297]], [[312, 351], [362, 401]]],
                    [[[249, 348], [359, 458]]],
                    [[[197, 296], [307, 406]]],
                    [[[205, 304], [315, 414]]],
                    [[[209, 298], [309, 398]]]]

for i, session in enumerate(sessions):
    print(session)
    data = load_data_1_session(h5_path, session)
    if session[:2] == 'aq':
        color = 'cornflowerblue'
    # else:
    #     color = 'salmon'

    data['trial_info']['which_half'] = np.zeros(data['spk'].shape[0])

    if use_preset_ind:
        for j in range(len(preset_inds_half[i])):
            selected_trials = np.flatnonzero(data['trial_info']['correct_trials'] > 0)
            data['trial_info']['which_half'][selected_trials[preset_inds_half[i][j][0][0]:preset_inds_half[i][j][0][1]]] = 1
            data['trial_info']['which_half'][selected_trials[preset_inds_half[i][j][1][0]:preset_inds_half[i][j][1][1]]] = 2
            decoding_result, _, feature_importance = decoding('which_half', time_win=[-300, 0], selected_trials=selected_trials)
            decoding_result_half[i] = decoding_result_half[i] + decoding_result
        decoding_result_half[i] = decoding_result_half[i]/len(preset_inds_half[i])
        for j in range(len(preset_inds_task[i])):
            selected_trials = np.flatnonzero(data['trial_info']['correct_trials'] > 0)
            selected_trials = np.concatenate([selected_trials[preset_inds_task[i][j][0][0]:preset_inds_task[i][j][0][1]],
                                              selected_trials[preset_inds_task[i][j][1][0]:preset_inds_task[i][j][1][1]]])
            decoding_result, _, feature_importance = decoding('task', time_win=[-300, 0], selected_trials=selected_trials)
            decoding_result_task[i] = decoding_result_task[i] + decoding_result
        decoding_result_task[i] = decoding_result_task[i]/len(preset_inds_task[i])

    else:
        selected_trials = np.flatnonzero((data['trial_info']['task'] == 1) | (data['trial_info']['task'] == 2))
        remove_switching_trials = 10
        switch_ind = [0]
        if data['trial_info']['task'][0] == 2:
            switch_ind = np.concatenate([switch_ind, np.flatnonzero(
                (data['trial_info']['task'][:-1] == 1) & (data['trial_info']['task'][1:] == 2))])
        elif data['trial_info']['task'][0] == 1:
            switch_ind = np.concatenate([switch_ind, np.flatnonzero(
                (data['trial_info']['task'][:-1] == 2) & (data['trial_info']['task'][1:] == 1))])
        n_block = 0
        inds_half, inds_task = [], []
        plt.figure(figsize=[16,12])
        for j, ind_j in enumerate(switch_ind):
            if ind_j==switch_ind[-1]:
                ind_end = data['spk'].shape[0]
            else:
                ind_end = switch_ind[j+1]
            print([ind_j, ind_end])

            ind_trials = (np.arange(data['spk'].shape[0]) >= ind_j) & (np.arange(data['spk'].shape[0]) < ind_end)
            selected_trials = np.flatnonzero((data['trial_info']['task']==1) & (data['trial_info']['correct_trials'] > 0) & ind_trials)
            # selected_trials = np.flatnonzero((data['trial_info']['task']==1) & (~np.isnan(data['trial_info']['memory_on'])) & ind_trials)
            if len(selected_trials)<20:
                continue
            data['trial_info']['which_half'][selected_trials[remove_switching_trials:int(len(selected_trials)/2)]] = 1
            data['trial_info']['which_half'][selected_trials[int(len(selected_trials)/2+remove_switching_trials):]] = 2
            selected_trials = np.concatenate([selected_trials[remove_switching_trials:int(len(selected_trials)/2)],selected_trials[int(len(selected_trials)/2+remove_switching_trials):]])
            decoding_result_1, _, feature_importance_1 = decoding('which_half', time_win=[-300, 0], selected_trials=selected_trials)

            selected_trials = np.flatnonzero((data['trial_info']['task']==2) & (data['trial_info']['correct_trials'] > 0) & ind_trials)
            # selected_trials = np.flatnonzero((data['trial_info']['task']==2) & (~np.isnan(data['trial_info']['memory_on'])) & ind_trials)
            if len(selected_trials)<20:
                continue
            data['trial_info']['which_half'][selected_trials[remove_switching_trials:int(len(selected_trials)/2)]] = 1
            data['trial_info']['which_half'][selected_trials[int(len(selected_trials)/2+remove_switching_trials):]] = 2
            selected_trials = np.concatenate([selected_trials[remove_switching_trials:int(len(selected_trials)/2)],selected_trials[int(len(selected_trials)/2+remove_switching_trials):]])
            decoding_result_2, _, feature_importance_2 = decoding('which_half', time_win=[-300, 0], selected_trials=selected_trials)

            n_block = n_block + 1
            decoding_result_half[i] = decoding_result_half[i] + (decoding_result_1+decoding_result_2)/2

            # selected_trials = np.flatnonzero((~np.isnan(data['trial_info']['memory_on'])) & (((data['trial_info']['task']==data['trial_info']['task'][0]) & (data['trial_info']['which_half']==2)) | ((data['trial_info']['task']==({1,2}-{data['trial_info']['task'][0]}).pop()) & (data['trial_info']['which_half']==1))) & ind_trials)
            selected_trials = np.flatnonzero((data['trial_info']['correct_trials'] > 0) & (((data['trial_info']['task']==data['trial_info']['task'][0]) & (data['trial_info']['which_half']==2)) | ((data['trial_info']['task']==({1,2}-{data['trial_info']['task'][0]}).pop()) & (data['trial_info']['which_half']==1))) & ind_trials)
            decoding_result, _, feature_importance_task = decoding('task', time_win=[-300, 0], selected_trials=selected_trials)
            decoding_result_task[i] = decoding_result_task[i] + decoding_result

            corr_temp = np.zeros(3)

            plt.subplot(2, 3, 3*j+1)
            plt.scatter(feature_importance_1, feature_importance_task)
            plt.plot([0,1], [0,1], 'k')
            corr_temp[0] = np.corrcoef(feature_importance_1, feature_importance_task)[0, 1]
            plt.title('{:.4f}'.format(corr_temp[0]))
            plt.xlabel('Half decoder (block {})'.format(j*2+1))
            plt.ylabel('Task decoder (block {} & block {}'.format(j*2+1, j*2+2))

            plt.subplot(2, 3, 3*j+2)
            plt.scatter(feature_importance_2, feature_importance_task)
            plt.plot([0,1], [0,1], 'k')
            corr_temp[1] = np.corrcoef(feature_importance_2, feature_importance_task)[0, 1]
            plt.title('{:.4f}'.format(corr_temp[1]))
            plt.xlabel('Half decoder (block {})'.format(j*2+2))
            plt.ylabel('Task decoder (block {} & block {}'.format(j*2+1, j*2+2))

            plt.subplot(2, 3, 3*j+3)
            plt.scatter(feature_importance_1, feature_importance_2)
            plt.plot([0,1], [0,1], 'k')
            corr_temp[2] = np.corrcoef(feature_importance_1, feature_importance_2)[0, 1]
            plt.title('{:.4f}'.format(corr_temp[2]))
            plt.xlabel('Half decoder (block {})'.format(j*2+1))
            plt.ylabel('Half decoder (block {})'.format(j*2+2))

            corr_feature_importance.append(corr_temp)

        plt.savefig('feature_importance_{}'.format(session))

        decoding_result_half[i] = decoding_result_half[i] / n_block
        decoding_result_task[i] = decoding_result_task[i] / n_block
        # plt.figure()
        # plt.plot(data['trial_info']['task'][data['trial_info']['correct_trials'] > 0])
        # plt.plot(data['trial_info']['which_half'][data['trial_info']['correct_trials'] > 0])

        # for j in range(100):
        #     # decoding_result[i, j], step_centers = decoding(decode_variable, time_win=[-300, 0], selected_trials=selected_trials)
        #     decoding_result_shuffle[i, j], step_centers = decoding('task', time_win=[-300, 0], selected_trials=selected_trials, shuffle=True)
        # plt.errorbar(i, decoding_result_shuffle[i, :].mean(axis=0), decoding_result_shuffle[i, :].std(axis=0), c=color, alpha=0.4, ls='', marker='o')
plt.figure()
plt.scatter(np.arange(7), decoding_result_half[:7], c='salmon', marker='o', label='decoding 1st/2nd half (AQ)')
plt.scatter(np.arange(7), decoding_result_task[:7], c='salmon', marker='x', label='decoding task (AQ)')
plt.scatter(np.arange(7, 15), decoding_result_half[7:], c='cornflowerblue', marker='o', label='decoding 1st/2nd half (HB)')
plt.scatter(np.arange(7, 15), decoding_result_task[7:], c='cornflowerblue', marker='x', label='decoding task (HB)')
plt.legend()
    # plt.close(h_fig)
plt.figure()
corr_feature_importance = np.stack(corr_feature_importance)
plt.hist(np.concatenate((corr_feature_importance[:, 0], corr_feature_importance[:, 1])), label='half - task')
plt.hist(corr_feature_importance[:, 2], label='half - half')
plt.xlabel('correlaion between feature-importances')
plt.legend()
plt.savefig('feature_importance_correlation_all'.format(session))

## PCA
from sklearn.decomposition import PCA
colors = ['white', 'mistyrose', 'lightcyan']
n_component = 4
for i, session in enumerate(sessions):
    print(session)
    data = load_data_1_session(h5_path, session)
    selected_trials = np.flatnonzero(data['trial_info']['correct_trials'])
    # selected_trials = np.flatnonzero(~np.isnan(data['trial_info']['edata_memory_on']))
    data_realigned = realign((data['trial_info']['edata_memory_on']-data['trial_info']['edata_first_display'])*1000, data_to_align=data['spk'], ts=data['ts'], time_win=[-300,0], selected_trials=selected_trials)
    X = data_realigned.mean(axis=1)

    task = data['trial_info']['task'][selected_trials]
    print([X[task==1].mean(), X[task==2].mean()])

    pca = PCA(n_components=n_component, whiten=True)
    pca.fit(X)
    Y = pca.transform(X)

    Y_detrended = sp.signal.detrend(Y, axis=0)

    h_fig = plt.figure()
    for t in range(len(selected_trials)):
        plt.axvline(t, color=colors[int(task[t])])
    for i_PC in range(n_component):
        plt.plot(pna.SmoothTrace(Y[:,i_PC]+1.5*i_PC, sk_std=5, fs=1.0, ts=None, axis=0), color='k', ls=':')
        plt.plot(pna.SmoothTrace(Y_detrended[:,i_PC]+1.5*i_PC, sk_std=5, fs=1.0, ts=None, axis=0), color='k')
    plt.xlabel('trials')
    plt.yticks(np.arange(n_component)*1.5, ['PC{}'.format(n) for n in range(n_component)])
    plt.title(session)

    plt.savefig('PCA_{}_detrended'.format(session))
    plt.close(h_fig)

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
decoding_result = np.zeros((len(sessions)))
# decoding_result = np.zeros((len(sessions), 100))
decoding_result_shuffle = np.zeros((len(sessions), 100))
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
        selected_trials = (broken_trials | correct_trials | wrong_trials) & (data['trial_info']['task']==2)
    elif decode_variable=='correct_trials':
        selected_trials = wrong_trials | correct_trials
    elif decode_variable=='task':
        selected_trials = correct_trials & (data['trial_info']['task']>0)
    # selected_trials_k = selected_trials & (data['trial_info']['task']==2)
    decoding_result[i], step_centers = decoding(decode_variable, time_win=[-300, 0], selected_trials=selected_trials)
    for j in range(100):
        # decoding_result[i, j], step_centers = decoding(decode_variable, time_win=[-300, 0], selected_trials=selected_trials)
        decoding_result_shuffle[i, j], step_centers = decoding(decode_variable, time_win=[-300, 0], selected_trials=selected_trials, shuffle=True)
    plt.plot(i, decoding_result[i], c=color, marker='o')
    # plt.errorbar(i, decoding_result[i, :].mean(axis=0), decoding_result[i, :].std(axis=0), c=color, marker='o')
    plt.errorbar(i, decoding_result_shuffle[i, :].mean(axis=0), decoding_result_shuffle[i, :].std(axis=0), c=color, alpha=0.4, ls='', marker='o')

plt.axhline(0.5, linestyle='--')
plt.savefig('decoding_{}_avoid.png'.format(decode_variable), dpi=300)

decoding_result_fixation_break = {'accuracy': decoding_result, 'accuracy_trial_shuffled': decoding_result_shuffle}
sp.io.savemat('decoding_result_fixation_break_avoid.mat', decoding_result_fixation_break)

##
spk_broken = realign(data['trial_info']['memory_on']-data['trial_info']['first_display'], data['spk'], ts=ts, time_win=[-300, 0], selected_trials=broken_trials)
spk_correct = realign(data['trial_info']['memory_on']-data['trial_info']['first_display'], data['spk'], ts=ts, time_win=[-300, 0], selected_trials=correct_trials)
plt.errorbar(np.arange(data['spk'].shape[2]), spk_broken.mean(axis=(0,1)), yerr=spk_broken.std(axis=(0,1))/np.sqrt(data['spk'].shape[1]),marker='o',label='fixation_broken')
plt.errorbar(np.arange(data['spk'].shape[2]), spk_correct.mean(axis=(0,1)), yerr=spk_correct.std(axis=(0,1))/np.sqrt(data['spk'].shape[1]),marker='o',label='correct')
# plt.plot(spk_broken.mean(axis=(0,1)),marker='o',label='fixation_broken')
# plt.plot(spk_correct.mean(axis=(0,1)),marker='o',label='correct')
plt.legend()


## LFP
# time_win = [-300,0]
time_win = [300,1000]
frequency_ranges = [7, 13], [13, 30], [30, 50]
frequency_labels = ['alpha', 'beta', 'gamma']
colors = ['white', 'mistyrose', 'lightcyan']
spcg_result = []
for i, session in enumerate(sessions):
    print(session)
    data = load_data_1_session(h5_path, session, do_not_load=[])
    std_lfp = np.nanstd(data['lfp'], axis=(1, 2))
    selected_trials = np.flatnonzero((data['trial_info']['correct_trials'] > 0) & (np.abs(std_lfp-np.nanmean(std_lfp)) < 2*np.nanstd(std_lfp)))

    data_realigned = realign((data['trial_info']['edata_memory_on']-data['trial_info']['edata_first_display'])*1000, data_to_align=data['lfp'], ts=data['ts'], time_win=time_win, selected_trials=selected_trials)


    spcg, spcg_t, spcg_f = pna.ComputeSpectrogram(data_realigned, fs=1000.0, t_ini=time_win[0]/1000, t_bin=(time_win[1]-time_win[0])/1000, t_step=(time_win[1]-time_win[0])/1000, f_lim=[3,100])
    spcg = np.swapaxes(spcg, 2, 3)
    spcg_result_temp = []
    for j in range(len(frequency_ranges)):
        spcg_result_temp.append(np.nanmean(np.log(spcg[:, (spcg_f>=frequency_ranges[j][0])&(spcg_f<frequency_ranges[j][1])]), axis=1))
    spcg_result.append(spcg_result_temp)

    h_fig = plt.figure()
    task = data['trial_info']['task'][selected_trials]
    for t in range(len(selected_trials)):
        plt.axvline(t, color=colors[int(task[t])])
    for j in range(len(frequency_ranges)):
        to_plot = np.nanmean(pna.SmoothTrace(np.squeeze(spcg_result[i][j]), sk_std=10, fs=1.0, ts=None, axis=0), axis=1)
        to_plot = np.nanmean(spcg_result[i][j], axis=1)
        plt.plot((to_plot-np.nanmean(to_plot))/np.nanstd(to_plot)+3*j, color='k')
    plt.yticks(np.arange(len(frequency_ranges))*3, frequency_labels)
    plt.xlabel('trials')
    plt.title('Delay period {}'.format(session))
    plt.savefig('LFP_by_trials_{}_delay'.format(session))
    plt.close(h_fig)


## Decode using mat_y data from Donatas
dates = ['aq_20161012','aq_20170914','aq_20170919','aq_20180404','aq_20180406','aq_20180415','aq_20180419','hb_20170330','hb_20170831','hb_20171207','hb_20171219','hb_20171222','hb_20180219','hb_20180221']
score_all = []
for date in dates:
    # X = np.load('./data/{}_no_nan_data'.format(date))
    # y = np.load('./data/{}_no_nan_label'.format(date))
    # # y = y[np.isnan(X.sum(axis=1))==0]
    # # X = X[np.isnan(X.sum(axis=1))==0, :]
    # X = X[:, np.isnan(X.sum(axis=0))==0]

    X = np.load('./data/{}_no_nan_data'.format(date))
    y = np.squeeze(np.load('./data/{}_no_nan_label'.format(date)))
    # y = y[np.isnan(X.sum(axis=1))==0]
    # X = X[np.isnan(X.sum(axis=1))==0, :]
    X = X[:, np.isnan(X.sum(axis=0))==0]
    selected_trials = y == 2
    X = X[selected_trials, :]
    y = y[selected_trials]
    y[1:int(len(y)/2)] = 1
    y[int(len(y)/2):-1] = 2

    if np.min(X.shape)>0:
        clf = svm.SVC(class_weight='balanced', kernel='linear', gamma='scale')
        scores = cross_val_score(clf, X, y, cv=5)
        # scores = cross_val_score(clf, X, y, cv=np.sum(y==0))
        score_all.append([X.shape[0], X.shape[1], np.mean(scores)])
    else:
        score_all.append([X.shape[0], X.shape[1], np.nan])
score_all = np.array(score_all)

for i, date in enumerate(dates):
    print([date, score_all[i,0], score_all[i,1], score_all[i,2]])