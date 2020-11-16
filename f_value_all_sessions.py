import h5py
import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

h5_path='D:/Dropbox/analysis/sgrating/sgrating_prelim_data.h5'
# h5_path='D:/Dropbox/analysis/sgrating/hb_20201008.h5'


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
    data_all = h5_read(f)

time_win_plot = [-300, 700]
ts_plot = np.arange(time_win_plot[0], time_win_plot[1], 1)


def realign(times, data_spk):
    ts = data['ts']
    time_win = time_win_plot
    new_data = []
    for i in range(data_spk.shape[0]):
        data_in_range = data_spk[i, (ts >= times[i] + time_win[0]) & (ts < times[i] + time_win[1])]
        if ts[0] > (times[i] + time_win[0]):
            data_in_range = np.concatenate((np.ones(
                (int(ts[0] - (times[i] + time_win[0]) - 1), data_in_range.shape[1])) * np.nan, data_in_range))
        if ts[-1] < (times[i] + time_win[1]):
            data_in_range = np.concatenate((data_in_range, np.ones(
                (int(times[i] + time_win[1] - ts[-1] - 1), data_in_range.shape[1])) * np.nan))

        new_data.append(data_in_range)
    return np.stack(new_data)


def select_visual_unit(data, ts, time_win_1, time_win_2):
    selected = []
    for u in range(data.shape[2]):
        fr_1 = data[:, (ts >= time_win_1[0]) & (ts < time_win_1[1]), u].mean(axis=1)
        fr_2 = data[:, (ts >= time_win_2[0]) & (ts < time_win_2[1]), u].mean(axis=1)
        _, p_u = sp.stats.ttest_ind(fr_1, fr_2)
        cohen_d = (fr_2 - fr_1).mean() / (fr_2 - fr_1).std()
        selected.append((p_u < 0.01) & (cohen_d > 0.5))
        # selected.append(p_u < 0.01)
    selected = np.stack(selected)
    return selected


def get_tuning(epoch, data, time_win_calculate, selected_units=None):
    if selected_units is None:
        selected_units = np.ones(data['spk'].shape[2])>0
    spk_realigned = realign(data['trial_info'][epoch], data_spk=data['spk'][:,:,selected_units])
    fr_all = np.nanmean(spk_realigned[:, (ts_plot>=time_win_calculate[0]) & (ts_plot<time_win_calculate[1])], axis=1)
    # fr_all = np.nanmean(spk_realigned[:, (ts_plot>=time_win_calculate[0]) & (ts_plot<time_win_calculate[1])], axis=1) - np.nanmean(spk_realigned[:, (ts_plot>=-300) & (ts_plot<0)], axis=1)

    orientations = np.unique(data['trial_info']['st1_orientation'])
    data_list = []
    for i in range(len(orientations)):
        trials = (data['trial_info']['st1_orientation']==orientations[i]) & (data['trial_info']['st1_sfreq']==4)
        selected_fr = fr_all[trials]
        data_list.append(selected_fr[(np.isnan(selected_fr).sum(axis=1))==0, :])
    return data_list


def get_information(data_list):
    F_all= []
    for u in range(data_list[0].shape[1]):
        F, P = sp.stats.f_oneway(data_list[0][:,u], data_list[1][:,u], data_list[2][:,u], data_list[3][:,u])
        # F, P = sp.stats.f_oneway(np.concatenate([data_i[0][:,u],data_i[2][:,u]]), np.concatenate([data_i[1][:,u],data_i[3][:,u]]))
        # F = np.std(np.concatenate(data_i)[:,u])/np.mean([np.std(data_i[i][:,u]) for i in range(len(data_i))])
        F_all.append(F)
    if len(F_all)>0:
        F_all = np.log(np.stack(F_all))
    return F_all


def decoding(data_list, method='SVM_population'):
    if data_list[0].shape[1] == 0:
        return []
    N_data = [len(x) for x in data_list]
    N_fold = 5
    N_units = data_list[0].shape[1]
    data_matrix = np.zeros([len(data_list), np.min(N_data), N_units])
    labels = np.zeros([len(data_list), np.min(N_data)])
    for i in range(len(data_list)):
        data_matrix[i] = data_list[i][np.random.choice(data_list[i].shape[0], np.min(N_data), replace=False)]
        labels[i] = np.ones(np.min(N_data)) * i
    data_matrix = np.reshape(data_matrix, [len(data_list) * np.min(N_data), N_units])
    labels = np.reshape(labels, [len(data_list) * np.min(N_data)])
    if method == 'SVM_single':
        scores = np.zeros([N_fold, N_units])
        for u in range(N_units):
            clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
            scores[:, u] = cross_val_score(clf, data_matrix[:, u][:, None], labels, cv=5)
        scores = scores.mean(axis=0)
    elif method == 'SVM_population':
        scores = np.zeros(N_fold)
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        scores = cross_val_score(clf, data_matrix, labels, cv=5)
    return scores




F_before, F_after, F_before_fov, F_after_fov, F_before_early, F_before_late, F_before_early_fov, F_before_late_fov = [], [], [], [], [], [], [], []
for i, date in enumerate(data_all.keys()):
    print((i, date))
    data = data_all[date]
    data['trial_info'] = pd.DataFrame.from_dict(data['trial_info'])
    data['spk'] = data['spk'] * 1000

    st1_units = select_visual_unit(realign(data['trial_info']['st1_on'], data['spk']), ts_plot, [-50, 50], [50, 150])
    fovea_units = select_visual_unit(realign(data['trial_info']['st1_acquired'], data['spk']), ts_plot, [-50, 50], [50, 150])

    # F_before = np.concatenate([F_before, get_information(get_tuning('st1_on', data, [50,550], st1_units))])
    # F_after = np.concatenate([F_after, get_information(get_tuning('st2_acquired', data, [50,550], st1_units))])
    # F_before_fov = np.concatenate([F_before_fov, get_information(get_tuning('st1_on', data, [50,550], fovea_units))])
    # F_after_fov = np.concatenate([F_after_fov, get_information(get_tuning('st2_acquired', data, [50,550], fovea_units))])
    # F_before_early = np.concatenate([F_before_early, get_information(get_tuning('st1_on', data, [50,300], st1_units))])
    # F_before_late = np.concatenate([F_before_late, get_information(get_tuning('st1_on', data, [300,550], st1_units))])
    # F_before_early_fov = np.concatenate([F_before_early_fov, get_information(get_tuning('st1_on', data, [50,300], fovea_units))])
    # F_before_late_fov = np.concatenate([F_before_late_fov, get_information(get_tuning('st1_on', data, [300,550], fovea_units))])

    F_before = np.concatenate([F_before, decoding(get_tuning('st1_on', data, [50,550], st1_units))])
    F_after = np.concatenate([F_after, decoding(get_tuning('st2_acquired', data, [50,550], st1_units))])
    F_before_fov = np.concatenate([F_before_fov, decoding(get_tuning('st1_on', data, [50,550], fovea_units))])
    F_after_fov = np.concatenate([F_after_fov, decoding(get_tuning('st2_acquired', data, [50,550], fovea_units))])
    F_before_early = np.concatenate([F_before_early, decoding(get_tuning('st1_on', data, [50,300], st1_units))])
    F_before_late = np.concatenate([F_before_late, decoding(get_tuning('st1_on', data, [300,550], st1_units))])
    F_before_early_fov = np.concatenate([F_before_early_fov, decoding(get_tuning('st1_on', data, [50,300], fovea_units))])
    F_before_late_fov = np.concatenate([F_before_late_fov, decoding(get_tuning('st1_on', data, [300,550], fovea_units))])


_, h_ax = plt.subplots(2, 2, sharex=True, sharey=True)
ax_cur = h_ax[0, 0]
plt.axes(ax_cur)
plt.plot(F_before, F_after, 'r+')
plt.plot(ax_cur.get_xlim(), ax_cur.get_xlim(), 'k--')
ax_cur.set_aspect('equal')
plt.axis('square')
plt.xlabel('F value (Before foveal view)')
plt.ylabel('F value (After foveal view)')
plt.title('Target units')

ax_cur = h_ax[0, 1]
plt.axes(ax_cur)
plt.plot(F_before_early, F_before_late, 'r+')
plt.plot(ax_cur.get_xlim(), ax_cur.get_xlim(), 'k--')
ax_cur.set_aspect('equal')
plt.axis('square')
plt.xlabel('F value (Before foveal view, early)')
plt.ylabel('F value (Before foveal view, late)')
plt.title('Target units')

ax_cur = h_ax[1, 0]
plt.axes(ax_cur)
plt.plot(F_before_fov, F_after_fov, 'r+')
plt.plot(ax_cur.get_xlim(), ax_cur.get_xlim(), 'k--')
ax_cur.set_aspect('equal')
plt.axis('square')
plt.xlabel('F value (Before foveal view)')
plt.ylabel('F value (After foveal view)')
plt.title('Foveal units')

ax_cur = h_ax[1, 1]
plt.axes(ax_cur)
plt.plot(F_before_early_fov, F_before_late_fov, 'r+')
plt.plot(ax_cur.get_xlim(), ax_cur.get_xlim(), 'k--')
ax_cur.set_aspect('equal')
plt.axis('square')
plt.xlabel('F value (Before foveal view, early)')
plt.ylabel('F value (Before foveal view, late)')
plt.title('Foveal units')