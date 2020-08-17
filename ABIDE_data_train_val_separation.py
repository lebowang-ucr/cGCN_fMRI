# Please refer to https://github.com/preprocessed-connectomes-project/abide for ABIDE preprocessed data
# This script is based on raw data with self preprocessing.
# Leave-one-site-out and 10-fold cross-validation are shown.

import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import h5py
from nilearn.signal import clean
import collections
import random
from sklearn.model_selection import StratifiedKFold


# In[2]:


def load_phenotypes(pheno_path):
    pheno = pd.read_csv(pheno_path, encoding = "ISO-8859-1")

    pheno['DX_GROUP'] = pheno['DX_GROUP'].apply(lambda v: int(v)-1)
    pheno['SITE_ID'] = pheno['SITE_ID'].apply(lambda v: re.sub('_[0-9]', '', v))
    pheno['SEX'] = pheno['SEX'].apply(lambda v: {1: "M", 2: "F"}[v])
#     pheno['MEAN_FD'] = pheno['func_mean_fd']
#     pheno['SUB_IN_SMP'] = pheno['SUB_IN_SMP'].apply(lambda v: v == 1)
#     pheno["STRAT"] = pheno[["SITE_ID", "DX_GROUP"]].apply(lambda x: "_".join([str(s) for s in x]), axis=1)

#     pheno.index = pheno['FILE_ID']
    # return pheno[['FILE_ID', 'DX_GROUP', 'SEX', 'SITE_ID', 'MEAN_FD', 'SUB_IN_SMP', 'STRAT']]
#     return pheno[['FILE_ID', 'DX_GROUP', 'SITE_ID']]
    return pheno[['SUB_ID', 'DX_GROUP', 'SITE_ID']]


# In[3]:


TR = {'CALTECH':2, 'CMU':2, 'KKI':2.5, 'LEUVEN':1.67, 'MAX_MUN':3, 'NYU':2, 
      'OHSU':2.5, 'OLIN':1.5, 'PITT':1.5, 'SBL':2.2, 'SDSU':2, 'STANFORD':2, 
     'TRINITY':2, 'UCLA':3, 'UM':2, 'USM':2, 'YALE':2}


# In[4]:


root = '/home/lbwang/abide_dataset_download/data/raw/ABIDE_I/all_rest/'
subjs = np.loadtxt(root+'ids.good', dtype=str)
subjs.sort()
subjs_site = []
# assert len(subjs) == len([f for f in os.listdir(root) if '.afni' in f])
file_path = '/home/lbwang/abide_dataset_download/data/raw/ABIDE_I/all_rest/%s.rest.afni/errts.fanaticor2std3.cc200.timeseries'

pheno_path = '/home/lbwang/abide_dataset_download/data/Phenotypic_ABIDE_I.csv'
pheno = load_phenotypes(pheno_path)
tmp_id = pheno['SUB_ID'].tolist()
tmp_label = pheno['DX_GROUP'].tolist()
tmp_site = pheno['SITE_ID'].tolist()
assert len(tmp_id) == len(tmp_label)
assert len(tmp_id) == len(tmp_site)
dic = collections.defaultdict(dict)
for i, sub in enumerate(tmp_id):
    dic[str(sub)]['site'] = tmp_site[i]
    dic[str(sub)]['label'] = tmp_label[i]

for sub in subjs:
    subjs_site.append(dic[str(sub)[2:]]['site'])
    
    
pheno_subj_id = pheno['SUB_ID'].tolist()
print('total subjects:', len(pheno_subj_id))
pheno_label = np.array(pheno['DX_GROUP'])
assert len(pheno_subj_id) == len(pheno_label)
print('available subjects:', len(subjs))

sites = list(set(pheno['SITE_ID'].tolist()))


# In[96]:


# dataset_name = 'abide_I_leave_one_site_out_self_preprocessing_CC200_norm_demean.h5'
dataset_name = 'abide_I_leave_one_site_out_self_preprocessing_CC200_filter_0.1_0.01_norm.h5'


# In[6]:


frames = 315

FC = np.zeros((len(subjs), 200, 200))
raw_data = np.zeros((len(subjs), frames, 200))

length = {}
label = []
for i, sub in enumerate(subjs):
    print(i, ':', sub)
    tmp = np.loadtxt(file_path%sub)
    length[sub] = tmp.shape[0]
    tmp_len = min(tmp.shape[0], frames)
    
# #     preprocess
# #     tmp_frame = min(frames, tmp.shape[0])
# #     tmp = tmp[:tmp_frame, :]
#     np.nan_to_num(tmp, copy=False, nan=0.0, posinf=0, neginf=0)
#     assert np.sum(np.isnan(tmp)) == 0
    
#     tmp_mean = np.mean(tmp, axis=0, keepdims=True)
#     assert tmp_mean.shape == (1, 200)
#     tmp_std = np.std(tmp, axis=0, keepdims=True)
#     assert tmp_std.shape == (1, 200)
#     tmp_std[tmp_std==0] = np.inf
#     assert np.sum(np.isnan(tmp)) == 0
#     tmp_preprocessing = (tmp - tmp_mean) / tmp_std
#     raw_data[i, :tmp_len, :] = tmp_preprocessing
    
# #     tmp_mean = np.mean(tmp, axis=1, keepdims=True)
# #     tmp_std = np.std(tmp, axis=1, keepdims=True)
# #     tmp_std[tmp_std==0]=1e-10
# #     raw_data[i, :tmp.shape[0], :] = (tmp-tmp_mean)/tmp_std
    
# nilearn  
    tmp_filter = clean(tmp, sessions=None, detrend=False, standardize=True, 
          confounds=None, low_pass=0.1, high_pass=0.01, t_r=TR[dic[sub[2:]]['site']], 
          ensure_finite=True)
#     tmp_filter = clean(tmp, sessions=None, detrend=False, standardize=True, 
#           confounds=None, low_pass=None, high_pass=None, t_r=None, 
#           ensure_finite=True)
    tmp_mean = np.mean(tmp_filter, axis=0, keepdims=True)
    assert tmp_mean.shape == (1, 200)
    tmp_filter_demean = tmp_filter - tmp_mean
    raw_data[i, :tmp_len, :] = tmp_filter_demean[:tmp_len]
    
    FC[i] = np.corrcoef(tmp_filter_demean.T)
#     assert np.sum(np.isnan(FC[i])) == 0
    FC[i, np.isnan(FC[i])] = 0
    assert np.sum(np.isnan(FC[i])) == 0
    label.append(dic[sub[2:]]['label'])
    
label = np.array(label)
assert label.shape[0] == raw_data.shape[0]

print('total subjects:', len(subjs))
print('No. of label 0:', np.sum(label == 0))
print('No. of label 1:', np.sum(label == 1))
assert np.sum(label == 0) + np.sum(label == 1) == label.shape[0]    


# # check one data

# In[8]:


N = raw_data.shape[0]
idx = random.randint(0, N)
print('plotting %d subj with total number of subjects (%d)'%(idx, N))
plt.subplot(1, 2, 1)
plt.plot(raw_data[idx])
plt.subplot(1, 2, 2)
plt.hist(raw_data[idx].reshape(-1, 1))
plt.show()


# # check the distribution of values

# In[9]:


plt.hist(raw_data.reshape(-1, 1))


# # check the distribution of the length of scans

# In[10]:


plt.hist(length.values(), bins=20)


# # output avg_FC

# In[11]:


FC_avg = np.mean(FC, axis=0)
np.save('avg_FC_' + dataset_name[:-3] ,FC_avg)
plt.imshow(FC_avg)


# # output data - leave one site out

# In[ ]:


N_allVal = 0
with h5py.File(dataset_name, 'a') as f:
    for leftSite in sites:
        print('leave site out:', leftSite)
        train_idx = []
        val_idx = []
        for i, subj in enumerate(subjs):
            if dic[subj[2:]]['site'] == leftSite:
                val_idx.append(i)
            else:
                train_idx.append(i)
        assert len(val_idx) + len(train_idx) == len(subjs)
        N_allVal += len(val_idx)
        print('No. of subjs in %s: %d'%(leftSite, len(val_idx)))
        f.create_group(leftSite)    
        x_train, y_train = raw_data[train_idx], label[train_idx]
        x_val, y_val = raw_data[val_idx], label[val_idx]   
        f[leftSite].create_dataset('x_train', data=x_train)
        f[leftSite].create_dataset('y_train', data=y_train) 
        f[leftSite].create_dataset('x_val', data=x_val)
        f[leftSite].create_dataset('y_val', data=y_val)
print('all subjs taken as Val dataset: ', N_allVal)
assert N_allVal == len(subjs)


# In[ ]:


N_allVal = 0
with h5py.File(dataset_name, 'r') as f:
    for leftSite in sites:
        x_train = f[leftSite]['x_train'][()]
        y_train = f[leftSite]['y_train'][()]
        x_val = f[leftSite]['x_val'][()]
        y_val = f[leftSite]['y_val'][()]
        assert x_train.shape[0] == y_train.shape[0]
        assert x_val.shape[0] == y_val.shape[0]
        assert x_train.shape[0] + x_val.shape[0] == len(subjs)
        assert y_train.shape[0] + y_val.shape[0] == len(subjs)
        N_allVal += x_val.shape[0]
print('all subjs taken as Val dataset: ', N_allVal)
assert N_allVal == len(subjs)


# # check if subjs from Val in Training

# In[ ]:


N_check = 0
with h5py.File(dataset_name, 'r') as f:
    for leftSite in sites:
        print('checking site:', leftSite)
        x_train = f[leftSite]['x_train'][()]
        x_val = f[leftSite]['x_val'][()]
        for iVal in range(x_val.shape[0]):
            N_check += 1
            for iTrain in range(x_train.shape[0]):
                assert not np.allclose(x_val[iVal], x_train[iTrain])
assert N_check == len(subjs)                
print('check subjs as val: ', N_check)


# # check data unbalance

# In[ ]:


N_total = 0
N_label1 = 0
with h5py.File(dataset_name, 'r') as f:
    for leftSite in sorted(sites):
        print('site:', leftSite)
        y_val = f[leftSite]['y_val'][()]
        N_total += y_val.shape[0]
        N_label1 += np.sum(y_val)
        print('\ttotal data: %d, label=1: %d'%(y_val.shape[0], np.sum(y_val)))
print('total subjs: %d, total label = 1: %d'%(N_total, N_label1))


# In[ ]:


print('done')


# # output data - 10 fold - stratified based on no. of subjs in each site and the portions of ASD in each site

# In[6]:


dataset_name = 'abide_I_10_fold_self_preprocessing_CC200_filter_0.1_0.01_norm_demean.h5'


# In[14]:


site_idx_label = {site: [] for site in sites}
totol_N = 0
setIdx = set()
for idx in range(len(subjs)):
    site_idx_label[subjs_site[idx]].append([idx, label[idx]])
for site in site_idx_label:
    site_idx_label[site] = np.array(site_idx_label[site])
    totol_N += site_idx_label[site].shape[0]
    setIdx.update(set(site_idx_label[site][:, 0]))
assert totol_N == len(subjs)
assert len(setIdx) == len(subjs)

print('No. of label==1:')
for site in sorted(sites):
    print(site, ':', np.sum(site_idx_label[site], 0)[1])


# In[132]:


fold = 10
site_idx_stratified_k_fold = {site: [] for site in sites}
skf = StratifiedKFold(n_splits=fold, shuffle=True)
for site in sites:
    for i, (_, val_index) in enumerate(skf.split(site_idx_label[site][:, [0]], site_idx_label[site][:, [1]])):
        site_idx_stratified_k_fold[site].append(list(site_idx_label[site][val_index, 0]))
setIdx = set()
for site in sites:
    for i in range(fold):
        setIdx.update(set(site_idx_stratified_k_fold[site][i]))
assert len(setIdx) == len(subjs)

# because the last fold in each site has the smallest, so shuffle each site
for site in sites:
    random.shuffle(site_idx_stratified_k_fold[site])
    
setIdx = set()
for site in sites:
    for i in range(fold):
        setIdx.update(set(site_idx_stratified_k_fold[site][i]))
assert len(setIdx) == len(subjs)

lengthFold = [0] * fold
labelOneFold = [0] * fold

for i in range(fold):
    for site in sites:
        lengthFold[i] += len(site_idx_stratified_k_fold[site][i])
        labelOneFold[i] += sum(label[site_idx_stratified_k_fold[site][i]])
        
unbalance = [0] * fold
for i in range(fold):
    tmp = labelOneFold[i] / lengthFold[i]
    unbalance[i] = max(tmp, 1 - tmp)
    
final_idx_on_raw_data = [[] for _ in range(fold)]
check_labelOneFold = [0] * fold
for fold_i in range(fold):
    for site in sites:
        final_idx_on_raw_data[fold_i] += site_idx_stratified_k_fold[site][fold_i]
    check_labelOneFold[fold_i] = sum(label[final_idx_on_raw_data[fold_i]])
assert check_labelOneFold == labelOneFold

check_total_idx = set()
for fold_i in range(fold):
    check_total_idx.update(set(final_idx_on_raw_data[fold_i]))
assert len(check_total_idx) == len(subjs)
        

plt.figure(figsize=(10, 5))        
fig, ax1 = plt.subplots()
color1 = 'red'
ax1.plot(lengthFold, color=color1)
ax1.set_ylabel('No. of subjs', color=color1)
ax1.set_xlabel('fold')

color2 = 'blue'
ax2 = ax1.twinx()
ax2.plot(labelOneFold, color=color2)
ax2.set_ylabel('No. of label_1', color=color2)

plt.subplots()
plt.plot(unbalance)
plt.xlabel('fold')
plt.ylabel('Unbalance')
plt.show()


# In[133]:


with h5py.File(dataset_name, 'a') as f:
    for fold_i in range(fold):
        f.create_group(str(fold_i))
        f[str(fold_i)].create_dataset('X', data=raw_data[final_idx_on_raw_data[fold_i]])
        f[str(fold_i)].create_dataset('Y', data=label[final_idx_on_raw_data[fold_i]])


# # fold unbalance

# In[7]:


N_fold = [0] * fold
N_label_1 = [0] * fold
with h5py.File(dataset_name, 'r') as f:
    for fold_i in range(fold):
        N_fold[fold_i] += f[str(fold_i)]['X'][()].shape[0]
        N_label_1[fold_i] += sum(f[str(fold_i)]['Y'][()])

unbalance = [0] * fold
for i in range(fold):
    tmp = N_label_1[i] / N_fold[i]
    unbalance[i] = max(tmp, 1 - tmp)
print(unbalance)

plt.figure(figsize=(10, 5))        
fig, ax1 = plt.subplots()
color1 = 'red'
ax1.plot(N_fold, color=color1)
ax1.set_ylabel('No. of subjs', color=color1)
ax1.set_xlabel('fold')

color2 = 'blue'
ax2 = ax1.twinx()
ax2.plot(N_label_1, color=color2)
ax2.set_ylabel('No. of label_1', color=color2)

plt.subplots()
plt.plot(unbalance)
plt.xlabel('fold')
plt.ylabel('Unbalance')
plt.show()

