import numpy as np
import pandas as pd

'''
We have a phenotypic spreadsheet with multiple visit dates per participant. We also have a spreadsheet with multiple
tau pet values for (some) participants. The goal is to find each participant's baseline visit data from the phenotypic
file, and then find a tau pet value that was collected within one year.
'''

# load baseline phenotypic data
pheno = pd.read_csv('/Users/natashaclarke/Documents/SIMEXP/cwas_neuro_sandbox/adni_cwas/adni_spreadsheet.csv', index_col=0, header=0)

# keep only the variables of interest
pheno = pheno.filter(['Subject ID','Phase','Sex','Research Group', 'Visit','Study Date','Age'], axis=1)

# convert 'study date' to 'session' in datetime format, to match other spreadsheets
pheno['session'] = pd.to_datetime(pheno['Study Date'])

# pull out only the subject id and asign it to the index
pheno_subj = []
for i in pheno['Subject ID']:
    subj = i.split('_')[2].lstrip("0") # remove leading zeros since it won't match ADNI IDs
    pheno_subj.append(subj)
    
pheno.index = pheno_subj
pheno.rename_axis('RID',inplace=True)
pheno.index = pheno.index.astype('int64')

# separate patients and controls, because (in theory) we can use any control visit as baseline, but
# for patients we want their actual baseline data
ps = ['AD', 'EMCI', 'LMCI', 'MCI', 'SMC']
pdf = pheno[pheno['Research Group'].isin(ps)] # df of patient diagnoses

cdf = pheno.loc[pheno['Research Group'] == 'CN'] # df of control diagnoses

# I think these visits are acceptable as baseline data, i.e. actual baseline +/-3 months, excluding
# any initial visits where patient continued from a previous phase
p_visits = ['ADNI Screening','ADNI2 Month 3 MRI-New Pt', 'ADNI2 Screening MRI-New Pt', 
                   'ADNIGO Month 3 MRI','ADNIGO Screening MRI']

pdf_bl = pdf[pdf['Visit'].isin(p_visits)]

# rejoin the patients to the controls
new_df = pd.concat([cdf,pdf_bl])

# select the earliest visit available for each participant
new_df.sort_values(['Subject ID', 'Age'], inplace=True) # sort by age
baseline = new_df[~new_df.duplicated(['Subject ID'], keep='first')] # select the first row

# calculate window for acceptable biomarker data, currently +- 12months. This throws an error but I'm
# not sure of an alternative?
baseline['date_lower'] = baseline['session'] - pd.DateOffset(months=12)
baseline['date_upper'] = baseline['session'] + pd.DateOffset(months=12)

# load flortaucipir spreadsheet
tau_df = pd.read_csv('/Users/natashaclarke/Documents/SIMEXP/cwas_neuro_sandbox/adni_cwas/UCBERKELEYAV1451_04_26_22.csv', index_col=0, header=0, low_memory=False)

# keep only relevant columns
tau_df = tau_df.filter(['EXAMDATE','META_TEMPORAL_SUVR'], axis=1)

# convert to datetime
tau_df['EXAMDATE'] = pd.to_datetime(tau_df['EXAMDATE'])

# sort both dfs by index, and tau df also by date. Also throws an error
baseline.sort_values(by='RID', inplace=True)
tau_df.sort_values(by=['RID', 'EXAMDATE'],inplace=True)

# group subjects in the tau df, look them up in the baseline df, and if they match return the tau
# value closest to the session date in baseline. Return a list of dfs, one per subject. This seems
# a silly and expensive way to do this! But, I ran into all sorts of problems with the datetime format
# and this is the only way I figured it out...
dfs = []
for i, grp in tau_df.groupby(level='RID'):
    for subject, session in zip(baseline.index, baseline.session):
        if i == subject:
            df = grp
            df.set_index('EXAMDATE', inplace=True)
            tau = pd.DataFrame(df['META_TEMPORAL_SUVR'][df.index[[df.index.get_loc(session, method='nearest')]]])
            tau['EXAMDATE'] = tau.index
            tau.index = [subject]
            tau.rename_axis('RID',inplace=True)
            dfs.append(tau)

# concatenate individual tau dfs
master_tau = pd.concat(dfs)

# merge tau data into baseline df
baseline = baseline.join(master_tau)

# create a new df with only tau values that were collected within a 12month window of the baseline visit
baseline_tau = baseline[(baseline.EXAMDATE > baseline.date_lower) & (baseline.EXAMDATE < baseline.date_upper)]

# save df
baseline_tau.to_csv('/Users/natashaclarke/Documents/SIMEXP/cwas_neuro_sandbox/adni_cwas/baseline_tau.csv')