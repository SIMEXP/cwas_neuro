import pandas as pd
import numpy as np

from pathlib import Path

# load data
adni_data = Path(__file__).parent.parent / 'data' / 'adni_spreadsheet.csv'
tau_data = Path(__file__).parent.parent / 'data' / 'UCBERKELEYAV1451_04_26_22.csv'

# functions
def get_baselines(file):

    ''' Load multiple visits for adni participants and return a df with only baselines '''

    # load baseline phenotypic data
    pheno = pd.read_csv(file, index_col=0, header=0)

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
    patient_diagnoses = ['AD', 'EMCI', 'LMCI', 'MCI', 'SMC']
    patient_df = pheno[pheno['Research Group'].isin(patient_diagnoses)] # df of patient diagnoses

    control_df = pheno.loc[pheno['Research Group'] == 'CN'] # df of control diagnoses

    # I think these visits are acceptable as baseline data, i.e. actual baseline +/-3 months, excluding
    # any initial visits where patient continued from a previous phase
    bl_visits = ['ADNI Screening','ADNI2 Month 3 MRI-New Pt', 'ADNI2 Screening MRI-New Pt', 
                   'ADNIGO Month 3 MRI','ADNIGO Screening MRI']

    patient_df_bl = patient_df[patient_df['Visit'].isin(bl_visits)]
    
    # rejoin the patients to the controls
    new_df = pd.concat([control_df,patient_df_bl])
    
    # select the earliest visit available for each participant
    new_df.sort_values(['Subject ID', 'Age'], inplace=True) # sort by age
    baseline_df = new_df[~new_df.duplicated(['Subject ID'], keep='first')] # select the first row
    
    # sort df by index
    baseline_df.sort_values(by='RID', inplace=True)
    
    # calculate window for acceptable biomarker data, currently +- 12months
    baseline_df.loc[:,('date_lower')] = baseline_df.loc[:,('session')] - pd.DateOffset(months=12)
    baseline_df.loc[:,('date_upper')] = baseline_df.loc[:,('session')] + pd.DateOffset(months=12)

    return (baseline_df)

def get_tau(file):

    ''' Load tau data and return df '''

    # load flortaucipir spreadsheet
    tau_df = pd.read_csv(file, index_col=0, header=0, low_memory=False)

    # keep only relevant columns
    tau_df = tau_df.filter(['EXAMDATE','META_TEMPORAL_SUVR'], axis=1)

    # convert to datetime
    tau_df['EXAMDATE'] = pd.to_datetime(tau_df['EXAMDATE'])

    # sort tau df by index and date. Also throws an error
    tau_df.sort_values(by=['RID', 'EXAMDATE'],inplace=True)
    
    return (tau_df)

def match_baseline_tau(baseline_df, tau_df):
    
    '''
    Group subjects in the tau df, look them up in the baseline df, and if they match return the tau
    value closest to the session date in baseline. Return a list of dfs, one per subject, merge into
    baseline df, and then keep only those within a 12 month window. This seems a silly and expensive way 
    to do this! But, I ran into all sorts of problems with the datetime format and this is the only way 
    I figured it out...
    '''
    
    tau_dfs_list = []
    for tau_id, group in tau_df.groupby(level='RID'):
        for baseline_id, session in zip(baseline_df.index, baseline_df.session):
            if tau_id == baseline_id:
                participant_df = group
                participant_df.set_index('EXAMDATE', inplace=True)
                participant_tau = pd.DataFrame(participant_df['META_TEMPORAL_SUVR'][participant_df.index[[participant_df.index.get_loc(session, method='nearest')]]])
                participant_tau['EXAMDATE'] = participant_tau.index
                participant_tau.index = [baseline_id]
                participant_tau.rename_axis('RID',inplace=True)
                tau_dfs_list.append(participant_tau)
                
    # concatenate individual tau dfs
    master_tau = pd.concat(tau_dfs_list)
    
    # merge tau data into baseline df
    baseline_df = baseline_df.join(master_tau)
    
    # create a new df with only tau values that were collected within a 12month window of the baseline visit
    baseline_tau = baseline_df[(baseline_df.EXAMDATE > baseline_df.date_lower) & (baseline_df.EXAMDATE < baseline_df.date_upper)]
                
    return (baseline_tau)

# run functions
baseline_df = get_baselines(adni_data)
tau_df = get_tau(tau_data)
baseline_tau = match_baseline_tau(baseline_df, tau_df)

# save baseline and matched tau data in data directory
baseline_tau.to_csv(Path("__file__").parent.parent / 'data' / 'baseline_tau.csv')