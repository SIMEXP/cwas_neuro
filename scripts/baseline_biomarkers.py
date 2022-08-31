import pandas as pd
import numpy as np

from functools import reduce
from pathlib import Path

# paths to files

adni_data = Path("__file__").resolve().parents[1] / 'data' / 'adni_spreadsheet.csv'
tau_data = Path("__file__").resolve().parents[1] / 'data' / 'UCBERKELEYAV1451_04_26_22.csv'
other_biomarker_data = Path("__file__").resolve().parents[1] / 'data' / 'ADNIMERGE.csv'

# functions

def get_baselines(file):
    
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
    baseline_df.loc[:,('date_lower')] = baseline_df.loc[:,('session')] - pd.DateOffset(months=18)
    baseline_df.loc[:,('date_upper')] = baseline_df.loc[:,('session')] + pd.DateOffset(months=18)

    return (baseline_df)

def load_biomarker_df(biomarker_data):

    # load data
    biomarker_df = pd.read_csv(biomarker_data, index_col=0, header=0, low_memory=False)

    # convert examdate to datetime
    biomarker_df['EXAMDATE'] = pd.to_datetime(biomarker_df['EXAMDATE'])

    # sort df by index and date
    biomarker_df.sort_values(by=['RID', 'EXAMDATE'],inplace=True)

    # create column from index (useful for later functions)
    biomarker_df['RID'] = biomarker_df.index
    
    return (biomarker_df)

def match_baselines_biomarker(biomarker_df, baseline_df, biomarker):
    
    df = pd.DataFrame(columns=['RID',biomarker,biomarker+'_EXAMDATE']) #create df
    common_ids = biomarker_df.index.intersection(baseline_df.index) #find ids common to the biomarker and baseline dfs
    biomarker_df = biomarker_df.set_index('EXAMDATE') #reindex, needed to use 'nearest' method

    for rid in common_ids:
        participant_df = biomarker_df[(biomarker_df['RID'] == rid)] #create df of all participants results
        participant_df = participant_df.dropna(subset=[biomarker]) #drop NaNs, since we only want nearest date with result available
        
        baseline = baseline_df.loc[rid] #create df of participants baseline data
        session = baseline['session'] #participant's baseline date

        if participant_df.empty:
            pass
        else:
            idx_nearest = participant_df.index.get_loc(session, method='nearest') #find the index of the closest test date to session
            nearest_date = participant_df.index[idx_nearest] #which date is it
            nearest_result = participant_df[biomarker][idx_nearest] #find the biomarker result associated with closest date

            df.loc[len(df)] = [rid,nearest_result,nearest_date] #add to df
    
    df = df.set_index('RID')        
    
    return (df)

def check_visit_window(biomarker_df, baseline_df, biomarker):
    
    '''
    Join closest biomarkers to baseline info, check if the result was collected on a date within the baseline
    window, and if not replace with NaN. Drop unwanted columns and return biomarker info again, ready to merge.
    '''
    
    # create new df, merging biomarker data into baseline df
    baseline_bio = baseline_df.join(biomarker_df)
    
    # create mask of date range, between lower and upper acceptable dates
    mask_date_range = (baseline_bio[biomarker+'_EXAMDATE'] > baseline_bio['date_lower']) & (baseline_bio[biomarker+'_EXAMDATE'] < baseline_bio['date_upper'])
    
    # fill values collected outside date range with NaN
    baseline_bio[biomarker][~mask_date_range] = np.nan
    
    cols_to_drop = ['Subject ID',
     'Phase',
     'Sex',
     'Research Group',
     'Visit',
     'Study Date',
     'Age',
     'session',
     'date_lower',
     'date_upper']
    
    baseline_bio = baseline_bio.drop(cols_to_drop, axis=1) #drop unwanted columns
    
    return (baseline_bio)

def get_biomarker(biomarker_df, baseline_df, biomarker):

    #runs two functions to match the baseline to closest biomarker result, and then checks if it falls in the window
    
    find_nearest_biomarker = match_baselines_biomarker(biomarker_df, baseline_df, biomarker)
    window_checked = check_visit_window(find_nearest_biomarker, baseline_df, biomarker)
    
    return (window_checked)

if __name__ == '__main__':
    
    #get baselines from adni spreadsheet
    baseline_df = get_baselines(adni_data)

    # load biomarker files
    tau_df = load_biomarker_df(tau_data)
    other_biomarkers_df = load_biomarker_df(other_biomarker_data)
    
    # create one df per biomarker with closest result within a one year window of participant baseline
    tau = get_biomarker(tau_df, baseline_df, 'META_TEMPORAL_SUVR')
    abeta = get_biomarker(other_biomarkers_df, baseline_df, 'ABETA')
    ptau = get_biomarker(other_biomarkers_df, baseline_df, 'PTAU')
    av45 = get_biomarker(other_biomarkers_df, baseline_df, 'AV45')
    fbb = get_biomarker(other_biomarkers_df, baseline_df, 'FBB')

    # create list of dataframes (baseline data and all individual biomarkers)
    data_frames = [baseline_df, tau, abeta, ptau, av45, fbb]

    # merge dataframes
    master_biomarkers = reduce(lambda left, right:
             pd.merge_asof(left, right, left_index=True, right_index=True),
             data_frames)
    
    # save baseline and matched biomarker data in data directory
    master_biomarkers.to_csv(Path("__file__").resolve().parents[1] / 'data' / 'master_biomarkers.csv')
