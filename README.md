# Connectome-wide association of resting-state functional connectivity and clinical diagnoses in heterogeneous neuropsychiatric and neurodegenerative conditions

Description to be added.

## Scripts and notebooks

`afc_from_niak.ipynb`: Notebook to calculate the Average Functional Connectivity (AFC) for the diagonal of a connectome. Adapted from the NIAK, for Python. Under development.

The following are used to find baseline only data for ADNI participants and strtify patients and controls according to biomarker test results.

`calculate_meta_roi.ipynb`: Calculate the 'meta ROI', which is the average tau uptake on PET (AV-1451) across 6 temporal regions, based on Jack et al. (2017). This is opposed to the ADNI central calculation 'META_TEMPORAL_ROI', which only includes 5 regions, excluding the parahippocampal gyrus.

`baseline_biomarkers.py`: Load phenotypic ADNI data, return the baseline visit data only, along with a 12 month window for acceptable biomarker test results. Currently returns the earliest visit available for controls even if it is not their technical baseline, since we do not have all baseline data. May change this to only baseline, but would lose some participants. To do: replace use of META_TEMPORAL_ROI with result from calculate_meta_roi.ipynb; tidy up functions; try with a 1 year ± 6 months biomarker test window, which is acceptable.

`biomarker_thresholding.ipynb`: Notebook to stratify patients and controls according to biomarker results within the 12 month window from their visit. Returns patients positive on any one of five tests, and controls negative on all five. Participants with no biomarker results (on any tests) are excluded. CSF (Aβ42 and Ptau) and PET (FBB and AV-45) thresholds have been determined according to the literature, ADNI team and discussion with other ADNI researchers. The tau PET threshold is calculated as study specific. All results within a 5% margin are excluded. To do: tidy up functions.
