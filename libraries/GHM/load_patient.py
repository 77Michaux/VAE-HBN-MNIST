import os
import sys
sys.path.insert(1, r"..\idh_rtgeom\src")

import numpy as np
from pydicom import dcmread
from idh_rtgeom import *

# Identify which patients have all structures 
def search_patient_valid(INDEXER, SOI, STANDARDIZER) :
    patients_ok = []
    for i, patient in enumerate(INDEXER.Patients_sorted):
        study = patient.Studies_sorted[0]
        plan = study.Plans_sorted[0]
        rtstruct_paths = plan.rtstruct_selected.paths or plan.rtstruct_candidates.paths
        rtstruct_raw = load_rtstruct(rtstruct_paths)
        rtstruct, report = rtstruct_raw.standardized(
            STANDARDIZER,
            return_report=True,
            )
        found_struct = {
            elem.name
            for elem in rtstruct.list_rois()
            if elem.name in SOI
        }
        missing = SOI - found_struct
        if not missing:
            print(f"Patient {i+1}/{len(INDEXER.Patients_sorted)} Tout y est")
            patients_ok.append(i)
        else:
            print(f"Patient {i+1}/{len(INDEXER.Patients_sorted)} Structures manquantes : {missing}")
    print("Indices des patients complets :", patients_ok)
    print(f"{len(patients_ok)} patients sur {len(INDEXER.Patients_sorted)}")
    
    return patients_ok

# Identify which patients have all structures  
def search_patient_with_missing_ROI(INDEXER, SOI, STANDARDIZER, index = []) :
    patients_ok = []
    for i, patient in enumerate(INDEXER.Patients_sorted):
        if i in index :
            study = patient.Studies_sorted[0]
            plan = study.Plans_sorted[0]
            rtstruct_paths = plan.rtstruct_selected.paths or plan.rtstruct_candidates.paths
            rtstruct_raw = load_rtstruct(rtstruct_paths)
            rtstruct, report = rtstruct_raw.standardized(
                STANDARDIZER,
                return_report=True,
                )
            found_struct = {
                elem.name
                for elem in rtstruct.list_rois()
                if elem.name in SOI
            }
            missing = SOI - found_struct
            if missing:
                print(f"Patient {i+1}/{len(INDEXER.Patients_sorted)} Structures manquantes : {missing}")
                patients_ok.append(i)
            else:
                print(f"Patient {i+1}/{len(INDEXER.Patients_sorted)} Tout y est")
    print("Indices des patients complets :", patients_ok)
    print(f"{len(patients_ok)} patients sur {len(INDEXER.Patients_sorted)}")
    
    return patients_ok


# Getting patient set from INDEXER, SOI and STANDARDIZER giving an INTERP_DOSE  
def load_patient_from_idx(INDEXER, patient_index, SOI, STANDARDIZER, INTERP_DOSE, NEW_SPACING) :
    patient_set = {}
    for k, i in enumerate(patient_index): # patient_index[batchrange[0]:batchrange[1]]
        print(f"Patient {k + 1 }/{len(patient_index)}...")
    
        patient = INDEXER.Patients_sorted[i]
    
        study = patient.Studies_sorted[0]
        plan = study.Plans_sorted[0]
    
        rtstruct_paths = plan.rtstruct_selected.paths or plan.rtstruct_candidates.paths
        rtdose_paths = [
        p for p in plan.rtdose.paths
        if str(
            getattr(dcmread(p, stop_before_pixels=True), "DoseSummationType", "")
        ).upper() == "PLAN"
        ]
    
        dose_raw = load_dose(rtdose_paths)
        dose = dose_raw.resample(new_spacing=NEW_SPACING)
        if INTERP_DOSE != True:
            dose = dose_raw
        rtstruct_raw = load_rtstruct(rtstruct_paths)
    
        rtstruct, report = rtstruct_raw.standardized(
            STANDARDIZER,
            return_report=True,
            )
        crops_per_organ = {}
        for roi_name in SOI:
            roi_crop = roi_extract_box(
                rtstruct.get_roi(roi_name),
                img=dose,
                pad_vox=0,
                masked_fill=np.nan,
            )
    
            crops_per_organ[roi_name] = {
                "DOSE": roi_crop.image_zyx,
                "MASK": roi_crop.mask_zyx,
            }
        patient_set[patient.patient_id] = crops_per_organ
    return patient_set

def get_batch(X, VOX, b, batchsize = 20, shape = None) :
    batch_1, batch_2 = batchsize *b, batchsize *(b+1) 
    
    VOX_IDX = np.where((VOX[0] >= batch_1) & (VOX[0] < batch_2))
    BATCH_IDX = VOX[:, VOX_IDX[0]]
    X_IDX = BATCH_IDX.copy()
    BATCH_IDX[0] = BATCH_IDX[0] -batch_1
    cur_batchsize = X_IDX[0].max() +1 -batch_1
    
    X_batch = np.zeros((cur_batchsize, 2, ) +tuple(shape), dtype = np.float32)
    X_batch[:, 0][tuple(BATCH_IDX)] = X[tuple(X_IDX)].astype(np.float32) /2**16 
    X_batch[:, 1][tuple(BATCH_IDX)] = 1.
    return X_batch


def get_batch_paroi(OARs, b, batchsize = 20, shape = None, mode = "batch") :
    ROI_OAR, ROI_PAR = OARs.keys()
    X_batch_1 = get_batch(OARs[ROI_OAR]["DOSE"], OARs[ROI_OAR]["VOX"], b, batchsize, shape)
    X_batch_2 = get_batch(OARs[ROI_PAR]["DOSE"], OARs[ROI_PAR]["VOX"], b, batchsize, shape)
    
    if mode == "0_3" :
        return np.concatenate([X_batch_1[:, :1], X_batch_2[:, 1:]], axis = 1)

    if mode == "split" :
        return np.concatenate([X_batch_1, X_batch_2], axis = 0)
        
    if mode == "all_split" :
        return np.concatenate([X_batch_1[:, :1], X_batch_1[:, 1:], X_batch_2[:, :1], X_batch_2[:, 1:]], axis = 0)
        
    return np.concatenate([X_batch_1, X_batch_2], axis = 1)



