import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Find shape from patient set 
## 
def get_patient_ROI_shapes(patient_set, ROI_shapes = None) :
    if ROI_shapes is None :
        ROI_shapes = {}
        
    for patient_id, crops_per_organ in patient_set.items():
        for roi_name, data in crops_per_organ.items():
    
            dose = data["DOSE"]
            mask = data["MASK"]
    
            if mask is None:
                continue
    
            if roi_name not in ROI_shapes:
                ROI_shapes[roi_name] = []
    
            ROI_shapes[roi_name].append(mask.shape)
    return ROI_shapes

def round_up_to_multiple(x, multiple=8):
    return int(np.ceil(x / multiple) * multiple)

def find_uniform_shapes(ROI_shapes, multiple = 8) :
    rows = []
    target_shapes_by_organ = {}
    
    for roi_name, shapes in ROI_shapes.items():
    
        shapes = np.asarray(shapes)
    
        min_shape = shapes.min(axis=0)
        max_shape = shapes.max(axis=0)
        median_shape = np.median(shapes, axis=0)
        p95_shape = np.percentile(shapes, 95, axis=0)
    
        target_shape = tuple(
            round_up_to_multiple(v, multiple=multiple)
            for v in max_shape
        )
    
        target_shapes_by_organ[roi_name] = target_shape
    
        rows.append({
            "roi_name": roi_name,
            "n": len(shapes),
    
            "min_shape": tuple(min_shape.astype(int)),
            "median_shape": tuple(np.ceil(median_shape).astype(int)),
            "p95_shape": tuple(np.ceil(p95_shape).astype(int)),
            "max_shape": tuple(max_shape.astype(int)),
    
            "target_shape": target_shape,
        })
    ROI_shape_info = pd.DataFrame(rows).sort_values("roi_name")
    return ROI_shape_info, target_shapes_by_organ



# Applied uniform shapes on ROI
def center_crop_or_pad_3d(arr, target_shape, fill_value=0):
    """
    Place une crop 3D dans une matrice cible.
    Si la crop est plus petite : padding centré.
    Si la crop est plus grande : crop centré.

    Parameters
    ----------
    arr : np.ndarray
        Array 3D de shape (z, y, x).
    target_shape : tuple[int, int, int]
        Shape cible (Z, Y, X).
    fill_value : float
        Valeur de remplissage.

    Returns
    -------
    out : np.ndarray
        Array 3D de shape target_shape.
    """

    arr = np.asarray(arr)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    target_shape = np.asarray(target_shape, dtype=int)
    source_shape = np.asarray(arr.shape, dtype=int)

    out = np.full(
        tuple(target_shape),
        fill_value,
        dtype=arr.dtype,
    )

    copy_shape = np.minimum(source_shape, target_shape)

    src_start = ((source_shape - copy_shape) // 2).astype(int)
    src_end = src_start + copy_shape

    dst_start = ((target_shape - copy_shape) // 2).astype(int)
    dst_end = dst_start + copy_shape

    out[
        dst_start[0]:dst_end[0],
        dst_start[1]:dst_end[1],
        dst_start[2]:dst_end[2],
    ] = arr[
        src_start[0]:src_end[0],
        src_start[1]:src_end[1],
        src_start[2]:src_end[2],
    ]

    return out

def build_tensors_by_organ(
    patient_set,
    target_shapes_by_organ,
    *,
    normalize_dose=True,
    dose_norm_mode="local_max",
    mask_threshold=0.5,
):
    """
    Convertit patient_set en tenseurs par organe.

    Parameters
    ----------
    patient_set : dict
        Dictionnaire patient_set[patient_id][roi_name]["DOSE" / "MASK"].
    target_shapes_by_organ : dict
        Dictionnaire target_shapes_by_organ[roi_name] = (Z, Y, X).
    normalize_dose : bool
        Si True, normalise la dose.
    dose_norm_mode : str
        "local_max" pour diviser chaque crop par son maximum.
        "none" pour ne pas normaliser.
    mask_threshold : float
        Seuil pour rebinariser les masques après placement dans la matrice cible.

    Returns
    -------
    tensors_by_organ : dict
        tensors_by_organ[roi_name] = np.ndarray, shape (N, 2, Z, Y, X).
    metadata_by_organ : dict
        metadata_by_organ[roi_name] = pd.DataFrame avec patient_id, shapes, etc.
    """

    samples_by_organ = {}
    metadata_rows_by_organ = {}

    for patient_id, crops_per_organ in patient_set.items():

        for roi_name, data in crops_per_organ.items():

            if roi_name not in target_shapes_by_organ:
                # print(f"[SKIP] {patient_id} - {roi_name}: no target shape")
                continue

            dose = np.asarray(data["DOSE"], dtype=np.float32)
            mask = np.asarray(data["MASK"], dtype=np.float32)

            if dose.shape != mask.shape:
                print(
                    f"[SKIP] {patient_id} - {roi_name}: "
                    f"dose shape {dose.shape} != mask shape {mask.shape}"
                )
                continue

            if not np.any(mask):
                print(f"[SKIP] {patient_id} - {roi_name}: empty mask")
                continue

            original_shape = dose.shape
            target_shape = target_shapes_by_organ[roi_name]

            dose = np.nan_to_num(
                dose,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            if normalize_dose:

                if dose_norm_mode == "local_max":
                    max_dose = float(np.max(dose))
                    if max_dose > 0:
                        dose = dose / max_dose

                elif dose_norm_mode is None:
                    pass

                else:
                    raise ValueError(
                        "Unknown dose_norm_mode. "
                        "Use 'local_max' or 'none'."
                    )

            dose_fixed = center_crop_or_pad_3d(
                dose,
                target_shape=target_shape,
                fill_value=0.0,
            )

            mask_fixed = center_crop_or_pad_3d(
                mask,
                target_shape=target_shape,
                fill_value=0.0,
            )

            mask_fixed = (mask_fixed > mask_threshold).astype(np.float32)

            x = np.stack(
                [
                    dose_fixed.astype(np.float32),
                    mask_fixed.astype(np.float32),
                ],
                axis=0,
            )

            if roi_name not in samples_by_organ:
                samples_by_organ[roi_name] = []
                metadata_rows_by_organ[roi_name] = []

            samples_by_organ[roi_name].append(x)

            metadata_rows_by_organ[roi_name].append({
                "patient_id": patient_id,
                "roi_name": roi_name,
                "original_shape": original_shape,
                "target_shape": target_shape,
                "mask_voxels_original": int(np.sum(mask > mask_threshold)),
                "mask_voxels_fixed": int(np.sum(mask_fixed > mask_threshold)),
                "dose_min_fixed": float(np.min(dose_fixed)),
                "dose_max_fixed": float(np.max(dose_fixed)),
                "dose_mean_in_mask_fixed": float(
                    np.mean(dose_fixed[mask_fixed > mask_threshold])
                ),
            })

    tensors_by_organ = {}
    metadata_by_organ = {}

    for roi_name, samples in samples_by_organ.items():

        tensors_by_organ[roi_name] = np.stack(samples, axis=0).astype(np.float32)
        metadata_by_organ[roi_name] = pd.DataFrame(metadata_rows_by_organ[roi_name])

    return tensors_by_organ, metadata_by_organ