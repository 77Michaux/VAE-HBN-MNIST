import torch
import json
import src.dataprep as dp
import numpy as np
import nibabel as nib
import torch.nn.functional as F

from time import monotonic_ns
from pydicom import dcmread, Dataset
from pathlib import Path
from tqdm import tqdm
from torchvision.transforms import Pad


def is_axial(dicom):
    """
    Check if patient is axial or not.

    Parameters
    ----------
    dicom_path: Path
        Specific dicom path.

    Returns
    -------
    Boolean
        Axial should be equal to 2, otherwise 0.
    """
    axes = dicom[0x0020, 0x0037]  #ImageOrientationPatient
    return sum(axes) == 2


def get_dicom_paths(
        root_dir: str | Path
) -> list:
    """
    Get all dicom paths from a cohort of interest.
    Parameters
    ----------
    root_dir: Path
    check_axial: bool

    Returns
    -------
    patients_dicom_list: iterable
        List of dicom paths.
    """
    cpt_non_axial = 0
    cpt_discarded = 0
    #logger.info("Importing all dicoms paths.")
    root_dir = Path(root_dir)
    patients_list = list(root_dir.iterdir())
    if Path(root_dir, "LICENSE") in patients_list:
        patients_list.remove(Path(root_dir, "LICENSE"))
    patients_list.sort()
    patients_dicom_list = []
    patients_progress = tqdm(patients_list,
                             ncols=70,
                             dynamic_ncols=True,
                             colour="BLUE")
    for patient in patients_progress:
        patient = Path(patient)
        dates_list = list(patient.iterdir())
        dates_list.sort()
        for date in dates_list:
            date = Path(patient, date)
            gated_list = list(date.iterdir())
            gated_list.sort()
            for gated in gated_list:
                gated = Path(patient, date, gated)
                dicoms_list = list(gated.iterdir())
                dicoms_list.sort()
                if len(dicoms_list) > 1:
                    patients_dicom_list.append(dicoms_list)
                else:
                    cpt_discarded += 1
    # logger.info(f"{cpt_discarded} DICOM sequences of length 1 were discarded.")
    # logger.info(f"{cpt_non_axial} non-axial DICOM sequences were then discarded.")
    return patients_dicom_list


def isCT(ds,
         tag_to_check: tuple,
         tag_ct: str,
         tag_cbct: str,
         dicom: str):
    try :
        tag_value = ds[tag_to_check].value
    except KeyError:
        print(dicom)
        return None

    if tag_value == tag_ct:
        return True
    elif tag_value == tag_cbct:
        return False
    else:
        return None


def check_size(ds,
               th_size: int):
    if ds[(0x0028, 0x0010)].value == th_size and ds[(0x0028, 0x0011)].value == th_size :
        return True
    else :
        return False


def NiiPadding(img, H=512, W=512):
    Wpad = int((W - img.shape[0]) / 2)
    Hpad = int((H - img.shape[1]) / 2)

    if Wpad > 0 and Hpad > 0:
        img = img.permute(2, 0, 1)
        img = Pad((Wpad, Hpad), fill=0, padding_mode='constant')(img).permute(1, 2, 0)

    return img

def nii_normalizing(img, p=2, dim=(0, 1)):
    return F.normalize(img, p, dim)


def loadAndSaveNII(data_out: Path,
                   patients_list:list,
                   data_name: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for serie, folders in enumerate(patients_list):
        label = str(folders[0].parts[-2]).upper()
        folder_progress = tqdm(folders, desc=label, ncols=70, colour='GREEN', dynamic_ncols=True, leave=True,
                               mininterval=.01)
        for pat, nii in enumerate(folder_progress):
            pat_name = nii.stem.split('.')[0]

            # load nii
            nii_img = nib.load(nii)
            nii_data = nii_img.get_fdata()

            # To tensor
            tensor_nii = torch.as_tensor(nii_data, device=device, dtype=torch.float32)

            # Padding
            tensor_nii = nii_normalizing(NiiPadding(tensor_nii, H=512, W=512), p=2, dim=(0, 1))

            # Save
            W, H, N = tensor_nii.shape
            tensor_nii = tensor_nii.detach().cpu()
            save_progress = tqdm(range(N), leave=False, ncols=70, colour='GREEN', dynamic_ncols=True, mininterval=.01)
            for s in save_progress:
                slice_nii = tensor_nii[:, :, s].unsqueeze(0)
                torch.save(slice_nii,
                           Path(data_out, label, f'{data_name}__{pat_name}__{s + 1}.pt'))


def loadAndSaveDCM(data_out: Path,
                   dicom_paths: list,
                   patient_list: list,
                   tag_to_check: tuple,
                   tag_ct: str,
                   tag_cbct: str,
                   theoretical_size: int,
                   data_name: str):
    img_progress = tqdm(dicom_paths,
                        ncols=70,
                        dynamic_ncols=True,
                        leave=True,
                        colour="BLUE",
                        desc="Source")
    dcm_files = []
    pt_files = []

    for serie, folder in enumerate(img_progress):
        pat_name = [p for p in patient_list if p in str(folder)][0]
        folder_progress = tqdm(folder, desc="Folder", ncols=70, colour="GREEN", dynamic_ncols=True, leave=False,
                               mininterval=.01)
        for d_slice, dicom in enumerate(folder_progress):
            # load dicom
            ds = dcmread(dicom, force=True)

            # check if axial
            if not is_axial(ds) or not check_size(ds, theoretical_size):
                continue

            # check if ct
            i = 1
            dicom_label = isCT(ds, tag_to_check[0], tag_ct, tag_cbct, dicom)
            while dicom_label is None and i < len(tag_to_check):
                dicom_label = isCT(ds, tag_to_check[i], tag_ct, tag_cbct, dicom)
                i += 1

            if dicom_label is None:
                continue

            # get pixel
            pixel_data = dp.get_pixels(ds, normalizing=True, remove_padding=True)

            # skip image if all pixels are black
            if len(np.unique(pixel_data)) == 1:
                continue

            # On renvoie un tenseur Float32 [1, H, W]
            img_tensor = torch.from_numpy(pixel_data).float().unsqueeze(0)

            # save
            dcm_name = str(data_name + "__" + pat_name + "__" + str(serie) + "__" + str(d_slice + 1))
            if dicom_label:
                torch.save(img_tensor,
                           Path(data_out, "CT", str(dcm_name + '.pt')))
            else:
                torch.save(img_tensor,
                           Path(data_out, "CBCT", str(dcm_name + '.pt')))

            dcm_files.append(str(dicom))
            pt_files.append(dcm_name)

    with open(Path(data_out, str(data_name + '_info.json')), 'w') as fp:
        json.dump(dict(zip(pt_files, dcm_files)),
                  fp)


def preprocess(
        data_dir: Path,
        data_out: Path,
        data_name: str = "4DLung"
):
    data_dir = Path(data_dir)
    data_out = Path(data_out)
    Path(data_out, "CT").mkdir(parents=True, exist_ok=True)
    Path(data_out, "CBCT").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    match data_name:
        case "all":
            preprocess(data_dir, data_out)
            preprocess(data_dir, data_out, data_name="Pancreas")
            preprocess(data_dir, data_out, data_name="HeadNeck")
        case "4DLung":
            print("Loading 4D-Lung data...")
            data_in = Path(data_dir, "manifest-ObLxS9Wd1073675925233948759", "4D-Lung")
            dicom_paths = get_dicom_paths(data_in)
            patient_list = [p.name for p in data_in.iterdir() if p.name != 'LICENSE']
            tag_to_check = ((0x0008, 0x0070),)
            tag_cbct = "Varian Imaging Laboratories, Switzerland"
            tag_ct = "ADAC"
            theoretical_size = 512
            loadAndSaveDCM(data_out, dicom_paths, patient_list,
                           tag_to_check, tag_ct, tag_cbct,
                           theoretical_size,
                           data_name)

        case "HeadNeck":
            print("Loading Head and Neck data...")
            data_in = Path(data_dir, "Head-Neck-CBCT-CT", "Head-Neck-CBCT-CT")
            nii_paths = list(data_in.iterdir())  # CT and CBCT folders
            patients_list = [list(p.iterdir()) for p in nii_paths]  # list of 2 lists (patients in ct folder and patients in cbct folder)
            loadAndSaveNII(data_out, patients_list, data_name)

        case "Pancreas":
            print("Loading Pancreas data...")
            data_in = Path(data_dir, "Pancreatic-CT-CBCT-SEG", "pancreatic_ct_cbct_seg")
            dicom_paths = get_dicom_paths(data_in)
            patient_list = [p.name for p in data_in.iterdir()]
            tag_to_check = ((0x0008, 0x1090), (0x0008, 0x103E))
            tag_cbct = "Patient Verification"
            tag_ct = "PANCREAS DI, iDose (3)"
            theoretical_size = 512
            loadAndSaveDCM(data_out, dicom_paths, patient_list,
                           tag_to_check, tag_ct, tag_cbct,
                           theoretical_size,
                           data_name)
        case _:
            raise Exception("Unknown dataset")
