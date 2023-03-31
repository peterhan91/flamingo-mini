import os
import pandas as pd
import argparse
import glob
from tqdm import tqdm
from PIL import Image
import h5py
import cv2
import numpy as np
from typing import *
from pathlib import Path


def load_data(filepath):
    dataframe = pd.read_csv(filepath)
    return dataframe

def get_cxr_paths_list(filepath): 
    dataframe = load_data(filepath)
    cxr_paths = dataframe['Path']
    return cxr_paths

'''
This function resizes and zero pads image 
'''
def preprocess(img, desired_size=320):
    old_size = img.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = img.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it

    new_img = Image.new('L', (desired_size, desired_size))
    new_img.paste(img, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
    return new_img

def img_to_hdf5(cxr_paths, out_filepath: str, resolution=320): 
    """
    Convert directory of images into a .h5 file given paths to all 
    images. 
    """
    dset_size = len(cxr_paths)
    failed_images = []
    with h5py.File(out_filepath,'w') as h5f:
        img_dset = h5f.create_dataset('cxr', shape=(dset_size, resolution, resolution))    
        for idx, path in enumerate(tqdm(cxr_paths)):
            try: 
                # read image using cv2
                img = cv2.imread(str(path))
                # convert to PIL Image object
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                # preprocess
                img = preprocess(img_pil, desired_size=resolution)     
                img_dset[idx] = img
            except Exception as e: 
                failed_images.append((path, e))
    print(f"{len(failed_images)} / {len(cxr_paths)} images failed to be added to h5.", failed_images)

def get_files(directory):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        for file in filenames:
            if file.endswith(".jpg"):
                files.append(os.path.join(dirpath, file))
    return files

def get_cxr_path_csv(out_filepath, directory):
    files = get_files(directory)
    file_dict = {"Path": files}
    df = pd.DataFrame(file_dict)
    df.to_csv(out_filepath, index=False)

def section_start(lines, section=' IMPRESSION'):
    for idx, line in enumerate(lines):
        if line.startswith(section):
            return idx
    return -1

def section_end(lines, section_start):
    num_lines = len(lines)

def getIndexOfLast(l, element):
    """ Get index of last occurence of element
    @param l (list): list of elements
    @param element (string): element to search for
    @returns (int): index of last occurrence of element
    """
    i = max(loc for loc, val in enumerate(l) if val == element)
    return i 

def write_report_csv(df, txt_folder, out_path, targets=["indication", "findings", "impression"]):
    imps = {"filename": [], "dicom_id":[]}
    for target in targets:
        imps[target] = []
    for n in tqdm(range(len(df))):
        dfs = df.iloc[n]
        study_num = str(dfs['study_id'])
        patient_num = str(dfs['subject_id'])
        dicom_num = str(dfs['dicom_id'])
        path = glob.glob(os.path.join(txt_folder, "*", 'p'+patient_num, 's'+study_num+'.txt'))
        assert len(path) == 1
        filename = study_num + '.txt'
        f = open(path[0], 'r')
        s = f.read()
        s_split = s.split()

        for idx, target in enumerate(targets):
            if target.upper()+":" in s_split:
                begin = getIndexOfLast(s_split, target.upper()+":") + 1
                end = None
                end_cand1 = None
                end_cand2 = None
                # remove recommendation(s) and notification
                if "RECOMMENDATION(S):" in s_split:
                    end_cand1 = s_split.index("RECOMMENDATION(S):")
                elif "RECOMMENDATION:" in s_split:
                    end_cand1 = s_split.index("RECOMMENDATION:")
                elif "RECOMMENDATIONS:" in s_split:
                    end_cand1 = s_split.index("RECOMMENDATIONS:")

                if "NOTIFICATION:" in s_split:
                    end_cand2 = s_split.index("NOTIFICATION:")
                elif "NOTIFICATIONS:" in s_split:
                    end_cand2 = s_split.index("NOTIFICATIONS:")

                if end_cand1 and end_cand2:
                    end = min(end_cand1, end_cand2)
                elif end_cand1:
                    end = end_cand1
                elif end_cand2:
                    end = end_cand2            

                if end == None:
                    imp = " ".join(s_split[begin:])
                else:
                    imp = " ".join(s_split[begin:end])
            else:
                # imp = 'NO IMPRESSION'
                imp = np.nan
                
            imps[target].append(imp)
            if idx == 0:
                imps["filename"].append(filename)
                imps["dicom_id"].append(dicom_num)
        
    df = pd.DataFrame(data=imps)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)



if __name__ == "__main__":
    
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--csv_out_path', type=str, default='data/cxr_paths.csv', help="Directory to save paths to all chest x-ray images in dataset.")
        parser.add_argument('--cxr_out_path', type=str, default='data/cxr.h5', help="Directory to save processed chest x-ray image data.")
        parser.add_argument('--dataset_type', type=str, default='mimic', choices=['mimic', 'chexpert-test'], help="Type of dataset to pre-process")
        parser.add_argument('--mimic_impressions_path', default='data/mimic_impressions_val.csv', help="Directory to save extracted impressions from radiology reports.")
        parser.add_argument('--chest_x_ray_path', default='/deep/group/data/mimic-cxr/mimic-cxr-jpg/2.0.0/files', help="Directory where chest x-ray image data is stored. This should point to the files folder from the MIMIC chest x-ray dataset.")
        parser.add_argument('--radiology_reports_path', default='../Stanford_MIT_CHEST/MIMIC-CXR-v2.0/reports/files', help="Directory radiology reports are stored. This should point to the files folder from the MIMIC radiology reports dataset.")
        args = parser.parse_args()
        return args
    
    args = parse_args()
    if args.dataset_type == "mimic":
        # Write Chest X-ray Image HDF5 File
        # get_cxr_path_csv(args.csv_out_path, args.chest_x_ray_path)
        # cxr_paths = get_cxr_paths_list(args.csv_out_path)
        df = pd.read_csv('../Stanford_MIT_CHEST/MIMIC-CXR-v2.0/csvs/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.0.0/mimic-cxr-2.0.0-split.csv')
        train_df = df[df['split']=='validate']
        #Write CSV File Containing Impressions for each Chest X-ray
        write_report_csv(train_df, args.radiology_reports_path, args.mimic_impressions_path)
        # convert cxr images into one h5df file
        cxr_paths = [os.path.join('../Stanford_MIT_CHEST/MIMIC-CXR-v2.0/mimic-cxr', x+'.jpg') for x in train_df['dicom_id']]
        img_to_hdf5(cxr_paths, args.cxr_out_path)


    elif args.dataset_type == "chexpert-test": 
        # Get all test paths based on cxr dir
        cxr_dir = Path(args.chest_x_ray_path)
        cxr_paths = list(cxr_dir.rglob("*.jpg"))
        cxr_paths = list(filter(lambda x: "view1" in str(x), cxr_paths)) # filter only first frontal views 
        cxr_paths = sorted(cxr_paths) # sort to align with groundtruth
        assert(len(cxr_paths) == 500)
       
        img_to_hdf5(cxr_paths, args.cxr_out_path)