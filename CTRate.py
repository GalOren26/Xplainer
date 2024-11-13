from functools import partial
import glob
import os
import re

import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import tqdm
from huggingface_hub import hf_hub_download
from descriptors import CTRate_disease_descriptors 
import nibabel as nib

class CTRateDataset(Dataset):
    def __init__(self, data_folder,labels ,meta_data,min_slices=20, resize_dim=500):
        self.repo_id = "ibrahimhamamci/CT-RATE"
        self.data_folder = data_folder
        self.min_slices = min_slices
        CTRate_disease_descriptors
        self.disease_keys=  list(CTRate_disease_descriptors.keys())
        # self.disease_keys=  ["Medical material","Cardiomegaly","Hiatal hernia","Emphysema",
        #                       "Atelectasis","Lung nodule","Lung opacity","Pleural effusion","Consolidation"]
        self.labels = labels
        self.patient_id_to_meta_info = self.prepare_samples()
        # self.accession_to_text = self.load_accession_text(csv_file)
        self.transform = transforms.Compose([
            transforms.Resize((resize_dim,resize_dim)),
            transforms.ToTensor()
        ])
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)
        self.meta_data=meta_data 
        
    def get_pattern_of_path(self,filename):
        # Extract relevant parts from filename
        parts = filename.split('_')
        subdir1 = f"{parts[0]}_{parts[1]}"
        # Regex pattern to match volume files in the subdirectory
        pattern = rf'^{subdir1}_'
        return pattern
    def get_path_from_file_name(self,custom_base, filename):
        # Extract relevant parts from filename
        parts = filename.split('_')
        dataset_type = parts[0]          # This will be "valid" or "train"
        subdir1 = f"{parts[0]}_{parts[1]}"  # For example, "valid_1041"
        subdir2 = f"{parts[0]}_{parts[1]}_{parts[2]}"  # For example, "valid_1041_a"
        # Construct the full path
        path= f"{custom_base}/{dataset_type}/{subdir1}/{subdir2}/{filename}"
        return path  
     
    def prepare_samples(self):
        patient_id_to_meta_info={}
        # Read labels once outside the loop
        test_df = pd.read_csv(self.labels)
        test_df['one_hot_labels'] = list(test_df[self.disease_keys].values)
        # run on folder and preaper arrey with paths  of scans for client and labels for this scan.
        visited_paaterns={}
        for file_name in tqdm.tqdm(test_df["VolumeName"]):
            pattern_patient=self.get_pattern_of_path(file_name)
            if pattern_patient in  visited_paaterns:
                continue    
            else: 
                visited_paaterns[pattern_patient]=True
            filtered_df = test_df[test_df["VolumeName"].str.contains(pattern_patient)]
            patiant_id = int(re.search(r'_(\d+)_', file_name).group(1))
            if patiant_id not in patient_id_to_meta_info:
                patient_id_to_meta_info[patiant_id] = {}
            for scan_name in filtered_df["VolumeName"]:
                # Extract the scan ID (e.g., a, b, c, etc.) from the filename
                scan_id = scan_name.split('_')[-2]  # Assuming scan ID is the second-to-last part
                # Initialize list for scan ID if it doesn’t exist
                if scan_id not in patient_id_to_meta_info[patiant_id]:
                    onehotlabels = filtered_df[filtered_df["VolumeName"] == scan_name]["one_hot_labels"].values[0]
                    patient_id_to_meta_info[patiant_id][scan_id] ={'image_paths':[], 'disease_keys':self.disease_keys, 'disease_values':onehotlabels}
                # Get the full path for the scan and append it to the scan_id list
                path=self.get_path_from_file_name(self.data_folder, scan_name)
                patient_id_to_meta_info[patiant_id][scan_id]['image_paths'].append(path)
        return patient_id_to_meta_info

    def load_accession_text(self, csv_file):
        #not used 
        df = pd.read_csv(csv_file)
        df_meta = pd.read_csv(self.meta_data)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['VolumeName']] = row["Findings_EN"],row['Impressions_EN']
        return accession_to_text
    ## download image from path and convert nii to tensor and delete temp file 
    def nii_img_to_tensor(self,path , transform):
        # read image data 
        temp_file_path=self.download_npz_from_hub(path)
        nii_img = nib.load(str(temp_file_path))
        img_data = nii_img.get_fdata()
        # read meta_data 
        df_meta = pd.read_csv(self.meta_data) 
        file_name = path.split("/")[-1]
        row = df_meta[df_meta['VolumeName'] == file_name]
        slope = float(row["RescaleSlope"].iloc[0])
        intercept = float(row["RescaleIntercept"].iloc[0])
        xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
        z_spacing = float(row["ZSpacing"].iloc[0])
        # Define the target spacing values
        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5
        current = (z_spacing, xy_spacing, xy_spacing)
        target = (target_z_spacing, target_x_spacing, target_y_spacing)
        img_data = slope * img_data + intercept
        img_data = img_data.transpose(2, 0, 1)
        tensor = torch.tensor(img_data)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        img_data = self.resize_array(tensor, current, target)
        img_data = img_data[0][0]
        img_data= np.transpose(img_data, (1, 2, 0))
        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)
        img_data = (((img_data ) / 1000)).astype(np.float32)
        slices=[]
        tensor = torch.tensor(img_data)
        # Get the dimensions of the input tensor
        target_shape = (480,480,240)
        # Extract dimensions
        h, w, d = tensor.shape

        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

        tensor = tensor.permute(2, 0, 1)

        tensor = tensor.unsqueeze(0).unsqueeze(0)

        return tensor
    def resize_array(self,array, current_spacing, target_spacing):
        """
        Resize the array to match the target spacing.

        Args:
        array (torch.Tensor): Input array to be resized.
        current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
        target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

        Returns:
        np.ndarray: Resized array.
        """
        # Calculate new dimensions
        original_shape = array.shape[2:]
        scaling_factors = [
            current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
        ]
        new_shape = [
            int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
        ]
        # Resize the array
        resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
        return resized_array

        
    def download_npz_from_hub(self,path):
        subfolder, filename = os.path.split(path)
        # Download the file temporarily
        temp_file_path=hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            cache_dir='./cache_scans',
            token=os.getenv("hf_token"),
            subfolder=subfolder,
            filename=filename,
            resume_download=True,
        )
        return temp_file_path

    def __len__(self):
        return len(self.patient_id_to_meta_info)

    def __getitem__(self, patient_id):
        return self.patient_id_to_meta_info[patient_id]
    
    
    #     ## download image from path and convert nii to tensor
    # def nii_img_to_tensor_old(self,path , transform):
        
    #     # img_data = np.load(path)['arr_0']
    #     temp_file_path=self.download_npz_from_hub(path)
        
    #     image = nib.load(temp_file_path)
    #     img_data = image.get_fdata() 
    #     img_data= np.transpose(img_data, (1, 2, 0))
    #     img_data = img_data*1000
    #     hu_min, hu_max = -1000, 200
    #     img_data = np.clip(img_data, hu_min, hu_max)

    #     img_data = (((img_data+400 ) / 600)).astype(np.float32)
    #     slices=[]
    #     tensor = torch.tensor(img_data)
    #     # Get the dimensions of the input tensor
    #     target_shape = (480,480,240)
    #     # Extract dimensions
    #     h, w, d = tensor.shape

    #     # Calculate cropping/padding values for height, width, and depth
    #     dh, dw, dd = target_shape
    #     h_start = max((h - dh) // 2, 0)
    #     h_end = min(h_start + dh, h)
    #     w_start = max((w - dw) // 2, 0)
    #     w_end = min(w_start + dw, w)
    #     d_start = max((d - dd) // 2, 0)
    #     d_end = min(d_start + dd, d)

    #     # Crop or pad the tensor
    #     tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    #     pad_h_before = (dh - tensor.size(0)) // 2
    #     pad_h_after = dh - tensor.size(0) - pad_h_before

    #     pad_w_before = (dw - tensor.size(1)) // 2
    #     pad_w_after = dw - tensor.size(1) - pad_w_before

    #     pad_d_before = (dd - tensor.size(2)) // 2
    #     pad_d_after = dd - tensor.size(2) - pad_d_before

    #     tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)


    #     tensor = tensor.permute(2, 0, 1)

    #     tensor = tensor.unsqueeze(0)
    #     os.remove(temp_file_path)
    #     return tensor