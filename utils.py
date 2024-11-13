from math import log, exp
import os

import numpy as np
import requests
from sklearn.metrics import roc_auc_score


def cos_sim_to_prob(sim):
    return (sim + 1) / 2  # linear transformation to 0 and 1

def calculate_auroc(all_disease_probs, gt_diseases):
    '''
    Calculates the AUROC (Area Under the Receiver Operating Characteristic curve) for multiple diseases.

    Parameters:
    all_disease_probs (numpy array): predicted disease labels, a multi-hot vector of shape (N_samples, 14)
    gt_diseases (numpy array): ground truth disease labels, a multi-hot vector of shape (N_samples, 14)

    Returns:
    overall_auroc (float): the overall AUROC score
    per_disease_auroc (numpy array): an array of shape (14,) containing the AUROC score for each disease
    '''

    per_disease_auroc = np.zeros((gt_diseases.shape[1],))  # num of diseases
    for i in range(gt_diseases.shape[1]):
        # Compute the AUROC score for each disease
        per_disease_auroc[i] = roc_auc_score(gt_diseases[:, i], all_disease_probs[:, i])

    # Compute the overall AUROC score
    overall_auroc = roc_auc_score(gt_diseases, all_disease_probs, average='macro')
    return overall_auroc, per_disease_auroc
def download_from_hub(url,local_path):
        filename = os.path.basename(url)
        directory=os.path.dirname(local_path)
        if os.path.isfile(local_path):
            print(f"File '{filename}' already exists, skipping download.")
        os.makedirs(directory, exist_ok=True)
        token=os.getenv('hf_token')
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(url, headers=headers, stream=True)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
            print(f"{filename} downloaded successfully.")
        else:
            print(f"Failed to download model: {response.status_code} - {response.text}")
            