import os
from pathlib import Path
from typing import List

from transformers import BertTokenizer, BertModel

from CTRate import CTRateDataset
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import tqdm
import pandas as pd
from utils import calculate_auroc
from descriptors import CTRate_disease_descriptors
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from CT_CLIP.ct_clip.ct_clip import CTCLIP

from math import exp, log
class CTClipInference(nn.Module):
    def __init__(
            self,
            CTClip: CTCLIP,
            *,
            batch_size,
            meta_data,
            data_folder: "external_valid",
            save_results_every = 100,
            save_model_every = 2000,
            results_folder = './results',
            labels = "labels.csv",
            accelerate_kwargs: dict = dict(),
            tokenizer= BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True),
            
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **accelerate_kwargs)
        self.CTClip = CTClip 
        self.tokenizer =tokenizer
        self.results_folder = results_folder
        self.register_buffer('steps', torch.Tensor([0]))
        self.batch_size = batch_size
        # Load the pre-trained weight
        # s
        self.ds = CTRateDataset(data_folder=data_folder,labels=labels,meta_data=meta_data)
        # Split dataset into train and validation sets
        self.dl = DataLoader(
            self.ds,
            num_workers=3,
            batch_size=1,
            shuffle = True,
        )
        #self.dl_iter=cycle(self.dl)
        # self.device = self.accelerator.device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.CTClip.to(self.device)
        self.save_model_every = save_model_every
        self.save_results_every = save_results_every
        self.result_folder_txt = self.results_folder
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        # caches for faster inference
        self.text_embedding_cache = {}
        self.image_embedding_cache = {}

    def infer(self):
        device = self.device
        with torch.no_grad():
                model=self.CTClip
                model.eval()
                all_descriptors = self.get_all_descriptors(CTRate_disease_descriptors)
                all_labels = []
                all_probs_predicted = []
                for batch in tqdm.tqdm(self.dl):
                        for scan_id in batch:
                            batch_img = batch[scan_id]
                            image_paths, labels, keys= batch_img['image_paths'],batch_img['disease_values'],batch_img['disease_keys']
                            diseases=keys
                            agg_probs = []
                            agg_negative_probs = []
                            for image_path in image_paths:
                                probs, negative_probs = self.get_descriptor_probs(image_path[0], all_descriptors)
                                agg_probs.append(probs)
                                agg_negative_probs.append(negative_probs)
                            probs = {}  # Aggregated
                            negative_probs = {}  # Aggregated
                            for key in agg_probs[0].keys():
                                probs[key] = sum([p[key] for p in agg_probs]) / len(agg_probs)  # Mean Aggregation
                            for key in agg_negative_probs[0].keys():
                                negative_probs[key] = sum([p[key] for p in agg_negative_probs]) / len(agg_negative_probs)  # Mean Aggregation
                                
                            disease_probs, negative_disease_probs = self.get_diseases_probs(CTRate_disease_descriptors, pos_probs=probs,
                                                                                                    negative_probs=negative_probs)
                            predicted_diseases, disease_idx_to_prob = self.get_predictions(disease_probs=disease_probs,  
                                                                                    negative_disease_probs=negative_disease_probs,keys=keys)
                            all_labels.append(labels)
                            all_probs_predicted.append(disease_idx_to_prob)

            # x=[]
            # Image=self.ds.nii_to_tensor(image_path[0])
            # Image=Image.to("cuda")
            # image_embedding = self.CTClip.getImageEmbedding(Image)
            # for pathology in CTRate_disease_descriptors.keys():
            #     text = [f"{pathology} is present.", f"{pathology} is not present."]
            #     text_tokens=self.tokenizer(
            #                                       text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)

            #     output = model(text_tokens, Image.cuda(),  device=device)
            #     softmax = torch.nn.Softmax(dim=0)
            #     x.append(softmax(output))
                all_labels = torch.stack(all_labels)
                all_probs_predicted = torch.stack(all_probs_predicted)
                # evaluation-get rid from catgories that doesnt hve at leat one positive samples 
                existing_mask = sum(all_labels, 0) > 0
                all_labels_clean = all_labels[:, existing_mask]
                all_probs_predicted_clean = all_probs_predicted[:, existing_mask]
                all_keys_clean = [key for idx, key in enumerate(diseases) if existing_mask[idx]]
                overall_auroc, per_disease_auroc = calculate_auroc(all_probs_predicted_clean, all_labels_clean)
                print(f"AUROC: {overall_auroc:.5f}\n")
                for idx, key in enumerate(all_keys_clean):
                    print(f'{key}: {per_disease_auroc[idx]:.5f}')
        
    def get_similarity_score_from_raw_data(self, image_embedding, query_texts: str) -> float:
        """Compute the cosine similarity score between an image and one or more strings.
        If multiple strings are passed, their embeddings are averaged before L2-normalization.
        :param image_path: Path to the input chest X-ray, either a DICOM or JPEG file.
        :param query_text: Input radiology text phrase.
        :return: The similarity score between the image and the text.
        """
        text_embeddings=[]
        for query_text in query_texts:
            if query_text in self.text_embedding_cache:
                text_embedding = self.text_embedding_cache[query_text]
            else:
                text_embedding = self.CTClip.getTextEmbedding(query_text)
                # Convert the tensor to a NumPy array
                text_embedding_np = text_embedding.cpu().detach().numpy()
                # Save the NumPy array as a .npz file
                save_path =f'{self.results_folder}/text/{query_text}.npz'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.savez(save_path, arr=text_embedding_np)
                self.text_embedding_cache[query_text] = text_embedding
            text_embeddings.append(text_embedding)
        text_embeddings = torch.stack(text_embeddings).squeeze(1)
        text_embeddings_np = text_embeddings.cpu().detach().numpy()
        cos_similarity = (image_embedding @ text_embeddings_np.T)*self.CTClip.temperature.exp().item() # temprature is trainable parmater for control the samilrty messure and we take exponent of it as int the orignaol implenation 
        return cos_similarity

    def get_descriptor_probs(self, image_path: Path, descriptors: List[str], do_negative_prompting=True, demo=False,):
        probs = {}
        negative_probs = {}
        if image_path in self.image_embedding_cache:
            image_embedding = self.image_embedding_cache[image_path]
        else:
            Image=self.ds.nii_to_tensor(image_path)
            Image=Image.to("cuda")
            image_embedding = self.CTClip.getImageEmbedding(Image)
            image_embedding = image_embedding.cpu().detach().numpy()
            # Save the NumPy array as a .npz file
            save_path = f'{self.results_folder}/image/{image_path}.npz'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, arr=image_embedding)
            if not demo:
                self.image_embedding_cache[image_path] = image_embedding
        # Default get_similarity_score_from_raw_data would load the image every time. Instead we only load once.
        prompts = np.array([f'There is {desc}' for desc in descriptors])
        scores= self.get_similarity_score_from_raw_data(image_embedding, prompts)
        # text_tokens=self.tokenizer(list(prompts), return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(self.device)
        # y=self.CTClip(text_tokens,Image, device=self.device)
        if do_negative_prompting:
                neg_prompts = np.array([f'There is no {desc}' for desc in descriptors])
                neg_scores = self.get_similarity_score_from_raw_data(image_embedding, neg_prompts)
        pos_probs = self.cos_sim_to_prob(scores)
        if do_negative_prompting:
            stacked_scores = torch.stack([torch.tensor(scores), torch.tensor(neg_scores)]) #dont use divide by 2 couse alredy take care before about remprture
            # Apply softmax across each pair of positive and negative scores
            probs_batch = torch.softmax(stacked_scores, dim=0).cpu().detach().numpy()
            # Extract positive and negative probabilities
            pos_probs = probs_batch[0]  # Positive probabilities for each descriptor
            neg_probs = probs_batch[1]  # Negative probabilities for each descriptor
            for i, desc in enumerate(descriptors):
                probs[desc] = pos_probs[0][i]
                negative_probs[desc] = neg_probs[0][i]
        else: 
              for i, desc in enumerate(descriptors):
                probs[desc] = pos_probs[i]
                
        return probs, negative_probs
    
    def cos_sim_to_prob(self,sim):
        return (sim + 1) / 2  # linear transformation to 0 and 1
    
    def get_all_descriptors(self, disease_descriptors):
        all_descriptors = set()
        for disease, descs in disease_descriptors.items():
            all_descriptors.update([f"{desc} indicating {disease}" for desc in descs])
        return all_descriptors

    def get_diseases_probs(self, disease_descriptors, pos_probs, negative_probs, prior_probs=None, do_negative_prompting=True):
        disease_probs = {}
        disease_neg_probs = {}
        for disease, descriptors in disease_descriptors.items():
            # Create descriptors with disease context
            full_descriptors = [f"{desc} indicating {disease}" for desc in descriptors]

            # Vectorized log-probabilities for descriptors
            desc_log_probs = torch.tensor([log(pos_probs[desc]) for desc in full_descriptors])
            disease_log_prob = torch.mean(desc_log_probs)
            disease_probs[disease] = exp(disease_log_prob.item())
            
            if do_negative_prompting:
                desc_neg_log_probs = torch.tensor([log(negative_probs[desc]) for desc in full_descriptors])
                disease_neg_log_prob = torch.mean(desc_neg_log_probs)
                disease_neg_probs[disease] = exp(disease_neg_log_prob.item())
        return disease_probs, disease_neg_probs

    # Negative vs Positive Prompting
    def get_predictions(self, disease_probs, negative_disease_probs,keys):
        predicted_diseases = []
        disease_idx_to_prob = {}
        keys=[key[0] for key in keys ]
        for disease in disease_probs.keys():
            # collect prob vwctor for AUC-ROC calauation later 
            disease_idx_to_prob[keys.index(disease)]=disease_probs[disease]
            if disease_probs[disease] > negative_disease_probs.get(disease, 0):
                #just sainty to cheak where it above some value ,it will later be be detramined bu y the auc-roc 
                predicted_diseases.append(disease)
        return predicted_diseases, disease_idx_to_prob
