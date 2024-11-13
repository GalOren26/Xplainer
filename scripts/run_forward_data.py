import os
from pathlib import Path
import sys
sys.path.append('/root/root/Xplainer')
from CT_CLIP.ct_clip.ct_clip import CTCLIP
from transformers import BertTokenizer, BertModel
from CTClipInference import CTClipInference
from CT_CLIP.transformer_maskgit.transformer_maskgit.ctvit import CTViT
from dotenv import load_dotenv
from utils import download_from_hub 
from pathlib import Path
tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
text_encoder.resize_token_embeddings(len(tokenizer))
load_dotenv()
import os
image_encoder = CTViT(
    dim = 512,
    codebook_size = 8192,
    image_size = 480,
    patch_size = 20,
    temporal_patch_size = 10,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 32,
    heads = 8
)

clip = CTCLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    dim_image = 294912, # in the begging we has,24*24*24*512 , we avarged across axiale axis and then flaten the tokens.
    dim_text = 768,
    dim_latent = 512,
    extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_mlm=False,
    downsample_image_embeds = False,
    use_all_token_embeds = False

)
clip.load_checkpoint(file_name="CT_VocabFine_v2.pt")

data_folder="dataset"

# download labels
lables_file_name="valid_labels.csv"
lables_url="https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/resolve/main/dataset/multi_abnormality_labels/valid_predicted_labels.csv"
local_path_lables = os.path.join(data_folder, lables_file_name)
download_from_hub(lables_url,local_path_lables)
# download meta-data
meta_name="meta_data.csv"
meta_url="https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/resolve/main/dataset/metadata/validation_metadata.csv"
local_path_meta = Path(os.path.join(data_folder, meta_name))
if not local_path_meta.exists():
    download_from_hub(meta_url,local_path_meta)
inference = CTClipInference(
    CTClip=clip,
    batch_size = 1,
    data_folder =data_folder,
    labels = local_path_lables,
    results_folder = "./results",
    tokenizer=tokenizer,
    meta_data=local_path_meta
)

inference.infer()
