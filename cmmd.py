import argparse
import numpy as np
from transformers import AutoProcessor, CLIPVisionModel

from PIL import Image
import os, sys, torch

from tqdm import tqdm

def mmd2(K_XX, K_XY, K_YY, m, unit_diagonal=False, mmd_est='unbiased'):
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
              + (Kt_YY_sum + sum_diag_Y) / (m * m)
              - 2 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m-1))

    return mmd2


def rbf(x1, x2, sigma):
         return np.exp(-sigma * np.abs(x2 - x1) ** 2)

def rbf_mmd(features_1, features_2, m, sigma=0.01):
    k_11 = rbf(features_1.cpu(), features_1.cpu(), sigma=sigma)
    k_22 = rbf(features_2.cpu(), features_2.cpu(), sigma=sigma)
    k_12 = rbf(features_1.cpu(), features_2.cpu(), sigma=sigma)
    return mmd2(k_11, k_12, k_22, m)

def CMMD(image_folder_path_X, image_folder_path_Y, model, processor, subset_size = 1000):
    image_list_X = os.listdir(image_folder_path_X)
    image_list_Y = os.listdir(image_folder_path_Y)
    
    model = model.eval()
    
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad() :
        m = len(image_list_X)
        
        num_batch = m // subset_size
        
        if m % subset_size > 0:
            num_batch += 1
        
        cmmd = 0.0
        for batch_i in tqdm(range(num_batch)):
            
            image_vectors_X = []
            image_vectors_Y = []
            
            start_index = batch_i * subset_size
            
            if (batch_i + 1) * subset_size <= m :
                end_index = (batch_i + 1) * subset_size
            else:
                end_index = m
            
            for data_idx in range(start_index, end_index):
                image_path_X = f'{image_folder_path_X}/{image_list_X[data_idx]}'
                image_path_Y = f'{image_folder_path_Y}/{image_list_Y[data_idx]}'
                
                image_X = Image.open(image_path_X)
                image_X = processor(images=image_X, return_tensors="pt")
                image_X = image_X.to(model.device)
                
                image_Y = Image.open(image_path_Y)
                image_Y = processor(images=image_Y, return_tensors="pt")
                image_Y = image_Y.to(model.device)
        
                image_embeds_X = model(**image_X).last_hidden_state[:,1:,:]
                image_embeds_Y = model(**image_Y).last_hidden_state[:,1:,:].flatten(1) # [1, 49, 768] -> [1, 49*768]
                
                image_vectors_X.append(image_embeds_X.flatten(1))
                image_vectors_Y.append(image_embeds_Y)

            image_vectors_X = torch.cat(image_vectors_X, dim=0) # [b, d]
            image_vectors_Y = torch.cat(image_vectors_Y, dim=0) # [b, d]
            
            batch_cmmd = rbf_mmd(
                image_vectors_X,
                image_vectors_Y,
                image_vectors_X.size(0)
            )
            cmmd += batch_cmmd
            print(f"{batch_i}'s subset, cmmd: {batch_cmmd}")
            
        cmmd /= num_batch
        
        return cmmd

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Script.")

    parser.add_argument(
        "--original_folder_path",
        dest="original_folder_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--generated_folder_path",
        dest="generated_folder_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--model_version",
        dest="model_version",
        type=str,
        default="openai/clip-vit-large-patch14-336" # https://huggingface.co/openai/clip-vit-large-patch14-336
    )

    parser.add_argument(
        "--subset_size",
        dest="subset_size",
        type=int,
        default=1000
    )
    
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    model = CLIPVisionModel.from_pretrained(args.model_version)
    processor = AutoProcessor.from_pretrained(args.model_version)

    with torch.no_grad():
        cmmd = CMMD(args.generated_folder_path, args.original_folder_path, model, processor, args.subset_size)

    print(f"Path of the reference dataset: {args.original_folder_path}, Path of the generated images : {args.generated_folder_path}")
    print(f"CMMD result: {cmmd}")