# CMMD 
Unofficial PyTorch implementation of CMMD (Maximum Mean Discrepancy distance using CLIP embeddings). 
CMMD is proposed in [Rethinking FID: Towards a Better Evaluation Metric for Image Generation](https://arxiv.org/abs/2401.09603). CMMD is considered a more effective metric than FID, designed to overcome the persistent challenges of FID. 

The [original implementation](https://github.com/google-research/google-research/tree/master/cmmd) is based on JAX and TensorFlow. This implementation replaced them PyTorch and does not need the [`scenic`](https://github.com/google-research/scenic) for computing CLIP embeddings.

* For CLIP embedding, we use CLIPImageEncoder while the cmmd paper use [`CLIPVisionModel`](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModel) from huggingface.

## Installed modules
* [torch 1.13.1 (with CUDA 11.7)](https://pytorch.org/get-started/previous-versions/)
* [numpy](https://numpy.org/install/)
* [transformers](https://github.com/huggingface/transformers)
* [pillow](https://pypi.org/project/pillow/)
* [tqdm](https://pypi.org/project/tqdm/)

## Run
```bash
python -u cmmd.py --original_folder_path /path/to/original/dataset  --generated_folder_path /path/to/generated/dataset
```

**Notes**:
* if you want to use other version of CLIP: 
```bash
python -u cmmd.py --original_folder_path /path/to/original/dataset  --generated_folder_path /path/to/generated/dataset --model_version model version
```
* if you want to use other subset size:
```bash
python -u cmmd.py --original_folder_path /path/to/original/dataset  --generated_folder_path /path/to/generated/dataset --subset_size subset size
```
## Acknowledgements
Thank you to [Sadeep](https://github.com/sadeepj) for helpful comments to re-implement the CMMD.
