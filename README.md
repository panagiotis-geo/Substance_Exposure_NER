# Substance_Exposure_NER
This repository contains the code for the paper "Supporting the working life exposome: annotating occupational exposure for enhanced literature search"

## Dataset
Download the dataset, the original split and the cross-validation splits from Zenodo (link)

## Token-based model
Use poetry to install the associated libraries.
To run the training script:
```
cd token_based
sh train.sh
```
`train.sh` has a parameter called `dataset_name` for the dataset folder containg the train, valid and test folders.

## Span-based model
Download the span-based model source code from https://github.com/nguyennth/joint-ib-models
Follow the setup procedure described in that respoitory.
To train and evalute the model:
```
python3 src/train.py --yaml experiments/ncbi/ncbi-train.yaml
python3 src/predict.py --yaml experiments/ncbi/ncbi-test.yaml
```
You need to make the proper adjustment to the .yaml files to point to the correct dataset folders.


## Citation
If you use this repository, please cite the paper:
```
@article{thompson2024exposure,
  title={Supporting the working life exposome: annotating occupational exposure for enhanced literature search},
  author={Thompson, Paul and Ananiadou, Sophia and Basinas, Ioannis and Brinchmann, Bendik C. and Cramer, Christine and Galea, Karen S. and Ge, Calvin and Georgiadis, Panagiotis and Kirkeleit, Jorunn and Kuijpers, Eelco and Nguyen, Nhung and Nuñez, Roberto and Schlünssen, Vivi and Stokholm, Zara Ann and Taher, Evana Amir and Tinnerberg, Håkan and Tongeren, Martie Van and Xie, Qianqian},
  year={2024},
}
```
