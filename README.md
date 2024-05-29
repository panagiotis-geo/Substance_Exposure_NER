# Substance Exposure NER
This repository contains the code to train the machine learning models described in the following article: <br />
"Supporting the working life exposome: annotating occupational exposure for enhanced literature search"

The article describes experiments using two different models:

- Token-based
- Span-based

## Datasets
The datasets used in the experiments are created by splitting the annotated corpus described in the article (see https://zenodo.org/records/11164272) in different way to facilitate training and evaluation of the models. Two types of experiments are described in the article:

- 10-fold cross validation
- Evaluation on held-out test data set

The `datasets` folders contains the datasets used for both types of models and both types of experiments.  The input format of the data is different for each type of model and is contained within the subfolders `span-based` and `token-based`. Each of these subfolders contains:

- `cross-validation` - the splits of the data used for the 10-fold cross validation. The results reported in the paper were obtained by training the models on the "train" set and evaluating using the "valid" set. 
- `train-valid-test` - Models were trained using the "train" set and the "valid" set was used to fine-tine the model parameters. The models were evaluated usong the held-out "test" set.   

The test set is the same for all the splits and the original dataset.

## Token-based model
Use poetry to install the associated libraries:
```
cd token_based
poetry install
```
The experiments were performed using Python version 3.10.14 and pytorch v1.13.1. <br />
To run the training script inside the `token_based/`: <br />
```
sh train.sh
```
`train.sh` has a parameter called `dataset_name` that needs to point to the dataset folder containg the train, valid and test folders.

## Span-based model
Download the span-based model source code from https://github.com/nguyennth/joint-ib-models <br />
Follow the setup procedure described in that repository. <br />
To train and evalute the model:
```
python3 src/train.py --yaml experiments/ncbi/ncbi-train.yaml
python3 src/predict.py --yaml experiments/ncbi/ncbi-test.yaml
```
Remember to make the proper adjustment to the .yaml files to point to the correct dataset folders.


## Citation
If you use this repository, please cite the paper:
```
@article{thompson2024exposure,
  title={Supporting the working life exposome: annotating occupational exposure for enhanced literature search},
  author={Thompson, Paul and Ananiadou, Sophia and Basinas, Ioannis and Brinchmann, Bendik C. and Cramer, Christine and Galea, Karen S. and Ge, Calvin and Georgiadis, Panagiotis and Kirkeleit, Jorunn and Kuijpers, Eelco and Nguyen, Nhung and Nuñez, Roberto and Schlünssen, Vivi and Stokholm, Zara Ann and Taher, Evana Amir and Tinnerberg, Håkan and Tongeren, Martie Van and Xie, Qianqian},
  year={2024},
}
```
