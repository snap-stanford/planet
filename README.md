# PlaNet: Predicting population response to drugs via clinical knowledge graph

PyTorch implementation of PlaNet, a geometric deep learning tool for predicting population response to drugs. PlaNet provides a new clinical knowledge graph that captures the relations between disease biology, drug chemistry, and population characteristics. Using this knowledge graph, PlaNet can take a population and drugs to be applied (e.g., a clinical trial) as an input and predict the efficacy and safety of the drugs for the population. For a detailed description of the algorithm, please see our manuscript ["Predicting population response to drugs via clinical knowledge graph"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10942490/).


## 1. Installation
### Environment
Run the following commands to create a conda environment:
```bash
conda create -n planet python=3.8
source activate planet

pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install transformers==4.4.2 datasets==2.7.1 tensorboard==2.11.0 pandas wandb scikit-learn seqeval matplotlib pyyaml seaborn anndata scanpy
pip install setuptools==58.2.0 numpy==1.22.2

pip install torch-scatter==2.0.9 torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-1.10.1+cu113.html
pip install torch-geometric==2.0.0
pip install ogb==1.3.0 --no-deps
```
After this installation, the `tokenizers` library version should be 0.10.3. In case you encounter an error about the `tokenizers` version, go to `<conda_path>/envs/<env_name>/lib/python3.8/site-packages/transformers/dependency_versions_check.py` and replace the `if pkg == "tokenizers": ...` block with `if pkg == "tokenizers": continue`.

The total install time should be within 10 minutes.

Hardware requirement: 100GB RAM and GPU of 40GB memory

### Data
You can download all the data (knowledge graph, clinical trial dataset, models, etc.) from [**here (data.zip)**](https://snap.stanford.edu/planet/data.zip). Unzip this, which will create a `./data` directory.

## 2. Demo

We provide a demo notebook for loading the PlaNet knowledge graph and clinical trials data, and running the PlaNet models:
```
notebooks/demo.ipynb
```

The expected run time should be ~10 minutes.


## 3. Model training
Go to `./gcn_models` directory. We train models to predict the efficacy, safety, and potential adverse events of a clinical trial.

To train a model for **efficacy prediction**, run commands in
```
../scripts/train_efficacy.sh
```
To train a model for **safety prediction**, run commands in
```
../scripts/train_safety.sh
```
To train a model for **adverse event prediction**, run commands in
```
../scripts/train_ae.sh
```

## 4. If you want to use PlaNet models for new clinical trial data
Overview: Running our models to predict for new clinical trials involves two steps:
- (1) parse the trial (`parsing_package/parse_trial.py`) so that the trial data is preprocessed and linked to the PlaNet knowledge graph
- (2) run the models to obtain safety or efficacy predictions (`notebooks/predict_for_new_clinial_trial.ipynb`)

Specifically, to do this, follow the steps below:
 - Download all the data and resources needed for trial data parsing from [**here (parsing_package.zip)**](https://snap.stanford.edu/planet/parsing_package.zip). Unzip this and put it in the `./parsing_package` directory. Install the dependencies by following `./parsing_package/README`
 - Go to `./parsing_package` directory and run `parse_trial.py` to process a new clinical trial (e.g., NCT02370680)
 - Finally, go to `./notebooks` directory and run `predict_for_new_clinial_trial.ipynb` to get AE, safety, and efficacy predictions for the new clinical trial.



## Citation
If you find our code and research useful, please consider citing:
```bib
@article{planet2023,
  author =  {Maria Brbi{\'c} and Michihiro Yasunaga and Prabhat Agarwal and Jure Leskovec},
  title =   {Predicting population response to drugs via clinical knowledge graph},
  year =    {2023},  
}
```
