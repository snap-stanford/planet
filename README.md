# PlaNet: Predicting population response to drugs via clinical knowledge graph

## 1. Installation
### Environment
Run the following commands to create a conda environment:
```bash
conda create -n planet python=3.8
source activate planet

pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install transformers==4.4.2 datasets==2.7.1 tensorboard==2.11.0 pandas wandb sklearn seqeval matplotlib pyyaml seaborn anndata scanpy
pip install setuptools==58.2.0 numpy==1.22.2

pip install torch-scatter==2.0.9 torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-1.10.1+cu113.html
pip install torch-geometric==2.0.0
pip install ogb==1.3.0 --no-deps
```
After this installation, the `tokenizers` library version should be 0.10.3. In case you encounter an error about the `tokenizers` version, go to `<conda_path>/envs/<env_name>/lib/python3.8/site-packages/transformers/dependency_versions_check.py` and replace the `if pkg == "tokenizers": ...` block with `if pkg == "tokenizers": continue`.

### Data
You can download all the data (knowledge graph, clinical trial dataset, models, etc.) from [**here (data.zip)**](https://nlp.stanford.edu/projects/myasu/PlaNet/data.zip). Unzip this, which will create a `./data` directory.


## 2. Demo

We provide a demo notebook for loading the PlaNet knowledge graph and clinical trials data, and running the PlaNet models:
```
notebooks/demo.ipynb
```


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



## Citation
If you find our code and research useful, please consider citing:
```bib
@article{planet2023,
  author =  {Maria Brbi{\'c} and Michihiro Yasunaga and Prabhat Agarwal and Jure Leskovec},
  title =   {Predicting population response to drugs via clinical knowledge graph},
  year =    {2023},  
}
```
