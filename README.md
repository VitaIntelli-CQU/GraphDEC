# GraphDec

GraphDEC, a graph neural network-based tool for deconvoluting proteomics data.

![model.jpg](https://github.com/VitaIntelli-CQU/GraphDEC/blob/main/model.jpg)

## Installation  

We recommend using Anaconda to create a new Python environment and activate it via

```
conda env create -f GraphDec_env.yaml
conda activate GraphDec_env
pip uninstall nvidia-cublas-cu11         # Perhaps this command needs to be executed.
```

## Quick Start

### Mixup training data and generate test data

Please refer to the contents in the "mixup" folder.

### Input

* **train_data:**   [scanpy anndata] Mixup-processed training data.
* **test_data:**    [scanpy anndata] Test data.

### Output

* **pred:**   [numpy array] Cell abundance prediction.

### For calling GraphDec programmatically

```python
model = GraphDec(train_data, test_data)
model.train()
pred = model.prediction(torch.FloatTensor(test_data.X.astype(np.float32)).cuda())
```

If you want to accurately reproduce the results in the paper, **use NVIDIA GeForce RTX 4090**

## Tutorial

### Tutorial 1: Cell abundance prediction on CyTOF proteome dataset

1. Download the compressed file Primary_All_Lin.zip from [the URL](https://data.mendeley.com/datasets/vs8m5gkyfn/1), and then decompress and extract the file `Final_Primary_All_Lin.h5ad`.

2. Refer to or directly execute [mixup/CyTOF_mix.ipynb](https://github.com/VitaIntelli-CQU/GraphDEC/tree/main/mixup/CyTOF_mix.ipynb) to generate Mixup-processed training data and test data. This step will generate the pre-processed file `CyTOF_200.h5ad`.

3. Refer to or directly execute [tutorial/CyTOF_Tutorial.ipynb](https://github.com/VitaIntelli-CQU/GraphDEC/tree/main/tutorial/CyTOF_Tutorial.ipynb).

### Tutorial 2: Cell abundance prediction from Rheus_macaques_ifgn proteome dataset to Human_ifgn proteome dataset

1. Download the files `Rheus_macaques_ifgn.h5ad` and `Human_ifgn.h5ad` from [the URL](https://github.com/single-cell-proteomic/SCPRO-HI/tree/main/Data/cross-species).

2. Refer to or directly execute [mixup/cross_species.ipynb](https://github.com/VitaIntelli-CQU/GraphDEC/tree/main/mixup/cross_species.ipynb) to generate Mixup-processed training data and test data. This step will generate the pre-processed file `Rheus_macaques_2_Human.h5ad`.

3. Refer to or directly execute [tutorial/Rheus_macaques_2_Human_Tutorial.ipynb](https://github.com/VitaIntelli-CQU/GraphDEC/tree/main/tutorial/Rheus_macaques_2_Human_Tutorial.ipynb).

### Tutorial 3: Cell abundance prediction on pbmc_data transcriptome dataset

1. Download the file `pbmc_data.h5ad` from [the URL](https://figshare.com/s/e59a03885ec4c4d8153f?file=15008006).

2. Refer to or directly execute [tutorial/pbmc_data_Tutorial.ipynb](https://github.com/VitaIntelli-CQU/GraphDEC/tree/main/tutorial/pbmc_data_Tutorial.ipynb).

### Tutorial 4: Cell abundance prediction on mouse_PDAC spatial proteome dataset

1. Download the compressed file datasets.zip from [the URL](https://zenodo.org/records/14233865), and then decompress and extract the file `mouse_PDAC/intersected_reference_proteomics.h5ad` and `mouse_PDAC/intersected_spatial_proteomics.h5ad`.

2. Refer to or directly execute [mixup/mouse_PDAC_mix.ipynb](https://github.com/VitaIntelli-CQU/GraphDEC/tree/main/mixup/mouse_PDAC_mix.ipynb) to generate Mixup-processed training data and test data. This step will generate the pre-processed file `mouse_PDAC.h5ad`.

3. Refer to or directly execute [tutorial/mouse_PDAC_Tutorial.ipynb](https://github.com/VitaIntelli-CQU/GraphDEC/tree/main/tutorial/mouse_PDAC_Tutorial.ipynb).
