# GraphDec

GraphDEC, a graph neural network-based tool for deconvoluting proteomics data.

![model.jpg](https://github.com/VitaIntelli-CQU/GraphDEC/blob/main/model.jpg)

## Installation

### Conda (Recommended)

We recommend using **Anaconda** to create and manage the Python environment. To set up the environment, run the following commands:

```bash
conda env create -f GraphDec_env.yaml
conda activate GraphDec_env
pip uninstall nvidia-cublas-cu11  # You may need to run this command depending on your system setup.
```

### Docker

We provide Docker images, processed HBreast_CYTOF data, and example files for running. You can execute the following code for simple installation and execution:

```bash
docker pull crpi-4pq8xgwdyvc1i5wt.cn-shenzhen.personal.cr.aliyuncs.com/graphdec_docker/graphdec_docker:1.2
docker tag crpi-4pq8xgwdyvc1i5wt.cn-shenzhen.personal.cr.aliyuncs.com/graphdec_docker/graphdec_docker:1.2 graphdec:1.2
docker run --gpus all -it --name dc2 -p 6166:6166/tcp -v /app --shm-size 10240m graphdec:1.2 /bin/bash
python train.py
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

### Tutorial 1: Cell abundance prediction on HBreast_CYTOF proteome dataset

1. Download the compressed file Primary_All_Lin.zip from [the URL](https://data.mendeley.com/datasets/vs8m5gkyfn/1), and then decompress and extract the file `Final_Primary_All_Lin.h5ad`.

2. Refer to or directly execute [mixup/CyTOF_mix.ipynb](https://github.com/VitaIntelli-CQU/GraphDEC/tree/main/mixup/CyTOF_mix.ipynb) to generate Mixup-processed training data and test data. This step will generate the pre-processed file `CyTOF_200.h5ad`.

3. Refer to or directly execute [tutorial/CyTOF_Tutorial.ipynb](https://github.com/VitaIntelli-CQU/GraphDEC/tree/main/tutorial/CyTOF_Tutorial.ipynb).

### Tutorial 2: Cell abundance prediction from Ifrhesus proteome dataset to IFHuman proteome dataset

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
