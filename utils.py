import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
import scanpy as sc


def ood_process(adata):

    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.umap(adata)
    
    adata2 = adata[adata.obs['batch'] != '0']
    data = adata2.obsm['X_umap']
    gmm = GaussianMixture(n_components=2, random_state=0)
    adata2.obs['gmm'] = gmm.fit_predict(data)
    a = adata[adata.obs['batch'] == '0']
    b = adata2[adata2.obs['gmm'] == 1]
    c = adata2[adata2.obs['gmm'] == 0]
    
    similarity_ab = cosine_similarity(a.X, b.X)
    similarity_ac = cosine_similarity(a.X, c.X)

    a.obs['gmm'] = np.array(['0'] * a.shape[0])
    if similarity_ab.mean() > similarity_ac.mean():
        b.obs['gmm'] = np.array(['0'] * b.shape[0])
        c.obs['gmm'] = np.array(['1'] * c.shape[0])
    else:
        b.obs['gmm'] = np.array(['1'] * b.shape[0])
        c.obs['gmm'] = np.array(['0'] * c.shape[0])

    adata_new = sc.concat((a, b, c))
    adata_new.uns['cell_types_all'] = adata.uns['cell_types_all']
    adata_new.uns['cell_types'] = adata.uns['cell_types']

    return adata_new


### evaluate metrics ###
def ccc(pred, gt):
    numerator = 2 * np.corrcoef(gt, pred)[0][1] * np.std(gt) * np.std(pred)
    denominator = np.var(gt) + np.var(pred) + (np.mean(gt) - np.mean(pred)) ** 2
    ccc_value = numerator / denominator
    return ccc_value

def compute_metrics(pred, gt):
    x = pred.reshape(-1)
    y = gt.reshape(-1)
    # x = pd.melt(pred)['value']
    # y = pd.melt(gt)['value']
    CCC = ccc(x, y)
    RMSE = sqrt(mean_squared_error(x, y))
    Corr = pearsonr(x, y)[0]
    return CCC, RMSE, Corr

def SaveLossPlot(SavePath, metric_logger, loss_type, output_prex):
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
    for i in range(len(loss_type)):
        plt.subplot(2, 3, i+1)
        plt.plot(metric_logger[loss_type[i]])
        plt.title(loss_type[i], x = 0.5, y = 0.5)
    imgName = os.path.join(SavePath, output_prex +'.png')
    plt.savefig(imgName)
    plt.close()

def SavePredPlot(SavePath, target_pred, ground_truth):
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
    celltypes = list(target_pred.columns)

    plt.figure(figsize=(5*(len(celltypes)+1), 5)) 

    eval_metric = []
    x = pd.melt(target_pred)['value']
    y = pd.melt(ground_truth)['value']
    eval_metric.append(ccc(x, y))
    eval_metric.append(sqrt(mean_squared_error(x, y)))
    eval_metric.append(pearsonr(x, y)[0])
    plt.subplot(1, len(celltypes)+1, 1)
    plt.xlim(0, max(y))
    plt.ylim(0, max(y))
    plt.scatter(x, y, s=2)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),"r--")
    text = f"$CCC = {eval_metric[0]:0.3f}$\n$RMSE = {eval_metric[1]:0.3f}$\n$Corr = {eval_metric[2]:0.3f}$"
    plt.text(0.05, max(y)-0.05, text, fontsize=8, verticalalignment='top')
    plt.title('All samples')
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')

    for i in range(len(celltypes)):
        eval_metric = []
        x = target_pred[celltypes[i]]
        y = ground_truth[celltypes[i]]
        eval_metric.append(ccc(x, y))
        eval_metric.append(sqrt(mean_squared_error(x, y)))
        eval_metric.append(pearsonr(x, y)[0])
        plt.subplot(1, len(celltypes)+1, i+2)
        plt.xlim(0, max(y))
        plt.ylim(0, max(y))
        plt.scatter(x, y, s=2)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x,p(x),"r--")
        text = f"$CCC = {eval_metric[0]:0.3f}$\n$RMSE = {eval_metric[1]:0.3f}$\n$Corr = {eval_metric[2]:0.3f}$"
        plt.text(0.05, max(y)-0.05, text, fontsize=8, verticalalignment='top')
        plt.title(celltypes[i])
        plt.xlabel('Prediction')
        plt.ylabel('Ground Truth')
    imgName = os.path.join(SavePath, 'pred_fraction_target_scatter.jpg')
    plt.savefig(imgName)
    plt.close()

def SavetSNEPlot(SavePath, ann_data, output_prex):
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
    sc.tl.pca(ann_data, svd_solver='arpack')
    sc.pp.neighbors(ann_data, n_neighbors=10, n_pcs=20)
    sc.tl.tsne(ann_data)
    with plt.rc_context({'figure.figsize': (8, 8)}):
        sc.pl.tsne(ann_data, color=list(ann_data.obs.columns), color_map='viridis',frameon=False)
        plt.tight_layout()
    plt.savefig(os.path.join(SavePath, output_prex + "_tSNE_plot.jpg"))
    ann_data.write(os.path.join(SavePath, output_prex + ".h5ad"))
    plt.close()
    