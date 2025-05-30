# Glioma-Segmentation-Using-GNN
This repository provides a new graph-based deep-learning approach for glioma semantic segmentation. It is based on constructing a graph starting from an MRI scan and then employing a Graph Neural Network.

## Dataset
The dataset used for our experimental activity is publicly available at: [https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).
It contains MRI scans of 110 patients with lower-grade glioma with the relative segmentation mask annotated by medical personnel. 

## Graph Construction
Given an MRI scan and its mask, the steps for the graph building are:
- segmentation of the image into superpixels using the Felzenszwalb algorithm;
- computation of the superpixel features averaging all the pixels within each superpixel.
- construction of a Region Adiacenjy Graph. The nodes are the superpixels; the edges are the spatial relationships between them. 

![Graph](images/graph.png)


## Glioma segmentation
First, we build graphs from MRI images and the related masks, as previously described. We use these graphs to train a very simple GNN, including two GCN layers with a dimensionality of 512 and a final layer with one unit with a Sigmoid activation function to perform the final node classification. We perform Batch Normalization and Dropout between the GCN layers to enhance model generalization and mitigate overfitting.After training the GNN, we perform node classification and build the final segmentation mask. 


![framwwork](images/train_test.jpg)

## Installation

To install the required dependencies, use:

```bash
pip install -r requirements.txt
```
## Usage
To convert an MRI scan into a graph use the following code:
```python
from img2graph import create_graph
img_path="insert the image path"
mask_path="insert the mask path"

scale=1
sigma=0.8
min_size=20

g=create_graph(img_path1,mask_path,scale,sigma,min_size,plot=True)
```

To create the graphs of MRI scans, first download the dataset and place it in a folder named dataset, create  a new folder named graphs and then run the following script:
```bash
python create_graphs.py
```

To replicate our experiment, first, create a new folder named experiment. Inside it, create a folder named gnn and a new folder named results to save the segmention obtained by our method. Lastly, run the following scripts:
```bash
python run_exp.py
python plot_results.py
```

![example](images/image_3.png)

If you use this model please cite:
```
@inproceedings{amato2024semantic,
  title={Semantic Segmentation of Gliomas on Brain MRIs by Graph Convolutional Neural Networks},
  author={Amato, Domenico and Calderaro, Salvatore and {Lo Bosco}, Giosu{\'e} and Rizzo, Riccardo and Vella, Filippo},
  booktitle={2024 International Conference on AI x Data and Knowledge Engineering (AIxDKE)},
  pages={143--149},
  year={2024},
  organization={IEEE}
}
```
