<!---
<h1 align="center">
A Simple and Powerful Global Optimization for Unsupervised Video Object Segmentation
</h1>

<div align="center">
<h3>
<a href="http://ponimatkin.github.io">Georgy Ponimatkin</a>,
<a href="https://nerminsamet.github.io">Nermin Samet</a>,
 <a href="https://youngxiao13.github.io">Yang Xiao</a>,
<a href="https://dulucas.github.io/">Yuming Du</a>,
<a href="http://imagine.enpc.fr/~marletr/">Renaud Marlet</a>,
<a href="https://vincentlepetit.github.io/">Vincent Lepetit</a>
<br>
<br>
WACV: Winter Conference on Applications of Computer Vision, 2023
<br>
<br>
<a href="https://arxiv.org/abs/2209.09341">[Paper]</a>
<a href="https://ponimatkin.github.io/ssl-vos/index.html">[Project page]</a>
<br>
</h3>
</div>
-->

# A Simple and Powerful Global Optimization for Unsupervised Video Object Segmentation

This repository contains the official PyTorch implementation of the following paper.

> [**A Simple and Powerful Global Optimization for Unsupervised Video Object Segmentation**](https://arxiv.org/abs/2209.09341),            
> [Georgy Ponimatkin](http://ponimatkin.github.io), [Nermin Samet](https://nerminsamet.github.io), [Yang Xiao](https://youngxiao13.github.io), [Yuming Du](https://dulucas.github.io/), [Renaud Marlet](http://imagine.enpc.fr/~marletr/), [Vincent Lepetit](https://vincentlepetit.github.io),        
> *WACV 2023. ([arXiv pre-print](https://arxiv.org/abs/2209.09341), [Project page](https://ponimatkin.github.io/ssl-vos/index.html))*  
 
## Preparing the environment and data
To prepare the environment run the following commands: 
```
conda env create --name ssl-vos python=3.8 pip
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
pip install scikit-learn opencv-python fast_slic imageio matplotlib scikit-image easydict path.py

python setup.py build
python setup.py install
```
To download the required assets run `bash get_raw_data.sh` and `bash download_models.sh`. Data can be prepared by 
running the following script 
```
python prepare_data.py
```

Please refer to [ARFlow](https://github.com/lliuz/ARFlow) and [RAFT](https://github.com/princeton-vl/RAFT) repositories 
in order to prepare rest of your environment.

## Running the pipeline
Our approach requires three steps to run produce segmentations. At first, extract DINO features and optical flows by via
```
python extract_dino_features.py --dataset davis
python generate_flow_arflow.py --dataset davis --step 1
```

The second step consists of extracting the initial eigenvectors, which can be done by
```
python generate_pic_eigenvectors.py --use-gpu --dataset davis
```

Global optimization then can be run via
```
python global_optimization.py --dataset davis
```

The masks can be generated from the obtained solution by running
```
python extract_masks.py --dataset davis --method name_of_the_folder_in_data/davis
```
<!---
## License

Our code is released under the MIT License (refer to the [LICENSE](readme/LICENSE) file for details). Our codebase is built using codebase of [DINO](https://github.com/facebookresearch/dino), [ARFlow](https://github.com/lliuz/ARFlow), [RAFT](https://github.com/princeton-vl/RAFT) 
and [MoSeg](https://github.com/charigyang/motiongrouping). Please refer to the License of these works for more detail.
-->

## Citation
If you use this code in your research, please cite the following paper:

> G. Ponimatkin, N. Samet, Y. Xiao, Y. Du, R. Marlet and V. Lepetit "A Simple and Powerful Global Optimization for Unsupervised Video Object Segmentation",
> In IEEE Winter Conference on Applications of Computer Vision (WACV), 2023.

BibTeX entry:

```
@inproceedings{ponimatkin2023sslvos, 
title= {A Simple and Powerful Global Optimization for Unsupervised Video Object Segmentation}, 
author={G. {Ponimatkin} and N. {Samet} and Y. {Xiao} and Y. {Du} and R. {Marlet} and V. {Lepetit}}, 
booktitle={Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV)}, 
year={2023}} }
```
