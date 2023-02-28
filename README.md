# RGBDNeRF: Neural Radiance Fields from Sparse RGB-D Images for High-Quality View Synthesis (IEEE TPAMI)

![Teaser image](./img/tpami.jpg)

## Abstract
The recently proposed neural radiance fields (NeRF) use a continuous function formulated as a multi-layer perceptron (MLP) to model the appearance and geometry of a 3D scene. This enables realistic synthesis of novel views, even for scenes with view dependent appearance. Many follow-up works have since extended NeRFs in different ways. However, a fundamental restriction of the method remains that it requires a large number of images captured from densely placed viewpoints for high-quality synthesis and the quality of the results quickly degrades when the number of captured views is insufficient. To address this problem, we propose a novel NeRF-based framework capable of high-quality view synthesis using only a sparse set of RGB-D images, which can be easily captured using cameras and LiDAR sensors on current consumer devices. First, a geometric proxy of the scene is reconstructed from the captured RGB-D images. Renderings of the reconstructed scene along with precise camera parameters can then be used to pre-train a network. Finally, the network is fine-tuned with a small number of real captured images. We further introduce a patch discriminator to supervise the network under novel views during fine-tuning, as well as a 3D color prior to improve synthesis quality. We demonstrate that our method can generate arbitrary novel views of a 3D scene from as few as 6 RGB-D images. Extensive experiments show the improvements of our method compared with the existing NeRF-based methods, including approaches that also aim to reduce the number of input images.

## Requirements and Installation

The code has been tested using the following environment:

* python 3.7
* pytorch >= 1.7.1
* CUDA 11.0

Other dependencies can be installed via

```bash
pip install -r requirements.txt
```

Then,  run

```bash
pip install --editable ./
```

Or if you want to install the code locally, run:

```bash
python setup.py build_ext --inplace
```

## Data

We have prepared a processed data [here](https://drive.google.com/drive/folders/1u_KSUJOROzg0Vx8jqZEeZaFyugeBI57g?usp=sharing) of the scene 'plant'.

## Training and Inference

* For training, please refer to the example script `run_scan_plant.sh`.

* For rendering, please refer to the example script `render.sh`.

## Acknowledgement
This code borrows heavily from [NSVF](https://github.com/facebookresearch/NSVF).

## Citation

If you found this code useful please cite our work as:

```
@article{yuan2022neural,
  title={Neural radiance fields from sparse RGB-D images for high-quality view synthesis},
  author={Yuan, Yu-Jie and Lai, Yu-Kun and Huang, Yi-Hua and Kobbelt, Leif and Gao, Lin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
```
