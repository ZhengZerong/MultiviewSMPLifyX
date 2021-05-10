## Multiview SMPLify-x

This is a multiview version of [SMPLify-x](https://smpl-x.is.tue.mpg.de/). 

I borrowed the code from the [official implementation](https://github.com/vchoutas/smplify-x), cleaned it, simplified it, and extended it to multiview setup. 

It is used for fitting SMPL models to 3D human scans in our [PaMIR](https://github.com/ZhengZerong/PaMIR) project. 


## License
As this repo is mainly borrowed from [SMPLify-x](https://github.com/vchoutas/smplify-x), it is released under the original license. 

Software Copyright License for **non-commercial scientific research purposes**.
Please read carefully the [terms and conditions](https://github.com/vchoutas/smplx/blob/master/LICENSE) and any accompanying documentation before you download and/or use the SMPL-X/SMPLify-X model, data and software, (the "Model & Software"), including 3D meshes, blend weights, blend shapes, textures, software, scripts, and animations. By downloading and/or using the Model & Software (including downloading, cloning, installing, and any other use of this github repository), you acknowledge that you have read these terms and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not download and/or use the Model & Software. Any infringement of the terms of this agreement will automatically terminate your rights under this [License](https://github.com/vchoutas/smplify-x/blob/master/LICENSE).

## Installation
First, clone this repo and install dependencies:
```cmd
git clone https://github.com/ZhengZerong/MultiviewSMPLifyX
cd MultiviewSMPLifyX
pip install -r requirements.txt
```

Then you need to follow the instructions of [SMPL-X](https://github.com/vchoutas/smplx#downloading-the-model) and prepare the SMPL model files in ```smplx/models```. 
The downloaded SMPL files should be cleaned according to the instruction at [this page](https://github.com/vchoutas/smplx/tree/master/tools). The final directory structure should look like this:
```
smplx
└── models
    └── smpl
        ├── SMPL_FEMALE.pkl
        ├── SMPL_MALE.pkl
        └── SMPL_NEUTRAL.pkl
```

The last step is preparing the trained VPoser models. Please download them from [this webpage](https://smpl-x.is.tue.mpg.de/) and place the ```*.pt``` files in ```vposer/models/snapshots/```. 

## Usage
This repo is used for fitting SMPL models to 3D human scans in our [PaMIR](https://github.com/ZhengZerong/PaMIR) project. 

To use it, please render the 3D scan from multiple viewpoints using the dataset generation code of PaMIR, 
and use [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to detect keypoints on the rendered images; 
see```./dataset_example``` for an example.
After that, run:
```bash
python main.py --config cfg_files/fit_smpl.yaml --data_folder ./dataset_example/image_data/rp_dennis_posed_004 --output_folder ./dataset_example/mesh_data/rp_dennis_posed_004/smpl
```


## Citation

If you find this Model & Software useful in your research we would kindly ask you to cite:

```
@ARTICLE{zheng2020pamir,
  author={Zheng, Zerong and Yu, Tao and Liu, Yebin and Dai, Qionghai},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={PaMIR: Parametric Model-Conditioned Implicit Representation for Image-based Human Reconstruction}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3050505}}


@inproceedings{SMPL-X:2019,
  title = {Expressive Body Capture: 3D Hands, Face, and Body from a Single Image},
  author = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}
```
