## Installation

### Prerequisites

- Ubuntu 16.04+
- Python 3.6+
- NumPy 1.19
- PyTorch (tested on 1.4.0)

### Installation

Our code is based on [visualDet3D](https://github.com/Owen-Liuyuxuan/visualDet3D), you can refer to [setup](https://github.com/Owen-Liuyuxuan/visualDet3D) for details. This repo is mainly developed with a single V100 GPU on our local environment (python=3.7, cuda=10.0, pytorch=1.4), and we recommend you to use anaconda to create a vitural environment:

```bash
conda create -n mono python=3.7 pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 numba tqdm -c pytorch -c conda-forge
```

```bash
conda activate mono
```

and other requirements:
```bash
pip3 install -r requirement.txt
```

Lastly, build ops from the root directory (deform convs and iou3d)
```bash
cd #MonoDTR_ROOT
./make.sh
```
