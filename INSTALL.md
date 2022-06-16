## Installation

### Prerequisites

- Ubuntu 16.04+
- Python 3.6+
- NumPy 1.19
- PyTorch (tested on 1.4.0)

### Installation

Our code is based on [visualDet3D](https://github.com/Owen-Liuyuxuan/visualDet3D), you can refer to [setup](https://github.com/Owen-Liuyuxuan/visualDet3D) for details. This repo is mainly developed with a single V100 GPU on our local environment (python=3.7, cuda=10.0, pytorch=1.4), and we recommend you to use anaconda to create a vitural environment:

```bash
conda create -n monodtr python=3.7
conda activate monndtr
```

Install PyTorch:

```bash
pip3 install torch==1.4.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html
```

and other requirements:
```bash
pip3 install -r requirements.txt
```

Lastly, build ops (deform convs and iou3d)
```bash
cd #MonoDTR_ROOT
./make.sh
```
