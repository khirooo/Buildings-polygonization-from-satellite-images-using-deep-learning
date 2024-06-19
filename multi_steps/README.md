# A deep learning pipline to tackle the problem of irregular building footprints

## Project Structure

projectRegularization/ \
│\
├── data/\
│ ├── init.py\
│ └── loader.py\
│\
├── models/\
│ ├── init.py\
│ └── networks.py\
│\
├── utils/\
│ ├── init.py\
│ ├── loss_crf.py\
│ └── helpers.py\
│\
├── scripts/\
│ ├── init.py\
│ ├── process_regularization.py\
│ └── train_network.py\
│\
├── checkpoints/\
│\
├── data/\
│ └── train_images/ # Folder containing training images\
│\
├── LICENSE\
├── README.md\
└── README.png\

## Installation

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the GAN model, run:

```bash
python scripts/train_network.py
```

### Regularization

To regularize building footprints, run:

```bash
python scripts/process_regularization.py
```

## Directory Setup

**Training Images**: Place your training images in the data/train_images directory.

**Checkpoints**: Model checkpoints will be saved in the checkpoints directory.


