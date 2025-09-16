# Image Super-Resolution Project

This project implements a deep learning model for single image super-resolution (SISR). It uses a Lightweight Feature Pyramid Network (LFPN) to upscale low-resolution images.

## Directory Structure

The project follows a specific directory structure for data, models, and results.

```
.
├── data/
│   ├── Train/
│   │   └── HR/
│   │       ├── image1.png
│   │       └── ...
│   └── Validation/
│       └── HR/
│           ├── image1.png
│           └── ...
├── models/
├── results/
├── train.py
└── README.md
```

### Data Directory

The `data` directory is organized as follows. You need to provide your own high-resolution (HR) images for training and validation.

Alternatively, you can use a standard benchmark dataset like **DIV2K**. It is a high-quality dataset commonly used for super-resolution tasks and can be downloaded from the following links:
- **Official Website:** [https://data.vision.ee.ethz.ch/cvl/DIV2K/](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- **GitHub Mirror:** [https://github.com/Benjamin-Wegener/DIV2K](https://github.com/Benjamin-Wegener/DIV2K)

Whether you use your own images or DIV2K, place the high-resolution (HR) images in the corresponding folders. The low-resolution (LR) counterparts are generated automatically during the training process.

```
data
└── Train
    └── HR
        # Contains high-resolution images for training.
        ├── 0001.jpeg
        ├── 0002.jpeg
        └── ...

└── Validation
    └── HR
        # Contains high-resolution images for validation.
        ├── 0001.jpeg
        ├── 0002.jpeg
        └── ...
```

- **`data/Train/HR`**: Place your high-resolution training images in this folder.
- **`data/Validation/HR`**: Place your high-resolution validation images in this folder.

*Note: If these folders do not exist, the `train.py` script will create them automatically on the first run.*

## Requirements

The project is built with Python and PyTorch. You can install the main dependencies using pip:

```bash
pip install torch torchvision pillow
```

## Usage

To start the training process, run the `train.py` script from the root directory of the project:

```bash
python train.py
```

The script will:
1.  Load the training and validation datasets from the `data` directory.
2.  Build the LFPN model.
3.  Start the training loop.
4.  Evaluate the model on the validation set after each epoch.
5.  Save the best-performing model and a log file into a new timestamped directory inside `results/`.

## Results

All training outputs, including the saved model checkpoints (`.pt` files) and log files (`log.txt`), will be stored in a newly created subdirectory within the `results` directory. The subdirectory is named with the timestamp of the training run, for example: `results/2000-01-01_12-00-00/`.