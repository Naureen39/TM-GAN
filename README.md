TM-GAN
======================

TM-GAN is a Tone Mapping GAN designed for HDR to LDR image conversion. This repository contains the implementation, training scripts, and utilities for TM-GAN.

### Install
```
git clone https://github.com/NaureenM/GAN_TMO
cd ./GAN_TMO
```

### Usage
#### Datasets
TM-GAN is trained on the Pair-Dataset. Ensure the dataset is structured as follows:
```
/dataset
    /train
        hdr_image_1.hdr
        ldr_image_1.png
        ...
    /val
        hdr_image_2.hdr
        ldr_image_2.png
        ...
```
Place the dataset folder in the base directory specified in the configuration file.

#### Transforms
Images are automatically transformed/pre-processed in the dataloader file. This includes resizing, normalization, and augmentation.

#### Training

To train TM-GAN, use the `Training.py` script:
```
python Training.py --config config_supervised.yaml [--gpu GPU_ID]
```

#### Arguments:
- `--config`: Path to the YAML configuration file (required).
- `--gpu`: CUDA device ID (optional, default is 0).

#### Results
**Pair Dataset:**

Results will include:
- Generated LDR images.
- Quantitative metrics such as TMQI scores.

Example results will be added soon.

#### Outputs
- Model checkpoints are saved in the `snapshot/` directory.
- Logs and metrics are stored in TensorBoard format.



## Links

- [Project Repository](https://github.com/Naureen39/TM-GAN)
- [Documentation](#) *(To be added)*
- [Dataset Source](#) *([For testing](http://markfairchild.org/HDR.html))*