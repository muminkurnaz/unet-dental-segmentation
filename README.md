# U-Net Model for Teeth Segmentation

This project explores the potential of the U-Net deep learning model for the segmentation of dental images. The applied model aims to provide dentists with more detailed and precise information about patients' dental structures, optimizing treatment planning.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Contribution](#contribution)
- [License](#license)

## Introduction

Oral health has a significant impact on overall health and quality of life. Advances in dental technology, particularly the use of digital imaging systems, have greatly enhanced the ability to examine patients' oral and dental structures in detail. However, the analysis of this data, diagnosis, and treatment planning processes remain manual and time-consuming.

Segmentation of X-ray images plays a crucial role in dental practice, facilitating the identification of teeth, tissues, and potential diseases, thus improving diagnostic accuracy. This project deeply examines the impact of the U-Net model on dental segmentation, providing faster, more accurate, and objective results to dentists.

## Features

- High-accuracy teeth segmentation.
- Achieved Jaccard index of 88.99% and Dice coefficient of 94.18%.
- Accuracy rate of 97.95% and F1 Score of 94.18%.
- Enhanced diagnostic accuracy and treatment planning for dentists.

## Architecture

The U-Net model architecture is described in the `unet_model_architecture.py` file, detailing the layers and structure of the model.


## Usage

### Preprocessing
The `veri_onislem.ipynb` notebook contains code for preprocessing the dataset.

### Training
To train the model, use the following command:
python unet_teeth_segmentation_code.py --mode train --data_path path_to_data --epochs num_epochs --batch_size batch_size

### Testing
To test the model, use the following command:
python unet_teeth_segmentation_code.py --mode test --model_path path_to_model --data_path path_to_data

### Jupyter Notebook

or an interactive approach, use the `unet_teeth_segmentation_code.ipynb` notebook which contains all the code for training and testing the model.

## Results
The experimental results demonstrate that the proposed model achieved impressive performance on the test dataset:

- Jaccard Index: 88.99%
- Dice Coefficient: 94.18%
- Accuracy: 97.95%
- F1 Score: 94.18%

These results indicate the model's high performance and segmentation success. `Results`


## Contribution
We welcome contributions! Please fork this repository and create a pull request or open an issue to discuss any improvements or suggestions.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more information.



