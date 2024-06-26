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

These results indicate the model's high performance and segmentation success.


## Contribution
We welcome contributions! Please fork this repository and create a pull request or open an issue to discuss any improvements or suggestions.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more information.











## GÖRSELLER
![çıktı1](https://github.com/muminkurnaz/unet-dental-segmentation/assets/112796390/9e04e3f4-63dc-42bc-b0f7-7d1d2011f429)
![accuracy](https://github.com/muminkurnaz/unet-dental-segmentation/assets/112796390/f5dbac7f-6c4b-4e5b-8bec-4bbc129740d7)
![metrikler](https://github.com/muminkurnaz/unet-dental-segmentation/assets/112796390/b32a7031-eaca-4cbe-b9d3-e2a59f93aab8)
![loss](https://github.com/muminkurnaz/unet-dental-segmentation/assets/112796390/2bcf752c-8d8c-40ad-a409-671ebfdd48c3)
![eğitim sonuçları](https://github.com/muminkurnaz/unet-dental-segmentation/assets/112796390/3efff0a5-9f78-46e2-ab64-451bf7b1c08f)
![çıktı7](https://github.com/muminkurnaz/unet-dental-segmentation/assets/112796390/4a3e129f-5f1e-4a44-b8a6-c8556d514b57)
![çıktı6](https://github.com/muminkurnaz/unet-dental-segmentation/assets/112796390/14e77c7b-af7c-481b-b558-a86aaad914f1)
![çıktı5](https://github.com/muminkurnaz/unet-dental-segmentation/assets/112796390/b9afb45e-8148-4baa-b4e3-65a6b8a39c64)
![çıktı4](https://github.com/muminkurnaz/unet-dental-segmentation/assets/112796390/afa98ab2-72f5-49f8-b6aa-25bd3c55a8d2)
![çıktı3](https://github.com/muminkurnaz/unet-dental-segmentation/assets/112796390/5fda6e13-e6d1-409c-b7f4-461db2b9c973)
![çıktı2](https://github.com/muminkurnaz/unet-dental-segmentation/assets/112796390/597d217c-3a3b-4f45-800f-c93c27629e6f)
