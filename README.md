# Retinal Blood Vessel Segmentation Algorithm
The purpose of this project is to train autoencoder model that can classify pixels from retinal images into two categories -- blood vessel and background.

## Double-Attention, Residual, Efficient U-Net
Model architecture is based on U-Net, which is improved by implementing residual connections and attention modules.
![DARE U-NET](architectures/DAREUNET.png "DARE U-Net")


## Results
Model performance was evaluated on 16 different databases which vary in terms of resolution, brightness and FOV. Most datasets use narrow-angle images, and 4 datasets are obtained by wide-angle funduscameras. Presented results were achieved on fine tuned models.

|Database       | Accuracy | Precision | Recall | Specificity | F1-score | IoU    |
|---            |---       |---        |---     | ---         | ---      | ---    |
|CHASE\_DB1     | 0.9742   | 0.7826    | 0.8477 | 0.9833      | 0.8133   | 0.6854 |
|AFIO           | 0.9613   | 0.7305    | 0.7566 | 0.978       | 0.7402   | 0.5897 |
|ORVS           | 0.976    | 0.7394    | 0.795  | 0.9854      | 0.7659   | 0.622  |
|LES-AV         | 0.979    | 0.8998    | 0.8102 | 0.9927      | 0.8527   | 0.7432 |
|STARE          | 0.9633   | 0.7195    | 0.8108 | 0.9755      | 0.7612   | 0.6204 |
|DRIVE          | 0.9684   | 0.7923    | 0.8236 | 0.9813      | 0.8068   | 0.6764 |
|DualModal2019  | 0.9793   | 0.8734    | 0.823  | 0.991       | 0.8468   | 0.7348 |
|IOSTAR         | 0.9725   | 0.8259    | 0.796  | 0.9863      | 0.8083   | 0.6792 |
|TREND          | 0.9706   | 0.6599    | 0.6804 | 0.9839      | 0.6677   | 0.5018 |
|DR~HAGIS       | 0.9817   | 0.7733    | 0.7737 | 0.9906      | 0.7729   | 0.632  |
|HRF            | 0.9735   | 0.8364    | 0.8585 | 0.9844      | 0.846    | 0.734  |
|ARIA           | 0.9528   | 0.6803    | 0.7834 | 0.9673      | 0.727    | 0.5719 |
|VAMPIRE        | 0.9827   | 0.7104    | 0.7964 | 0.989       | 0.751    | 0.6013 |
|RECOVERY       | 0.9627   | 0.8046    | 0.8046 | 0.9794      | 0.8046   | 0.6731 |
|PRIME-FP20     | 0.9947   | 0.7832    | 0.7592 | 0.9975      | 0.7664   | 0.6228 |
|Own dataset    | 0.9867   | 0.8934    | 0.849  | 0.9944      | 0.8707   | 0.771  |


## Application
Models can be trained or used via specialized methods, or by running `main.py`. Configurations related to image preprocessing, filepaths other hyperparameters can be configured via `config.py`.

To run application, create folders with datasets in `Datasets/` directory. Image division and augmentation will be done automatically and saved in folder declared in `config.py`. 

Program can be run with IDE or via CLI. Supported arguments are listed in `utils.py`.

## Pre-trained models
Repository contains pre-trained models which can be used for inference or further fine-tuning. To run segmentation task you have to adjust `config.py` to match pre-trained model filepath and set output images directory. Then run `test.py`. Console will display segmentation metrics and segmentation results with annotated missclafficiations will be saved in specified directory.