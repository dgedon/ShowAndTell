# Show And Tell

This repository is a reimplementation of the paper [Vinyals et. al. - Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) for the [WASP](https://wasp-sweden.org/) course Scalable Data Science and Distributed Machine Learning.

The goal of the implementation is to generate captions for a natural image. The architecture is based on a encoder decoder structure with a pre-trained ResNet-152 (on ImageNet) as encoder and a LSTM for the decoder, where the image features are included as the initial input. During training an embedding of the true caption are the following LSTM inputs. During inference the first generated token is recursively fed as the next input.

We train the model on MSCOCO 2014. The data can be downloaded by
```bash
mkdir data
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip -P ./data/
wget http://images.cocodataset.org/zips/train2014.zip -P ./data/
wget http://images.cocodataset.org/zips/val2014.zip -P ./data/

unzip ./data/captions_train-val2014.zip -d ./data/
rm ./data/captions_train-val2014.zip
unzip ./data/train2014.zip -d ./data/
rm ./data/train2014.zip 
unzip ./data/val2014.zip -d ./data/ 
rm ./data/val2014.zip 
```

To train the model run the following
```bash
python train.py --folder PATH/TO/OUTPUT/FOLDER
```
This does automatically also generate an image caption for an image placed in the location which can be changed by `--path_test_image PATH/TO/TEST/Image`. All the results are stored in the folder `folder`. Note here that the vocabulary is generated from the training captions and stored in `folder` for future usage. Similar the training and validation images and automatically pre-processed and stored in `\data` for future usage. Hence, running the code the first time directly includes this pre-processing step. Running the code again, does not pre-process but load the already preprocessed data. 

To test the model and generate an image rung
```bash
python train.py -no_training True --path_test_image PATH/TO/TEST/IMAGE
```