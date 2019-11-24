# fake-news-detection
Training a neural net to detect fake news on social media.

## Background
We are recreating the [EANN model](https://doi.org/10.1145/3219819.3219903) first outlined by Wang et al. at KDD.

## Data
Twitter data: https://github.com/MKLab-ITI/image-verification-corpus

## Preprocessing
As outlined in the EANN paper, the neural net was trained on tweets. This data come from the MediaEval 2016 dataset which contains information on tweets--including post features and user features.

This corpus is first stripped of any tweets without any text or without any images--as this net is trained to detect fake tweets by incorporating both text and image. 
