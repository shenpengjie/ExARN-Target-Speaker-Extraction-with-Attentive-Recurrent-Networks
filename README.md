# ExARN-Target-Speaker-Extraction-with-Attentive-Recurrent-Networks
## Guides
This is a project focused on target speaker extraction which is based on pytorch.This paper's primary contributions are twofold: First, it presents a novel attentive fusion module, specifically tailored to enhance the efficiency of using registered speaker information, with the added benefit of significantly reducing computational and parameter requirements relative to the conventional HMSA. Second, it demonstrates that our method surpasses the benchmark model in addressing the SC challenge, achieving superior performance on the wsj0-2mix-extr dataset.

## ExARN system
<div align=center><img src="https://github.com/shenpengjie/ExARN-Target-Speaker-Extraction-with-Attentive-Recurrent-Networks/blob/main/img/model.png"></div>

## Attentive block
<div align=center><img src="https://github.com/shenpengjie/ExARN-Target-Speaker-Extraction-with-Attentive-Recurrent-Networks/blob/main/img/self-attention.png"></div>

## How to start training
***.yaml is the training configuration file, `TR_SPEECH_PATH represents` the training set path, `CV_SPEECH_PATH` represents the validation set path, `TEST_PATH` represents the test set path, and `SPEECH_LST` represents the list of speakers in the training set and validation set.

If you want to train a model from scratch, execute the script: `python train.py -y ARN_nc.yaml `
<br>If you want to load a model and train it, execute the script: `python train.py -y ARN_nc.yaml -m your_model_path`



