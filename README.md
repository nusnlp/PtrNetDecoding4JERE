# PtrNetDecoding4JERE

This repository contains the source code of the paper "Effective Modeling of Encoder-Decoder Architecture for Joint Entity and Relation Extraction" published in AAAI 2020.

### Datasets ###

NYT24 and NYT29 datasets used for experiments in the paper can be downloaded from the following link:

https://drive.google.com/drive/folders/1RPD9kuHUHp4O3gQLLD1CgDPigAlRiR7L?usp=sharing

### Requirements ###

1) python3.5
2) pytorch 1.1.0
3) CUDA 8.0

### How to run ###

#### Word Decoding Model ####

python3.5 word_decoder.py gpu_id random_seed source_data_dir target_data_dit train/test

python3.5 word_decoder.py 0 1023 NYT29/ NYT29/word_decode_model train

python3.5 word_decoder.py 0 1023 NYT29/ NYT29/word_decode_model test

#### Pointer Network-based Decoding Model #### 

python3.5 ptrnet_decoder.py gpu_id random_seed source_data_dir target_data_dit train/test

python3.5 ptrnet_decoder.py 0 1023 NYT29/ NYT29/ptrnet_decode_model train

python3.5 ptrnet_decoder.py 0 1023 NYT29/ NYT29/ptrnet_decode_model test

### Publication ###

https://arxiv.org/abs/1911.09886

If you use the source code or models from this work, please cite our paper:

```
@inproceedings{nayak2019ptrnetdecoding,
  author    = {Nayak, Tapas and Ng, Hwee Tou},
  title     = {Effective Modeling of Encoder-Decoder Architecture for Joint Entity and Relation Extraction},
  booktitle = {Proceedings of The Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2020}
}
```


