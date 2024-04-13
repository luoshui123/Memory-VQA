# Memory-VQA
Memory-VQA: Video Quality Assessment of UGC Based on Human Memory System
## Description
This is a repository for the model proposed in the paper "Memory-VQA: Video Quality Assessment of UGC Based on Human Memory System". 
## Usage

### Install Requirements
```
pytorch
opencv
scipy
pandas
torchvision
torchvideo
```

### Download databases
[LSVQ](https://github.com/baidut/PatchVQ)
[KoNViD-1k](http://database.mmsp-kn.de/konvid-1k-database.html)
[Youtube-UGC](https://media.withyoutube.com/)

### Train models on large datasets(LSVQ dataset)
1. Memory Perception
```shell
python -u extract_frame_LSVQ.py
```
2. Memory Encoding
```shell
   python -u extract_SlowFast_features_LSVQ.py 
```
3. Memory Storage, Memory Retrieval and Memory Reconstruction
```shell
 python -u trainSwTransformer.py
```
### Test the model on public datasets
Pre-training weights for the model will be published after the paper is accepted.

Test on the public VQA database
```shell
python -u test_LSVQ.py
```
