
# ViFin

An implementation of ViFin

Wenqiang Chen, Lin Chen, Meiyi Ma, Farshid Salemi Parizi, Shwetak Patel, and John Stankovic. 2021. ViFin: Harness Passive Vibration to Continuous Micro Finger Writing with a Commodity Smartwatch. Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 5, 1, Article 46 (March 2021), 24 pages. https://doi.org/10.1145/3448119

## Dependencies

```
python==3.6.7
TensorFlow==2.2.0
NumPy==1.19.0
hdf5storage==0.1.18
```

## Data Processing

Run `matlab/DataProcessing.m` with Matlab to generate data for training the model.

## Run

After generating processed data, run python/main.py to train the model.

**Noted that the code can only be used for personal learning.**
