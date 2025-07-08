## Training
We trained all models using four RTX 3090 (24GB) GPUs.
```
CONFIG=ProtoOcc_1key # (ProtoOcc_1key / ProtoOcc_longterm / ProtoOcc_semanticKITTI)

./tools/dist_train.sh projects/configs/ProtoOcc/${CONFIG}.py 4 --work-dir ./work_dirs/${CONFIG}
```

## Evaluation
If you want to get the pretrained weights, download them from [Here](https://drive.google.com/drive/folders/1-hHITEyUVnbEHaI80u6C6ZiUmdXLoFjy?usp=drive_link).  
To measure inference speed, uncomment `# fp16 = dict(loss_scale='dynamic')` in the config file.  
```
CONFIG=ProtoOcc_1key # (ProtoOcc_1key / ProtoOcc_longterm / ProtoOcc_semanticKITTI)

bash tools/dist_test.sh ./projects/configs/${CONFIG}.py ./work_dirs/${CONFIG}/${CONFIG}.pth 1 --eval bboxx
```

