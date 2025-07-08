CONFIG=ProtoOcc_1key # (ProtoOcc_1key / ProtoOcc_longterm / ProtoOcc_semanticKITTI)

./tools/dist_train.sh projects/configs/ProtoOcc/${CONFIG}.py 4 --work-dir ./work_dirs/${CONFIG}
