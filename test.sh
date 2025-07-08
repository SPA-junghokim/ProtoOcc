CONFIG=ProtoOcc_1key # (ProtoOcc_1key / ProtoOcc_longterm / ProtoOcc_semanticKITTI)

bash tools/dist_test.sh ./projects/configs/${CONFIG}.py ./work_dirs/${CONFIG}/${CONFIG}.pth 1 --eval bboxx