@REM 0.test_prepare_CP_convert_to_png.py 

@REM 0.train_prepare_CP_convert_to_png.py

@REM 1_prepare_CP_for_nnunetv2.py

@REM 2_plan_and_preprocess.py

@REM python 3_train_network.py 100 2d all -tr nnUNetTrainerAttUNet
@REM python 3_train_network.py 100 2d all -tr nnUNetTrainerUNetPlain
@REM python 3_train_network.py 100 2d all -tr nnUNetTrainerIBUNet_5_4_3
@REM python 3_train_network.py 100 2d all -tr nnUNetTrainerIBUNet_5_4
@REM python 3_train_network.py 100 2d all -tr nnUNetTrainerIBUNet_5
@REM python 3_train_network.py 100 2d all -tr nnUNetTrainerNoDeepSupervision

@REM python 3_train_network.py 100 2d all -tr nnUNetTrainerBN

@REM python 4_predict.py -i ../data_result/nnUNet_raw/Dataset100_CPSegmentation/imagesTs -o ../data_result\nnUNet_raw\Dataset100_CPSegmentation\attunet_pred -c 2d -d 100 -f all  -tr nnUNetTrainerAttUNet


python 4_predict.py -i ../data_result/nnUNet_raw/Dataset101_CPSegmentation/imagesTs -o ../data_result\nnUNet_raw\Dataset101_CPSegmentation\nodeep_pred -c 2d -d 101 -f all  -tr nnUNetTrainerNoDeepSupervision

python 4_predict.py -i ../data_result/nnUNet_raw/Dataset101_CPSegmentation/imagesTs -o ../data_result\nnUNet_raw\Dataset101_CPSegmentation\attunet_pred -c 2d -d 101 -f all  -tr nnUNetTrainerAttUNet

python 4_predict.py -i ../data_result/nnUNet_raw/Dataset101_CPSegmentation/imagesTs -o ../data_result\nnUNet_raw\Dataset101_CPSegmentation\plainunet_pred -c 2d -d 101 -f all  -tr nnUNetTrainerUNetPlain

python 4_predict.py -i ../data_result/nnUNet_raw/Dataset101_CPSegmentation/imagesTs -o ../data_result\nnUNet_raw\Dataset101_CPSegmentation\IBunet_5_4_3_pred -c 2d -d 101 -f all  -tr nnUNetTrainerIBUNet_5_4_3

python 4_predict.py -i ../data_result/nnUNet_raw/Dataset101_CPSegmentation/imagesTs -o ../data_result\nnUNet_raw\Dataset101_CPSegmentation\IBunet_5_4_pred -c 2d -d 101 -f all  -tr nnUNetTrainerIBUNet_5_4

python 4_predict.py -i ../data_result/nnUNet_raw/Dataset101_CPSegmentation/imagesTs -o ../data_result\nnUNet_raw\Dataset101_CPSegmentation\IBunet_5_pred -c 2d -d 101 -f all  -tr nnUNetTrainerIBUNet_5

@REM python 4_predict.py -i ../data_result/nnUNet_raw/Dataset100_CPSegmentation/imagesTs -o ../data_result\nnUNet_raw\Dataset100_CPSegmentation\predictions_2 -c 2d -d 100 -f 2


@REM python 4_predict.py -i ../data_result/nnUNet_raw/Dataset100_CPSegmentation/imagesTs -o ../data_result\nnUNet_raw\Dataset100_CPSegmentation\predictions_3 -c 2d -d 100 -f 3


@REM python 4_predict.py -i ../data_result/nnUNet_raw/Dataset100_CPSegmentation/imagesTs -o ../data_result\nnUNet_raw\Dataset100_CPSegmentation\predictions_4 -c 2d -d 100 -f 4


@REM python 4_predict.py -i ../data_result/nnUNet_raw/Dataset100_CPSegmentation/imagesTs -o ../data_result\nnUNet_raw\Dataset100_CPSegmentation\predictions_5 -c 2d -d 100 -f 5


@REM python 4_predict.py -i ../data_result/nnUNet_raw/Dataset100_CPSegmentation/imagesTs -o ../data_result\nnUNet_raw\Dataset100_CPSegmentation\predictions_final -c 2d -d 100 -f all