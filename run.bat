@REM 0.test_prepare_CP_convert_to_png.py 

@REM 0.train_prepare_CP_convert_to_png.py

@REM 1_prepare_CP_for_nnunetv2.py

@REM 2_plan_and_preprocess.py

@REM 3_train_network.py 100 2d all -tr nnUNetTrainerUNetPP


python 4_predict.py -i ../data_result/nnUNet_raw/Dataset101_CPSegmentation/imagesTs -o ../data_result\nnUNet_raw\Dataset101_CPSegmentation\predictions_1 -c 2d -d 101 -f 1


python 4_predict.py -i ../data_result/nnUNet_raw/Dataset101_CPSegmentation/imagesTs -o ../data_result\nnUNet_raw\Dataset101_CPSegmentation\predictions_2 -c 2d -d 101 -f 2


python 4_predict.py -i ../data_result/nnUNet_raw/Dataset101_CPSegmentation/imagesTs -o ../data_result\nnUNet_raw\Dataset101_CPSegmentation\predictions_3 -c 2d -d 101 -f 3


python 4_predict.py -i ../data_result/nnUNet_raw/Dataset101_CPSegmentation/imagesTs -o ../data_result\nnUNet_raw\Dataset101_CPSegmentation\predictions_4 -c 2d -d 101 -f 4


python 4_predict.py -i ../data_result/nnUNet_raw/Dataset101_CPSegmentation/imagesTs -o ../data_result\nnUNet_raw\Dataset101_CPSegmentation\predictions_5 -c 2d -d 101 -f 5


python 4_predict.py -i ../data_result/nnUNet_raw/Dataset101_CPSegmentation/imagesTs -o ../data_result\nnUNet_raw\Dataset101_CPSegmentation\predictions_final -c 2d -d 101 -f all