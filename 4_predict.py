from nnunetv2.inference.predict_from_raw_data import predict_entry_point as nnUNetv2_predict 
import os
import shutil   

#D:/Users/dwb/anaconda3/envs/torch2.7.1/python.exe g:/nnUnetV2/code/4_predict.py -i G:\nnUnetV2\data_result\nnUNet_raw\Dataset100_CPSegmentation\imagesTr -o G:\nnUnetV2\data_result\nnUNet_raw\Dataset100_CPSegmentation\imagesTr_pred -d 100 -c 2d -f 1

if __name__ == "__main__":

    os.environ['OMP_NUM_THREADS'] = r'1'
    os.environ['MKL_NUM_THREADS'] = r'1'
    os.environ['OPENBLAS_NUM_THREADS'] = r'1'
    # reduces the number of threads used for compiling. More threads don't help and can cause problems
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = r'1'
    # multiprocessing.set_start_method("spawn")
    nnUNetv2_predict()