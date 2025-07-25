from nnunetv2.inference.predict_from_raw_data import predict_entry_point as nnUNetv2_predict 
import os
import shutil   

if __name__ == "__main__":
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['MKL_NUM_THREADS'] = '1'
    # os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # # reduces the number of threads used for compiling. More threads don't help and can cause problems
    # os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = 1
    # # multiprocessing.set_start_method("spawn")
    nnUNetv2_predict()