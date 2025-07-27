from nnunetv2.run.run_training import run_training_entry as nnUNetv2_train  
import os
import shutil   

#python.exe 3_train_network.py 100 2d all
#python.exe 3_train_network.py 100 2d 1

if __name__ == "__main__":
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['MKL_NUM_THREADS'] = '1'
    # os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # # reduces the number of threads used for compiling. More threads don't help and can cause problems
    # os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = 1
    # # multiprocessing.set_start_method("spawn")
    nnUNetv2_train()