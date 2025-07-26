from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess_entry as nnUNetv2_plan_and_preprocess   


# python.exe 2_plan_and_preprocess.py -d 101

import shutil   

if __name__ == "__main__":
    nnUNetv2_plan_and_preprocess()