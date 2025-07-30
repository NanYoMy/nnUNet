import sys
from tools.dir import invert_sort_glob as sort_glob
from tools.dir import mkdir_if_not_exist
from visulize.color import *
from visulize.grid_img import generate_img
from visulize.pngutils import SaveNumpy2Png
import os
from PIL import Image
import numpy as np
from medpy.metric import dc, jc

op = SaveNumpy2Png()

# case="CenterB_Case1003"
# case="CenterC_Case1003"
# case="CenterD_Case1003"


all_path = []
# base_dir="F:/dwb/myopsnew"
base_dir = "../data_result/nnUNet_raw/Dataset101_CPSegmentation/"


# for i in ['attunet_pred','IBunet_5_4_3_pred','IBunet_5_4_pred','IBunet_5_pred',"nodeep_pred","plainunet_pred",'predictions_final']:
DSC = {}
IOU = {}
methods = ["plainunet_pred", 'attunet_pred', "nodeep_pred",'IBunet_5_4_3_pred',  'predictions_final','IBunet_5_4_pred','IBunet_5_pred','predictions_3']
centers = ['CenterB', 'CenterC', 'CenterD']

# Initialize dictionaries to store results for each method and center
for i in methods:
    DSC[i] = {}
    IOU[i] = {}
    for center in centers:
        DSC[i][center] = {label: [] for label in range(1, 7)}  # Initialize lists for 6 labels
        IOU[i][center] = {label: [] for label in range(1, 7)}

for i in methods:
    for center in centers:
        for case in range(1001, 1010):
            pred_files = sort_glob(f"{base_dir}/{i}/{center}_Case{case}*png")
            gd_files = sort_glob(f"{base_dir}/labelsTs/{center}_Case{case}*png")

            if not pred_files or not gd_files:
                print(f"Warning: Missing prediction or ground truth files for {center}_Case{case} in {i}. Skipping.")
                continue

            for pred_path, gd_path in zip(pred_files, gd_files):
                try:
                    pred_img = Image.open(pred_path)
                    gd_img = Image.open(gd_path)

                    pred_array = np.array(pred_img)
                    gd_array = np.array(gd_img)

                    # Flatten the arrays for easier calculation
                    pred_flat = pred_array.flatten()
                    gd_flat = gd_array.flatten()

                    # Get unique labels present in either image
                    labels = np.unique(np.concatenate((pred_flat, gd_flat)))
                    labels = labels[(labels != 0)]  # Exclude background label 0

                    for label in range(1, 7):
                        pred_binary = (pred_flat == label).astype(np.bool)
                        gd_binary = (gd_flat == label).astype(np.bool)

                        # Calculate Dice coefficient and IoU
                        dsc_val = dc(gd_binary, pred_binary)
                        iou_val = jc(gd_binary, pred_binary)

                        DSC[i][center][label].append(dsc_val)
                        IOU[i][center][label].append(iou_val)

                    print(f"Calculated DSC and IOU for {center}_Case{case} in {i}")

                except Exception as e:
                    print(f"Error processing {center}_Case{case} in {i}: {e}")
                    continue

# Generate LaTeX table for each center
for center in centers:
    print(f"\\multicolumn{{8}}{{c}}{{\\textbf{{Results for {center}}}}}\\\\")  # LaTeX header for center
    print("\\begin{tabular}{l" + "c" * 7 + "}")
    print("\\toprule")
    print("Method", end="")
    for label in range(1, 7):
        print(f" & Label {label}", end="")
    print(" & Mean \\\\")
    print("\\midrule")

    for i in methods:
        print(i, end="")
        all_labels_dsc = []
        for label in range(1, 7):
            if DSC[i][center][label]:
                mean_dsc = np.mean(DSC[i][center][label])
                print(f" & {mean_dsc:.3f}", end="")
                all_labels_dsc.append(mean_dsc)
            else:
                print(" & - ", end="")
        if all_labels_dsc:
            print(f" & {np.mean(all_labels_dsc):.3f} \\\\")
        else:
            print(" & - \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\n")

numimages = len(all_path)  # Ignore name of program and column input