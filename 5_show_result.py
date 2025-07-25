import sys
from tools.dir import invert_sort_glob as  sort_glob
from tools.dir import mkdir_if_not_exist
from visulize.color import *
from visulize.grid_img import generate_img
from visulize.pngutils import SaveNumpy2Png
import os

op =SaveNumpy2Png()

case="1001"

all_path=[]
# base_dir="F:/dwb/myopsnew"
base_dir= "../data_result/nnUNet_raw/Dataset100_CPSegmentation/"

ori_c0= sort_glob(f"{base_dir}/imagesTr/CenterA_Case{case}*png")
assert  len(ori_c0)>0
gd= sort_glob(f"{base_dir}/labelsTr/CenterA_Case{case}*png")


numrows=0
all_path=[]

#图像
# tmp=op.save_imgs(ori_c0)
# all_path.extend(tmp)

#金标准
tmp = op.merge_two_png_imgs_with_text(ori_c0, gd, gd)
all_path.extend(tmp)

for i in ['imagesTr_pred']:

    pred= sort_glob(f"{base_dir}/{i}/output_{case}.nii.gz")
    numrows=len(pred)
    print(f"pred len:{len(pred)}")
    tmp = op.merge_two_png_imgs_with_text(ori_c0, pred, gd)

    all_path.extend(tmp)



# gd= sort_glob(f"{base_dir}/gd/Case{subject_ID}*.nii.gz")



numimages = len(all_path) # Ignore name of program and column input

numcols=numimages//numrows

generate_img(all_path, numrows, numcols, f"{base_dir}/comparison.png")
print(f"{base_dir}/comparison.png")


#自动打开输出文件夹
output_dir = os.path.dirname(f"{base_dir}/comparison.png")
if os.path.exists(output_dir):
    os.system(f"explorer {output_dir}")  # Windows系统使用explorer
else:
    print(f"Error: Directory not found: {output_dir}")