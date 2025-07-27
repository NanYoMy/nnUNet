import sys
from tools.dir import invert_sort_glob as  sort_glob
from tools.dir import mkdir_if_not_exist
from visulize.color import *
from visulize.grid_img import generate_img
from visulize.pngutils import SaveNumpy2Png
import os

op =SaveNumpy2Png()

case="1002"

all_path=[]
# base_dir="F:/dwb/myopsnew"
base_dir= "../data_result/nnUNet_raw/Dataset101_CPSegmentation/"

ori_c0= sort_glob(f"{base_dir}/imagesTs/CenterB_Case{case}*png")
assert  len(ori_c0)>0
gd= sort_glob(f"{base_dir}/labelsTs/CenterB_Case{case}*png")

numrows=0
all_path=[]

#金标准
tmp = op.merge_two_png_imgs_with_text(ori_c0, gd, gd)
all_path.extend(tmp)

for i in ['predictions_final']:

    pred= sort_glob(f"{base_dir}/{i}/CenterB_Case{case}*png")
    numrows=len(pred)
    print(f"pred len:{len(pred)}")
    tmp = op.merge_two_png_imgs_with_text(ori_c0, pred, gd)

    all_path.extend(tmp)


numimages = len(all_path) # Ignore name of program and column input

numcols=numimages//numrows

generate_img(all_path, numrows, numcols, f"{base_dir}/comparison.png")
print(f"{base_dir}/comparison.png")

