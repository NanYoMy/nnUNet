import os
import SimpleITK as sitk
import numpy as np
from PIL import Image
from tools.dir import mk_or_cleardir

# Set input and output directories
input_dir = '../datasets/CP'
output_dir_input = '../datasets/CP_PNG/training/input'
output_dir_output = '../datasets/CP_PNG/training/output'


# Create output directories if they don't exist
mk_or_cleardir(output_dir_input)
mk_or_cleardir(output_dir_output)


# Loop through subdirectories (CenterA-D)
for center_dir in os.listdir(input_dir):
    center_dir_path = os.path.join(input_dir, center_dir)
    if os.path.isdir(center_dir_path):
        # Loop through NIfTI files in each subdirectory
        for file in os.listdir(center_dir_path):
            if file.endswith('.nii.gz'):
                # Load the NIfTI file using SimpleITK
                img = sitk.ReadImage(os.path.join(center_dir_path, file))
                data = sitk.GetArrayFromImage(img)

                # Get the sagittal slices
                # sagittal_slices = np.rot90(data, 1, (0, 2))

                # Save the slices as PNG images
                for i, slice in enumerate(data):
                    # For image files, convert slice to uint8 and scale to 0-255 if it's a float array
                    if not file.endswith('_gd.nii.gz'):
                        slice = (slice - slice.min()) / (slice.max() - slice.min()) * 255
                        slice = slice.astype(np.uint8)
                        

                    img = Image.fromarray(slice)


                    if file.endswith('_gd.nii.gz'):
                        img = img.convert('L')
                        output_file = os.path.join(output_dir_output, f'{center_dir}_{file[:-10]}_s{i:03d}.png')
                    else:
                        img = img.convert('RGB')
                        output_file = os.path.join(output_dir_input, f'{center_dir}_{file[:-7]}_s{i:03d}.png')
                    img.save(output_file)