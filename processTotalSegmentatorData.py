# Sort the TotalSegmentator data into a format compatible with nnUNet
import numpy as np
import nibabel as nib
import os
import re
import shutil

local = False
if local:
    input_folder = "/Users/katecevora/Documents/PhD/data/TotalSegmentator"
    output_folder = "/Users/katecevora/Documents/PhD/data/TotalSegmentator"
    images_folder = os.path.join(input_folder, "imagesTr")
    labels_folder = os.path.join(input_folder, "labelsTr")
else:
    input_folder = "/vol/biomedic3/kc2322/data/Totalsegmentator_dataset"
    output_folder = "/vol/biomedic3/kc2322/data/TotalSegmentator_nnUNet/nnUNet_raw/Dataset300_Full"
    images_folder = os.path.join(output_folder, "imagesTr")
    labels_folder = os.path.join(output_folder, "labelsTr")

segmentation_files = ["kidney_right.nii.gz", "kidney_left.nii.gz", "liver.nii.gz", "pancreas.nii.gz"]


labs = ["case_0670.nii.gz",
        "case_1358.nii.gz",
        "case_1124.nii.gz",
        "case_0889.nii.gz",
        "case_1157.nii.gz",
        "case_1024.nii.gz",
        "case_0866.nii.gz"]


def check_label(name):
    lab_nii = nib.load(os.path.join(labels_folder, name))
    lab = lab_nii.get_fdata()

    # Fix mis-labelled regions
    lab[lab > 4] = 4

    lab_fixed_nii = nib.Nifti1Image(lab.astype(np.float32), lab_nii.affine)
    nib.save(lab_fixed_nii, os.path.join(labels_folder, name))


def main():
    # List folders
    fldrs = os.listdir(input_folder)

    # Find folder sxxx
    for fldr in fldrs:
        if re.match(r"s\d{4}", fldr):
            # open the image to determine the size
            print("Processing case {}".format(fldr))
            img_nii = nib.load(os.path.join(input_folder, fldr, "ct.nii.gz"))

            # Create an empty container to store the combined label
            lab = np.zeros(img_nii.shape)

            # Copy and rename ct.nii.gz with name in nnUNet format
            new_img_name = "case_{}_0000.nii.gz".format(fldr[-4:])
            new_lab_name = "case_{}.nii.gz".format(fldr[-4:])

            src = os.path.join(input_folder, fldr, "ct.nii.gz")
            dest = os.path.join(output_folder, "imagesTr", new_img_name)
            shutil.copy(src, dest)

            # Open label files and combine into one
            for i in range(len(segmentation_files)):
                seg_nii = nib.load(os.path.join(input_folder, fldr, "segmentations", segmentation_files[i]))
                seg = seg_nii.get_fdata()

                # Add the segmentation to the label file
                lab += (i+1) * seg

            # Check that we did not have any overlap of labels between different organs
            if np.max(lab) > len(segmentation_files):
                print("We have label overlap!")

                # Fix mis-labelled regions
                lab[lab > 4] = 4

            # Save combined label file in labelsTr
            lab_nii = nib.Nifti1Image(lab.astype(np.float32), seg_nii.affine)
            nib.save(lab_nii, os.path.join(labels_folder, new_lab_name))


if __name__ == "__main__":
    main()