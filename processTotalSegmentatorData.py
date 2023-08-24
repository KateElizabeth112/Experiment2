# Sort the TotalSegmentator data into a format compatible with nnUNet
# TotalSegmentator data comprises an image file and a separate segmentation file for each region
# Therefore we need to iterate over segmentation files and add them to a combined array
import numpy as np
import nibabel as nib
import os
import re
import shutil
import SimpleITK as sitk

local = True
if local:
    input_folder = "/Users/katecevora/Documents/PhD/data/TotalSegmentator_raw/Totalsegmentator_dataset"
    output_folder = "/Users/katecevora/Documents/PhD/data/TotalSegmentator"
    images_folder = os.path.join(output_folder, "imagesTr")
    labels_folder = os.path.join(output_folder, "labelsTr")
else:
    input_folder = "/rds/general/user/kc2322/home/data/TotalSegmentator_raw/Totalsegmentator_dataset"
    output_folder = "/rds/general/user/kc2322/home/data/TotalSegmentator/nnUNet_raw/Dataset300_Full"
    images_folder = os.path.join(output_folder, "imagesTr")
    labels_folder = os.path.join(output_folder, "labelsTr")

segmentation_files = ["kidney_right.nii.gz", "kidney_left.nii.gz", "liver.nii.gz", "pancreas.nii.gz"]


labs = ["case_0670.nii.gz",
        "case_1358.nii.gz",
        "case_1124.nii.gz",
        "case_0889.nii.gz",
        "case_1157.nii.gz",
        "case_1024.nii.gz",
        "case_0866.nii.gz",
        "case_1167.nii.gz"]


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

    # Debug
    no_foreground_counter = 0

    # Find folder sxxx
    for fldr in fldrs:
        if re.match(r"s\d{4}", fldr):
            # open the image to determine the size
            print("Processing case {}".format(fldr))

            img_sitk = sitk.ReadImage(os.path.join(input_folder, fldr, "ct.nii.gz"))
            img_np = sitk.GetArrayFromImage(img_sitk)

            # Create an empty container to store the combined label
            lab_np = np.zeros(img_np.shape)

            # Open individual label files and combine into one
            for i in range(len(segmentation_files)):
                # seg_nii = nib.load(os.path.join(input_folder, fldr, "segmentations", segmentation_files[i]))
                seg_sitk = sitk.ReadImage(os.path.join(input_folder, fldr, "segmentations", segmentation_files[i]))

                # seg = seg_nii.get_fdata
                seg_np = sitk.GetArrayFromImage(seg_sitk)

                # Add the segmentation to the label file
                lab_np += (i + 1) * seg_np

            # Check that we have labels for at least one foreground region
            if np.unique(lab_np).shape[0] == 1:
                no_foreground_counter += 1
                print("Background only")
            else:
                # Copy and rename ct.nii.gz with name in nnUNet format
                new_img_name = "case_{}_0000.nii.gz".format(fldr[-4:])
                new_lab_name = "case_{}.nii.gz".format(fldr[-4:])

                # copy across the image to its new destination
                src = os.path.join(input_folder, fldr, "ct.nii.gz")
                dest = os.path.join(output_folder, "imagesTr", new_img_name)
                shutil.copy(src, dest)

                # Check that we did not have any overlap of labels between different organs
                if np.max(lab_np) > len(segmentation_files):
                    print("We have label overlap!")
                    max = int(len(segmentation_files))
                    # Fix mis-labelled regions
                    lab_np[lab_np > max] = max

                # Save combined label file in labelsTr
                lab_sitk = sitk.GetImageFromArray(lab_np)
                lab_sitk.CopyInformation(img_sitk)

                sitk.WriteImage(lab_sitk, os.path.join(labels_folder, new_lab_name))

    print("Number of images with no foreground: {}".format(no_foreground_counter))


if __name__ == "__main__":
    main()