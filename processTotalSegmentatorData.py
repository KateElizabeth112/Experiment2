# Sort the TotalSegmentator data into a format compatible with nnUNet
# TotalSegmentator data comprises an image file and a separate segmentation file for each region
# Therefore we need to iterate over segmentation files and add them to a combined array
import numpy as np
import os
import re
import shutil
import SimpleITK as sitk
import pandas as pd
import pickle as pkl

local = False
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


def main():
    # List folders
    fldrs = os.listdir(input_folder)

    # Debug
    no_foreground_counter = 0
    no_foreground_list = []
    no_orthonormal_counter = 0
    no_orthonormal_list = []

    # list for patients with ok image files
    patients = []
    genders = []

    meta = pd.read_csv(os.path.join(input_folder, "meta.csv"), sep=";")
    ids_m = np.array(meta[meta["gender"] == "m"]["image_id"].values)
    ids_f = np.array(meta[meta["gender"] == "f"]["image_id"].values)

    # Find folder sxxx
    for fldr in fldrs:
        if re.match(r"s\d{4}", fldr):
            # open the image to determine the size
            print("Processing case {}".format(fldr))

            try:
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
                    no_foreground_list.append(fldr[-4:])
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

                    # add the patient ID to a list of images with the gender
                    # 0 = male, 1 = female
                    patients.append(fldr[-4:])
                    if ("s" + fldr[-4:]) in ids_m:
                        genders.append(0)
                    elif ("s" + fldr[-4:]) in ids_f:
                        genders.append(1)
                    else:
                        print("Cannot find patient ID in metadata")

            except:
                no_orthonormal_counter += 1
                no_orthonormal_list.append(fldr[-4:])
                continue

    print("Number of images with no foreground: {}".format(no_foreground_counter))
    print("Number of images with no orthonormal: {}".format(no_orthonormal_counter))

    # Save lists
    info = {"patients": patients,
            "genders": genders,
            "no_foreground": no_foreground_list,
            "no_orthonormal": no_orthonormal_list}

    f = open(os.path.join(output_folder, "info.pkl"), "wb")
    pkl.dump(info, f)
    f.close()


if __name__ == "__main__":
    main()