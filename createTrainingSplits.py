# A script to create train/test splits from the total segmentator dataset
import pandas as pd
import numpy as np
import os
import pickle as pkl
import shutil

local = False
if local:
    input_folder = "/Users/katecevora/Documents/PhD/data/TotalSegmentator_nnUNet"
    input_images_folder = os.path.join(input_folder, "imagesTr")
    input_labels_folder = os.path.join(input_folder, "labelsTr")

    output_folder = "/Users/katecevora/Documents/PhD/data/TotalSegmentator_nnUnet"
    splits_folder = os.path.join(output_folder, "splits")
else:
    input_folder = "/vol/biomedic3/kc2322/data/TotalSegmentator_nnUNet/nnUNet_raw/Dataset300_Full"
    input_images_folder = os.path.join(input_folder, "imagesTr")
    input_labels_folder = os.path.join(input_folder, "labelsTr")

    output_folder = "/vol/biomedic3/kc2322/data/TotalSegmentator_nnUNet/nnUNet_raw"
    splits_folder = os.path.join("/vol/biomedic3/kc2322/data/TotalSegmentator_nnUNet", "splits")


tr_size = 338
tr_size_half = tr_size / 2
ts_size_half = 50

def generate_sets():
    # Extract relevant meta data and create numpy arrays of male and female IDs
    meta = pd.read_csv(os.path.join(input_folder, "meta.csv"), sep=";")
    ids_m = np.array(meta[meta["gender"] == "m"]["image_id"].values)
    ids_f = np.array(meta[meta["gender"] == "f"]["image_id"].values)

    # open the no foreground images and sort into male and female ids
    f = open(os.path.join(input_folder, "no_fg_ids.pkl"), "rb")
    no_foreground_list = pkl.load(f)
    f.close()

    no_fg_m = []
    no_fg_f = []
    for case in list(no_foreground_list):
        if case in ids_m:
            # add to the no fg list
            no_fg_m.append(case)

            # remove from list of useable ids
            ids_m = np.delete(ids_m, np.where(ids_m == case))
        else:
            # add to the no fg list
            no_fg_f.append(case)

            # remove from list of useable ids
            ids_f = np.delete(ids_f, np.where(ids_f == case))


    # randomly shuffle indices
    np.random.shuffle(ids_m)
    np.random.shuffle(ids_f)

    ids_tr_m = ids_m[:tr_size]
    ids_ts_m = ids_m[tr_size:tr_size + ts_size_half]
    ids_tr_f = ids_f[:tr_size]
    ids_ts_f = ids_f[tr_size:]

    ids_ts = np.concatenate((ids_ts_f, ids_ts_m), axis=0)

    # Set 1 train: 225 men, 225 women
    # Set 1 test: 49 men, 49 women
    ids_tr = np.concatenate((ids_tr_f[:int(tr_size / 2)], ids_tr_m[:int(tr_size / 2)]), axis=0)

    set_1_ids = {"train": ids_tr, "test": ids_ts}
    f = open(os.path.join(splits_folder, "set1_splits.pkl"), "wb")
    pkl.dump(set_1_ids, f)
    f.close()

    # Set 2 train: 0 men, 450 women
    # Set 2 test: 49 men, 49 women
    set_2_ids = {"train": ids_tr_f, "test": ids_ts}
    f = open(os.path.join(splits_folder, "set2_splits.pkl"), "wb")
    pkl.dump(set_2_ids, f)
    f.close()

    # Set 3 train: 450 men, 0 women
    # Set 3 test: 49 men, 49 women
    set_3_ids = {"train": ids_tr_m, "test": ids_ts}
    f = open(os.path.join(splits_folder, "set3_splits.pkl"), "wb")
    pkl.dump(set_3_ids, f)
    f.close()


def copy_images(dataset_name, ids_tr, ids_ts):
    os.mkdir(os.path.join(output_folder, dataset_name))

    output_imagesTr = os.path.join(output_folder, dataset_name, "imagesTr")
    output_labelsTr = os.path.join(output_folder, dataset_name, "labelsTr")
    output_imagesTs = os.path.join(output_folder, dataset_name, "imagesTs")
    output_labelsTs = os.path.join(output_folder, dataset_name, "labelsTs")

    os.mkdir(output_imagesTr)
    os.mkdir(output_labelsTr)
    os.mkdir(output_imagesTs)
    os.mkdir(output_labelsTs)

    # copy over the files from Training Set
    for case in list(ids_tr):
        print("Case {}".format(case))
        img_name = "case_" + case[1:] + "_0000.nii.gz"
        lab_name = "case_" + case[1:] + ".nii.gz"

        # Copy across images
        shutil.copyfile(os.path.join(input_images_folder, img_name), os.path.join(output_imagesTr, img_name))

        # Copy across labels
        shutil.copyfile(os.path.join(input_labels_folder, lab_name), os.path.join(output_labelsTr, lab_name))

    # copy over the files from Test Set
    for case in list(ids_ts):
        img_name = "case_" + case[1:] + "_0000.nii.gz"
        lab_name = "case_" + case[1:] + ".nii.gz"

        # Copy across images
        shutil.copyfile(os.path.join(input_images_folder, img_name), os.path.join(output_imagesTs, img_name))

        # Copy across labels
        shutil.copyfile(os.path.join(input_labels_folder, lab_name), os.path.join(output_labelsTs, lab_name))


def main():
    #generate_sets()

    # Sort the case IDs according to the sets
    # Set1
    f = open(os.path.join(splits_folder, "set1_splits.pkl"), "rb")
    set_1_ids = pkl.load(f)
    f.close()

    ids_tr = set_1_ids["train"]
    ids_ts = set_1_ids["test"]

    print("Working on Set 1....")
    copy_images("Dataset301_Set1", ids_tr, ids_ts)

    # Set2
    f = open(os.path.join(splits_folder, "set2_splits.pkl"), "rb")
    set_2_ids = pkl.load(f)
    f.close()

    ids_tr = set_2_ids["train"]
    ids_ts = set_2_ids["test"]

    print("Working on Set 1....")
    copy_images("Dataset302_Set2", ids_tr, ids_ts)

    # Set3
    f = open(os.path.join(splits_folder, "set3_splits.pkl"), "rb")
    set_3_ids = pkl.load(f)
    f.close()

    ids_tr = set_3_ids["train"]
    ids_ts = set_3_ids["test"]

    print("Working on Set 1....")
    copy_images("Dataset303_Set3", ids_tr, ids_ts)


if __name__ == "__main__":
    main()