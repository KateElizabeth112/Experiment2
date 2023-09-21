# A script to create train/test splits from the total segmentator dataset
import pandas as pd
import numpy as np
import os
import pickle as pkl
import shutil

local = False

if local:
    root_folder = "/Users/katecevora/Documents/PhD/data/TotalSegmentator"
else:
    root_folder = "/rds/general/user/kc2322/home/data/TotalSegmentator/"

input_folder = os.path.join(root_folder, "nnUNet_raw/Dataset300_Full")
output_folder = os.path.join(root_folder, "nnUNet_raw")
input_images_folder = os.path.join(input_folder, "imagesTr")
input_labels_folder = os.path.join(input_folder, "labelsTr")
splits_folder = os.path.join(root_folder, "splits")

output_datasets = ["Dataset401_Set1", "Dataset402_Set2", "Dataset403_Set3"]
splits = ["set401_splits.pkl", "set402_splits.pkl", "set403_splits.pkl"]


def generate_folds():
    f = open(os.path.join(root_folder, "info.pkl"), "rb")
    info = pkl.load(f)
    f.close()

    patients = np.array(info["patients"])
    genders = np.array(info["genders"])       # male = 0, female = 1

    # split into male and female IDs
    ids_m = patients[genders == 0]
    ids_f = patients[genders == 1]

    # randomly shuffle indices
    np.random.shuffle(ids_m)
    np.random.shuffle(ids_f)

    block_size = np.floor(ids_f.shape[0] / 9)
    dataset_size = int(block_size * 9)

    print("Dataset size: {}".format(dataset_size))

    # create 9 training blocks overall (these will form 5 folds)
    blocks_f = []
    blocks_m = []

    for i in range(9):
        blocks_f.append(ids_f[i*block_size:(i+1)*block_size])
        blocks_m.append(ids_m[i * block_size:(i + 1) * block_size])

    # create 5 training folds for three datasets
    # fold 0
    ts = np.concatenate((blocks_f[0], blocks_m[0]), axis=0)
    tr1_f = np.concatenate((blocks_f[0:0], blocks_f[1:5]), axis=0)
    tr1_m = np.concatenate((blocks_m[0:0], blocks_m[1:5]), axis=0)
    tr1 = np.concatenate((tr1_f, tr1_m), axis=0)

    tr2 = np.concatenate((blocks_f[0:0], blocks_f[1:9]), axis=0)
    tr3 = np.concatenate((blocks_m[0:0], blocks_m[1:9]), axis=0)

    print(tr1.shape, tr2.shape, tr3.shape, ts.shape)

    # fold 1
    ts = np.concatenate((blocks_f[1], blocks_m[1]), axis=0)
    tr1_f = np.concatenate((blocks_f[0:1], blocks_f[2:5]), axis=0)
    tr1_m = np.concatenate((blocks_m[0:1], blocks_m[2:5]), axis=0)
    tr1 = np.concatenate((tr1_f, tr1_m), axis=0)

    tr2 = np.concatenate((blocks_f[0:1], blocks_f[2:9]), axis=0)
    tr3 = np.concatenate((blocks_m[0:1], blocks_m[2:9]), axis=0)

    print(tr1.shape, tr2.shape, tr3.shape, ts.shape)

    # fold 2
    ts = np.concatenate((blocks_f[2], blocks_m[2]), axis=0)
    tr1_f = np.concatenate((blocks_f[0:2], blocks_f[3:5]), axis=0)
    tr1_m = np.concatenate((blocks_m[0:2], blocks_m[3:5]), axis=0)
    tr1 = np.concatenate((tr1_f, tr1_m), axis=0)

    tr2 = np.concatenate((blocks_f[0:2], blocks_f[3:9]), axis=0)
    tr3 = np.concatenate((blocks_m[0:2], blocks_m[3:9]), axis=0)

    print(tr1.shape, tr2.shape, tr3.shape, ts.shape)

    # fold 3
    ts = np.concatenate((blocks_f[3], blocks_m[3]), axis=0)
    tr1_f = np.concatenate((blocks_f[0:3], blocks_f[4:5]), axis=0)
    tr1_m = np.concatenate((blocks_m[0:3], blocks_m[4:5]), axis=0)
    tr1 = np.concatenate((tr1_f, tr1_m), axis=0)

    tr2 = np.concatenate((blocks_f[0:3], blocks_f[4:9]), axis=0)
    tr3 = np.concatenate((blocks_m[0:3], blocks_m[4:9]), axis=0)

    print(tr1.shape, tr2.shape, tr3.shape, ts.shape)

    # fold 4
    ts = np.concatenate((blocks_f[4], blocks_m[4]), axis=0)
    tr1_f = np.concatenate((blocks_f[0:4], blocks_f[5:5]), axis=0)
    tr1_m = np.concatenate((blocks_m[0:4], blocks_m[5:5]), axis=0)
    tr1 = np.concatenate((tr1_f, tr1_m), axis=0)

    tr2 = np.concatenate((blocks_f[0:4], blocks_f[5:9]), axis=0)
    tr3 = np.concatenate((blocks_m[0:4], blocks_m[5:9]), axis=0)

    print(tr1.shape, tr2.shape, tr3.shape, ts.shape)



def generate_sets():
    f = open(os.path.join(input_folder, "info.pkl"), "rb")
    info = pkl.load(f)
    f.close()

    patients = np.array(info["patients"])
    genders = np.array(info["genders"])       # male = 0, female = 1

    # split into male and female IDs
    ids_m = patients[genders == 0]
    ids_f = patients[genders == 1]

    # randomly shuffle indices
    np.random.shuffle(ids_m)
    np.random.shuffle(ids_f)

    # define training and test set size
    n_f = ids_f.shape[0]
    ts_size = 100
    tr_size = int(n_f - (ts_size / 2))

    print("Training set size: {}".format(tr_size))
    print("Test set size: {}".format(ts_size))

    ids_tr_m = ids_m[:tr_size]
    ids_ts_m = ids_m[tr_size:tr_size + int(ts_size / 2)]
    ids_tr_f = ids_f[:tr_size]
    ids_ts_f = ids_f[tr_size:]

    ids_ts = np.concatenate((ids_ts_f, ids_ts_m), axis=0)

    # Set 1 train: 225 men, 225 women
    # Set 1 test: 49 men, 49 women
    ids_tr = np.concatenate((ids_tr_f[:int(tr_size / 2)], ids_tr_m[:int(tr_size / 2)]), axis=0)

    set_1_ids = {"train": ids_tr, "test": ids_ts}
    f = open(os.path.join(splits_folder, splits[0]), "wb")
    pkl.dump(set_1_ids, f)
    f.close()

    # Set 2 train: 0 men, 450 women
    # Set 2 test: 49 men, 49 women
    set_2_ids = {"train": ids_tr_f, "test": ids_ts}
    f = open(os.path.join(splits_folder, splits[1]), "wb")
    pkl.dump(set_2_ids, f)
    f.close()

    # Set 3 train: 450 men, 0 women
    # Set 3 test: 49 men, 49 women
    set_3_ids = {"train": ids_tr_m, "test": ids_ts}
    f = open(os.path.join(splits_folder, splits[2]), "wb")
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
        img_name = "case_" + case + "_0000.nii.gz"
        lab_name = "case_" + case + ".nii.gz"

        # Copy across images
        shutil.copyfile(os.path.join(input_images_folder, img_name), os.path.join(output_imagesTr, img_name))

        # Copy across labels
        shutil.copyfile(os.path.join(input_labels_folder, lab_name), os.path.join(output_labelsTr, lab_name))

    # copy over the files from Test Set
    for case in list(ids_ts):
        img_name = "case_" + case + "_0000.nii.gz"
        lab_name = "case_" + case + ".nii.gz"

        # Copy across images
        shutil.copyfile(os.path.join(input_images_folder, img_name), os.path.join(output_imagesTs, img_name))

        # Copy across labels
        shutil.copyfile(os.path.join(input_labels_folder, lab_name), os.path.join(output_labelsTs, lab_name))


def main():
    #generate_sets()
    generate_folds()

    # Sort the case IDs according to the sets
    #for j in range(3):
        #f = open(os.path.join(splits_folder, splits[j]), "rb")
        #ids = pkl.load(f)
        #f.close()

        #ids_tr = ids["train"]
        #ids_ts = ids["test"]

        #print("Working on Set {}....".format(j))
        #copy_images(output_datasets[j], ids_tr, ids_ts)



if __name__ == "__main__":
    main()