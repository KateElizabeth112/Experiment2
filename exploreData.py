import numpy as np
import pickle as pkl
import nibabel as nib
import os

root_dir = "/Users/katecevora/Documents/PhD/data/TotalSegmentator_nnUNet/"
images_dir = os.path.join(root_dir, "imagesTr")
labels_dir = os.path.join(root_dir, "labelsTr")



def main():
    cases = os.listdir(labels_dir)

    no_foreground_list = []

    x = 0
    for case in cases:
        if case.endswith(".nii.gz"):
            lab = nib.load(os.path.join(labels_dir, case)).get_fdata()

            if np.unique(lab).shape[0] == 1:
                no_foreground_list.append("s" + case[5:9])

        x += 1
        if x % 10 == 0:
            print(x)

    # Save the list of images with no foreground as an array
    f = open(os.path.join(root_dir, "no_fg_ids.pkl"), "wb")
    pkl.dump(np.array(no_foreground_list), f)
    f.close()

    print("Done")


if __name__ == "__main__":
    main()