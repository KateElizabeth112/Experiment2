import numpy as np
import pickle as pkl
import nibabel as nib
import os

root_dir = "/Users/katecevora/Documents/PhD/data/TotalSegmentator_nnUNet/nnUNet_raw/Dataset303_Set3"
case = "case_1403.pkl"

def main():
    f = open(os.path.join(root_dir, case), 'rb')
    meta = pkl.load(f)
    f.close()

    data = np.load(os.path.join(root_dir, "case_1403.npz"))

    lst = data.files
    for item in lst:
        print(item)
        print(data[item])

    # check out the label
    lab = nib.load(os.path.join(root_dir, "case_1403.nii.gz"))


    print("Done")

if __name__ == "__main__":
    main()