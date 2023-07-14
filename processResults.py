import numpy as np
import nibabel as nib
import os

root_dir = "/Users/katecevora/Documents/PhD/data/TotalSegmentator_nnUNet/inference/"
task = "Task301"
fold = "fold0"

preds_dir = os.path.join(root_dir, task, fold)
gt_dir = "/Users/katecevora/Documents/PhD/data/TotalSegmentator_nnUNet/nnUNet_raw/Dataset301_Set1/labelsTs"


def multiChannelDice(pred, gt, channels):

    dice = []

    for channel in range(channels):
        a = np.zeros(pred.shape)
        a[pred == channel] = 1

        b = np.zeros(gt.shape)
        b[gt == channel] = 1

        dice.append(np.sum(a[b == 1])*2.0 / (np.sum(a) + np.sum(a)))

    return dice


def main():
    # open metadata and get a list of male and female IDs

    cases = os.listdir(preds_dir)
    for case in cases:
        if case.endswith(".nii.gz"):
            print(case)

            pred = nib.load(os.path.join(preds_dir, case)).get_fdata()
            gt = nib.load(os.path.join(gt_dir, case)).get_fdata()

            if np.unique(gt).sum() == 0:
                print("Only background")

            # Get Dice and NSD
            dice = multiChannelDice(pred, gt, 5)

            print(dice)


if __name__ == "__main__":
    main()