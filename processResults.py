import numpy as np
import nibabel as nib
import os
import pickle as pkl

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

    return np.array(dice)


def main():
    # get a list of male and female IDs
    f = open(os.path.join("/Users/katecevora/Documents/PhD/data/TotalSegmentator_nnUNet/", "case_ids.pkl"), "rb")
    [idx_men, idx_women] = pkl.load(f)
    f.close()

    dice_men = []
    dice_women = []

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

            if "s" + case[5:9] in idx_women:
                dice_women.append(dice)
            else:
                dice_men.append(dice)

    dice_men = np.array(dice_men)
    dice_women = np.array(dice_women)

    av_dice_men = np.nanmean(dice_men, axis=1)
    av_dice_women = np.nanmean(dice_women, axis=1)

    print("Average dice for men: \t {0:.3f} \t {1:.3f} \t {2:.3f} \t {3:.3f}".format(av_dice_men[1], av_dice_men[2], av_dice_men[3], av_dice_men[4]))
    print("Average dice for women: \t {0:.3f} \t {1:.3f} \t {2:.3f} \t {3:.3f}".format(av_dice_women[1], av_dice_women[2], av_dice_women[3], av_dice_women[4]))


if __name__ == "__main__":
    main()