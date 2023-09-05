# Script to calculate evaluation metrics for total segmentator dataset predictions
import numpy as np
import nibabel as nib
import os
import pickle as pkl
from monai.metrics import compute_hausdorff_distance
import argparse

# argparse
parser = argparse.ArgumentParser(description="Just an example",  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-t", "--task", default="Dataset301_Set1", help="Task to evaluate")
args = vars(parser.parse_args())

# set up variables
task = args["task"]

local = False
if local:
    root_dir = "/Users/katecevora/Documents/PhD/data/TotalSegmentator"
else:
    root_dir = "/rds/general/user/kc2322/home/data/TotalSegmentator"

fold = "all"

preds_dir = os.path.join(root_dir, "inference", task, fold)
gt_dir = os.path.join(root_dir, "nnUNet_raw", task, "labelsTs")

labels = {"background": 0,
          "right kidney": 1,
          "left kidney": 2,
          "liver": 3,
          "pancreas": 4}

n_channels = int(len(labels))

def oneHotEncode(array):
    array_dims = len(array.shape)
    array_max = 15
    one_hot = np.zeros((array_max + 1, array.shape[0], array.shape[1], array.shape[2]))

    for i in range(0, array_max + 1):
        one_hot[i, :, :, :][array==i] = 1

    return one_hot


def computeHDDIstance(pred, gt):
    # To use the MONAI function pred must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32].
    # The values should be binarized.
    # gt: ground truth to compute mean the distance. It must be one-hot format and first dim is batch.
    # The values should be binarized.

    # Convert to one hot
    # covert predictions to one hot encoding
    pred_one_hot = oneHotEncode(pred)
    gt_one_hot = oneHotEncode(gt)

    # expand the number of dimensions to include batch
    pred_one_hot = np.expand_dims(pred_one_hot, axis=0)
    gt_one_hot = np.expand_dims(gt_one_hot, axis=0)

    hd = compute_hausdorff_distance(pred_one_hot, gt_one_hot, include_background=False, distance_metric='euclidean', percentile=None,
                               directed=False, spacing=None)

    return hd


def multiChannelDice(pred, gt, channels):

    dice = []

    for channel in range(channels):
        a = np.zeros(pred.shape)
        a[pred == channel] = 1

        b = np.zeros(gt.shape)
        b[gt == channel] = 1

        dice.append(np.sum(a[b == 1])*2.0 / (np.sum(a) + np.sum(a)))

    return np.array(dice)


def calculateMetrics():
    # get a list of male and female IDs
    f = open(os.path.join(root_dir, "splits", "set1_splits.pkl"), "rb")
    set_1_ids = pkl.load(f)
    f.close()

    ids_ts = set_1_ids["test"]
    n_ids = len(ids_ts)
    idx_women = ids_ts[0:int(n_ids / 2)]
    idx_men = ids_ts[int(n_ids / 2):]

    dice_men = []
    dice_women = []
    hd_men = []
    hd_women = []

    cases = os.listdir(preds_dir)
    for case in cases:
        if case.endswith(".nii.gz"):
            print(case)

            pred = nib.load(os.path.join(preds_dir, case)).get_fdata()
            gt = nib.load(os.path.join(gt_dir, case)).get_fdata()

            if np.unique(gt).sum() == 0:
                print("Only background")

            # Get Dice and NSD
            dice = multiChannelDice(pred, gt, n_channels)

            hd = computeHDDIstance(pred, gt)

            if int(case[5:9]) in idx_women:
                dice_women.append(dice)
                hd_women.append(hd)
            elif int(case[5:9]) in idx_men:
                dice_men.append(dice)
                hd_men.append(hd)
            else:
                print("Not in list")

    print("Number of men: {}".format(len(dice_men)))
    print("Number of women: {}".format(len(dice_women)))

    dice_men = np.array(dice_men)
    dice_women = np.array(dice_women)
    hd_men = np.array(hd_men)
    hd_women = np.array(hd_women)

    f = open(os.path.join(preds_dir, "dice_and_hd.pkl"), "wb")
    pkl.dump({"dice_men": dice_men,
              "dice_women": dice_women,
              "hd_men": hd_men,
              "hd_women": hd_women}, f)
    f.close()


def printResults():
    datasets = ["Dataset701_Set1", "Dataset702_Set2", "Dataset703_Set3"]

    # lists to store the results for each dataset
    av_dice_men = []
    std_dice_men = []
    av_hd_men = []
    std_hd_men = []

    av_dice_women = []
    std_dice_women = []
    av_hd_women = []
    std_hd_women = []

    for ds in datasets:
        preds_dir = os.path.join(root_dir, "inference", ds, fold)
        f = open(os.path.join(preds_dir, "dice_and_hd.pkl"), "rb")
        metrics = pkl.load(f)
        f.close()

        # Dice
        av_dice_men.append(np.nanmean(metrics["dice_men"], axis=1))
        std_dice_men.append(np.nanstd(metrics["dice_men"], axis=1))
        av_dice_women.append(np.nanmean(metrics["dice_women"], axis=1))
        std_dice_women.append(np.nanstd(metrics["dice_women"], axis=1))

        # Hausdorff
        hd_men = np.squeeze(metrics["hd_men"])
        hd_women = np.squeeze(metrics["hd_women"])

        # Replace infs with nans so we can compute average using nanmean
        hd_men[hd_men == np.inf] = np.nan
        hd_women[hd_women == np.inf] = np.nan

        av_hd_men.append(np.nanmean(hd_men, axis=1))
        std_hd_men.append(np.nanstd(hd_men, axis=1))
        av_hd_women.append(np.nanmean(hd_women, axis=1))
        std_hd_women.append(np.nanstd(hd_women, axis=1))

    organs = list(labels.keys())
    n_channels = len(labels)

    # First print out the information for men
    for i in range(n_channels):
        print(organs[i] + " & {0:.3f} ({1:.3f}) & {2:.3f} ({3:.3f}) & {4:.3f} ({5:.3f})".format(av_dice_men[0][i],
                                                                                                std_dice_men[0][i],
                                                                                                av_dice_men[1][i],
                                                                                                std_dice_men[1][i],
                                                                                                av_dice_men[2][i],
                                                                                                std_dice_men[2][
                                                                                                    i]) + r" \\")

    print('')

    # Then print out the information for women
    for i in range(n_channels):
        print(organs[i] + " & {0:.3f} ({1:.3f}) & {2:.3f} ({3:.3f}) & {4:.3f} ({5:.3f})".format(av_dice_women[0][i],
                                                                                                std_dice_women[0][i],
                                                                                                av_dice_women[1][i],
                                                                                                std_dice_women[1][i],
                                                                                                av_dice_women[2][i],
                                                                                                std_dice_women[2][
                                                                                                    i]) + r" \\")

    print('')

    # First print out the information for men
    for i in range(n_channels):
        print(organs[i] + " & {0:.3f} & {1:.3f} & {2:.3f}".format(av_dice_men[0][i],
                                                                  av_dice_men[1][i],
                                                                  av_dice_men[2][i]) + r" \\")

    print('')

    # Then print out the information for women
    for i in range(n_channels):
        print(organs[i] + " & {0:.3f} & {1:.3f} & {2:.3f}".format(av_dice_women[0][i],
                                                                  av_dice_women[1][i],
                                                                  av_dice_women[2][i]) + r" \\")


def main():
    calculateMetrics()
    printResults()


if __name__ == "__main__":
    main()