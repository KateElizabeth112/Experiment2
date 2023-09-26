# Script to make combined plots from different experiments
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy import stats

root_dir = "/Users/katecevora/Documents/PhD/data/TotalSegmentator"

lblu = "#add9f4"
lred = "#f36860"

labels = {"background": 0,
          "right kidney": 1,
          "left kidney": 2,
          "liver": 3,
          "pancreas": 4}

n_channels = int(len(labels))


def plotDice(dice_men1, dice_women1, dice_men2, dice_women2, dice_men3, dice_women3, organ, save_path):
    plt.clf()

    # Delete NaNs
    dice_men1 = dice_men1[~np.isnan(dice_men1)]
    dice_women1 = dice_women1[~np.isnan(dice_women1)]
    dice_men2 = dice_men2[~np.isnan(dice_men2)]
    dice_women2 = dice_women2[~np.isnan(dice_women2)]
    dice_men3 = dice_men3[~np.isnan(dice_men3)]
    dice_women3 = dice_women3[~np.isnan(dice_women3)]

    data = [dice_men1, dice_men2, dice_men3, dice_women1, dice_women2, dice_women3]

    labels = ['Balanced', 'Female Training Set', 'Male Training Set', 'Balanced', 'Female Training Set', 'Male Training Set']

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.manager.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=False, sym='+', vert=True, whis=1.5, showfliers=False)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.2)

    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title='Dice scores for {}'.format(organ),
        xlabel='',
        ylabel='Dice Score',
    )

    # Now fill the boxes with desired colors
    box_colors = [lblu, lblu, lblu, lred, lred, lred]
    num_boxes = len(data)
    medians = np.empty(num_boxes)

    for i in range(num_boxes):
        box = bp['boxes'][i]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i]))
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        median_x = []
        median_y = []
        for j in range(2):
            median_x.append(med.get_xdata()[j])
            median_y.append(med.get_ydata()[j])
            ax1.plot(median_x, median_y, 'k')
        medians[i] = median_y[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]), color='k', marker='*', markeredgecolor='k', markersize=10)

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 1.0
    bottom = 0.2
    #ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(labels, rotation=45, fontsize=8)

    # Finally, add a basic legend
    fig.text(0.80, 0.38, 'Male Test Set',
             backgroundcolor=box_colors[0], color='black', weight='roman',
             size='small')
    fig.text(0.80, 0.345, 'Female Test Set',
             backgroundcolor=box_colors[3],
             color='white', weight='roman', size='small')
    fig.text(0.80, 0.295, '*', color='black',
             weight='roman', size='large')
    fig.text(0.815, 0.300, ' Average Value', color='black', weight='roman',
             size='small')

    plt.axvline(x=3.5, color='k', linestyle="dashed", linewidth=1)

    plt.savefig(save_path)
    #plt.show()



def printDice(dice_men1, dice_women1, dice_men2, dice_women2, dice_men3, dice_women3):
    # Calculate the deltas from the baseline experiment
    organs = list(labels.keys())

    for i in range(1, n_channels):
        organ = organs[i]

        # Delete NaNs
        dice_men1_i = dice_men1[:, i][~np.isnan(dice_men1[:, i])]
        dice_women1_i = dice_women1[:, i][~np.isnan(dice_women1[:, i])]
        dice_men2_i = dice_men2[:, i][~np.isnan(dice_men2[:, i])]
        dice_women2_i = dice_women2[:, i][~np.isnan(dice_women2[:, i])]
        dice_men3_i = dice_men3[:, i][~np.isnan(dice_men3[:, i])]
        dice_women3_i = dice_women3[:, i][~np.isnan(dice_women3[:, i])]

        # Baseline results
        av_dice_men1 = np.mean(dice_men1_i)
        av_dice_women1 = np.mean(dice_women1_i)

        # Experiment 2 (all female training set)
        av_dice_men2 = np.mean(dice_men2_i)
        av_dice_women2 = np.mean(dice_women2_i)

        delta_men2 = -(av_dice_men1 - av_dice_men2) * 100
        delta_women2 = -(av_dice_women1 - av_dice_women2) * 100

        (_, p_men2) = stats.ttest_ind(dice_men1_i, dice_men2_i, equal_var=False)
        (_, p_women2) = stats.ttest_ind(dice_women1_i, dice_women2_i, equal_var=False)

        # Experiment 3 (male training set)
        av_dice_men3 = np.mean(dice_men3_i)
        av_dice_women3 = np.mean(dice_women3_i)

        delta_men3 = -(av_dice_men1 - av_dice_men3) * 100
        delta_women3 = -(av_dice_women1 - av_dice_women3) * 100

        (_, p_men3) = stats.ttest_ind(dice_men1_i, dice_men3_i, equal_var=False)
        (_, p_women3) = stats.ttest_ind(dice_women1_i, dice_women3_i, equal_var=False)

        print(organ + " & {0:.2f} ({1:.2f}) & {2:.2f} ({3:.2f}) & {4:.2f} ({5:.2f}) & {6:.2f} ({7:.2f})".format(delta_men2,
                                                                                                          p_men2,
                                                                                                          delta_men3,
                                                                                                          p_men3,
                                                                                                          delta_women2,
                                                                                                          p_women2,
                                                                                                          delta_women3,
                                                                                                          p_women3) + r" \\")


def main():
    # first get relevant metrics from all three experiments
    experiments = ["Dataset501_Fold0", "Dataset502_Fold0", "Dataset503_Fold0"]

    # Experiment 1
    f = open(os.path.join(root_dir, "inference", experiments[0], "all", "dice_and_hd.pkl"), "rb")
    metrics1 = pkl.load(f)
    f.close()

    dice_men1 = metrics1["dice_men"]
    dice_women1 = metrics1["dice_women"]

    # Experiment 2
    f = open(os.path.join(root_dir, "inference", experiments[1], "all", "dice_and_hd.pkl"), "rb")
    metrics2 = pkl.load(f)
    f.close()

    dice_men2 = metrics2["dice_men"]
    dice_women2 = metrics2["dice_women"]

    # Experiment 3
    f = open(os.path.join(root_dir, "inference", experiments[2], "all", "dice_and_hd.pkl"), "rb")
    metrics3 = pkl.load(f)
    f.close()

    dice_men3 = metrics3["dice_men"]
    dice_women3 = metrics3["dice_women"]

    # Now make some plots
    organs = list(labels.keys())

    for i in range(1, n_channels):
        organ = organs[i]

        if organ == "prostate/uterus":
            organ = "prostate or uterus"

        save_path = os.path.join(root_dir, "plots", "Dataset500", "{}_dice.png".format(organ))

        plotDice(dice_men1[:, i],
                 dice_women1[:, i],
                 dice_men2[:, i],
                 dice_women2[:, i],
                 dice_men3[:, i],
                 dice_women3[:, i],
                 organ,
                 save_path)

    # Now process dice scores for all three experiments into a tabular format
    printDice(dice_men1, dice_women1, dice_men2, dice_women2, dice_men3, dice_women3)


if __name__ == "__main__":
    main()