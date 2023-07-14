# Script to explore metadata for TotalSegmentator and Kits dataser
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

working_dir = "/Users/katecevora/Documents/PhD/data/TotalSegmentator_nnUNet"

def main():
    ts_meta = pd.read_csv(os.path.join(working_dir, "meta.csv"), sep=";")

    study_types = list(pd.unique(ts_meta["study_type"]))
    study_types_trimmed = []
    for typ in study_types:
        # trim any trailing whitespace
        typ_strip = typ.strip()
        if not(typ_strip in study_types_trimmed):
            study_types_trimmed.append(typ_strip)

    institutes = list(pd.unique(ts_meta["institute"]))

    # Split by gender
    ts_meta_m = ts_meta[ts_meta["gender"] == "m"]
    ts_meta_f = ts_meta[ts_meta["gender"] == "f"]

    # save
    ts_meta_m.to_csv(os.path.join(working_dir, "meta_m.csv"), index=False)
    ts_meta_f.to_csv(os.path.join(working_dir, "meta_f.csv"), index=False)

    idx_men = list(ts_meta_m["image_id"].values)
    idx_women = list(ts_meta_f["image_id"].values)

    f = open(os.path.join(working_dir, "case_ids.pkl"), "wb")
    pkl.dump([idx_men, idx_women], f)
    f.close()

    num_m = ts_meta_m.shape[0]
    num_f = ts_meta_f.shape[0]

    # Print some statistics
    print("Attributes: {}".format(list(ts_meta.columns)))
    print("Number of patients: {}".format(ts_meta.shape[0]))
    print("Number of study types: {}".format(len(study_types_trimmed)))
    print("Study types: {}".format(study_types_trimmed))
    print("Number of institutes: {}".format(len(institutes)))
    print("Institutes: {}".format(institutes))
    print("Number of men: {}".format(num_m))
    print("Number of women: {}".format(num_f))

    # Make some plots
    # Age distribution by gender
    plt.clf()
    plt.title("Age Distribution By Gender")
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    plt.hist(ts_meta_m["age"].values, label='M', alpha=0.6, bins=bins)
    plt.hist(ts_meta_f["age"].values, label='F', alpha=0.6, bins=bins)
    plt.legend()
    plt.xlabel("Age")
    plt.savefig(os.path.join(working_dir, "plots", "age_distribution_by_gender.png"))

    # Institution by gender
    institute_counts_m = []
    institute_counts_f = []
    for inst in institutes:
        institute_counts_m.append(ts_meta_m[ts_meta_m["institute"] == inst].shape[0] / num_m)
        institute_counts_f.append(ts_meta_f[ts_meta_f["institute"] == inst].shape[0] / num_f)

    bar_locs = np.arange(0, len(institutes))

    plt.clf()
    plt.title("Institutions by Gender")
    plt.bar(bar_locs, institute_counts_m, label="M", alpha=0.6)
    plt.bar(bar_locs, institute_counts_f, label="F", alpha=0.6)
    plt.xlabel("Institution")
    plt.ylabel("Proportion")
    plt.xticks(bar_locs, institutes)
    plt.legend()
    plt.savefig(os.path.join(working_dir, "plots", "institutions_by_gender.png"))

    # Study types by gender
    study_counts_m = []
    study_counts_f = []
    for study in study_types_trimmed:
        study_counts_m.append(ts_meta_m[ts_meta_m["study_type"] == study].shape[0] / num_m)
        study_counts_f.append(ts_meta_f[ts_meta_f["study_type"] == study].shape[0] / num_f)

    bar_locs = np.arange(0, len(study_types_trimmed))
    plt.clf()
    plt.title("Study types by Gender")
    plt.bar(bar_locs, study_counts_m, label="M", alpha=0.6)
    plt.bar(bar_locs, study_counts_f, label="F", alpha=0.6)
    plt.ylabel("Proportion")
    plt.xlabel("Study Type")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "plots", "study_type_by_gender.png"))

    # Study types by age


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
