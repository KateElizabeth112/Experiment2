# Script to sort kits metadata from json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

data_folder = "/Users/katecevora/Documents/PhD/data/KiTS19"

comorbidities_dict = {
      "myocardial_infarction": 0,
      "congestive_heart_failure": 1,
      "peripheral_vascular_disease": 2,
      "cerebrovascular_disease": 3,
      "dementia": 4,
      "copd": 5,
      "connective_tissue_disease": 6,
      "peptic_ulcer_disease": 7,
      "uncomplicated_diabetes_mellitus": 8,
      "diabetes_mellitus_with_end_organ_damage": 9,
      "chronic_kidney_disease": 10,
      "hemiplegia_from_stroke": 11,
      "leukemia": 12,
      "malignant_lymphoma": 13,
      "localized_solid_tumor": 14,
      "metastatic_solid_tumor": 15,
      "mild_liver_disease": 16,
      "moderate_to_severe_liver_disease": 17,
      "aids": 18
}

def main():
    # Opening JSON file
    f = open('meta.json')
    # returns JSON object as a dictionary
    data = json.load(f)
    f.close()

    # Build a python dataframe to hold information
    df_meta = pd.DataFrame(data)

    # Transform comorbidities
    ids = df_meta["case_id"].values
    cms = comorbidities_dict.keys()

    # Create a new dataframe for comorbidities that we will append to the original df
    df_cm = pd.DataFrame(columns=cms)
    for id in ids:
        i = int(id[-3:])
        cm_row_orig = dict(df_meta[df_meta["case_id"] == id]["comorbidities"].values[0])
        cm_row_new = []
        for cm in cms:
            if cm_row_orig[cm] == True:
                cm_row_new.append(True)
            else:
                cm_row_new.append(False)

        df_cm.loc[len(df_cm)] = cm_row_new

    # Attach to original dataframe
    df_meta = pd.concat([df_meta, df_cm], axis=1)

    # Split by gender
    df_meta_m = df_meta[df_meta["gender"] == "male"]
    df_meta_f = df_meta[df_meta["gender"] == "female"]

    num_m = df_meta_m.shape[0]
    num_f = df_meta_f.shape[0]

    # Print some statistics
    print("Attributes: {}".format(list(df_meta.columns)))
    print("Number of patients: {}".format(df_meta.shape[0]))

    print("Number of men: {}".format(num_m))
    print("Number of women: {}".format(num_f))

    # Age distribution by sex
    plt.clf()
    plt.title("Age Distribution By Gender")
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    plt.hist(df_meta_m["age_at_nephrectomy"].values, label='male', alpha=0.6, bins=bins)
    plt.hist(df_meta_f["age_at_nephrectomy"].values, label='female', alpha=0.6, bins=bins)
    plt.legend()
    plt.ylabel("Counts")
    plt.xlabel("Age")
    plt.savefig(os.path.join(data_folder, "plots", "age_distribution_by_sex.png"))

    # BMI distribution by sex
    plt.clf()
    plt.title("BMI Distribution By Gender")
    bins = np.arange(15, 55, 5)
    plt.hist(df_meta_m["body_mass_index"].values, label='male', alpha=0.6, bins=bins)
    plt.hist(df_meta_f["body_mass_index"].values, label='female', alpha=0.6, bins=bins)
    plt.legend()
    plt.ylabel("Counts")
    plt.xlabel("BMI")
    plt.savefig(os.path.join(data_folder, "plots", "bmi_distribution_by_sex.png"))

    # Comorbidities distribution by gender
    cms = comorbidities_dict.keys()
    counts_f = []
    counts_m = []

    for cm in cms:
        counts_m.append(np.sum(df_meta_m[cm].values) / num_m)
        counts_f.append(np.sum(df_meta_f[cm].values) / num_f)

    bar_locs = np.arange(0, len(cms))

    plt.clf()
    plt.title("Comorbidities by Gender")
    plt.bar(bar_locs, counts_m, label="M", alpha=0.6)
    plt.bar(bar_locs, counts_f, label="F", alpha=0.6)
    plt.ylabel("Proportion")
    plt.xticks(bar_locs, cms, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_folder, "plots", "comorbidities_by_sex.png"))





if __name__ == "__main__":
    main()