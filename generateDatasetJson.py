# Generate the dataset.json file
from nnunet.dataset_conversion.utils import generate_dataset_json
import argparse
import os

# argparse
parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-r", "--root_dir", default='/vol/biomedic3/kc2322/data/TotalSegmentator_nnUNetv1', help="Root directory for nnUNet")
parser.add_argument("-n", "--dataset_name", default='TotalSegmentator', help="Name of the dataset")
parser.add_argument("-t", "--task_id", default="Task303", help="ID of the task")
args = vars(parser.parse_args())

# set up variables
ROOT_DIR = args['root_dir']
DS_NAME = args['dataset_name']
TASK_ID = args['task_id']

output_file = os.path.join(ROOT_DIR, "nnUNet_raw_data_base/nnUNet_raw_data/{}/dataset.json".format(TASK_ID))
imagesTr_dir = os.path.join(ROOT_DIR, "nnUNet_raw_data_base/nnUNet_raw_data/{}/imagesTr".format(TASK_ID))
imagesTs_dir = os.path.join(ROOT_DIR, "nnUNet_raw_data_base/nnUNet_raw_data/{}/imagesTs".format(TASK_ID))

labels = {0: "background",
          1: "right kidney",
          2: "left kidney",
          3: "liver",
          4: "pancreas"}

modalities = ("C")

generate_dataset_json(str(output_file), str(imagesTr_dir), str(imagesTs_dir), modalities, labels, DS_NAME)