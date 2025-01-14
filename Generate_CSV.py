import os
import csv
from ethnic_classifications import ethnic_groups

# Path to the dataset
dataset_path = "FAGE"
output_csv = "ethnic_labels.csv"

# Prepare to save labels
data = []

# Iterate over Train and Test folders
for split in ["Training", "Test"]:
    split_path = os.path.join(dataset_path, split)
    for person_folder in os.listdir(split_path):
        person_path = os.path.join(split_path, person_folder)
        if os.path.isdir(person_path):
            label = ethnic_groups.get(person_folder)  # Get ethnic group or mark as Unknown
            for image_file in os.listdir(person_path):
                if image_file.endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(split, person_folder, image_file)
                    data.append([image_path, label])

# Write labels to a CSV file
with open(output_csv, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image_path", "label"])
    writer.writerows(data)

print(f"Labels saved to {output_csv}.")