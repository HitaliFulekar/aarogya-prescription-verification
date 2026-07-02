import os
import pandas as pd
from ocr_utils import extract_text

data = []

dataset_path = "dataset"

for category in os.listdir(dataset_path):

    category_path = os.path.join(dataset_path, category)

    for subfolder in os.listdir(category_path):

        subfolder_path = os.path.join(category_path, subfolder)

        for file in os.listdir(subfolder_path):

            file_path = os.path.join(subfolder_path, file)

            try:
                text = extract_text(file_path)

                data.append({
                    "filename": file,
                    "text": text,
                    "label": category
                })

                print("DONE:", file)

            except Exception as e:
                print("ERROR:", file, e)

df = pd.DataFrame(data)

df.to_csv("dataset.csv", index=False)

print("CSV SAVED SUCCESSFULLY")