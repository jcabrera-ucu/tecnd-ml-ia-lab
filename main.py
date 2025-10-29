import sys
import os
import cv2
import pandas as pd

def categorize_age(age):
    if age <= 12:
        return 'child'
    elif age <= 25:
        return 'young'
    elif age <= 50:
        return 'adult'
    else:
        return 'senior'


def load_dataset(path):
    dataset = []

    for filename in os.listdir(path):
        age, gender, race, date = filename.split('.')[0].split('_')

        age = int(filename.split('_')[0])
        img = cv2.imread(os.path.join('UTKFace', filename))
        if img is not None:
            img = cv2.resize(img, (128, 128))

        dataset.append({
            "image": img,
            "age": int(age),
            "gender": "male" if gender == '0' else 'female',
            "race": race,
            "date": date,
        })

    return dataset



if __name__ == '__main__':
    path = sys.argv[1]

    dataset = load_dataset(path)
    # df = pd.DataFrame({'age': [x["age"] for x in dataset]})
    main()