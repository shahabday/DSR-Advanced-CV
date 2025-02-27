from pathlib import PurePath, Path
import os
import cv2
import numpy as np
import pandas as pd 
from tqdm.notebook import tqdm

def IoU(regionA, regionB):
    
    #evaluate the intersection points 
    xA = np.maximum(regionA['x1'], regionB['x1'])
    yA = np.maximum(regionA['y1'], regionB['y1'])
    xB = np.minimum(regionA['x2'], regionB['x2'])
    yB = np.minimum(regionA['y2'], regionB['y2'])

    # compute the area of intersection rectangle
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    regionAArea = (regionA['x2'] - regionA['x1'] + 1) * (regionA['y2'] - regionA['y1'] + 1)
    regionBArea = (regionB['x2'] - regionB['x1'] + 1) * (regionB['y2'] - regionB['y1'] + 1)

    #compute the union 
    unionArea = (regionAArea + regionBArea - interArea)

    # return the intersection over union value
    return interArea / unionArea


def data_loader():
    REPO_DIR = Path(os.getcwd()).parent
    
    train_labels = pd.read_csv(REPO_DIR / "data/module_2/train_labels.csv")
    test_labels = pd.read_csv(REPO_DIR / "data/module_2/test_labels.csv")
    
    train_dataset = {}
    test_dataset = {}
    
    
    for i in tqdm(range(train_labels.shape[0])):
        img_path = REPO_DIR / "data/module_2/train" / train_labels.iloc[i]['filename']
        
        if img_path.parts[-1].endswith("png") and train_labels.iloc[i]['filename'] not in train_dataset:
            obj = {}
            img = cv2.cvtColor(cv2.imread(img_path.__str__()), cv2.COLOR_BGR2RGB)
            obj['img'] = img

            obj['objects'] = [{"x1":train_labels.iloc[i]['xmin'],
                               "x2":train_labels.iloc[i]['xmax'],
                               "y1":train_labels.iloc[i]['ymin'],
                               "y2":train_labels.iloc[i]['ymax']}]
            
            train_dataset[train_labels.iloc[i]['filename']] = obj
        else:
            train_dataset[train_labels.iloc[i]['filename']]['objects'].append({"x1":train_labels.iloc[i]['xmin'],
                               "x2":train_labels.iloc[i]['xmax'],
                               "y1":train_labels.iloc[i]['ymin'],
                               "y2":train_labels.iloc[i]['ymax']})

    for i in tqdm(range(test_labels.shape[0])):
        img_path = REPO_DIR / "data/module_2/test" / test_labels.iloc[i]['filename']
        
        if img_path.parts[-1].endswith("png") and test_labels.iloc[i]['filename'] not in test_dataset:
            obj = {}

            img = cv2.cvtColor(cv2.imread(img_path.__str__()), cv2.COLOR_BGR2RGB)
            obj['img'] = img

            obj['objects'] = [{"x1":train_labels.iloc[i]['xmin'],
                               "x2":train_labels.iloc[i]['xmax'],
                               "y1":train_labels.iloc[i]['ymin'],
                               "y2":train_labels.iloc[i]['ymax']}]
            
            test_dataset[train_labels.iloc[i]['filename']] = obj
        else:
            test_dataset[train_labels.iloc[i]['filename']]['objects'].append({"x1":train_labels.iloc[i]['xmin'],
                               "x2":train_labels.iloc[i]['xmax'],
                               "y1":train_labels.iloc[i]['ymin'],
                               "y2":train_labels.iloc[i]['ymax']})
    
    return train_dataset, test_dataset