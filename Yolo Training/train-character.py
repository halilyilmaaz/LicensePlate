import numpy as np
from darkflow import tfnets
import cv2

options = {"model": "cfg/yolo-character.cfg", 
           "load": "bin/yolo.weights",
           "batch": 8,
           "epoch": 100,
           "gpu": 0.9,
           "train": True,
           "annotation": "./data/AnnotationsXML/characters/",
           "dataset": "./data/Images/characters/"}

tfnet = tfnets(options)
tfnet.train()
tfnet.savepb()

