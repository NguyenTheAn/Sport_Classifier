#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
import os


# In[27]:


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to our input video")
ap.add_argument("-o", "--output", required=True,
    help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=128,
    help="size of queue for averaging")
args = vars(ap.parse_args())


# In[28]:


path = os.getcwd()
model =  load_model(os.path.join(path, "Model/model2.h5"))
lb = pickle.loads(open ("lb.pickle", "rb").read())


# In[29]:


mean = np.array([123.68, 116.779, 103.939], dtype="float32")
qu = deque(maxlen=args["size"])


# In[30]:


vid = cv2.VideoCapture(os.path.join(path, "Input", args["input"]))
writer = None
(W, H) = (None, None)


# In[21]:

while True:
    grabbed, frame = vid.read()
    if not grabbed:
        break;
    if H is None or W is None:
        (W, H) = frame.shape[:2]
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224)).astype("float32")
    frame -= mean
    frame = np.reshape(frame, (1, 224, 224, 3))
    preds = model.predict(frame)
    qu.append(preds)
    result = np.array(qu).mean(axis = 0)
    index = np.argmax(result)
    label = lb.classes_[index]
    string = "The activity is: {}".format(label)
    cv2.putText(output, string, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
        1.25, (50, 205, 154), 5)
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(os.path.join(path, "Output", args["output"]), fourcc, 30, (W, H), True)
    writer.write(output)
    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF
 
    if key == ord("q"):
        break
        
writer.release()
vid.release()
cv2.destroyAllWindows()
