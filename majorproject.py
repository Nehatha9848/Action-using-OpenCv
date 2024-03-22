# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 22:04:37 2020

@author: BALU
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 21:27:54 2020

@author: BALU
"""


# coding: utf-8

# In[3]:

import numpy as np
import cv2
import glob
from random import shuffle
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import datetime
import pickle


def split_data(data, percentaje):    
    
    shuffle(data)
    train_n = int(percentaje*len(data))
    
    train, test = np.split(data, [train_n])

    s_train = list(zip(*train))
    print(s_train)
    s_test = list(zip(*test))
    print(s_test)
    
    samples_train = (s_train[0])
    labels_train = (s_train[1])

    samples_test = (s_test[0])
    labels_test = (s_test[1])
    print("done")
    
    return samples_train, labels_train, samples_test, labels_test


def draw_flow(img, flow, step=16):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def calc_hist(flow):

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees = 1)
    
    q1 = ((0 < ang) & (ang <= 45)).sum()
    q2 = ((45 < ang) & (ang <= 90)).sum()
    q3 = ((90 < ang) & (ang <= 135)).sum()
    q4 = ((135 < ang) & (ang <= 180)).sum()
    q5 = ((180 < ang) & (ang <= 225)).sum()
    q6 = ((225 <= ang) & (ang <= 270)).sum()
    q7 = ((270 < ang) & (ang <= 315)).sum()
    q8 = ((315 < ang) & (ang <= 360)).sum()
    
    hist = [q1, q2, q3, q4 ,q5, q6, q7 ,q8]
    
    return (hist)


def process_video(fn, samples):

    video_hist = []
    hog_list = []
    sum_desc = []
    bins_n = 10

    cap = cv2.VideoCapture(fn)
    ret, prev = cap.read()
    print(ret)
            
    prevgray = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
    cv2.imshow('pr',prevgray)

    hog = cv2.HOGDescriptor()


    while True:
           
        ret, img = cap.read()
        
        if not ret : break
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prevgray,gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        prevgray = gray

        bins = np.hsplit(flow, bins_n)

        out_bins = []
        for b in bins:
            out_bins.append(np.vsplit(b, bins_n))

        frame_hist = []
        for col in out_bins:

            for block in col:
                frame_hist.append(calc_hist(block))
                                 
        video_hist.append(np.matrix(frame_hist) )

    # average per frame
    sum_desc = video_hist[0]
    for i in range(1, len(video_hist)):
        sum_desc = sum_desc + video_hist[i] 
    
    ave = sum_desc / len(video_hist)

    # max per bin
    maxx = np.amax(video_hist, 0)
    maxx = np.matrix(maxx)
    
    fn = fn.lower()
    print(fn)

    if "walking" in fn :
        label =1
    if "running" in fn :
        label =2
    
    
    ave_desc = np.asarray(ave)
    a_desc = []
    a_desc.append(np.asarray(ave_desc, dtype = np.uint8).ravel())

    max_desc = np.asarray(maxx)
    m_desc = []
    m_desc = np.asarray(max_desc, dtype = np.uint8).ravel()

    return a_desc, label, m_desc


# In[4]:
import os

if __name__ == '__main__':
        
    path =r"E:\major\Mixed data"
    print("y")
   # path = '/Users/soledad/Box Sync/Fall 15/I590 - Collective Intelligence/CV Project/240x320/'
    
#     folders = glob.glob(path+ "/*")
    folders=[path]
    
    walk_data = []
    run_data = []

    samples = 10
    
    
    a = datetime.datetime.now()
    
    for act in folders:
        
            fileList = glob.glob(act + "/*.avi")  
            print(len(fileList))

            for f in fileList:
                f = f.lower()
                print (f)
                
    
                if 'walking' in f:
                    video_desc, label, maxx = process_video(f, samples)
                
                    if (label) != 0 :
                        walk_data.append([video_desc[0], label, maxx])
                if 'running' in f:
                    video_desc, label, maxx = process_video(f, samples)
                
                    if (label) != 0 :
                        run_data.append([video_desc[0], label, maxx])


                            

    b = datetime.datetime.now()
    
    print (b-a)


# In[2]:

percentaje = 0.85

clf = svm.SVC(kernel = 'rbf', C = 1000, gamma = 0.0000001)

gnb = GaussianNB()
mnb = MultinomialNB()

svm = 0
nb1 = 0
nb2 = 0

# all_data = happy_data + sad_data + fear_data + surprise_data + disgust_data + angry_data
     
times = 10

for i in range(0,times):
    # happiness
    walk_samples_train=[]
    walk_labels_train=[]
    walk_samples_test = []
    walk_labels_test = []
    run_samples_train = []
    run_labels_train = []
    run_samples_test = []
    run_labels_test = []
    if len(walk_data) > 0:
        walk_samples_train, walk_labels_train, walk_samples_test, walk_labels_test = split_data(walk_data, percentaje)
       
    if len(run_data) > 0:
        run_samples_train, run_labels_train, run_samples_test, run_labels_test = split_data(run_data, percentaje)
      
    train_set = walk_samples_train
    train_set += run_samples_train
    test_set = walk_samples_test
    test_set += run_samples_test
    labels_train = walk_labels_train
    labels_train += run_labels_train
    labels_test = walk_labels_test
    labels_test += run_labels_test
     

    # train_set, labels_train, test_set, labels_test = split_data(all_data, percentaje)    
    print("yes")
    clf.fit(train_set, labels_train)
    
    
    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    
    
    
    clf2 = pickle.load(open(filename, 'rb'))
    
    predicted = clf2.predict(test_set) 
    
    err1 = (labels_test == predicted).mean()
    
        
    print ('accuracy svm: %.2f %%' % (err1*100))

#     folder = '/Users/soledad/Box Sync/Fall 15/I590 - Collective Intelligence/CV Project/Code/Emotion_Out/'


    



# In[ ]:




# In[ ]:


