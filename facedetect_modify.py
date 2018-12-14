#!/usr/bin/env python

'''
face detection using haar cascades

USAGE:
    facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

# local modules
from video import create_capture
from common import clock, draw_str
import pyttsx 
import time 
import datetime 
engine = pyttsx.init() # Python TTs for the text to speech capability 
WorkDay = []
today = datetime.date.today() 
WorkDay.append(today) 
x1 = 0 
x2 = 0  
y1 = 0 
y2 = 0 
# Valiable added into the system for the function that will be tracking 
Facename = " "
positionX = 0 
positionY = 0 

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
def capturepict(): 
    cv.imshow('img1',img) #display the captured image
    cv.imwrite('img1'+ str(WorkDay[0]) +'.png',img) 

if __name__ == '__main__':
    import sys, getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[1]
    except:
        video_src = 0
    args = dict(args)  
    cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "../../data/haarcascades/haarcascade_eye.xml")

    cascade = cv.CascadeClassifier(cascade_fn)
    nested = cv.CascadeClassifier(nested_fn)

    cam = create_capture(video_src, fallback='synth:bg=../data/lena.jpg:noise=0.05')

    while True:
        ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)

        t = clock()
        rects = detect(gray, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        if not nested.empty():
            for x1, y1, x2, y2 in rects:
                #Get the value and computational for the detection function
                capturepict()  # Capture the picture when see human face  
                print('Face detected')
                print('Face size:'+ str(x1)+","+str(y1)) # Face size output function 
                print('Eyes size:'+ str(x2)+","+str(y2)) # Eyes size output function 
                engine.say("Hello what is your name") 
                time.sleep(0.3) 
                engine.say("I'm the sentry mode of the system please identify your self")
                engine.runAndWait() 
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                subrects = detect(roi.copy(), nested)
                draw_rects(vis_roi, subrects, (255, 0, 0))
        dt = clock() - t

        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv.imshow('facedetect', vis)

        if cv.waitKey(5) == 27:
            break
    cv.destroyAllWindows()
