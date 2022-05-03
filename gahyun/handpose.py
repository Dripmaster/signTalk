import os

import cv2
import mediapipe as mp
import numpy as np
import json

import pandas as pd

def csvSave(x,name):
    df = pd.DataFrame(x)
    df.head()
    r = df.to_csv(name)
    return r

def csvLoad(name):
    df = pd.read_csv(name,index_col=0)
    return df

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


videoPath = 'D:/sighTest/Sign-Language-Recognition--MediaPipe-DTW-master/data/videos'
morList = os.listdir(videoPath)
fps = 30
mCount = 0
for m in morList:
    videoList = os.listdir(os.path.join(videoPath,m))
    for v in videoList:
        handsType = []
        myHands = []
        errorFrame = []
        videoFilePath = os.path.join(videoPath,m,v)
        cap = cv2.VideoCapture(videoFilePath)
        width = int(cap.get(3))  # 가로 길이 가져오기
        height = int(cap.get(4))  # 세로 길이 가져오기
        with mp_hands.Hands(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    hc = len(handsType)
                    tempHandsType = list()
                    errorHand = ""
                    for hand_landmarks in results.multi_hand_landmarks:
                        myHand = []
                        for id, lm in enumerate(hand_landmarks.landmark):
                            myHand.append((int(lm.x * width), int(lm.y * height)))
                        myHands.append(myHand)

                    for hand in results.multi_handedness:
                        handType = hand.classification[0].label
                        handsType.append(handType)
                        tempHandsType.append(handType)
                    if len(tempHandsType) > 1 and tempHandsType[0] == tempHandsType[1]:
                        handsType[-1] = 'Right'
                        handsType[-2] = 'Left'
                        tempHandsType[0] = 'Left'
                        tempHandsType[1] = 'Right'
                    if len(handsType) - hc < 2:
                        if tempHandsType[0] == 'Left':  # 오른손 없음
                            errorHand = 'Right'

                            lastHand = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                                        (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                                        (0, 0)]
                            myHands.append(lastHand)
                            handsType.append('Right')
                        else:  # 왼손 없음
                            errorHand = 'Left'

                            lastHand = myHands[-1].copy()
                            myHands[-1] = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                                           (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                                           (0, 0), (0, 0), (0, 0)]
                            myHands.append(lastHand)
                            handsType[-1] = 'Left'
                            handsType.append('Right')
                        errorFrameCount = len(handsType) / 2 - 1
                        # print("양손검출실패:",errorHand,"/ frame:",errorFrameCount)
                        errorFrame.append((errorFrameCount, errorHand))
                    elif not (tempHandsType[0] == 'Left' and tempHandsType[1] == 'Right'):
                        # print("왼손오른손순서 틀림:",tempHandsType,"/ frame:",len(handsType)/2-1) # Right-Left
                        tempHand = myHands[-1].copy()  # Left
                        myHands[-1] = myHands[-2].copy()
                        myHands[-2] = tempHand
                        handsType[-1] = 'Right'
                        handsType[-2] = 'Left'
                else:
                    print("detect fail")

        for f, d in errorFrame:
            fc = int(f * 2)
            if d == 'Right':
                fc = int(f * 2) + 1
            for i in range(len(myHands[fc])):
                c = len(myHands)
                if c <=fc+2:
                    myHands[fc][i] = (int((myHands[fc - 2][i][0])),
                                      int((myHands[fc - 2][i][1])))
                elif 0 >fc-2:
                    myHands[fc][i] = (int((myHands[fc + 2][i][0])),
                                      int((myHands[fc + 2][i][1])))
                else:
                    myHands[fc][i] = (int((myHands[fc - 2][i][0] + myHands[fc + 2][i][0]) / 2),
                                  int((myHands[fc - 2][i][1] + myHands[fc + 2][i][1]) / 2))

        handkeypoints = dict()
        idx = 0
        for i in range(int(len(myHands) / 2)):
            handkeypoints[idx] = myHands[i * 2] + myHands[i * 2 + 1]
            idx += 1
        df = pd.DataFrame(handkeypoints)
        df = df.transpose()
        savePath = os.path.join('D:/aiData/keypoints/',m)
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        savePath= os.path.join('D:/aiData/keypoints/',m,v+'.csv')
        csvSave(df, savePath)
        mCount+=1
        print((mCount/len(morList*5))*100,"%")
cap.release()
#out.release()
cv2.destroyAllWindows()


