import json
import math
import os
import saveCSV
import pandas as pd
import cv2
import numpy as np
import mediapipe as mp
def dist(v):
    return math.sqrt(v[0]**2 + v[1]**2)

def innerProduct(v1, v2):
    # 벡터 v1, v2의 크기 구하기
    distA = dist(v1)
    distB = dist(v2)

    # 내적 1 (x1x2 + y1y2)
    ip = v1[0] * v2[0] + v1[1] * v2[1]

    # 내적 2 (|v1|*|v2|*cos x)
    ip2 = distA * distB

    # cos x값 구하기
    cost = ip / ip2

    # x값(라디안) 구하기 (cos 역함수)
    x = math.acos(cost)

    # x값을 x도로 변환하기
    return (ip, x)
def getXY(landmark):
    return (landmark.x,landmark.y)

def csvSave(x,name):
    df = pd.DataFrame(x)
    df.head()
    r = df.to_csv(name)
    return r

def csvLoad(name):
    df = pd.read_csv(name,index_col=0)
    return df

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


path_dir = "D:/aiData/수어 영상/1.Training/morpheme/03/"
video_path = "C:/AIDATA/수어 영상/1.Training/03/"
file_list = os.listdir(path_dir)
file_count = len(file_list)

morphemes = {}
morphemesStart = {}
morphemesEnd = {}
morphemesCount = 0;
useMorphemeList = ['한국농아인협회','내리다','응급실','어디','카드','계산','서울역','영수증','주세요','화장실','배','아프다']
resultlist = []
for s in file_list:
    if "F" in s:
        resultlist.append(s)
#file_list = resultlist

for f in file_list:
    data = json.load(open(path_dir + f, encoding='UTF8'))
    mors = len(data['data'])
    if mors <1 :
        continue

    m_temp = []
    m_start_temp = []
    m_end_temp = []
    for i in range(mors):
        if data['data'][i]['attributes'][0]['name'] not in useMorphemeList:
            continue
        m_temp.append(data['data'][i]['attributes'][0]['name'])
        m_start_temp.append(data['data'][i]['start'])
        m_end_temp.append( data['data'][i]['end'])
    if len(m_temp) <1:
        continue
    morphemes[data['metaData']['name']] = m_temp
    morphemesStart[data['metaData']['name']] = m_start_temp
    morphemesEnd[data['metaData']['name']] = m_end_temp
print(len(morphemes))
mor = morphemes.values()
morphemesSet = set()
for m in mor:
    for n in m:
        morphemesSet.add(n)
morphemesDict = dict()
i = 0
for m in morphemesSet:
    morphemesDict[m] = i
    i = i+1
morphemesCount = len(morphemesSet)
#saveCSV.csvSave([morphemesDict],"D:/aiData/morphemesSentenceDict.csv")
fileCount=0

morphemesDict_swap = {}
for k, v in morphemesDict.items():
    morphemesDict_swap[v] = k

morCounts = np.zeros(len(morphemesDict))

for m in mor:
    for n in m:
        morCounts[morphemesDict[n]] = morCounts[morphemesDict[n]]+1
#print(morphemesDict_swap[morCounts.argmax()])
#top_2_idx = np.argsort(morCounts)[-20:]
#for t in top_2_idx:
#    print(morphemesDict_swap[t],morCounts[t]/5,t)
mCount = 0
for fname in morphemes.keys():
    videoFilePath = os.path.join(video_path,fname)
    for i in range(len(morphemes[fname])):
        m = morphemes[fname][i]

        failFlag = 0
        cap = cv2.VideoCapture(videoFilePath)

        vfps = cap.get(cv2.CAP_PROP_FPS)
        startFrame = int(vfps * morphemesStart[fname][i])
        endFrame = int(vfps * morphemesEnd[fname][i])

        width = int(cap.get(3))  # 가로 길이 가져오기
        height = int(cap.get(4))  # 세로 길이 가져오기
        handsType = []
        myHands = []
        errorFrame = []
        handAngels = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
        with mp_hands.Hands(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break
                if failFlag > 5:
                    break

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                with mp_pose.Pose(
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as pose:
                    poseResults = pose.process(image)
                if not poseResults.pose_landmarks:
                    failFlag = failFlag+ 10
                    print('pose fail')
                    continue
                _,L1 = innerProduct(getXY(poseResults.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]),
                             getXY(poseResults.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]))
                _,L2 = innerProduct(getXY(poseResults.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]),
                             getXY(poseResults.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]))
                _, R1 = innerProduct(getXY(poseResults.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]),
                                     getXY(poseResults.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]))
                _, R2 = innerProduct(getXY(poseResults.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]),
                                     getXY(poseResults.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]))


                if L1 == 0 or L2 == 0 or R1 == 0 or R2 == 0:
                    print('pose 0')
                    failFlag = failFlag+ 10
                    continue
                handAngleLeft = (L1, L2)
                handAngleRight = (R1, R2)
                handAngels.append(handAngleLeft)
                handAngels.append(handAngleRight)
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    hc = len(handsType)
                    tempHandsType = list()
                    errorHand = ""
                    handCountLen = len(results.multi_hand_landmarks)
                    if(handCountLen >2):
                        handCountLen = 2
                    for handidx in range(handCountLen):
                        myHand = []
                        for id, lm in enumerate(results.multi_hand_landmarks[handidx].landmark):
                            myHand.append((int(lm.x * width), int(lm.y * height)))
                        myHands.append(myHand)

                    for handidx in range(handCountLen):
                        handType = results.multi_handedness[handidx].classification[0].label
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
                        #print(handsType)
                        #print("detect one fail:", fname, "errorFrame:", (errorFrameCount, errorHand), (endFrame - startFrame))
                        if(errorFrameCount-1 in errorFrame):
                            failFlag = failFlag+5
                        failFlag = failFlag + 1
                        # print("양손검출실패:",errorHand,"/ frame:",errorFrameCount)
                        errorFrame.append((errorFrameCount, errorHand))
                    elif not (tempHandsType[0] == 'Left' and tempHandsType[1] == 'Right'):
                        #print("왼손오른손순서 틀림:",tempHandsType,"/ frame:",len(handsType)/2-1) # Right-Left
                        tempHand = myHands[-1].copy()  # Left
                        myHands[-1] = myHands[-2].copy()
                        myHands[-2] = tempHand
                        handsType[-1] = 'Right'
                        handsType[-2] = 'Left'

                else:
                    print("detect fail:"+fname)
                    failFlag = failFlag+ 1

                if failFlag >5:
                    break
                if cap.get(cv2.CAP_PROP_POS_FRAMES) >= endFrame:
                    break
        if failFlag >5:
            mCount += 1
            print('fail:',failFlag)
            continue
        for f, d in errorFrame:
            fc = int(f * 2)
            if d == 'Right':
                fc = int(f * 2) + 1
            for i in range(len(myHands[fc])):
                c = len(myHands)
                if c <= fc + 2:
                    myHands[fc][i] = (int((myHands[fc - 2][i][0])),
                                      int((myHands[fc - 2][i][1])))
                elif 0 > fc - 2:
                    myHands[fc][i] = (int((myHands[fc + 2][i][0])),
                                      int((myHands[fc + 2][i][1])))
                else:
                    myHands[fc][i] = (int((myHands[fc - 2][i][0] + myHands[fc + 2][i][0]) / 2),
                                      int((myHands[fc - 2][i][1] + myHands[fc + 2][i][1]) / 2))
        normHands = []
        for f in myHands:
            maxDisX = 1
            maxDisY = 1
            normHand = []
            for i in range(len(f)):

                disX = f[i][0] - f[0][0]
                if abs(maxDisX) < abs(disX):
                    maxDisX = abs(disX)

                disY = f[i][1] - f[0][1]
                if abs(maxDisY) < abs(disY):
                    maxDisY = abs(disY)

            for k in range(len(f)):
                #f[k] = (((f[k][0] - f[0][0]) / maxDisX), ((f[k][1] - f[0][1]) / maxDisY))
                disX = (f[k][0] - f[0][0])/maxDisX
                disY = (f[k][1] - f[0][1])/maxDisY
                normHand.append((disX,disY))
                #f[k] = (disX,disY)
            normHands.append(normHand)
        myHands = normHands
        handkeypoints = dict()
        idx = 0
        for i in range(int(len(myHands) / 2)):
            myHands[i * 2][0] = handAngels[i*2]
            myHands[i * 2+1][0] = handAngels[i*2+1]
            handkeypoints[idx] = myHands[i * 2] + myHands[i * 2 + 1]
            idx += 1
        df = pd.DataFrame(handkeypoints)
        df = df.transpose()
        savePath = os.path.join('D:/aiData/keypoints/', m)
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        savePath = os.path.join('D:/aiData/keypoints/', m, m+'-'+fname + '.csv')
        csvSave(df, savePath)
        mCount += 1
        print(mCount,"/", len(morphemes.keys()))
cap.release()
#out.release()
cv2.destroyAllWindows()