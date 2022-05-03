import cv2
import mediapipe as mp
import json
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
handsType=[]

cap = cv2.VideoCapture('./video/NIA_SL_SEN0001_REAL01_D.mp4')

width = int(cap.get(3)) # 가로 길이 가져오기
height = int(cap.get(4)) # 세로 길이 가져오기
fps = 30

fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
#out = cv2.VideoWriter('NIA_SL_SEN0001_REAL01_D_hand.avi', fcc, fps, (width, height))

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
    myHands = []
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        myHand = []
        for id, lm in enumerate(hand_landmarks.landmark):
# h, w, c = image.shape
# cx, cy = int(lm.x * w), int(lm.y * h)
# res = {id, cx, cy}
          myHand.append((int(lm.x * width), int(lm.y * height)))
          myHands.append(myHand)

          mp_drawing.draw_landmarks(
              image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

      for hand in results.multi_handedness:
        handType = hand.classification[0].label
        handsType.append(handType)

      # mp_drawing.draw_landmarks(
      #     image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)

    cv2.imshow('MediaPipe Hands', image)
    #out.write(image)
    if cv2.waitKey(5) & 0xFF == 27:
        break


list(zip(myHands, handsType))
for val1, val2 in zip(myHands, handsType):
    print(val1, val2)

jsonString = json.dumps(list(zip(myHands, handsType)))
print(jsonString)

with open("'NIA_SL_SEN0001_REAL01_D_keypoints.json", "w") as f:
    json.dump(jsonString, f)

    #for hand, handType in zip(myHands, handsType):
      #for ind in [0, 5, 6, 7, 8]:
        #cv2.circle(fps, hand[ind], 15)


cap.release()
#out.release()
cv2.destroyAllWindows()
