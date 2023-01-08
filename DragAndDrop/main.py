import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import cvzone as cvz


cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)


class DragRect():

    def __init__(self, pos, dim=[200, 200],colourR=(255,0,255)):
        self.pos = pos
        self.dim = dim
        self.colourR=colourR

    def update(self, cursor):
        cx, cy = self.pos
        w, h = self.dim
        
        if (cx-w//2 < cursor[0] < cx+w//2 and cy-h//2 < cursor[1] < cy+h//2):
            self.colourR = (0, 255, 0)
            self.pos = cursor[0], cursor[1]
        
        

rectList=[]
for i in range(5):
    rectList.append(DragRect([i*250+150, 150]))


# cx, cy, w, h = 100, 100, 200, 200


while True:
    success, img = cap.read()
    img = cv.flip(img, 1)
    hands, img = detector.findHands(img)  # with draw
    # hands = detector.findHands(img, draw=False)  # without draw

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right
        fingers1 = detector.fingersUp(hand1)

        if lmList1:
            length, info, img = detector.findDistance(
                lmList1[8][:2], lmList1[12][:2], img)
            # print(length)
            for rect in rectList:
                if (length < 55):
                    cursor = lmList1[8]
                    rect.update(cursor)
                else:
                    rect.colourR=(255,0,255)
                

    for rect in rectList:
        cx, cy = rect.pos
        w, h = rect.dim
        cv.rectangle(img, (cx-w//2, cy-h//2),
                    (cx+w//2, cy+h//2), rect.colourR, cv.FILLED)
        cvz.cornerRect(img, (cx-w//2, cy-h//2,w,h), 20,rt=0)
    cv.imshow("output", img)
    if (cv.waitKey(20) & 0xFF == ord('x')):
        break

cap.release()
cv.destroyAllWindows()
