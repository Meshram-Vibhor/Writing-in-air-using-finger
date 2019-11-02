import cv2
import numpy as np

def nothing(x):
    pass

x1 = []             ##List to store x coordinates
y1 = []             ##List to store y coordinates
Kernal = np.ones((5,5), np.uint8)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)     ##Change resolution of the camera
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
_, frame =cap.read()
print(frame.shape)
while(1):
    ret, frame = cap.read()         ##Read frame
    frame = cv2.flip(frame, +1)     ##Mirror image frame
    if not ret:
        break
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)     ##BGR to HLS conversion
    if cv2.waitKey(1) == ord('s'):
        break

    lb = np.array([0, 0, 81])               ##Masking parameters
    ub = np.array([185, 255, 255])

    mask = cv2.inRange(frame2, lb, ub)      ##Create mask
    cv2.imshow('Mask', mask)

    res = cv2.bitwise_and(frame, frame, mask = mask)        ##apply mask on original image
    cv2.imshow('Res', res)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, Kernal)        ##Apply opening morphology
    cv2.imshow('Opening', opening)

    dilation = cv2.dilate(opening, Kernal, iterations=5)            ##Apply dilation morphology
    cv2.imshow('Dilation', dilation)

    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE,      ##FInd contours
                                           cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 1500:         ##Find contour with largest area
            cnt = contour
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])           ##Find centroid
    cy = int(M['m01'] / M['m00'])
    cv2.circle(frame, (cx, cy), 5, [50, 120, 255], -1)      ##Draw centroid
    extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])           ##Find topmost point of object
    print(cx - extTop[0])
    if abs(cy - extTop[1]) > 200 and abs(cx - extTop[0]) < 150:     ##Distance between centroid and topmost point
        x1.append(extTop[0])
        y1.append(extTop[1])

    for i in range(len(x1)):
        cv2.circle(frame, (x1[i], y1[i]), 4, (255, 155, 100), 5)        ##Draw circle
    cv2.imshow('Resuting Image', frame)

cap.release()       ##Release memory
cv2.destroyAllWindows()     ##Destroy all windows