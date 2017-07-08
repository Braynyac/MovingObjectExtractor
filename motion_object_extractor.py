import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
counter = 0
corners = []
while True:
    time1 = time.time()
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    w, h, c = frame.shape
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if counter > 0:
        delta_frame = np.multiply(cv2.absdiff(gray_frame, previous_frame), 2)
        mean_delta = np.mean(delta_frame)
        if cv2.countNonZero(delta_frame) != 0:
            _, delta_threshold = cv2.threshold(delta_frame, 100, 255, cv2.THRESH_BINARY)
            if cv2.countNonZero(delta_threshold) != 0:
                edges = cv2.Canny(delta_threshold, mean_delta * .75, mean_delta * 1.25)
                corners_ = cv2.goodFeaturesToTrack(edges, w*h, .01, 0)
                if corners_ is not None:
                    corners = np.array(corners_, np.int32)
                    extraction_img = np.zeros((w, h), np.uint8)
                    convex_hull = cv2.convexHull(corners)
                    cv2.fillPoly(extraction_img, [convex_hull], (255, 255, 255))
                    extraction = cv2.bitwise_and(frame, frame, mask=extraction_img)
                    cv2.polylines(frame, [convex_hull], True, (255, 255, 255), 3)
                    for corner in corners:
                        x, y = corner.ravel()
                        cv2.circle(frame, (x, y), 1, 255, -1)
                    cv2.imshow('extraction', extraction)
        cv2.imshow('frame', frame)
    previous_frame = gray_frame
    counter += 1
    k = cv2.waitKey(30) & 0xFF
    time2 = time.time()
    fps = (time2-time1)**-1
    print('FPS:', fps)
    if k == ord('q'):
        break
cv2.destroyAllWindows()
