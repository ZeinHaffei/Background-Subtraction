import cv2
import numpy as np
import time

# backSub = cv2.createBackgroundSubtractorKNN()
backSub = cv2.createBackgroundSubtractorMOG2()
# backSub = cv2.createBackgroundSubtractorGMG()

# video = cv2.VideoCapture(0)
video = cv2.VideoCapture('vtest.avi')
# video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not video.isOpened():
    print("Cannot open the file specified")

t = 0
iteration = 0


while True:
    beginning_time = time.time()

    success, frame = video.read()
    if frame is None:
        break

    # frame = cv2.resize(frame, (800, 600))
    fgMask = backSub.apply(frame)

    kernel = np.ones((2, 2), np.uint8)

    fgMask = cv2.erode(fgMask, kernel, iterations=2)
    fgMask = cv2.dilate(fgMask, kernel, iterations=2)

    cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv2.putText(frame, str(video.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    fgMask[np.abs(fgMask) < 250] = 0

    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)

    ending_time = time.time()
    loop_time = ending_time - beginning_time

    iteration += 1
    t += loop_time

    print('%.2f' % loop_time)
    fps = int(1 / loop_time)

    key = cv2.waitKey(60) & 0xFF

    # if the `q` key is pressed, break from the loop
    if key == ord("q"):
        break

average = t / iteration
print('%.5f' % average)

video.release()
cv2.destroyAllWindows()