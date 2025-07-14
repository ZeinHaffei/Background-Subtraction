import numpy as np
import cv2
import time

video = cv2.VideoCapture('vtest.avi')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,  15))
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

iteration = 0
t = 0
while True:
    beginning_time = time.time()

    success, frame = video.read()

    width, height = frame.shape[0:2]
    print(width, height)
    if not success:
        break

    fgmask = fgbg.apply(frame)

    # Noise is removed with morphological opening
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('Original frame', frame)
    cv2.imshow('frame', fgmask)

    ending_time = time.time()
    loop_time = ending_time - beginning_time

    iteration += 1
    t += loop_time

    print('%.2f' % loop_time)
    # fps = int(1 / loop_time)

    key = cv2.waitKey(60) & 0xFF

    # if the `q` key is pressed, break from the loop
    if key == ord("q"):
        break

average = t / iteration
print('%.5f' % average)

video.release()
cv2.destroyAllWindows()