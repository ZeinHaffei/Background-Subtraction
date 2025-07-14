import numpy as np
import cv2
import time

# cap = cv2.VideoCapture(0)
video = cv2.VideoCapture('vtest.avi')
success, first_frame = video.read()

# Save the first image as reference
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray, (21, 21), 0)

t = 0
iteration = 0

while True:
    beginning_time = time.time()

    success, frame = video.read()

    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # In each iteration, calculate absolute difference between current frame and reference frame
    difference = cv2.absdiff(gray, first_gray)

    # Apply thresholding to eliminate noise
    thresh = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1] # 25 is the threshold value, 255 is the max threshold
    thresh = cv2.dilate(thresh, None, iterations=2)

    # # in order to show contours
    # contours, hirarcy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

    cv2.imshow('Original', frame)
    cv2.imshow("thresh", difference)

    # # concatenate image Horizontally
    # Hori = np.concatenate((gray, thresh), axis=1)
    # cv2.imshow('HORIZONTAL', Hori)
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
print('%.5f' %average)
video.release()
cv2.destroyAllWindows()