import cv2
import time

backSub = cv2.bgsegm.createBackgroundSubtractorMOG()
# in order to use bgsegm kernel we should have contrib installed

video = cv2.VideoCapture('vtest.avi')
success, frame = video.read()

t = 0
iteration = 0

while success:

    beginning_time = time.time()

    success, frame = video.read()

    if not success:
        break

    fg_mask = backSub.apply(frame)

    # Apply thresholding to eliminate noise
    # thresh = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)[1] # 25 is the threshold value, 255 is the max threshold
    # thresh = cv2.dilate(thresh, None, iterations=2)

    cv2.imshow('Original Frame', frame)
    cv2.imshow('FG Mask', fg_mask)

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