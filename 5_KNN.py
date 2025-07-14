import cv2
import time

backSub = cv2.createBackgroundSubtractorKNN()

video = cv2.VideoCapture('sp1-1.mp4')
success, frame = video.read()

t = 0
iteration = 0

while True:

    beginning_time = time.time()

    success, frame = video.read()
    if not success:
        break

    fg_mask = backSub.apply(frame)

    cv2.imshow('Original Frame', frame)
    cv2.imshow('FG Mask', fg_mask)

    ending_time = time.time()
    loop_time = ending_time - beginning_time

    iteration += 1
    t += loop_time

    # print('%.2f' % loop_time)
    print(loop_time)
    # fps = int(1 / loop_time)

    key = cv2.waitKey(60) & 0xFF

    # if the `q` key is pressed, break from the loop
    if key == ord("q"):
        break

average = t / iteration
# print('%.5f' % average)
print(average)

video.release()
cv2.destroyAllWindows()