import cv2
import time


class VideoProcessor:
    def __init__(self, video_path, subtractor_type):
        self.video_path = video_path
        self.subtractor_type = subtractor_type

        self.mask_template = None

        self.cap = cv2.VideoCapture(video_path)
        self.bg_subtractor = self.create_subtractor(subtractor_type)

        # cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

    def create_subtractor(self, subtractor_type):
        if subtractor_type == 'KNN':
            return cv2.createBackgroundSubtractorKNN()
        elif subtractor_type == 'MOG2':
            return cv2.createBackgroundSubtractorMOG2()
        elif subtractor_type == 'MOG':
            return cv2.bgsegm.createBackgroundSubtractorMOG()
        elif subtractor_type == 'GMG':
            return cv2.bgsegm.createBackgroundSubtractorGMG()
        elif subtractor_type == 'MANUAL':
            return True
        else:
            raise ValueError("Invalid subtractor_type. Use 'KNN' , 'MOG', 'MOG2, 'GMG' or 'MANUAL.")

    def process_video(self):
        t = 0
        iteration = 0

        if self.subtractor_type == 'MANUAL':
            _, first_frame = self.cap.read()

            # Save the first image as reference
            first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            first_gray = cv2.GaussianBlur(first_gray, (21, 21), 0)

        while True:
            beginning_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break

            masked_frame = cv2.bitwise_and(frame, frame, mask=self.mask_template)
            if self.subtractor_type != 'MANUAL':
                foreground_mask = self.bg_subtractor.apply(masked_frame)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                # In each iteration, calculate absolute difference between current frame and reference frame
                foreground_mask = cv2.absdiff(gray, first_gray)

            # cv2.imshow('Foreground Mask', foreground_mask)  # Show the foreground mask with ROI applied
            # cv2.imshow('Video', frame)  # Display the original frame

            ending_time = time.time()
            loop_time = ending_time - beginning_time

            iteration += 1
            t += loop_time

            if cv2.waitKey(60) & 0xFF == 27:  # Press Esc to exit
                break

        average = t / iteration
        print('%.5f' % average)

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Background Subtraction with ROI Selection')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('subtractor_type', type=str, choices=['KNN', 'MOG', 'MOG2', 'GMG', 'MANUAL'],
                        help='Background subtractor type')

    args = parser.parse_args()

    video_processor = VideoProcessor(args.video_path, args.subtractor_type)

    video_processor.process_video()
