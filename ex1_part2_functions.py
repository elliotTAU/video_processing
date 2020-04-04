import numpy as np
import pandas as pd
from scipy.spatial import distance
import cv2
from scipy import signal
def write_video(video_name,cap):
    writer_fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_out = cv2.VideoWriter(video_name, writer_fourcc, fps, (frame_width, frame_height), isColor=False)
    return video_out

def Q2A(video_path):
    cap = cv2.VideoCapture(video_path)
    # Read until video is completed
    video_out = write_video('Vid1_Binary.avi',cap)
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret2, frame = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
            video_out.write(frame)
            cv2.imshow('Frame', frame)

            # Press q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    video_out.release()
    cv2.destroyAllWindows()
    return

def Q2B(video_path):
    cap = cv2.VideoCapture(video_path)
    # Read until video is completed
    video_out = write_video('Vid2_Grayscale.avi',cap)
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display and write the resulting frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            video_out.write(frame)
            cv2.imshow('Frame', frame)

            # Press q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    video_out.release()
    cv2.destroyAllWindows()
    return

def Q2C(video_path):
    cap = cv2.VideoCapture(video_path)
    # Read until video is completed
    video_out = write_video('Vid3_Sobel.avi',cap)
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            sobel_x = cv2.Sobel(gray,-1,1,0,ksize=3)
            sobel_y = cv2.Sobel(gray,-1,0,1,ksize=3)
            #frame  = np.square(np.square(np.absolute(sobel_x)) + np.square(np.absolute(sobel_y)))
            frame   = np.absolute(sobel_x) + np.absolute(sobel_y)

            video_out.write(frame)
            cv2.imshow('Frame', frame)

            # Press q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    video_out.release()
    cv2.destroyAllWindows()
    return


def main():

    video_path = r'C:\Users\elellouc\Desktop\TAU\Semestre 7\Video processing\HW1\HW1_files\atrium.avi'
    #Q2A(video_path)
    #Q2B(video_path)
    Q2C(video_path)

    return

main()
