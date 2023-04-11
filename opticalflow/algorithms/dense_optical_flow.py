import os
import cv2
import numpy as np

'''
Direction of flow represented by hue, same color same direction
Magnitude of flow by value of HSV color, darker -> smaller magnitude, brighter -> larger magnitude
'''
def dense_optical_flow(method, video_root, video_name, params=[], to_gray=False):
    of_frames = []
    of_values = []
    # Read the video and first frame
    cap = cv2.VideoCapture(os.path.join(video_root, video_name))
    ret, old_frame = cap.read()
    ref = old_frame
    h, w, _ = old_frame.shape
    size = (w, h)
    # create HSV & make Value a constant
    #hsv = np.zeros_like(old_frame)
    #hsv[..., 1] = 255
 
    # Preprocessing for exact method
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    curr_frame = 0
    while True:
        # Read the next frame
        ret, new_frame = cap.read()
        hsv = np.zeros_like(ref)
        hsv[..., 1] = 255
        #frame_copy = new_frame
        
        if not ret:
            break
        temp = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
        curr_frame += 1
        print(f"Reading Frame {curr_frame}/{total_frames}")

        # Preprocessing for exact method
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
 
        # Calculate Optical Flow
        flow = method(old_frame, new_frame, None, *params)

        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        #print(ang)
        # Use Hue and Value to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
 
        # Convert HSV image into BGR 
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        #cv2.imshow("frame", frame_copy)
        #cv2.imshow("optical flow", bgr)
        #k = cv2.waitKey(25) & 0xFF
        #if k == ord('e'):
        #    break
        #elif k == ord('s'):
        #    cv2.imwrite('Optical_image.png', frame_copy)
        #    cv2.imwrite('hsv_converted_image.png', bgr)

        # Update the previous frame

        old_frame = new_frame
        #temp = cv2.cvtColor(old_frame, cv2.COLOR_GRAY2BGR)
        dense_flow = cv2.addWeighted(temp, 1, bgr, 2, 0)
        #of_frames.append(bgr)
        of_frames.append(dense_flow)
        of_values.append(bgr)
    return of_frames, of_values, size
