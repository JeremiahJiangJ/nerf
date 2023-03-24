import cv2
import os
import numpy as np

'''
Lucas-Kanade Algorithm For Sparse Optical Flow
Main Idea: Based on local motion constancy assumption where nearby pixels have the same displacement direction
The assumption helps to get the approximated solution for the equation with 2 variables

Assume that the neighbouring pixels have the same motion vector [deltaX, deltaY].
Take a fixed size window to create a system of equations.
Let the pixel coordinates in the chose window of n elements be pi = (xi, yi)
'''
def lucas_kanade_method(video_root, video_name):
	of_frames = []
	# Read in the video
	cap = cv2.VideoCapture(os.path.join(video_root,video_name))
	
	# Parameters for ShiTomasi corner detection
	feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)

	# Parameters for Lucas Kanade Optical Flow
	lk_params = dict(
		winSize = (15, 15),
		maxLevel = 2,
		criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	# Create random colors
	color = np.random.randint(0, 255, (100, 3))

	# Take first frame and find corners in it
	ret, old_frame = cap.read()
	h, w, _ = old_frame.shape
	size = (w, h)
	old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
	p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

	# Create a mask image for drawing
	mask = np.zeros_like(old_frame)
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	curr_frame = 0
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		curr_frame += 1
		print(f"Reading Frame {curr_frame}/{total_frames}")
		h, w, _ = frame.shape
		size = (w, h)
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Calculate Optical Flow
		p1, st, _= cv2.calcOpticalFlowPyrLK(
			old_gray, frame_gray, p0, None, **lk_params)

		# Select good points
		if p1 is not None:
			good_new = p1[st == 1]
			good_old = p0[st == 1]

		# Draw the tracks
		for i, (new, old) in enumerate(zip(good_new, good_old)):
			a, b = new.ravel()
			c, d = old.ravel()
			mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
			frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

		# Display
		img = cv2.add(frame, mask)
		#cv2.imshow("frame", img)
		#k = cv2.waitKey(25) & 0xFF
		#if k == 27:
		#	break

		old_gray = frame_gray.copy()
		p0 = good_new.reshape(-1, 1, 2)
		of_frames.append(img)
	return of_frames, size