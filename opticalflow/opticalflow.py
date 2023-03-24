from genericpath import isfile
import cv2
import os
import argparse
from algorithms.sparse_optical_flow import lucas_kanade_method
from algorithms.dense_optical_flow import dense_optical_flow
from utils.save_video_from_frames import save_video_from_frames
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--video_root", type = str, default = "D:/nerf/opticalflow/videos/", help = "root directory of video")
	parser.add_argument("--video_name", type = str, required = True, help = "name of video")
	parser.add_argument("--save_path", type = str, default = "D:/nerf/opticalflow/opticalflow_videos/", help = "diretory to save video to")
	#parser.add_argument("--video_path", type = str, required = True, default = "./videos/IMG_5420.mp4", help = "path to video")
	parser.add_argument("--algorithm", type = str, default = "farneback", choices= ["lk_sparse",
																					"lk_dense",
																					"farneback",
																					"rlof"],
						help = "Optical flow algorithm to use")
	parser.add_argument("--save_vid", action='store_true')

	args = parser.parse_args()
	video_root = args.video_root
	video_name = args.video_name
	algorithm = args.algorithm
	save_path = args.save_path
	save_vid = args.save_vid

	if os.path.isfile(os.path.join(video_root, video_name)) == False:
		print(f"{os.path.join(video_root, video_name)} is INVALID")
		return

	print(f"Generating optical flow for {video_name} using {algorithm}")
	if algorithm == "lk_sparse":
		frames, size = lucas_kanade_method(video_root, video_name)

	elif algorithm == "lk_dense":
		method = cv2.optflow.calcOpticalFlowSparseToDense
		frames, size = dense_optical_flow(method, video_root, video_name, to_gray=True)

	elif algorithm == "farneback":
		method = cv2.calcOpticalFlowFarneback
		params = [0.5, 3, 15, 3, 5, 1.2, 0] #default Farneback's algorithm parameters
		frames, size = dense_optical_flow(method, video_root, video_name, params, to_gray=True)

	elif algorithm == "rlof":
		method = cv2.optflow.calcOpticalFlowDenseRLOF
		#method = cv2.estimateAffinePartial2D
		frames, size = dense_optical_flow(method, video_root, video_name)

	if args.save_vid:
		save_name = f"{video_name}_"
		save_video_from_frames(video_name, save_path, frames, size, algorithm)

if __name__ == "__main__":
	main()

