import os
import cv2

def save_video_from_frames(video_name, save_path, frames, frame_size, algo_used, save_ext="mp4"):
	if os.path.isdir(save_path) == False:
		os.mkdir(save_path)
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	#og_ext = video_name.split(".")[-1]
	[name, og_ext] = video_name.split(".")
	save_name = f"{name}_{algo_used}.{save_ext}"
	#save_name = video_name.replace(og_ext, save_ext)

	out = cv2.VideoWriter(os.path.join(save_path, save_name), fourcc, 30, frame_size, True)
	for i in frames:
		out.write(i)
	out.release()
