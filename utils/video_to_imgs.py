import cv2
import time
import os
import argparse
import subprocess


def get_iframes(video_path, save_path):
    try:
        os.mkdir(save_path)
    except OSError:
        pass
    
    command = f'ffmpeg -i {video_path} -vf "select=eq(pict_type\,I)" -vsync vfr -qscale:v 1 {save_path}/IMG_%04d.jpg'
    out = subprocess.check_output(command)

def change_fps(video_path, save_path, target_fps):
    #try:
    #    os.mkdir(os.path.join(save_path,".."))
    #except OSError:
    #    pass
    #get_fps_cmd = f'ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate {video_path}'
    #fps = subprocess.check_output(get_fps_cmd, shell=True)
    #fps=fps.decode('utf-8')
    #for v in fps:
    #    print(v)
    #print(fps)
    downsample_cmd = f'ffmpeg -i {video_path} -filter:v fps={target_fps} {save_path}'
    out = subprocess.check_output(downsample_cmd)

def video_to_imgs(video_path, imgs_path):
    try:
        os.mkdir(imgs_path)
    except OSError:
        pass

    start_time = time.time()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    #print(cap.get(cv2.CAP_PROP_FPS))
    #downsampled_fps = int(fps/factor)
    #print(downsampled_fps)
    #cap.set(cv2.CAP_PROP_FPS, downsampled_fps)
    #print(cap.get(cv2.CAP_PROP_FPS))
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print(f"Number of frames: {frame_count}")

    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imwrite(os.path.join(imgs_path, f"IMG_{str(count+1).zfill(4)}.jpg"), frame)
        count += 1
        print(f"Converting frame {count}/{frame_count}")

        if count > frame_count - 1:
            end_time = time.time()
            cap.release()
            print(f"Took {end_time - start_time} seconds to convert {count} frames")
            break
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type = str, required = True, help = "Path to video for conversion")
    parser.add_argument("--save_path", type = str, required = True, help = "Save directory after video conversion to imgs")
    parser.add_argument("--get_iframes", action='store_true', help = "Specify whether to extract iFrames or not")
    parser.add_argument("--convert_fps", type = int, default=0, help = "Specify the fps of the video that you want to change to")
    parser.add_argument("--only_iframes", action='store_true')
    args = parser.parse_args()

    video_path = args.video_path
    video_name = video_path.split('/')[-1].split(".")[0]
    if args.convert_fps != 0:
        video_path  = os.path.join(os.path.join(args.video_path,'..', f'{video_name}_{args.convert_fps}.mp4'))
        change_fps(args.video_path, video_path, args.convert_fps)
    
    if(args.only_iframes == False):
        video_to_imgs(video_path, args.save_path)

    if(args.get_iframes):
        iframes_save_path = os.path.join(args.save_path, '..', 'iframes')
        print(iframes_save_path)
        get_iframes(video_path,  iframes_save_path)

if __name__ == "__main__":
    main()

