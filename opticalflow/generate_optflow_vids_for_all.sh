algorithms=("lk_sparse" "lk_dense" "farneback")

videos=()
search_dir=./videos
for video in $(ls $search_dir)
do
	videos+=("${video}")
done

for video in "${videos[@]}"; do
	for algorithm in "${algorithms[@]}"; do
		python opticalflow.py --video_name "$video" --algorithm "$algorithm" --save_vid
	done
done
