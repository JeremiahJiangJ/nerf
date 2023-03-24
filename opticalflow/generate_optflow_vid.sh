video=$1
algorithms=("lk_sparse" "lk_dense" "farneback")
for algorithm in "${algorithms[@]}"; do
	python opticalflow.py --video_name "$video" --algorithm "$algorithm" --save_vid
done