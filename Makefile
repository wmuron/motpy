demo:
	python examples/2d_multi_object_tracking.py

install-develop:
	python setup.py develop

test:
	python -m pytest ./tests

env-create:
	conda create --name env_motpy python=3.7 -y

env-activate:
	# might fail from makefile
	conda activate env_motpy

env-install:
	pip install -r requirements.txt && pip install -r requirements_dev.txt

clean:
	autoflake --in-place --remove-unused-variables ./motpy/*.py ./tests/*.py

static-check:
	mypy --ignore-missing-imports motpy

demo-mot16:
	# note it requires downloading MOT16 dataset before running
	python examples/mot16_challange.py --dataset_root=~/Downloads/MOT16 --seq_id=11

demo-webcam:
	python examples/webcam_face_tracking.py

demo-video:
	python examples/detect_and_track_in_video.py \
            --video_path=./assets/video.mp4 \
            --detect_labels=['car','truck'] \
            --tracker_min_iou=0.2 \
            --architecture=fasterrcnn \
            --device=cuda
