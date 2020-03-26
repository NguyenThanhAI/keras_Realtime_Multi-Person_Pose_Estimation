In this repo, we write ```run_pose_annotator.py``` to extract keypoints and create videos with information.
Running:
```shell script
python run_pose_annotator.py --dataset_dir=$DATASET_DIR --output_dir=$OUTPUT_DIR --is_save_video={True, False} --video_save_dir=$VIDEO_SAVE_DIR --model=model/keras/model.h5 --extract_every_num_frames=EXTRACT_EVERY_NUM_FRAMES
```

* ```--dataset_dir```: Directory includes videos of all classes of MMAct dataset. A dataset with this structure:

```
/data4t
 │   ├── thanhnc 
 │   │   ├── MMAct
 │   │   │   ├── videos
 │   │   │   │   ├── subject1
 │   │   │   │   │   ├── cam1
 │   │   │   │   │   │   ├── scene1
 │   │   │   │   │   │   │   ├── session1
 │   │   │   │   │   │   │   │   ├── carrying.mp4
 │   │   │   │   │   │   │   │   ├── ...
 │   │   │   │   │   │   │   ├── ...
 │   │   │   │   │   │   ├── ...
 │   │   │   │   │   ├── ...
 │   │   │   │   ├── ...
 │   │   │   │   ├── subject20
 │   │   │   │   │   ├── cam1
 │   │   │   │   │   │   ├── scene1
 │   │   │   │   │   │   │   ├── session1
 │   │   │   │   │   │   │   │   ├── carrying.mp4
 │   │   │   │   │   │   │   │   ├── ...
 │   │   │   │   │   │   │   ├── ...
 │   │   │   │   │   │   ├── ...
 │   │   │   │   │   ├── ...  
```
In this example, ```$DATASET_DIR=/data4t/thanhnc/MMAct/videos```

*```--output_dir```: Directory includes json files with the same structure in ```$DATASET_DIR```. If ```$OUTPUT_DIR``` hasn't existed, it will be created.
Example, if ```$OUTPUT_DIR=/data4t/thanhnc/MMAct_annotator```. The folder structure will look like:
```
/data4t
 │   ├── thanhnc 
 │   │   ├── MMAct_annotator
 │   │   │   ├── subject1
 │   │   │   │   ├── cam1
 │   │   │   │   │   ├── scene1
 │   │   │   │   │   │   ├── session1
 │   │   │   │   │   │   │   ├── carrying.json
 │   │   │   │   │   │   │   ├── ...
 │   │   │   │   │   │   ├── ...
 │   │   │   │   │   ├── ...
 │   │   │   │   ├── ...
 │   │   │   ├── ...
 │   │   │   ├── subject20
 │   │   │   │   ├── cam1
 │   │   │   │   │   ├── scene1
 │   │   │   │   │   │   ├── session1
 │   │   │   │   │   │   │   ├── carrying.json
 │   │   │   │   │   │   │   ├── ...
 │   │   │   │   │   │   ├── ...
 │   │   │   │   │   ├── ...
 │   │   │   │   ├── ...  
```
* ```--is_save_video```: Save videos with keypoints information or not
* ```--video_save_dir```: Directory includes output videos with the same structure in ```$DATASET_DIR```. If ```$VIDEO_SAVE_DIR``` hasn't existed, it will be created.
Example, if ```$OUTPUT_DIR=/data4t/thanhnc/MMAct_save_videos```. The folder structure will look like:
```
/data4t
 │   ├── thanhnc 
 │   │   ├── MMAct_save_videos
 │   │   │   ├── subject1
 │   │   │   │   ├── cam1
 │   │   │   │   │   ├── scene1
 │   │   │   │   │   │   ├── session1
 │   │   │   │   │   │   │   ├── carrying.json
 │   │   │   │   │   │   │   ├── ...
 │   │   │   │   │   │   ├── ...
 │   │   │   │   │   ├── ...
 │   │   │   │   ├── ...
 │   │   │   ├── ...
 │   │   │   ├── subject20
 │   │   │   │   ├── cam1
 │   │   │   │   │   ├── scene1
 │   │   │   │   │   │   ├── session1
 │   │   │   │   │   │   │   ├── carrying.json
 │   │   │   │   │   │   │   ├── ...
 │   │   │   │   │   │   ├── ...
 │   │   │   │   │   ├── ...
 │   │   │   │   ├── ...  
```
Note: if ```--is_save_video=False```, ```--video_save_dir``` will be ignored.

* ```--model```: Path to checkpoint. By default, ```--model=model/keras/model.h5```
*  ```--extract_every_num_frames```: Extracting keypoints every ```--extract_every_num_frames``` as well as saves to video if ```--is_save_video=True```

Example:
```shell script
python run_pose_annotator.py --dataset_dir=/data4t/thanhnc/MMAct/videos --output_dir=/data4t/thanhnc/MMAct_annotator --is_save_video=True --video_save_dir=/data4t/thanhnc/MMAct_save_videos --model=model/keras/model.h5 --extract_every_num_frames=1
```