import os
import cv2

images_dir = "image_to_video"
output = "demo.avi"

image_list = []
for dirs, _, files in os.walk(images_dir):
    for file in files:
        if file.endswith(".jpg"):
            image_list.append(os.path.join(dirs, file))

print(image_list)

frame_array = []
for image in image_list:
    img = cv2.imread(image)
    height, width = img.shape[:2]
    frame_array.append(img)

out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))

for i in range(len(frame_array)):
    out.write(frame_array[i])

out.release()
