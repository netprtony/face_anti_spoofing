import numpy as np
import argparse
import cv2
import os
# python3 face_extract.py --input videos/real.mp4 --output dataset/real
# python3 face_extract.py --input videos/fake.mp4 --output dataset/fake
# Cac tham so dau vao
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
	help="path to input video")
ap.add_argument("-o", "--output", type=str, required=True,
	help="path to output directory of cropped faces")
ap.add_argument("-d", "--detector", type=str, default='face_detector',
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip", type=int, default=1,
	help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())

# Load model ssd nhan dien mat
print("[INFO] loading face detector...")
# protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
# modelPath = os.path.sep.join([args["detector"],
# 	"res10_300x300_ssd_iter_140000.caffemodel"])
protoPath = "face_detector/deploy.prototxt"
modelPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Doc file video input
vs = cv2.VideoCapture(args["input"])
read = 0
saved = 0

# Lap qua cac frame cua video
while True:
	(grabbed, frame) = vs.read()
	if not grabbed or frame is None:
		print("[INFO] No more frames to read or invalid frame.")
		break

	read += 1
	if read % args["skip"] != 0:
		continue

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	net.setInput(blob)
	detections = net.forward()

	if len(detections) > 0:
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		if confidence > args["confidence"]:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			startX = max(0, startX)
			startY = max(0, startY)
			endX = min(w, endX)
			endY = min(h, endY)

			if endX - startX == 0 or endY - startY == 0:
				print(f"[WARNING] Invalid face region at frame {read}. Skipping...")
				continue

			face = frame[startY:endY, startX:endX]
			face_resized = cv2.resize(face, (128, 128), interpolation=cv2.INTER_AREA)
			filename = f"{os.path.splitext(os.path.basename(args['input']))[0]}_{saved}.png"
			p = os.path.join(args["output"], filename)
			cv2.imwrite(p, face_resized)
			saved += 1
			print("[INFO] saved {} to disk".format(p))

vs.release()
cv2.destroyAllWindows()