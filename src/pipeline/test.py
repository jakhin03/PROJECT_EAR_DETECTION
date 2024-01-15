import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
detector_model = YOLO('yolo_detector.pt')
recognition_model = YOLO('yolo_classifier.pt')

# define a video capture object
vid = cv2.VideoCapture(0)

while(True):
    fps = vid.get(cv2.CAP_PROP_FPS)
    print('fps:', fps)
    # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    # Capture the video frame by frame
    # _, frame = vid.read()
    frame = cv2.imread('./001 (14).jpg')
    # Run YOLOv8 inference on the frame
    detected_roi = detector_model(frame)

    # Visualize the results on the frame
    annotated_frame = detected_roi[0].plot()

    recognition_result = recognition_model(frame)  # predict on an image
    # recognized_subject = recognition_result[0].names[recognition_result[0].probs.argmax().item()]
    if  recognition_result[0].boxes:
        ids = recognition_result[0].boxes.id.cpu().numpy().astype(int)
        for id_ in ids:
            cv2.putText(img=annotated_frame, 
                    text=id_,
                    org=(250, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=2,
                    lineType=2)   

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

    # the 'q' button is set as the quitting button you may use any desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
