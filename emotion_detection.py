import cv2
from deepface import DeepFace

# create a face detector object
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# create an emotion detector object
emotion_detector = DeepFace.build_model('Emotion')

# function to detect emotions in an image
def detect_emotions_in_image(image_path):
    # load the image
    img = cv2.imread(image_path)

    # detect the face
    face = DeepFace.detectFace(img, detector_backend='opencv')

    # detect the emotion
    emotion = DeepFace.analyze(face, actions=['emotion'], model=emotion_detector)

    # print the emotion scores
    print(emotion["emotion"])

    # show the image with the bounding box and emotion text
    cv2.imshow("Emotion Detection", cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# function to detect emotions in real-time from camera
def detect_emotions_in_realtime():
    # create a VideoCapture object for the camera
    cap = cv2.VideoCapture(0)

    # loop over frames from the camera
    while True:
        # read a frame from the camera
        ret, frame = cap.read()

        # convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # loop over the faces and detect emotions
        for (x, y, w, h) in faces:
            # extract the face ROI
            face = frame[y:y+h, x:x+w]

            # detect the emotion
            emotion = DeepFace.analyze(face, actions=['emotion'], model=emotion_detector)

            # draw the bounding box and emotion text on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion['dominant_emotion'], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # display the resulting frame
        cv2.imshow('Emotion Detection', frame)

        # break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the VideoCapture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

# main program loop
while True:
    # ask the user whether to upload an image or detect emotions in real-time
    print("Enter 'i' to upload an image or 'c' for real-time camera emotion detection, or 'q' to quit:")
    choice = input()

    # perform the selected task
    if choice == 'i':
        # ask the user for the path to the image file
        image_path = input("Enter the path to the image file: ")

        # detect emotions in the image
        detect_emotions_in_image(image_path)

    elif choice == 'c':
        # detect emotions in real-time from camera
        detect_emotions_in_realtime()

    elif choice == 'q':
        break

    else:
        print("Invalid choice. Please try again.")
