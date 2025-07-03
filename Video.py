import cv2

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_haar_cascade = cv2.CascadeClassifier(cascade_path)

capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Error: Could not open video device.")
    exit()

if face_haar_cascade.empty():
    print("Error: Failed to load Haar cascade.")
    exit()

while True:
    ret, image = capture.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)
    print(f"Faces detected: {len(faces)}")

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", image)

    if cv2.waitKey(30) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()
