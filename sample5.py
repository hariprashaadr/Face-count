import cv2
import numpy as np

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained gender detection model
gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt','gender_net.caffemodel')
gender_list = ['Male', 'Female']

def get_gender(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                 (78.426, 87.768, 114.895), swapRB=False)
    gender_net.setInput(blob)
    preds = gender_net.forward()
    return gender_list[preds[0].argmax()]

def detect_from_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Could not read image!")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    male_count = 0
    female_count = 0

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        try:
            face_resized = cv2.resize(face_img, (227, 227))
            gender = get_gender(face_resized)

            color = (255, 0, 0) if gender == 'Male' else (255, 0, 255)
            if gender == 'Male':
                male_count += 1
            else:
                female_count += 1

            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except:
            continue

    total = male_count + female_count
    cv2.putText(img, f"Total: {total}  Male: {male_count}  Female: {female_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Gender Detection - Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_from_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        male_count = 0
        female_count = 0

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            try:
                face_resized = cv2.resize(face_img, (227, 227))
                gender = get_gender(face_resized)

                color = (255, 0, 0) if gender == 'Male' else (255, 0, 255)
                if gender == 'Male':
                    male_count += 1
                else:
                    female_count += 1

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            except:
                continue

        total = male_count + female_count
        cv2.putText(frame, f"Total: {total}  Male: {male_count}  Female: {female_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Gender Detection - Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    print("\n--- Gender Detection ---")
    print("1. Detect from image")
    print("2. Detect from camera")
    choice = input("Choose mode (1 or 2): ")

    if choice == '1':
        img_path = input("Enter path to image: ")
        detect_from_image(img_path)
    elif choice == '2':
        detect_from_camera()
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
