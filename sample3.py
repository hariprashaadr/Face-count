import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def count_faces_in_image():
    image_path = input("Enter image file path: ")
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found!")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    print(f"Faces detected in image: {len(faces)}")

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Image - Face Count", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def count_faces_in_video():
    video_path = input("Enter video file path: ")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(frame, f'Count: {len(faces)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Video - Face Count", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    print(f"Faces detected in video: {len(faces)}")
    cap.release()
    cv2.destroyAllWindows()

def count_faces_from_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not accessible.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(frame, f'Count: {len(faces)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Camera - Face Count", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === Main Menu ===
def main():
    while True:
        print("\n--- Face Counting Menu ---")
        print("1. Count faces in an image")
        print("2. Count faces in a video")
        print("3. Count faces from camera")
        print("4. Exit")

        choice = input("Choose an option (1-4): ")

        if choice == '1':
            count_faces_in_image()
        elif choice == '2':
            count_faces_in_video()
        elif choice == '3':
            count_faces_from_camera()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid option. Try again!")

if __name__ == "__main__":
    main()
