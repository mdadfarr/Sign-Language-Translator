import mediapipe as mp
import cv2

mp_draw = mp.solutions.drawing_utils


hands = mp.solutions.hands.Hands()

video = cv2.VideoCapture(0)

while (video.isOpened()):
    success, frame = video.read()
    
    # MediaPipe expects the image in RGB format, but OpenCV gives you frames in BGR format 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
video.release()
cv2.destroyAllWindows()








    
