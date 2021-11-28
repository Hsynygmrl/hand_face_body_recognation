# import mediapipe as mp
# import cv2

# mp_drawing = mp.solutions.drawing_utils
# mp_holistic = mp.solutions.holistic

# cap = cv2.VideoCapture(0)
# with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:

#     while cap.isOpened():
#         scss,frame = cap.read()
#         # recolor Feed
#         image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#         # Make Detections
#         results = holistic.process(image)
#         # print(results.face_landmarks)
#         # print(results.pose_landmarks)

#         # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_ladmarks

#         # Recolor image back to BGR for rendering
#         image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
#         # Draw face landmarks
#         mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holistic.FACEMESH_TESSELATION) # FACE_CONNECTIONS ölmüş
        
#         # right hand
#         mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
#         # left hand
#         mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
#         # pose detections
#         mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS)

#         cv2.imshow('Holistic Model Detections',image)
#         if cv2.waitKey(10) & 0xFF==27:
#             break
#         # print(mp_holistic.POSEMESH_TESSELATION)
# cv2.destroyAllWindows()
# cap.release()

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

mp_drawing.DrawingSpec(color=(0,0,255),thickness=2,circle_radius=2)
cap = cv2.VideoCapture('mmg.jpeg')
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        scss,frame = cap.read()
        # recolor Feed
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)
        # print(results.pose_landmarks)

        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_ladmarks

        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holistic.FACEMESH_TESSELATION, # FACE_CONNECTIONS ölmüş
        mp_drawing.DrawingSpec(color=(50,50,0),thickness=2,circle_radius=4), # circlar için
        mp_drawing.DrawingSpec(color=(50,50,50),thickness=2,circle_radius=1)) # çizgiler için        
        # 2. right hand
        mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(255,0,0),thickness=2,circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(255,0,0),thickness=2,circle_radius=1))
        
        # 3. left hand
        mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(255,0,0),thickness=2,circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(255,0,0),thickness=2,circle_radius=1))
        
        # 4.pose detections
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,255,255),thickness=1,circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(0,255,255),thickness=1,circle_radius=1))

        cv2.imshow('Holistic Model Detections',image)
        plt.imshow(image)
        if cv2.waitKey(10) & 0xFF==27:
            break
        # print(mp_holistic.POSEMESH_TESSELATION)
cv2.destroyAllWindows()
cap.release()