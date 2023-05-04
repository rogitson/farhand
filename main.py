import cv2
import mediapipe as mp
import numpy as np
from farhand import HandTracking
from farhand_utils import CvFps, draw_fps, streamer_view
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


def main():
    cap = cv2.VideoCapture(0)

    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # FPS Measurement
    cvfps = CvFps(10)

    ht = HandTracking(img_w=cap_width, img_h=cap_height,
                      pad_w=0, pad_h=0, crop_size=512, mode="higher")

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose, \
        mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            fps = cvfps.get()

            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw the pose annotation on the image.
            annotated = image.copy()
            mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # Skip iteration if no pose detected
            if not results.pose_landmarks:
                continue

            scores = [
                landmark.visibility for landmark in results.pose_landmarks.landmark]
            keypoints = np.array([(landmark.x, landmark.y)
                                  for landmark in results.pose_landmarks.landmark])

            ht.setBody(keypoints, scores)

            hand_crop_region = ht.get_crop_region()
            if hand_crop_region:
                hand_cropped = ht.crop_and_resize(
                    image, hand_crop_region)

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                hand_cropped.flags.writeable = False
                hand_cropped = cv2.cvtColor(hand_cropped, cv2.COLOR_BGR2RGB)
                results = hands.process(hand_cropped)

                # Draw the hand annotations on the image.
                hand_cropped.flags.writeable = True
                hand_cropped = cv2.cvtColor(hand_cropped, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            hand_cropped,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
            else:
                hand_cropped = np.zeros_like(image)

            image = streamer_view(annotated, hand_cropped)
            image = draw_fps(image, fps)
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


if __name__ == "__main__":
    main()
