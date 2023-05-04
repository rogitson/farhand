import cv2
import mediapipe as mp
import numpy as np
from collections import namedtuple
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# All values are in pixel. The region is a square of size 'size' pixels
CropRegion = namedtuple(
    'CropRegion', ['xmin', 'ymin', 'xmax',  'ymax', 'size'])

# Dictionary that maps from joint names to keypoint indices.
BODY_KP = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_pinky': 17,
    'right_pinky': 18,
    'left_index': 19,
    'right_index': 20,
    'left_thumb': 21,
    'right_thumb': 22,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_foot_index': 31,
    'right_foot_index': 32
}


class HandTracking:
    """
    Hand Tracking class
    """

    def __init__(self, img_w, img_h, pad_w, pad_h, crop_size, mode="group", score_thresh=0.2, scale=1.0, hands_up_only=True):

        self.img_w = img_w
        self.img_h = img_h
        self.pad_w = pad_w
        self.pad_h = pad_h
        self.crop_size = crop_size
        self.mode = mode
        self.score_thresh = score_thresh
        self.scale = scale
        self.hands_up_only = hands_up_only

    def setBody(self, keypoints_normalized, scores):
        self.keypoints_norm = keypoints_normalized
        self.scores = scores
        self.keypoints = np.array([(kp[0] * self.img_w, kp[1] * self.img_h)
                                  for kp in keypoints_normalized])

    def estimate_focus_zone_size(self):
        """
        This function is called if at least the segment "wrist_elbow" is visible.
        We calculate the length of every segment from a predefined list. A segment length
        is the distance between the 2 endpoints weighted by a coefficient. The weight have been chosen
        so that the length of all segments are roughly equal. We take the maximal length to estimate
        the size of the focus zone. 
        If no segment are vissible, we consider the body is very close 
        to the camera, and therefore there is no need to focus. Return 0
        To not have at least one shoulder and one hip visible means the body is also very close
        and the estimated size needs to be adjusted (bigger)
        """
        segments = [
            ("left_shoulder", "left_elbow", 2.3),
            ("left_elbow", "left_wrist", 2.3),
            ("left_shoulder", "left_hip", 1),
            ("left_shoulder", "right_shoulder", 1.5),
            ("right_shoulder", "right_elbow", 2.3),
            ("right_elbow", "right_wrist", 2.3),
            ("right_shoulder", "right_hip", 1),
        ]
        lengths = []
        for s in segments:
            if self.scores[BODY_KP[s[0]]] > self.score_thresh and self.scores[BODY_KP[s[1]]] > self.score_thresh:
                l = np.linalg.norm(
                    self.keypoints[BODY_KP[s[0]]] - self.keypoints[BODY_KP[s[1]]])
                lengths.append(l)
        if lengths:
            if (self.scores[BODY_KP["left_hip"]] < self.score_thresh and
                self.scores[BODY_KP["right_hip"]] < self.score_thresh or
                self.scores[BODY_KP["left_shoulder"]] < self.score_thresh and
                    self.scores[BODY_KP["right_shoulder"]] < self.score_thresh):
                coef = 1.5
            else:
                coef = 1.0
            # The size is made even
            return 2 * int(coef * self.scale * max(lengths) / 2)
        else:
            return 0

    def get_focus_zone(self):
        """
        Return a tuple (focus_zone, label)
        'body' = instance of class Body
        'focus_zone' is a zone around a hand or hands, depending on the value 
        of self.mode ("left", "right", "higher" or "group") and on the value of self.hands_up_only.
            - self.mode = "left" (resp "right"): we are looking for the zone around the left (resp right) wrist,
            - self.mode = "group": the zone encompasses both wrists,
            - self.mode = "higher": the zone is around the higher wrist (smaller y value),
            - self.hands_up_only = True: we don't take into consideration the wrist if the corresponding elbow is above the wrist,
        focus_zone is a list [left, top, right, bottom] defining the top-left and right-bottom corners of a square. 
        Values are expressed in pixels in the source image C.S.
        The zone is constrained to the squared source image (= source image with padding if necessary). 
        It means that values can be negative.
        left and right in [-pad_w, img_w + pad_w]
        top and bottom in [-pad_h, img_h + pad_h]
        'label' describes which wrist keypoint(s) were used to build the zone : "left", "right" or "group" (if built from both wrists)

        If the wrist keypoint(s) is(are) not present or is(are) present but self.hands_up_only = True and
        wrist(s) is(are) below corresponding elbow(s), then focus_zone = None.
        """

        def zone_from_center_size(x, y, size):
            """
            Return zone [left, top, right, bottom] 
            from zone center (x,y) and zone size (the zone is square).
            """
            half_size = size // 2
            size = half_size * 2
            if size > self.img_w:
                x = self.img_w // 2
            x1 = x - half_size
            if x1 < -self.pad_w:
                x1 = -self.pad_w
            elif x1 + size > self.img_w + self.pad_w:
                x1 = self.img_w + self.pad_w - size
            x2 = x1 + size
            if size > self.img_h:
                y = self.img_h // 2
            y1 = y - half_size
            if y1 < -self.pad_h:
                y1 = -self.pad_h
            elif y1 + size > self.img_h + self.pad_h:
                y1 = self.img_h + self.pad_h - size
            y2 = y1 + size
            return [x1, y1, x2, y2]

        def get_one_hand_zone(hand_label):
            """
            Return the zone [left, top, right, bottom] around the hand given by its label "hand_label" ("left" or "right")
            Values are expressed in pixels in the source image C.S.
            If the wrist keypoint is not visible, return None.
            If self.hands_up_only is True, return None if wrist keypoint is below elbow keypoint.
            """
            wrist_kp = hand_label + "_wrist"
            wrist_score = self.scores[BODY_KP[wrist_kp]]
            if wrist_score < self.score_thresh:
                return None
            x, y = self.keypoints[BODY_KP[wrist_kp]]
            if self.hands_up_only:
                # We want to detect only hands where the wrist is above the elbow (when visible)
                elbow_kp = hand_label + "_elbow"
                if self.scores[BODY_KP[elbow_kp]] > self.score_thresh and \
                        self.keypoints[BODY_KP[elbow_kp]][1] < self.keypoints[BODY_KP[wrist_kp]][1]:
                    return None
            # Let's evaluate the size of the focus zone
            size = self.estimate_focus_zone_size()
            if size == 0:
                # The hand is too close. No need to focus
                return [-self.pad_w, -self.pad_h, self.frame_size-self.pad_w, self.frame_size-self.pad_h]
            return zone_from_center_size(x, y, size)

        if self.mode == "group":
            zonel = get_one_hand_zone("left")
            if zonel:
                zoner = get_one_hand_zone("right")
                if zoner:
                    xl1, yl1, xl2, yl2 = zonel
                    xr1, yr1, xr2, yr2 = zoner
                    x1 = min(xl1, xr1)
                    y1 = min(yl1, yr1)
                    x2 = max(xl2, xr2)
                    y2 = max(yl2, yr2)
                    # Global zone center (x,y)
                    x = int((x1+x2)/2)
                    y = int((y1+y2)/2)
                    size_x = x2-x1
                    size_y = y2-y1
                    size = 2 * (max(size_x, size_y) // 2)
                    return (zone_from_center_size(x, y, size), "group")
                else:
                    return (zonel, "left")
            else:
                return (get_one_hand_zone("right"), "right")
        elif self.mode == "higher":
            if self.scores[BODY_KP["left_wrist"]] > self.score_thresh:
                if self.scores[BODY_KP["right_wrist"]] > self.score_thresh:
                    if self.keypoints[BODY_KP["left_wrist"]][1] > self.keypoints[BODY_KP["right_wrist"]][1]:
                        hand_label = "right"
                    else:
                        hand_label = "left"
                else:
                    hand_label = "left"
            else:
                if self.scores[BODY_KP["right_wrist"]] > self.score_thresh:
                    hand_label = "right"
                else:
                    return (None, None)
            return (get_one_hand_zone(hand_label), hand_label)
        else:  # "left" or "right"
            return (get_one_hand_zone(self.mode), self.mode)

    def get_crop_region(self):
        """Function that gets the crop region based on the pose keypoints.

        Returns:
            CropRegion: tuple that contains the crop region (xmin, ymin, xmax, ymax) and the crop size.
        """
        focus_zone, _ = self.get_focus_zone()
        if focus_zone is None:
            return None
        return CropRegion(*focus_zone, self.crop_size)

    def crop_and_resize(self, frame, crop_region):
        """Crops and resize the image to prepare for the model input."""
        cropped = frame[int(max(0, crop_region.ymin)):int(min(self.img_h, crop_region.ymax)), int(
            max(0, crop_region.xmin)):int(min(self.img_w, crop_region.xmax))]

        if crop_region.xmin < 0 or crop_region.xmax >= self.img_w or crop_region.ymin < 0 or crop_region.ymax >= self.img_h:
            # Padding is necessary
            cropped = cv2.copyMakeBorder(cropped,
                                         int(max(0, -crop_region.ymin)),
                                         int(max(0, crop_region.ymax-self.img_h)),
                                         int(max(0, -crop_region.xmin)),
                                         int(max(0, crop_region.xmax-self.img_w)),
                                         cv2.BORDER_CONSTANT)

        cropped = cv2.resize(
            cropped, (crop_region.size, crop_region.size), interpolation=cv2.INTER_AREA)
        return cropped
