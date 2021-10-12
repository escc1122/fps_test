import cv2
import tensorflow as tf
import tensorflow_hub as hub

keypoint_threshold = 0.3
# [鼻子，左眼，右眼，左耳，右耳朵，左肩，右肩，左肘，右肘，左手腕，右手腕，左髖，右髖，左膝，右膝，左腳踝，右腳踝]
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


def multi_pose():
    import time
    import numpy as np

    model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
    movenet = model.signatures['serving_default']
    input_source = "test1.mp4"
    cap = cv2.VideoCapture(input_source)
    hasFrame, frame = cap.read()
    while cv2.waitKey(1) < 0:
        t = time.time()
        hasFrame, frame = cap.read()
        cv2_image = np.copy(frame)
        if not hasFrame:
            cv2.waitKey()
            break
        image = tf.Variable(cv2_image, name='x')
        image = tf.expand_dims(image, axis=0)
        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        image = tf.cast(tf.image.resize_with_pad(image, 192, 192), dtype=tf.int32)
        outputs = movenet(image)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints = outputs['output_0']

        image_width = cv2_image.shape[1]
        image_height = cv2_image.shape[0]

        for i in range(0, keypoints.shape[1]):
            target_key_points = {}
            target_key_points_2 = {}
            for joint in KEYPOINT_DICT.keys():
                target_key_points[joint] = [
                    keypoints[0, i, KEYPOINT_DICT[joint] * 3 + 0] * image_height,
                    keypoints[0, i, KEYPOINT_DICT[joint] * 3 + 1] * image_width,
                    keypoints[0, i, KEYPOINT_DICT[joint] * 3 + 2],
                ]
                target_key_points_2[KEYPOINT_DICT[joint]] = [
                    keypoints[0, i, KEYPOINT_DICT[joint] * 3 + 0] * image_height,
                    keypoints[0, i, KEYPOINT_DICT[joint] * 3 + 1] * image_width,
                    keypoints[0, i, KEYPOINT_DICT[joint] * 3 + 2],
                ]
            count = 0
            cv2_image = plt_line(cv2_image, target_key_points_2)
        cv2.imshow('image', cv2_image)

# 單人姿勢
def single_pose():
    import time
    import numpy as np
    # 判斷的畫面大小,以中心
    cut_size = 256
    # 預測圖片需要的大小
    predict_size = 192
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

    # model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/1")
    # predict_size = 256
    movenet = model.signatures['serving_default']
    input_source = "test2.mp4"
    cap = cv2.VideoCapture(input_source)
    hasFrame, frame = cap.read()

    # 儲存預測結果
    vid_writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10,
                                 (frame.shape[1], frame.shape[0]))

    while cv2.waitKey(1) < 0:
        t = time.time()
        hasFrame, frame = cap.read()
        cv2_image = np.copy(frame)
        if not hasFrame:
            cv2.waitKey()
            break

        # 判斷的畫面大小 start
        left = int((cv2_image.shape[1] - cut_size) / 2)
        right = int(left + cut_size)
        down = int((cv2_image.shape[0] - cut_size) / 2)
        top = int(down + cut_size)
        predict_image = cv2_image[down:top, left:right]
        # 判斷的畫面大小 end

        image = tf.Variable(predict_image, name='x')
        image = tf.expand_dims(image, axis=0)
        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        # image = tf.cast(tf.image.resize_with_pad(image, 192, 192), dtype=tf.int32)
        image = tf.cast(tf.image.resize_with_pad(image, predict_size, predict_size), dtype=tf.int32)
        outputs = movenet(image)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints = outputs['output_0']

        # 取得預測點的座標
        target_key_points, target_key_points_2 = get_predict_point(keypoints, cut_size, cut_size, left, down)
        # 畫線
        cv2_image = plt_line(cv2_image, target_key_points_2)
        # 顯示預覽
        cv2.imshow('image', cv2_image)
        # 儲存影片
        vid_writer.write(cv2_image)
    vid_writer.release()


# 取得預測的點
def get_predict_point(keypoints, image_height, image_width, left, down):
    target_key_points = {}
    target_key_points_2 = {}
    for joint in KEYPOINT_DICT.keys():
        target_key_points[joint] = [
            keypoints[0, 0, KEYPOINT_DICT[joint], 0] * image_height + down,
            keypoints[0, 0, KEYPOINT_DICT[joint], 1] * image_width + left,
            keypoints[0, 0, KEYPOINT_DICT[joint], 2]
        ]
        target_key_points_2[KEYPOINT_DICT[joint]] = [
            keypoints[0, 0, KEYPOINT_DICT[joint], 0] * image_height + down,
            keypoints[0, 0, KEYPOINT_DICT[joint], 1] * image_width + left,
            keypoints[0, 0, KEYPOINT_DICT[joint], 2]
        ]
    return target_key_points, target_key_points_2


# 利用預測的點畫線
def plt_line(cv2_image, target_key_points_2):
    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        if (target_key_points_2[edge_pair[0]][2] > keypoint_threshold and
                target_key_points_2[edge_pair[1]][2] > keypoint_threshold):
            x_start = int(target_key_points_2[edge_pair[0]][1])
            y_start = int(target_key_points_2[edge_pair[0]][0])
            x_end = int(target_key_points_2[edge_pair[1]][1])
            y_end = int(target_key_points_2[edge_pair[1]][0])
            cv2.line(cv2_image, (x_start, y_start), (x_end, y_end), (0, 255, 255), 2)
    return cv2_image


if __name__ == '__main__':
    single_pose()
