import cv2


def main():
    # Import TF and TF Hub libraries.
    import tensorflow as tf
    import tensorflow_hub as hub

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

    # Load the input image.
    # image_path = 'images/test4.jpg'
    image_path = 'images/single.jpeg'
    cv2_image = cv2.imread(image_path)
    # cv2_image = cv2.resize(cv2_image, (768, 768))
    # image_width = cv2_image.shape[1]
    # image_height = cv2_image.shape[0]
    # image = tf.Variable(cv2_image, name='x')
    image = tf.io.read_file(image_path)
    image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.cast(tf.image.resize_with_pad(image, 192, 192), dtype=tf.int32)

    # Download the model from TF Hub.
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

    # tf.saved_model.save(model, "tf_model")

    # model = tf.saved_model.load("tf_model")

    # print(list(model.signatures.keys()))  # ["serving_default"]

    # model = hub.load("\movenet_singlepose_lightning\saved_model.pb")
    # model = tf.saved_model.load("\movenet_singlepose_lightning")
    # f = imported.signatures["serving_default"]

    movenet = model.signatures['serving_default']

    # Run model inference.
    outputs = movenet(image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints = outputs['output_0']
    print(keypoints)

    image_width = cv2_image.shape[1]
    image_height = cv2_image.shape[0]
    target_keypoints = {}
    target_keypoints2 = {}
    keypoint_threshold = 0.11
    for joint in KEYPOINT_DICT.keys():
        target_keypoints[joint] = [
            keypoints[0, 0, KEYPOINT_DICT[joint], 0] * image_height,
            keypoints[0, 0, KEYPOINT_DICT[joint], 1] * image_width,
            keypoints[0, 0, KEYPOINT_DICT[joint], 2]
        ]
        target_keypoints2[KEYPOINT_DICT[joint]] = [
            keypoints[0, 0, KEYPOINT_DICT[joint], 0] * image_height,
            keypoints[0, 0, KEYPOINT_DICT[joint], 1] * image_width,
            keypoints[0, 0, KEYPOINT_DICT[joint], 2]
        ]
    i = 0
    for joint in KEYPOINT_DICT.keys():
        y = target_keypoints[joint][0]
        x = target_keypoints[joint][1]
        s = target_keypoints[joint][2]
        # cv2.circle(cv2_image, (int(x), int(y)), 3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        # cv2.putText(cv2_image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
        #             lineType=cv2.LINE_AA)
        i = i + 1

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        if (target_keypoints2[edge_pair[0]][2] > keypoint_threshold and
                target_keypoints2[edge_pair[1]][2] > keypoint_threshold):
            x_start = int(target_keypoints2[edge_pair[0]][1])
            y_start = int(target_keypoints2[edge_pair[0]][0])
            x_end = int(target_keypoints2[edge_pair[1]][1])
            y_end = int(target_keypoints2[edge_pair[1]][0])
            cv2.line(cv2_image, (x_start, y_start), (x_end, y_end), (0, 255, 255), 2)

    cv2.imshow('image', cv2_image)
    cv2.waitKey(0)


def test():
    image_path = 'images/test7.jpg'
    image = cv2.imread(image_path)
    cv2.circle(image, (5, 5), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    # cv2.circle(image, 0, 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    cv2.imshow('image', image)
    cv2.waitKey(0)


def test2():
    import tensorflow as tf
    model = tf.saved_model.load("tf_model")
    print(list(model.signatures.keys()))  # ["serving_default"]


if __name__ == '__main__':
    main()
