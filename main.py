import cv2
import numpy as np


def is_in_bounds(x_value):
    if x_value >= -(10 ** 8) & x_value <= (10 ** 8):
        return True
    return False


cam = cv2.VideoCapture('Lane Detection Test Video 01.mp4')

while True:

    ret, frame = cam.read()
    (height, width, color) = frame.shape
    ratio = height / width

    # exercise 2
    new_height = height * ratio / 2
    new_width = width * ratio / 2
    frame = cv2.resize(frame, (int(new_width), int(new_height)))
    original_frame = frame.copy()

    # exercise 3
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey_scale_frame = frame.copy()
    cv2.imshow('grey scale', grey_scale_frame)

    # exercise4
    (height, width) = frame.shape

    upper_left = (width * 0.47, height * 0.75)
    upper_right = (width * 0.53, height * 0.75)
    lower_left = (0, height)
    lower_right = (width, height)

    trapezoid_points = np.array([upper_left, upper_right, lower_right, lower_left], np.int32)

    trapezoid_frame = np.zeros_like(frame)
    cv2.fillPoly(trapezoid_frame, [trapezoid_points], 1)
    cv2.imshow('trapezoid', trapezoid_frame * 255)

    frame = frame * trapezoid_frame
    cv2.imshow('trapezoid frame', frame)

    # Exercise 5
    screen_points = np.array([(0, 0), (width, 0), (width, height), (0, height)], np.float32)
    magical_matrix = cv2.getPerspectiveTransform(np.float32(trapezoid_points), screen_points)

    stretched_trapezoid_frame = cv2.warpPerspective(frame, magical_matrix, (width, height))
    cv2.imshow('stretched_trapezoid_frame', stretched_trapezoid_frame)

    # Exercise 6
    stretched_trapezoid_frame = cv2.blur(stretched_trapezoid_frame, ksize=(5, 5))
    cv2.imshow('blurred', stretched_trapezoid_frame)

    # Exercise 7
    sobel_vertical = np.float32([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    sobel_horizontal = np.transpose(sobel_vertical)

    frame_f = np.float32(stretched_trapezoid_frame)

    frame_1 = cv2.filter2D(frame_f, -1, sobel_vertical)
    frame_2 = cv2.filter2D(frame_f, -1, sobel_horizontal)

    combined = np.sqrt(frame_1 ** 2 + frame_2 ** 2)

    frame = cv2.convertScaleAbs(combined)
    cv2.imshow('sobel', frame)

    # Exercise 8
    threshold, binarize = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow('binarized', binarize)

    # Exercise 9
    copy_frame = binarize.copy()
    copy_frame[:, 0:int(width * 0.05)] = 0
    copy_frame[:, int(width * 0.90):width] = 0

    left_xs = []
    left_ys = []

    right_xs = []
    right_ys = []

    half = int(width / 2)
    first_half = copy_frame[:, 0:half]
    second_half = copy_frame[:, half:width]

    left_coord = np.argwhere(first_half >= 255)
    print("left_coord", left_coord)
    right_coord = np.argwhere(second_half >= 255)

    left_ys, left_xs = zip(*left_coord)
    right_coord[:, 1] += width // 2
    right_ys, right_xs = zip(*right_coord)

    frame = copy_frame.copy()

    line_left = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)
    line_right = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)

    left_top_y = 0
    left_top_x = 0
    x = int(-line_left[0] / line_left[1])
    if -(10 ** 6) <= x <= (10 ** 6):
        left_top_x = x

    right_top_y = 0
    right_top_x = 0
    x = int(-line_right[0] / line_right[1])
    if -(10 ** 6) <= x <= (10 ** 6):
        right_top_x = x

    left_bottom_y = height
    left_bottom_x = 0
    x = int((left_bottom_y - line_left[0]) / line_left[1])
    if -(10 ** 6) <= x <= (10 ** 6):
        left_bottom_x = x

    right_bottom_y = height
    right_bottom_x = 0
    x = int((right_bottom_y - line_right[0]) / line_right[1])
    if -(10 ** 6) <= x <= (10 ** 6):
        right_bottom_x = x

    cv2.line(frame, (int(left_top_x), int(left_top_y)), (int(left_bottom_x), int(left_bottom_y)), (200, 0, 0), 5)
    cv2.line(frame, (int(right_top_x), int(right_top_y)), (int(right_bottom_x), int(right_bottom_y)), (100, 0, 0), 5)
    cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 0, 0), 1)
    cv2.imshow("Lines", frame)

    # Exercise 11
    left_lane_frame = np.zeros_like(original_frame)
    cv2.line(left_lane_frame, (left_top_x, left_top_y), (left_bottom_x, left_bottom_y), (255, 0, 0), 3)

    magical_matrix = cv2.getPerspectiveTransform(screen_points, np.float32(trapezoid_points))
    cv2.warpPerspective(left_lane_frame, magical_matrix, (width, height), left_lane_frame)

    left_lane_points = np.argwhere(left_lane_frame == 255)

    right_lane_frame = np.zeros_like(original_frame)
    cv2.line(right_lane_frame, (right_top_x, right_top_y), (right_bottom_x, right_bottom_y), (255, 0, 0), 3)

    cv2.warpPerspective(right_lane_frame, magical_matrix, (width, height), right_lane_frame)
    right_lane_points = np.argwhere(right_lane_frame == 255)

    final_frame = original_frame.copy()

    for point in left_lane_points:
        final_frame[point[0], point[1]] = (50, 50, 250)
    for point in right_lane_points:
        final_frame[point[0], point[1]] = (50, 250, 50)

    # Exercise 12

    if ret is False:
        break

    cv2.imshow('Final', final_frame)
    cv2.imshow('Lines', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
