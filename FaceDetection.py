import cv2
from numpy import array
import math
import numpy as np


def face_detect(test_img_path, cascPath, show_result = False, save_dir = None):

    img = cv2.imread(test_img_path)
    faceCascade = cv2.CascadeClassifier(cascPath)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    if show_result:
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Faces found", img)
        cv2.waitKey(0)

    if save_dir is not None:
        count = 1
        for (x, y, w, h) in faces:
            crop_img = img[y: y + h, x: x + w]
            # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
            cv2.imwrite(save_dir + '/face_' + str(count) + '.jpg', crop_img)
            count += 1

    return len(faces)

'''Aligned the faces before using this function, otherwise the result is not satisfying'''
def FaceMatching(aligned_face1_path, aligned_face2_path):
    aligned_face1 = cv2.imread(aligned_face1_path)
    aligned_face2 = cv2.imread(aligned_face2_path)
    gray_face1 = cv2.cvtColor(aligned_face1, cv2.COLOR_BGR2GRAY)
    gray_face2 = cv2.cvtColor(aligned_face2, cv2.COLOR_BGR2GRAY)

    mat1 = array(gray_face1)
    mat2 = array(gray_face2)

    embeding_mat1 = np.zeros((len(mat1), len(mat1[0])))
    embeding_mat2 = np.zeros((len(mat2), len(mat2[0])))


    final_mat = np.zeros((len(mat1), len(mat1[0])))

    sum1 = 0
    sum2 = 0

    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            sum1 += mat1[i][j] ** 2

    for i in range(len(mat2)):
        for j in range(len(mat2[0])):
            sum2 += mat2[i][j] ** 2

    sum1 = math.sqrt(sum1)
    sum2 = math.sqrt(sum2)

    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            embeding_mat1[i][j] = float(mat1[i][j]) / float(sum1)

    for i in range(len(mat2)):
        for j in range(len(mat2[0])):
            embeding_mat2[i][j] = float(mat2[i][j]) / sum2

    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            final_mat[i][j] = (embeding_mat1[i][j] - embeding_mat2[i][j]) ** 2

    sim_score = 1 - math.sqrt(final_mat.sum())

    return sim_score



if __name__ == '__main__':
    FaceMatching('2.png', '6.png')
    # test_img_path = 'test2.png'
    # cascPath = '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml'

    # face_detect(test_img_path, cascPath, True, None)
