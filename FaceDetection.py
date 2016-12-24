import cv2

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


if __name__ == '__main__':
    test_img_path = 'test2.png'
    cascPath = '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml'

    face_detect(test_img_path, cascPath, True, None)
