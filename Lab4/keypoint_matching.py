import cv2
import matplotlib.pyplot as plt


def get_matches(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.2 * n.distance:
            good.append([m])

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[0:10], img2, flags=2, matchColor=-1)
    plt.imshow(img3), plt.show()

    return good


img1 = cv2.imread('boat1.pgm', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('boat2.pgm', cv2.IMREAD_GRAYSCALE)
get_matches(img1, img2)
