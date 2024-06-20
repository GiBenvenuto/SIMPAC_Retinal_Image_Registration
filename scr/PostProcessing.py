import cv2
import numpy as np
import glob
from scr.ops import mkdir
#from Utils import evaluations as ev
import skimage.io as sk


def post_processing(path, cat, ind):
    image = cv2.imread(path)
    gray = image[:,:,1]
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    output = cv2.connectedComponentsWithStats(thresh, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    mask = np.zeros(gray.shape, dtype="uint8")
    for i in range(0, numLabels):

        '''if i == 0:
             text = "examining component {}/{} (background)".format(i + 1, numLabels)
        else:
             text = "examining component {}/{}".format( i + 1, numLabels)
             print("[INFO] {}".format(text))'''

        area = stats[i, cv2.CC_STAT_AREA]

        if (area < 10):
            #print("[INFO] keeping connected component '{}'".format(i))
            componentMask = (labels == i).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, componentMask)


    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)


    pos = cv2.bitwise_and(mask, thresh)
    #cv2.imshow("Image", thresh)
    #cv2.imshow("Characters", mask)
    #cv2.imshow("Pos", pos)
    cv2.imwrite("TESTES/Teste_512_1603_cc_allcats_inv/" + str(cat) + "/{:02d}_z.tif".format(ind + 1), pos)
    #cv2.waitKey(0)

def fill_holes(img, min_hole_size):
    img = np.invert(img)
    img = img.astype('uint8')
    output = cv2.connectedComponentsWithStats(img, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    mask = np.zeros(img.shape, dtype="uint8")
    for i in range(0, numLabels):
        area = stats[i, cv2.CC_STAT_AREA]

        if (area < min_hole_size):
            componentMask = (labels == i).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, componentMask)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)
    img = cv2.bitwise_and(mask, img)

    return np.invert(img)

def cca_img(image, dir_path, j):
    #gray = image[:,:,1]
    image = np.clip(image, 0., 1.)
    image = image*255
    image = image.astype("uint8")
    _, thresh = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)

    output = cv2.connectedComponentsWithStats(thresh, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    mask = np.zeros(image.shape, dtype="uint8")
    for i in range(0, numLabels):

        '''if i == 0:
             text = "examining component {}/{} (background)".format(i + 1, numLabels)
        else:
             text = "examining component {}/{}".format( i + 1, numLabels)
             print("[INFO] {}".format(text))'''

        area = stats[i, cv2.CC_STAT_AREA]

        if (area < 20):
            #print("[INFO] keeping connected component '{}'".format(i))
            componentMask = (labels == i).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, componentMask)


    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)


    pos = cv2.bitwise_and(mask, thresh)
    pos = fill_holes(pos, 5)
    cv2.imwrite(dir_path + "/{:02d}_zcca.tif".format(j + 1), pos)




if __name__ == "__main__":
    mkdir("../DataBases/Testes_Journal/Paiva/Proposed_CCA30")
    aux = glob.glob("../DataBases/Testes_Journal/Paiva/Proposed/*.tif")
    for j in range(0, len(aux)):
        img = cv2.imread(aux[j])
        img = cv2.resize(img, (512, 512), cv2.INTER_CUBIC)
        img = img[:, :, 1]
        img = img / 255

        cca_img(img, "../DataBases/Testes_Journal/Paiva/Proposed_CCA30", j)

        #skimage.io.imsave("../DataBases/Testes_Journal/Paiva/Proposed_OpenCloseFilter/{:02d}_z.png".format(j + 1), op)








