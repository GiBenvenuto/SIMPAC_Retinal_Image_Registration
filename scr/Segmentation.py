import glob

import numpy as np
import cv2

def segmentation (path = None, img = None):

    if path is None:
        if img is None:
            print("ERROR - Missing Path and IMG")
            exit(0)
    else:
        img = cv2.imread(path)

    #img = cv2.imread("1_bad.JPG")

    II0=[]
    II0.append(img)
    num_pyr = 0

    while len(II0[-1])>900:
        II0.append(cv2.pyrDown(II0[-1]))
        num_pyr = num_pyr + 1

    II = cv2.cvtColor(II0[-1], cv2.COLOR_RGB2GRAY)
    #cv2.imshow("Teste", II)
    #cv2.waitKey(0)
    mask = (II > 5)
    mask = 1*mask

    BW1 = IUWT_segmentation_for_testing(II0[-1], [2, 3], 0.15, 200, 20, True, mask)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(60,60))

    nmask = np.invert(mask)
    nmask = nmask.astype('uint8')
    border = cv2.dilate(nmask, se)
    border = (border > 254)
    border = 1*border
    BW1 = (BW1 <= 254)
    BW1 = 1*BW1
    BW1 = BW1 - border
    BW1 = (BW1 > 0)
    BW1 = 1 * BW1
    BW1 = BW1.astype('float')
    #BW1 = cv2.resize(BW1, (512,512), cv2.INTER_CUBIC)
    #cv2.imshow("Teste", BW1)

    #cv2.imwrite("teste.png", BW1)
    #cv2.waitKey(0)

    return BW1*255

def segmented_all():
    paths = glob.glob("../DataBases/paiva_dataset/images/*.jpg")
    j = 1
    k = 1
    for p in paths:
        #if j%2==0:
           text = p.split(sep="/")
           name = text[-1].split(sep="\\")
           #name = name[1].split(sep=".")

           img = segmentation(p)
           img = img*255
           cv2.imwrite("../DataBases/paiva_dataset/seg/" + name[1], img)
         #  k = k + 1
           j = j + 1


def IUWT_segmentation_for_testing(im, levels, percent, remove, fill, dark, bw_mask=None):
    if len(im[2]) > 1:
        im = im[:,:,1]

    #Create a mask if necessary
    bw_mask = cv2.erode(im>20, np.ones(3)) if bw_mask is None else bw_mask

    w = iuwt_vessels(im, levels)
    #cv2.imwrite("../DataBases/Testes_Journal/Teste_1.jpg", w*255)
    bw = percentage_segment(w, percent, dark, bw_mask)

    bw = clean_segmented_image(bw, remove, fill)



    return bw

def iuwt_vessels(im, levels, padding=None):
    #Default padding
    padding = 'symmetric' if padding is None else padding

    #First smoothing level = input image
    s_in = im.astype('float')

    #Inititalise output
    w = 0

    #B3 spline coefficients for filter

    b3 = [1, 4, 6, 4, 1]
    b3 = np.divide(b3, 16)

    #Compute transform
    for ii in range(1, levels[-1] + 1):
       #Create convolution kernel
       h = dilate_wavelet_kernel(b3, 2**(ii - 1) - 1)

       #Convolve and subtract to get wavelet level
       s_out = cv2.filter2D(s_in, ddepth=-1, kernel=np.transpose(h) * h)

       #Add wavelet level if in LEVELS
       if ii in levels:
          w = w + s_in - s_out

       #Update input for new iteration
       s_in = s_out

    return w

def dilate_wavelet_kernel(h, spacing):

    #Preallocate the expanded filter
    h2 = np.zeros((1, np.size(h) + spacing * (np.size(h) - 1)))

    #Ensure output kernel orientation is the same
    h=np.array([h])
    if h.shape[0] > h.shape[1]:
        h2 = np.transpose(h2)

    #Put in the coefficients
    j = 0
    for i in range(0, h2.size, spacing + 1):
        h2[0][i] = h[0][j]
        j = j + 1

    return h2


def clean_segmented_image(bw, min_object_size, min_hole_size):
    #Remove small objects, if necessary
    if min_object_size > 0:
        img = bw[0]
        img = img.astype('uint8')
        output = cv2.connectedComponentsWithStats(img, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        mask = np.zeros(img.shape, dtype="uint8")
        for i in range(0, numLabels):
            area = stats[i, cv2.CC_STAT_AREA]

            if (area < min_object_size):
                componentMask = (labels == i).astype("uint8") * 255
                mask = cv2.bitwise_or(mask, componentMask)
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)
        bw_clean = cv2.bitwise_and(mask, img)
    else:
        bw_clean = bw

    if min_hole_size > 0:
        img = np.invert(bw_clean)
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
        bw_clean = cv2.bitwise_and(mask, img)

    return bw_clean


def percentage_segment(im, proportion, dark, bw_mask, sorted_pix=None):

    #Sort pixels in image, if a sorted array is not already available
    if sorted_pix is None:
        if bw_mask.any():
            img = bw_mask.ravel()
            #img = img.astype('uint8')
            im_ravel = im.ravel()
            sorted_pix = []
            j = 0
            for i in img:
                if i == 1:
                    sorted_pix.append(im_ravel[j])
                j = j + 1

            sorted_pix = np.array(sorted_pix)
            sorted_pix.sort()


        else:
            sorted_pix = im.ravel().sort()

    #Convert to a proportion if we appear to have got a percentage
    if proportion > 1:
        proportion = proportion / 100
        print('PERCENTAGE_SEGMENT:THRESHOLD', 'The threshold exceeds 1; it will be divided by 100.')

    #Invert PERCENT if DARK
    if dark:
        proportion = 1 - proportion

    #Get threshold
    [threshold, sorted_pix] = percentage_threshold(sorted_pix, proportion, True)

    #Threshold to get darkest or lightest objects
    if dark:
        bw = im <= threshold
    else:
        bw = im > threshold

    #Apply mask
    if bw_mask.any():
        bw = np.bitwise_and(bw, bw_mask)


    return bw, sorted_pix


def percentage_threshold(data, proportion, sorted=None):

    #Need to make data a vector
    if not(isinstance(data, list)):
        data = list(data)

    #If not told whether data is sorted, need to check

    if sorted is None:
        data[0].sort()
        data_sorted = data
    else:
        data_sorted = data

    #Calculate  threshold value
    if proportion > 1:
        proportion = proportion / 100

    proportion = 1 - proportion
    thresh_ind = round(proportion * np.size(data_sorted))

    if thresh_ind > np.size(data_sorted):
        threshold = np.inf
    elif thresh_ind < 1:
        threshold = -np.inf
    else:
        threshold = data_sorted[thresh_ind]

    return threshold, data_sorted

if __name__ == "__main__":
    img = segmentation("../DataBases/Testes_Journal/A08_2.jpg")
    img = img * 255
    cv2.imwrite("../DataBases/Testes_Journal/A08_2_seg.png", img)
    #segmented_all()



