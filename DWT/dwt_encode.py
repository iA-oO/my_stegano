import cv2
import numpy as np
from PIL import Image
from pywt import wavedec2
from pywt import waverec2

def embed_size(img):
    w = format(wt_w, "08b")
    h = format(wt_h, "08b")

    for i in range(10):
        pixel = list(img[-1][i])
        pixel[0] &= 0xFE
        pixel[1] &= 0xFE
        img[-1][i] = tuple(pixel)

    count = 1
    for i in range(9, -1, -1):
        if count > len(w) and count > len(h):
            break
        pixel = list(img[-1][i])
        if count <= len(w):
            pixel[0] |= int(w[-count])
        if count <= len(h):
            pixel[1] |= int(h[-count])
        img[-1][i] = tuple(pixel)
        count += 1
    return img

if __name__ == '__main__':
    img1 = cv2.imread('image1.bmp')
    img2 = cv2.imread('image2.bmp')
    im_h, im_w, _ = img1.shape
    wt_h, wt_w, _ = img2.shape
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    L = max(im_h, im_w)
    l = round(L / 4) + 2
    img = cv2.resize(img1, (L, L))
    wt = cv2.resize(img2, (l, l))

    b, g, r = cv2.split(img)
    a1 = 0.1
    a2 = 0.1
    a3 = 0.1
    a4 = 0.2
    [CA3, (CH3, CV3, CD3), (CH2, CV2, CD2), (CH1, CV1, CD1)] = wavedec2(b, 'db2', level=3)
    [ca, (ch, cv, cd)] = wavedec2(wt, 'db2', level=1)
    CA3 = CA3 + a1 * ca
    CH3 = CH3 + a2 * ch
    CV3 = CV3 + a3 * cv
    CD3 = CD3 + a4 * cd
    B = waverec2([CA3, (CH3, CV3, CD3), (CH2, CV2, CD2), (CH1, CV1, CD1)], 'db2')

    img3 = np.zeros(img.shape, dtype=np.uint8)
    img3[:, :, 0] = r
    img3[:, :, 1] = g
    img3[:, :, 2] = B
    img3 = Image.fromarray(img3)
    img3 = img3.resize((im_w, im_h))
    img3.save('image3.bmp')
    img3 = cv2.imread('image3.bmp')
    embed_size(img3)
    cv2.imwrite('image3.bmp', img3)
