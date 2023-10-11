import cv2
import numpy as np
from PIL import Image
from pywt import wavedec2
from pywt import waverec2

def get_size(pixels):
    n1 = ""
    n2 = ""
    pixels = pixels[-1][:10]
    for pixel in pixels:
        r_lsb = pixel[0] & 1
        g_lsb = pixel[1] & 1
        n1 += str(r_lsb)
        n2 += str(g_lsb)
    num1 = int(n1, 2)
    num2 = int(n2, 2)
    return num1, num2

if __name__ == '__main__':
    img1 = cv2.imread('image1.bmp')
    img2 = cv2.imread('image3.bmp')
    im_w, im_h, _ = img1.shape
    wt_w, wt_h = get_size(img2)
    b, g, r = cv2.split(img1)
    B, G, R = cv2.split(img2)
    [ca3, (ch3, cv3, cd3), (ch2, cv2, cd2), (ch1, cv1, cd1)] = wavedec2(b, 'db2', level=3)
    [CA3, (CH3, CV3, CD3), (CH2, CV2, CD2), (CH1, CV1, CD1)] = wavedec2(B, 'db2', level=3)    

    a1 = 0.1
    a2 = 0.1
    a3 = 0.1
    a4 = 0.2
    ca = (CA3 - ca3) / a1
    ch = (CH3 - ch3) / a2
    cv = (CV3 - cv3) / a3
    cd = (CD3 - cd3) / a4
    wt = waverec2([ca, (ch, cv, cd)], 'db2')    
    img3 = np.zeros((len(wt), len(wt[0]), 3), dtype=np.uint8)
    img3[:, :, 0] = wt
    img3[:, :, 1] = wt
    img3[:, :, 2] = wt
    img3 = Image.fromarray(img3)
    img3 = img3.resize((wt_w, wt_h))
    img3.save('image4.bmp')