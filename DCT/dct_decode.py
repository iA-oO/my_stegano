import numpy as np
import cv2
from scipy.fft import dctn

def decode(img):

# 分n*m块，每块8*8
    h, w, _ = img.shape
    n = h // 8
    m = w // 8
    blocks = np.zeros([n, m], dtype = int).tolist()

    for i in range(n):
        for j in range(m):
            block = np.zeros([8, 8], dtype = int).tolist()
            blocks[i][j] = block

# 分块，dct
    count = 0
    for i in range(n):
        for j in range(m):
            block = img [i*8:(i+1)*8, j*8:(j+1)*8]
            if count > 0:
                block = dctn(block).tolist()
            blocks[i][j] = block
            count += 1

    n1 = ''
    n2 = ''

    for i in range(1, -1, -1):
        for j in range(7, -1, -1):
            a = format((int(blocks[0][0][i][j][0]) & 0b1), '01b')
            b = format((int(blocks[0][0][i][j][1]) & 0b1), '01b')
            n1 += a
            n2 += b

    return (int(n1, 2), int(n2, 2), blocks)


def debed(img, wt):
    count = 0
    for i in range(im_h // 8):
        for j in range(im_w // 8):
            if count == wt_h * wt_w:
                break
            if count == 0:
                r = round(abs(img[0][1][-1][-1][0]))
                g = round(abs(img[0][1][-1][-1][1]))
                b = round(abs(img[0][1][-1][-1][2]))
            if count != 0:
                r = round(abs(img[i][j][-1][-1][0]))
                g = round(abs(img[i][j][-1][-1][1]))
                b = round(abs(img[i][j][-1][-1][2]))
            pixel = [r, g, b]
            wt[count // wt_w][count % wt_w] = pixel
            count += 1
    return wt

def putin_zero(img):
    if im_h % 8 != 0:
        for i in range(im_h):
            for j in range(-1, -1-(im_w % 8), -1):
                img[i][j] = [0, 0, 0]
        for i in range(-1, -1-(im_h % 8), -1):
            for j in range(im_w):
                img[i][j] = [0, 0, 0]
    return img

if __name__ == "__main__":
    img1 = cv2.imread('image3.bmp', 1)
    wt_h, wt_w, _ = decode(img1)
    im_h, im_w, _ = img1.shape
    _, _, blocks = decode(putin_zero(img1))   
    img2 = np.zeros([wt_h, wt_w], dtype = int).tolist()
    cv2.imwrite('image4.bmp', np.array(debed(blocks, img2)))