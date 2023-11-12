import numpy as np
import cv2
from scipy.fftpack import dct, idct


def dearnold(img):
    r, c = img.shape
    p = np.zeros((r, c), np.uint8)
    a = 1
    b = 1
    for i in range(r):
        for j in range(c):
            x = ((a * b + 1) * i - b * j) % r
            y = (-a * i + j) % c
            p[x, y] = img[i, j]
    return p


def dct2(a):
    return dct(dct(np.array(a).T, norm='ortho').T, norm='ortho')


def extract_8bits_symmetric(pixel_block):
    recovered_dct_coefficients = dct2(pixel_block)

    extracted_bits_binary = ''
    for i in range(8):
        if i < 4:
            x1, y1 = i, 7 - i
            x2, y2 = 7 - i, i
        else:
            x1, y1 = 10 - i, i - 4
            x2, y2 = i - 3, 11 - i
        bit = '1' if recovered_dct_coefficients[x1, y1] > recovered_dct_coefficients[x2, y2] else '0'
        extracted_bits_binary += bit

    extracted_bits = int(extracted_bits_binary, 2)

    return extracted_bits


def decode(img):

    h, w = img.shape
    n = h // 8
    m = w // 8
    blocks = np.zeros([n, m], dtype = int).tolist()

    for i in range(n):
        for j in range(m):
            block = np.zeros([8, 8], dtype = int).tolist()
            blocks[i][j] = block

    count = 0
    for i in range(n):
        for j in range(m):
            block = img [i*8:(i+1)*8, j*8:(j+1)*8]
            blocks[i][j] = block
            count += 1

    n = ''

    for i in range(1, -1, -1):
        for j in range(7, -1, -1):
            a = format((int(blocks[0][0][i][j]) & 0b1), '01b')
            n += a

    return (int(n, 2), blocks)


def putin_zero(img):
    if im_h % 8 != 0:
        for i in range(im_h):
            for j in range(-1, -1-(im_w % 8), -1):
                img[i][j] = 0
        for i in range(-1, -1-(im_h % 8), -1):
            for j in range(im_w):
                img[i][j] = 0
    return img


def debed(img, wt):
    count = 0
    for i in range(im_h // 8):
        for j in range(im_w // 8):
            if count == shape * shape:
                break
            if count == 0:
                pixel = round(abs(img[0][1][-1][1]))
            if count != 0:
                pixel = extract_8bits_symmetric(img[i][j])
            wt[count // shape][count % shape] = pixel
            count += 1
    return wt


if __name__ == "__main__":
    img1 = cv2.imread('image3.bmp', 0)
    shape, _ = decode(img1)
    im_h, im_w = img1.shape
    _, blocks = decode(putin_zero(img1))
    img2 = np.zeros([shape, shape], dtype = int).tolist()
    cv2.imwrite('image4.bmp', dearnold(np.array(debed(blocks, img2))))