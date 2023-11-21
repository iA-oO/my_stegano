import cv2
import zlib
import numpy as np
from scipy.fftpack import dct


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


def decompress_data_from_byte_array(byte_array):
    compressed_data = bytes(byte_array)
    decompressed_data = zlib.decompress(compressed_data)
    return np.array(list(decompressed_data))


def dct2(a):
    return dct(dct(np.array(a).T, norm='ortho').T, norm='ortho')


def extract_4bits_symmetric(pixel_block):
    recovered_dct_coefficients = dct2(pixel_block)

    extracted_bits_binary = ''
    for i in range(4):
        x1, y1 = i, 7 - i
        x2, y2 = 7 - i, i
        bit = '1' if recovered_dct_coefficients[x1, y1] > recovered_dct_coefficients[x2, y2] else '0'
        extracted_bits_binary += bit

    return extracted_bits_binary


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

    n_1 = ''
    n_2 = ''

    for i in range(1, -1, -1):
        for j in range(7, -1, -1):
            a = format((int(blocks[0][0][i][j]) & 0b1), '01b')
            n_1 += a

    for i in range(len(blocks[0][0])-2, len(blocks[0][0])):
        for j in range(len(blocks[0][0][0])-8, len(blocks[0][0][0])):
            a = format((int(blocks[0][0][i][j]) & 0b1), '01b')
            n_2 += a

    return (int(n_1, 2), int(n_2, 2), blocks)


def putin_zero(img):
    if im_h % 8 != 0:
        for i in range(im_h):
            for j in range(-1, -1-(im_w % 8), -1):
                img[i][j] = 0
        for i in range(-1, -1-(im_h % 8), -1):
            for j in range(im_w):
                img[i][j] = 0
    return img


def extract(img_1, img_2):
    count = 0
    for i in range(im_h // 8):
        for j in range(im_w // 8):
            if count == l * 2 + 1:
                break
            if count != 0 and count % 2 == 1:
                pixel_1 = extract_4bits_symmetric(img_1[i][j])
            if count != 0 and count % 2 == 0:
                pixel_2 = extract_4bits_symmetric(img_1[i][j])
                pixel = int(pixel_1 + pixel_2, 2)
                img_2[(count - 1) // 2] = pixel
            count += 1
    return img_2


if __name__ == "__main__":
    img1 = cv2.imread('image3.bmp', 0)
    l, shape, _ = decode(img1)
    im_h, im_w = img1.shape
    _, _, blocks = decode(putin_zero(img1))
    img2 = np.zeros([l], dtype = int).tolist()
    cv2.imwrite('image4.bmp', dearnold(decompress_data_from_byte_array(extract(blocks, img2)).reshape(shape, shape)))