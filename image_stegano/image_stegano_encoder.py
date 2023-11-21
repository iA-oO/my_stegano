import cv2
import zlib
import numpy as np
from scipy.fftpack import dct, idct


def compress_data_to_byte_array(data):
    compressed_data = zlib.compress(bytes(data))
    byte_array = list(compressed_data)
    return byte_array


def arnold(img):
    r, c = img.shape
    p = np.zeros((r, c), np.uint8)
    a = 1
    b = 1
    for i in range(r):
        for j in range(c):
            x = (i + b * j) % r
            y = (a * i + (a * b + 1) * j) % c
            p[x, y] = img[i, j]
    return p


def dct2(a):
    return dct(dct(np.array(a).T, norm='ortho').T, norm='ortho')


def idct2(a):
    return idct(idct(np.array(a).T, norm='ortho').T, norm='ortho')


def embed_4bits_symmetric(pixel_block, bits_to_embed, threshold=44):
    bits_to_embed_binary = '{:04b}'.format(bits_to_embed)
    dct_coefficients = dct2(pixel_block)

    for i, bit in enumerate(bits_to_embed_binary):
        x1, y1 = i, 7 - i
        x2, y2 = 7 - i, i
        if bit == "1":
            dct_coefficients[x1, y1] = max(dct_coefficients[x2, y2], dct_coefficients[x1, y1] + threshold)
            dct_coefficients[x2, y2] = min(dct_coefficients[x1, y1], dct_coefficients[x2, y2] - threshold)
        else:
            dct_coefficients[x1, y1] = min(dct_coefficients[x2, y2], dct_coefficients[x1, y1] - threshold)
            dct_coefficients[x2, y2] = max(dct_coefficients[x1, y1], dct_coefficients[x2, y2] + threshold)

    camouflaged_pixel_block = idct2(dct_coefficients)

    camouflaged_pixel_block_rounded = np.round(camouflaged_pixel_block).astype(int)
    camouflaged_pixel_block_clipped = np.clip(camouflaged_pixel_block_rounded, 0, 255)

    return camouflaged_pixel_block_clipped


def embed_size(img, n_1, n_2):
    for i in range(2):
        for j in range(8):
                img[0][0][i][j] = int(img[0][0][i][j]) & 0xFE

    for i in range(-1, -3, -1):
        for j in range(-1, -9, -1):
                img[0][0][i][j] = int(img[0][0][i][j]) & 0xFE

    block = img[0][0]

    for i in range(2):
        for j in range(8):
            if n_1 == 0:
                break
            block[i][j] |= (n_1 & 0b1)
            n_1 = n_1 >> 1

    for i in range(-1, -3, -1):
        for j in range(-1, -9, -1):
            if n_2 == 0:
                break
            block[i][j] |= (n_2 & 0b1)
            n_2 = n_2 >> 1

    img[0][0] = block


def embed_numbers(img, num):
    count = 0
    n = im_w // 8
    while 1:
        if count == l * 2 + 1:
            break
        if count != 0 and count % 2 == 1:
            img[count // n][count % n] = embed_4bits_symmetric(img[count // n][count % n], int((format(num[(count - 1) // 2], "08b")[0:4]), 2))
        if count != 0 and count % 2 == 0:
            img[count // n][count % n] = embed_4bits_symmetric(img[count // n][count % n], int((format(num[(count - 1) // 2], "08b")[4:8]), 2))                   
        count += 1


def encode(img, number):

    n = im_h // 8
    m = im_w // 8
    blocks = np.zeros([n, m], dtype = int).tolist()

    for i in range(n):
        for j in range(m):
            block = np.zeros([8, 8], dtype = int).tolist()
            blocks[i][j] = block

    for i in range(n):
        for j in range(m):
            block = img [i*8:(i+1)*8, j*8:(j+1)*8].tolist()
            blocks[i][j] = block

    embed_size(blocks, l, shape)
    embed_numbers(blocks, number)

    return blocks


def merge_blocks(blocks):
    n = im_h // 8
    m = im_w // 8
    img2 = np.zeros([im_h, im_w, 3], dtype = int).tolist()

    for i in range(n):
        for j in range(m):
            for k in range(8):
                for p in range(8): 
                    img2[i*8+k][j*8+p] = blocks[i][j][k][p]
    return img2


def del_zero(img3, img1):
    if im_h % 8 != 0:
        for i in range(im_h):
            for j in range(-1, -1-(im_w % 8), -1):
                img3[i][j] = img1[i][j]
        for i in range(-1, -1-(im_h % 8), -1):
            for j in range(im_w):
                img3[i][j] = img1[i][j]
    return img3


if __name__ == "__main__":
    img1 = cv2.imread('image1.bmp', 0)
    img2 = cv2.imread('image2.bmp', 0)
    im_h, im_w = img1.shape
    shape, _ = img2.shape

    img2 = arnold(img2)

    compressed_pixel_data = compress_data_to_byte_array(img2)
    l = len(compressed_pixel_data)

    img3 = merge_blocks(encode(img1, compressed_pixel_data))

    cv2.imwrite('image3.bmp', np.array(del_zero(img3, img1)))