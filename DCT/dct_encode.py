import numpy as np
import cv2
from scipy.fft import dctn, idctn

def embed_size(img, n1, n2):

    # 第一小块前16像素lsb置0
    for i in range(2):
        for j in range(8):
            if img[0][0][i][j][0] >= 0:
                img[0][0][i][j][0] = int(img[0][0][i][j][0]) & 0xFE
            else:
                img[0][0][i][j][0] = -(abs(int(img[0][0][i][j][0])) & 0xFE)
            if img[0][0][i][j][1] >= 0:
                img[0][0][i][j][1] = int(img[0][0][i][j][1]) & 0xFE
            else:
                img[0][0][i][j][1] = -(abs(int(img[0][0][i][j][1])) & 0xFE)

    block = img[0][0]
    for i in range(2):
        for j in range(8):
            if n1 == 0 and n2 == 0:
                break
            block[i][j][0] |= (n1 & 0b1)
            block[i][j][1] |= (n2 & 0b1)
            n1 = n1 >> 1
            n2 = n2 >> 1

    img[0][0] = block

def embed_pixel(img, r, g, b, count):
    n = im_w // 8
    img[count // n][count % n][-1][-1][0] = r
    img[count // n][count % n][-1][-1][1] = g
    img[count // n][count % n][-1][-1][2] = b

def embed_pixels(img, wt):
    count = 0
    for i in range(wt_h):
        for j in range(wt_w):
            if count != 0:
                r = wt[i][j][0]
                g = wt[i][j][1]
                b = wt[i][j][2]
                embed_pixel(img, r, g, b, count)
            count += 1


def encode(img, wt):

# 分n*m块，每块8*8
    n = im_h // 8
    m = im_w // 8
    blocks = np.zeros([n, m], dtype = int).tolist()

    for i in range(n):
        for j in range(m):
            block = np.zeros([8, 8], dtype = int).tolist()
            blocks[i][j] = block

# 分块，dct
    count = 0
    for i in range(n):
        for j in range(m):
            block = img [i*8:(i+1)*8, j*8:(j+1)*8].tolist()
            if count > 0:
                block = dctn(block).tolist()
            blocks[i][j] = block
            count += 1

    embed_size(blocks, wt_h, wt_w)
    embed_pixels(blocks, wt)


#idct
    count = 0
    for i in range(n):
        for j in range(m):
            if count > 0:
                blocks[i][j] = idctn(blocks[i][j]).tolist()
            count += 1
    return blocks

def to_int(blocks):
    for i in range(len(blocks)):
        for j in range(len(blocks[0])):
            for k in range(len(blocks[0][0])):
                for n in range(len(blocks[0][0][0])):
                    for m in range(3):
                        blocks[i][j][k][n][m] = abs(round(blocks[i][j][k][n][m]))
    return blocks

def merge_blocks(blocks):
    n = im_h // 8
    m = im_w // 8
    img2 = np.zeros([im_h, im_w, 3], dtype = int).tolist()

    for i in range(n):
        for j in range(m):
            for k in range(8):
                for l in range(8): 
                    img2[i*8+k][j*8+l] = blocks[i][j][k][l]
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
    img1 = cv2.imread('image1.bmp', 1)
    img2 = cv2.imread('image2.bmp', 1)
    im_h, im_w, _ = img1.shape
    wt_h, wt_w, _ = img2.shape
    img3 = merge_blocks(to_int(encode(img1, img2)))
    cv2.imwrite('image3.bmp', np.array(del_zero(img3, img1)))