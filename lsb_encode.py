from PIL import Image

def embed_size(img, wt):
    im_pixels = list(img.getdata())
    w = format(wt.size[0], "08b")
    h = format(wt.size[1], "08b")

    for i in range(10):
        pixel = list(im_pixels[i])
        pixel[0] &= 0xFE
        pixel[1] &= 0xFE
        im_pixels[i] = tuple(pixel)

    count = 1
    for i in range(9, -1, -1):
        if count > len(w) and count > len(h):
            break
        pixel = list(im_pixels[i])
        if count <= len(w):
            pixel[0] |= int(w[-count])
        if count <= len(h):
            pixel[1] |= int(h[-count])
        im_pixels[i] = tuple(pixel)
        count += 1
    img.putdata(im_pixels)
    return img

def embed(wt_pixel, im_pixels, count1):
    count2 = 0
    for i in range(10 + count1 * 4, 14 + count1 * 4):
        pixel = im_pixels[i]
        r = (pixel[0] & 0xFC) | int(wt_pixel[0][0 + count2 * 2 : 2 + count2 * 2], 2)
        g = (pixel[1] & 0xFC) | int(wt_pixel[1][0 + count2 * 2 : 2 + count2 * 2], 2)
        b = (pixel[2] & 0xFC) | int(wt_pixel[2][0 + count2 * 2 : 2 + count2 * 2], 2)
        count2 += 1
        im_pixels[i] = tuple([r, g, b])
    return im_pixels

def encode(img, wt):
    im_pixels = list(img.getdata())
    count = 0
    for i in range(wt.size[0]):
        for j in range(wt.size[1]):
            pixel = wt.getpixel((i, j))
            wt_r = format(pixel[0], "08b")
            wt_g = format(pixel[1], "08b")
            wt_b = format(pixel[2], "08b")
            bin_pixel = [wt_r, wt_g, wt_b]
            im_pixels = embed(bin_pixel, im_pixels, count)
            count += 1
    return im_pixels

if __name__ == "__main__":
    """
    只能使用长，宽<1023的水印
    原始图片的大小不能小于水印图片大小的四倍
    """
    img1 = Image.open("原始图片.bmp")
    img2 = Image.open("水印图片.bmp")
    img1 = embed_size(img1, img2)
    img1.putdata(encode(img1, img2))
    img1.save("加密图片.bmp")