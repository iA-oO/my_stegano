from PIL import Image
def get_size(pixels):
    n1 = ""
    n2 = ""
    pixels = pixels[:10]
    for pixel in pixels:
        r_lsb = pixel[0] & 1
        g_lsb = pixel[1] & 1
        n1 += str(r_lsb)
        n2 += str(g_lsb)
    num1 = int(n1, 2)
    num2 = int(n2, 2)
    return num1, num2

def decode(img):
    count = 0
    wt_pixels = []
    wt_r = ""
    wt_g = ""
    wt_b = ""
    for j in range(10, w * h * 4 + 11):
        pixel = img[j]
        r = format(pixel[0], "08b")[-2:]
        g = format(pixel[1], "08b")[-2:]
        b = format(pixel[2], "08b")[-2:]
        wt_r += r
        wt_g += g
        wt_b += b
        count += 1
        if count == 4:
            wt_pixel = tuple([int(wt_r, 2), int(wt_g, 2), int(wt_b, 2)])
            wt_pixels.append(wt_pixel)
            wt_r = ""
            wt_g = ""
            wt_b = ""
            count = 0
    return wt_pixels

def show_wt(wt, wt_pixels):
    count = 0
    for i in range(w):
        for j in range(h):
            wt.putpixel((i, j), wt_pixels[count])
            count += 1
    wt.save("解码后得到的水印图片.bmp")


if __name__ == "__main__":
    img = Image.open("加密图片.bmp")
    im_pixels = list(img.getdata())
    w, h = get_size(im_pixels)
    wt = Image.new("RGB",(w, h))
    wt = show_wt(wt, decode(im_pixels))