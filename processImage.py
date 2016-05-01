import os
import sys

from scipy import misc

path = sys.argv[1]
filename = sys.argv[2]
tmp_pixels = sys.argv[3]

image = misc.imread(os.path.join(path, filename), flatten=0, mode="RGB")

with open(tmp_pixels, 'w+') as file:
    for y in range(0, image.shape[0]):
        for x in range(0, image.shape[1]): 
            file.write(str(image[y][x][0]) + ',')
            file.write(str(image[y][x][1]) + ',')
            file.write(str(image[y][x][2]) + ';')
        file.write('\n')

# for y in range(0, image.shape[0]):
#     for x in range(0, image.shape[1]): 
#         for pixel_color in range(0, image.shape[2]):
#             image[y][x][pixel_color] = ~image[y][x][pixel_color]

# misc.imshow(image)

# for x in range(0,100):
#     for y in range(0,100):
#         for color in range(0,3):
#             image[x][y][color] = 0

