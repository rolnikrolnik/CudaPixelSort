import os
from scipy import misc
path = "/home/rolnik/Documents/cuda"
image = misc.imread(os.path.join(path, "example.bmp"), flatten=0, mode="RGB")

with open('pixels', 'w+') as file:
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

