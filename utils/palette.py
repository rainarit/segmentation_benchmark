def get_voc_palette(num_classes):
    n = num_classes
    palette = [0]*(n*3)
    for j in range(0,n):
            lab = j
            palette[j*3+0] = 0
            palette[j*3+1] = 0
            palette[j*3+2] = 0
            i = 0
            while (lab > 0):
                    palette[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                    palette[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                    palette[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                    i = i + 1
                    lab >>= 3
    return palette

COCO_palette = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
                1, 64, 20, 63, 7, 72]