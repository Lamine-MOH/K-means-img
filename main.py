import math
import random
import sys
import cv2 as cv
import numpy as np

#       ###### data transform #####


def img_to_data(img, height, width):
    data = []

    for i in range(height):
        for j in range(width):
            data.append(dict(blue=img[i][j][0],
                             green=img[i][j][1],
                             red=img[i][j][2],
                             cluster=0))
    return data


def data_to_img(data, height, width):
    img = []
    data_index = 0

    for i in range(height):
        img.append([])
        for j in range(width):
            img[i].append([data[data_index]['blue'], data[data_index]
                          ['green'], data[data_index]['red']])
            data_index = data_index+1
    return np.array(img)


def order_data(old_data, height, width):
    new_data = []

    for i in range(height):
        for j in range(width):
            test = 0
            for k in range(len(old_data)):
                if(old_data[k]['x'] == i and old_data[k]['y'] == j):
                    new_data.append(dict(blue=old_data[k]['blue'],
                                         green=old_data[k]['green'],
                                         red=old_data[k]['red']))
                    test = 1
                    break

            if (test == 0):
                new_data.append(dict(blue=255, green=255, red=255))
    return new_data


#       ##### K mean algorithm #####
def random_points(data, points_num):
    length = len(data)
    points = []

    for i in range(points_num):
        while True:
            point_index = random.randint(0, length - 1)
            if data[point_index] in points:
                continue
            break
        points.append(data[point_index])
    return points

# return the cluster of an pixel


def clustering_data(pixel, points):
    pixel_distances = []

    for point in points:
        pixel_distances.append(round(math.sqrt(math.pow(pixel['blue'] - point['blue'], 2) +
                                               math.pow(pixel['green'] - point['green'], 2) +
                                               math.pow(pixel['red'] - point['red'], 2))))

    cluster = 0
    min_distance = pixel_distances[0]
    for i in range(1, len(pixel_distances)):
        if pixel_distances[i] < min_distance:
            cluster = i
            min_distance = pixel_distances[i]

    return cluster

# return the clusters of the data


def get_clusters(data, k_value):
    clusters = []

    for i in range(k_value):
        clusters.append([])

    for pixel in data:
        clusters[pixel['cluster']].append(pixel)

    return clusters

# return the middle point of the cluster


def cluster_middle_point(cluster):
    cluster_distances = []

    for point in cluster:
        distances = 0
        for pixel in cluster:
            distances = distances + math.sqrt(math.pow(pixel['blue'] - point['blue'], 2) +
                                              math.pow(pixel['green'] - point['green'], 2) +
                                              math.pow(pixel['red'] - point['red'], 2))
        cluster_distances.append(distances)

    middle_point_index = 0
    min_distance = cluster_distances[0]
    for i in range(1, len(cluster_distances)):
        if cluster_distances[i] < min_distance:
            middle_point_index = i
            min_distance = cluster_distances[i]

    return cluster[middle_point_index]

# change the color of the pixels to the cluster middle point color


def unit_colors(data, colors):
    for pixel in data:
        pixel['blue'] = colors[pixel['cluster']]['blue']
        pixel['green'] = colors[pixel['cluster']]['green']
        pixel['red'] = colors[pixel['cluster']]['red']

# return the clusters images array


def get_clusters_img(data, k_value, height, width, fill_color={'blue': 255, 'green': 255, 'red': 255}):
    clusters_data = []

    for i in range(k_value):
        clusters_data.append([])

    for pixel in data:
        for i in range(k_value):
            if pixel['cluster'] == i:
                clusters_data[i].append(pixel)
            else:
                clusters_data[i].append(fill_color)

    clusters_img = []
    for cluster in clusters_data:
        clusters_img.append(data_to_img(cluster, height, width))

    return clusters_img

# k means algorithme


def k_means_img(data, k_value):
    points = random_points(data, k_value)

    # repeating the process until the center of all clusters not change
    while True:
        # clustering the pixels
        for pixel in data:
            pixel['cluster'] = clustering_data(pixel, points)

        clusters = get_clusters(data, k_value)

        test = 0
        for i in range(k_value):
            middle_point = cluster_middle_point(clusters[i])
            if points[i] != middle_point:
                test = 1
                points[i] = middle_point
            else:
                break

        # break the loop if the middle point not changing
        if test == 0:
            break

    colors = [dict(blue=0, green=0, red=0),
              dict(blue=255, green=0, red=0),
              dict(blue=0, green=255, red=0),
              dict(blue=0, green=0, red=255),
              dict(blue=255, green=255, red=0),
              dict(blue=255, green=0, red=255),
              dict(blue=0, green=255, red=255),
              dict(blue=100, green=200, red=255),
              dict(blue=255, green=100, red=200),
              dict(blue=200, green=255, red=100),
              dict(blue=255, green=200, red=100)]
    # change the pixels color to the cluster middle point color
    unit_colors(data, colors)

    #
    return data


def main():
    fileName = sys.argv[1]

    print("loading the img")
    img = cv.imread(f"img/{fileName}")
    
    cv.imwrite("result/before.png", img)
    
    height = img.shape[0]
    width = img.shape[1]

    k_value = 4
    if(len(sys.argv)>2):
        k_value = int(sys.argv[2]) # set the k num from the input
        
    data = img_to_data(img, height, width)  
    new_data = k_means_img(data, k_value)

    new_img = data_to_img(data, height, width)
    clusters_img = get_clusters_img(new_data, k_value, height, width)

    print("saving the images")
    # save the images
    cv.imwrite("result/before.png", img)
    cv.imwrite("result/after.png", new_img)
    for i in range(len(clusters_img)):
        cv.imwrite(f"result/cluster-{i}.png", clusters_img[i])

    # show the images if ask
    if(len(sys.argv) > 3):
        resize_value = 3

        img = cv.resize(img, (0, 0), fx=resize_value, fy=resize_value)
        cv.imshow("before", img)

        #new_img = cv.resize(new_img, (0,0), fx=resize_value, fy=resize_value)
        new_img = cv.resize(cv.imread("result/after.png"),
                            (0, 0), fx=resize_value, fy=resize_value)
        cv.imshow("after", new_img)

        for i in range(k_value):
            name = f"cluster-{i}"
            path = f"result/{name}.png"
            cluster_img = cv.imread(path)

            cluster_img = cv.resize(
                cluster_img, (0, 0), fx=resize_value, fy=resize_value)
            cv.imshow(name, cluster_img)

        # wait until click any key to exit
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
