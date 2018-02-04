#!/usr/bin/env python3

import time
import scipy.stats
import numpy as np


def get_mode(list_ratings):
    return scipy.stats.mode(list_ratings)[0][0]


def get_mean(list_ratings):
    return np.mean(list_ratings)


def get_median(list_ratings):
    return np.median(list_ratings)


def write_computed_ratings(image_ratings, computation_fn, filename):
    with open(filename, 'w') as f:
        f.write('Image,Rating')
        for image_name in image_ratings:
            f.write('\n{},{}'.format(image_name, computation_fn(image_ratings[image_name])))


if __name__ == '__main__':

    image_ratings = {}

    with open('All_ratings.csv', 'rU') as f:
        # Drop the column descriptions
        next(f)
        for line in f:
            rater, image_name, rating, orig_rating = line.split(',')
            if image_name in image_ratings:
                image_ratings[image_name].append(int(rating))
            else:
                image_ratings[image_name] = [int(rating)]

    write_computed_ratings(image_ratings, get_mode, 'ratings_mode.csv')
    write_computed_ratings(image_ratings, get_mean, 'ratings_mean.csv')
    write_computed_ratings(image_ratings, get_median, 'ratings_median.csv')
