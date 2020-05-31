import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image


def display_image(img):
    '''
    Read images
    '''
    myimage = Image.open(img)
    return myimage


def create_sliding_matrices(color_array, height, width):
    '''
    Computes the array of all sliding 2x2 submatrices from a given matrix.
    '''
    matrix_array = np.zeros(shape = (height - 1, width - 1, 2, 2))
    col_mat = np.array(color_array).astype(int).reshape((height, width))
            
    for i in range(0, height - 1):
        for j in range(0, width - 1):
            matrix_array[i, j] = col_mat[np.ix_([i, i + 1],[j, j + 1])].astype(int)
    return(matrix_array)


def compute_permutation(matrix):
    '''
    Computes the permutation associated with the 2x2 matrix.
    '''
    permutation = np.argsort(matrix.flatten())
    return permutation


def permutataion_distribution(matrix_list):
    '''
    Returns the distribution of matrix permutaions.        
    '''
    list_of_perm = []
    num_of_perm = []
    distribution = np.zeros(24)
    for i in matrix_list:
        mat_perm = compute_permutation(np.array(i))
        indicator = 0
        for j in range(0, len(list_of_perm)):
            if np.all(mat_perm == list_of_perm[j]):
                num_of_perm[j] = num_of_perm[j] + 1
                indicator = 1
        if indicator == 0:
            list_of_perm.append(mat_perm)
            num_of_perm.append(1)
            
    distribution = np.array(num_of_perm) / sum(num_of_perm)
    return num_of_perm, list_of_perm, distribution

def compute_entropy(distribution):
    '''
    Compute normalised Shannon entropy from distribution
    '''
    Shannon_entropy = -np.sum(distribution * np.log2(distribution))
    return Shannon_entropy


def normalise_entropy(distribution):
    '''
    Compute normalised Shannon entropy from distribution
    '''
    Shannon_norm = -np.sum(distribution * np.log2(distribution)) / np.log2(len(distribution))
    return Shannon_norm


def compute_complexity(distribution):
    '''Statistical Complexity from distribution'''
    comp = compute_divergence(distribution) * normalise_entropy(distribution) / find_norm_constant(distribution)
    return comp


def compute_divergence(distribution):
    '''Jensen-Shannon divergence'''
    n = len(distribution)
    uniform = np.array([1/n] * n)
    div = compute_entropy((distribution + uniform)/2) - compute_entropy(distribution)/2 - compute_entropy(uniform)/2
    return div


def find_norm_constant(distribution):
    '''Normalisation constant D*'''
    n = len(distribution)
    const = -0.5 * (((n + 1) / n) * np.log(n + 1) + np.log(n) - 2 * np.log(2 * n))
    return const


def produce_entropy_complexity(name):
    '''
    Entropy and complexity computations.
    '''
    img1 = cv2.imread(name)
    myimage = display_image(name)
    
    painting_width, painting_height = myimage.size
    b, g, r = cv2.split(img1)
    
    b = pd.Series(b.flatten(), dtype='int')
    g = pd.Series(g.flatten(), dtype='int')
    r = pd.Series(r.flatten(), dtype='int')
    
    mean_channel = (b + g + r)/3
    sliding_matrix = create_sliding_matrices(mean_channel,
                                             painting_height,
                                             painting_width).reshape(((painting_height - 1) * (painting_width-1),2,2))
    res1 = permutataion_distribution(sliding_matrix)
    entropy = normalise_entropy(res1[2])
    complexity = compute_complexity(res1[2])
    return entropy, complexity


dataPath = './output'
pathDir = os.listdir(dataPath)
cnt = len(pathDir)

res = []
for img in pathDir:
    a,b = produce_entropy_complexity(os.path.join(dataPath,img))
    res.append([a,b])
    print("img {} complexity {} entropy {}".format(img, b, a))
    
avg_c, avg_e = 0, 0
for ele in res:
    avg_c += ele[1]
    avg_e += ele[0]
print("\nTest finish\n avg complexity: {} avg entropy: {}".format(avg_c/cnt,avg_e/cnt))