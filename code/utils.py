import cv2
import numpy
import timeit
from sklearn import neighbors, svm, cluster
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import SVC 
from timeit import default_timer as timer
import os

from sklearn.feature_extraction import image
from sklearn import metrics
from sklearn.cluster import KMeans

import numpy as np


def load_data():
    test_path = '../data/test/'
    train_path = '../data/train/'
    
    train_classes = sorted([dirname for dirname in os.listdir(train_path)], key=lambda s: s.upper())
    test_classes = sorted([dirname for dirname in os.listdir(test_path)], key=lambda s: s.upper())
    train_labels = []
    test_labels = []
    train_images = []
    test_images = []
    for i, label in enumerate(train_classes):
        for filename in os.listdir(train_path + label + '/'):
            image = cv2.imread(train_path + label + '/' + filename, 0)#cv2.CV_LOAD_IMAGE_GRAYSCALE
            train_images.append(image)
            train_labels.append(i)
    for i, label in enumerate(test_classes):
        for filename in os.listdir(test_path + label + '/'):
            image = cv2.imread(test_path + label + '/' + filename, 0)#cv2.CV_LOAD_IMAGE_GRAYSCALE
            test_images.append(image)
            test_labels.append(i)
            
    return train_images, test_images, train_labels, test_labels


def KNN_classifier(train_features, train_labels, test_features, num_neighbors):
    # outputs labels for all testing images
    # train_features is an N x d matrix, where d is the dimensionality of the
    # feature representation.
    # train_labels is an N x 1 cell array, where each entry is a string
    # indicating the ground truth category for each training image.
    # test_features is an M x d matrix, where d is the dimensionality of the
    # feature representation. You can assume M = N unless you've modified the
    # starter code.
    # predicted_categories is an M x 1 cell array, where each entry is a string
    # indicating the predicted category for each test image.
    
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    knn.fit(train_features, train_labels)
    predicted_categories = knn.predict(test_features)
    
    return predicted_categories


def SVM_classifier(train_features, train_labels, test_features, is_linear, svm_lambda):
    # this function will train a linear svm for every category (i.e. one vs all)
    # and then use the learned linear classifiers to predict the category of
    # every test image. every test feature will be evaluated with all 15 svms
    # and the most confident svm will "win". confidence, or distance from the
    # margin, is w*x + b where '*' is the inner product or dot product and w and
    # b are the learned hyperplane parameters.

    # train_features is an n x d matrix, where d is the dimensionality of
    # the feature representation.
    # train_labels is an n x 1 cell array, where each entry is a string 
    # indicating the ground truth category for each training image.
    # test_features is an m x d matrix, where d is the dimensionality of the
    # feature representation. (you can assume m=n unless you modified the 
    # starter code)
    # is_linear is a boolean. If true, you will train linear SVMs. Otherwise, you 
    # will use SVMs with a Radial Basis Function (RBF) Kernel.
    # lambda is a scalar, the value of the regularizer for the SVMs
    # predicted_categories is an m x 1 cell array, where each entry is a sstring
    # indicating the predicted category for each test image.

    if(is_linear):
      svm_model_linear = SVC(kernel='linear', C=1).fit(train_features, train_labels)
      predicted_categories = svm_model_linear.predict(test_features)
    else:
      svm_model_linear = SVC(kernel='rbf', C=1).fit(train_features, train_labels)
      predicted_categories = svm_model_linear.predict(test_features)      
    return predicted_categories


def imresize(input_image, target_size):
    # resizes the input image to a new image of size [target_size, target_size]. normalizes the output image
    # to be zero-mean, and in the [-1, 1] range.

    resize_scale = (target_size, target_size)
    resized = cv2.resize(input_image, resize_scale, interpolation=cv2.INTER_AREA)
    norm_and_resized = cv2.normalize(resized, None, -1, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    return norm_and_resized.flatten()


def reportAccuracy(true_labels, predicted_labels):
    # generates and returns the accuracy of a model
    # true_labels is a n x 1 cell array, where each entry is a string
    # and n is the size of the testing set.
    # predicted_labels is a n x 1 cell array, where each entry is a 
    # string, and n is the size of the testing set. these labels 
    # were produced by your system
    # label_dict is a 15x1 cell array where each entry is a string
    # containing the name of that category
    # accuracy is a scalar, defined in the spec (in %)
    
    return 100 * metrics.accuracy_score(true_labels, predicted_labels)


def computeSift(image):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors
  
def computeSurf(image):
    surf = cv2.xfeatures2d.SURF_create()
    keypoints, descriptors = surf.detectAndCompute(image, None)
    return keypoints, descriptors

def computeOrb(image):
    orb = cv2.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors
  
def buildDict(train_images, dict_size, feature_type, clustering_type):
    # this function will sample descriptors from the training images,
    # cluster them, and then return the cluster centers.
    # train_images is a n x 1 array of images
    # dict_size is the size of the vocabulary,
    # feature_type is a string specifying the type of feature that we are interested in.
    # Valid values are "sift", "surf" and "orb"
    # clustering_type is one of "kmeans" or "hierarchical"
    # the output 'vocabulary' should be dict_size x d, where d is the 
    # dimention of the feature. each row is a cluster centroid / visual word.
    
    final_descriptors = []
    dimension = 0

    for image in train_images:
      keypoints, descriptors = None, []
      
      if feature_type == "sift":
        keypoints, descriptors = computeSift(image)
        dimension = 128
      elif feature_type == "surf":
        keypoints, descriptors = computeSurf(image)
        dimension = 64
      elif feature_type == "orb":
        keypoints, descriptors = computeOrb(image)
        dimension = 32

      for descriptor in descriptors:
        final_descriptors.append(descriptor)

    if clustering_type == "kmeans":
      kmeans = KMeans(n_clusters=dict_size)
      kmeans.fit(final_descriptors)
      return kmeans.cluster_centers_ 
    
    elif(clustering_type == "hierarchical"):
      # use a subset of descriptors to reduce memory usage
      subset_size = 10000
      agg_cluster = AgglomerativeClustering(n_clusters=dict_size, affinity='euclidean', linkage='ward')
      random_subarray = np.random.random_integers(len(final_descriptors)-1, size=subset_size)

      random_descriptors = []
      for i in random_subarray:
        random_descriptors.append(final_descriptors[i])
      agg_cluster.fit_predict(random_descriptors)

      cluster_count = [0] * dict_size
      cluster_values = [[0] * dimension for _ in range(dict_size)]
      for i in range(len(random_descriptors)):
        cluster_count[agg_cluster.labels_[i]] += 1 
        # get the total counts for each dimension in each cluster center
        for j in range(dimension):
          cluster_values[agg_cluster.labels_[i]][j] += random_descriptors[i][j]
      # get the average values for each cluster center
      for i in range(len(random_descriptors)):
        for j in range(dimension):
          (cluster_values[agg_cluster.labels_[i]])[j] /= cluster_count[agg_cluster.labels_[i]]
          
      cluster_centers = []
      for i in cluster_values:
        cluster_centers.append(random_descriptors[KNN_classifier(random_descriptors, [x for x in range(subset_size)], [i], 1)[0]])
      return cluster_centers


def computeBow(image, vocabulary, feature_type):
    # extracts features from the image, and returns a BOW representation using a vocabulary
    # image is 2D array
    # vocabulary is an array of size dict_size x d
    # feature type is a string (from "sift", "surf", "orb") specifying the feature
    # used to create the vocabulary
    # BOW is the new image representation, a normalized histogram
    
    keypoints, descriptors = None, None

    if feature_type == "sift":
      keypoints, descriptors = computeSift(image)
    elif feature_type == "surf":
      keypoints, descriptors = computeSurf(image)
    elif feature_type == "orb":
      keypoints, descriptors = computeOrb(image)
    
    # create labels for each of the cluster centers
    vocab_labels = [x for x in range(len(vocabulary))]
    histogram = [0] * len(vocabulary)
    histogram_count = {}

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(vocabulary, vocab_labels)
    predicted_categories = knn.predict(descriptors)
    
    # count occurances of each cluster center
    for predicted in predicted_categories:
      if predicted in histogram_count:
        histogram_count[predicted] += 1
      else:
        histogram_count[predicted] = 1
    
    for word in range(len(vocab_labels)):
      if vocab_labels[word] in histogram_count:
        histogram[word] = histogram_count[vocab_labels[word]]/len(vocab_labels)
      
    return histogram



def tinyImages(train_features, test_features, train_labels, test_labels):
    # train_features is a nx1 array of images
    # test_features is a nx1 array of images
    # train_labels is a nx1 array of integers, containing the label values
    # test_labels is a nx1 array of integers, containing the label values
    # label_dict is a 15x1 array of strings, containing the names of the labels
    # classResult is a 18x1 array, containing accuracies and runtimes

    # each image rescale size and number of neighbors for training
    scale_arr = [8, 16, 32]
    neighbor_arr = [1, 3, 6]
    classResult = []
    
    for scale in scale_arr:
      start = timer()
      resized_features = [imresize(x, scale) for x in train_features]
      retest_features = [imresize(x, scale) for x in test_features]
      resize_time = timer() - start
      
      for neighbor in neighbor_arr:
        start = timer()
        predicted_labels = KNN_classifier(resized_features, train_labels, retest_features, neighbor)
        accuracy = reportAccuracy(test_labels, predicted_labels)
        classify_time = timer() - start
        
        classResult.append(accuracy)
        classResult.append(resize_time + classify_time)
    
    return classResult