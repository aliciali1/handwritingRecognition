import math
import random
from collections import defaultdict
import heapq


# returns Euclidean distance between vectors and b
def euclidean(a,b):
    squared_diffs = 0
    for i in range(len(a)):
        squared_diffs += (a[i] - b[i]) ** 2
    
    dist = math.sqrt(squared_diffs)
    return(dist)
        
# returns Cosine Similarity between vectors and b
def cosim(a,b):
    dot_product = 0
    a_mag = 0
    b_mag = 0
    
    for i in range (len(a)):
        dot_product += a[i] * b[i] 
        a_mag += math.pow(a[i], 2)
        b_mag += math. pow(b[i],2)
    a_mag = math.sqrt(a_mag)
    b_mag = math.sqrt(b_mag)
    if a_mag == 0 or b_mag == 0:
        return 0.0
    dist = dot_product/(a_mag*b_mag)
    return(dist)

# returns Hamming distance between vectors and b
def hamming(a,b):
    dist = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            dist += 1
    return(dist)

# returns Pearson Correkation between vectors and b
def pearson(a,b):
    n = len(a)
    a_sum = sum(a)
    b_sum = sum(b)
    a_sum_squared = sum(x ** 2 for x in a)
    b_sum_squared = sum(y ** 2 for y in b)
    sum_ab = sum(x*y for x, y in zip(a,b))
    numerator = (n * sum_ab) - (a_sum * b_sum)
    denominator = ((n * a_sum_squared - a_sum ** 2) * (n * b_sum_squared - b_sum ** 2)) ** 0.5      
    if denominator == 0:
        return 0.0
    dist = numerator/denominator

    return(dist)

# returns a list of labels for the query dataset based upon labeled observations in the dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(data,query,metric):
    # data = [list(d) for d in data]
    # for i in range(len(data)):
    #     # data[i][1] = [float(x) for x in data[i][1]]
    #     # data[i][0][0] = int(data[i][0][0])
    for i in range(len(query)):
        query[i] = [float(x) for x in query[i]]

    k = 5  
    labels = []
    # print(query,'query')
    for query_point in query:
        # print(query_point,'query_point')
        if metric == "euclidean":
            distances = [(euclidean(query_point, attrib), label[0]) for label, attrib in data]
            distances.sort(key=lambda x: x[0])  
        elif metric == "cosim":
            similarities = [(cosim(query_point, attrib), label[0]) for label, attrib in data]
            distances = sorted(similarities, key=lambda x: x[0], reverse=True)  
        
        distance_list = distances[:k]
        top_k_labels = [label for _, label in distance_list]
       # print(top_k_labels,'top_k_labels')

        label_count = {}
        for label in top_k_labels:
            label_count[label] = label_count.get(label, 0) + 1
        most_common_label = max(label_count, key=label_count.get)

        labels.append(most_common_label)
    return labels

#movie_rating_knn - reformats movie data so it can be r
#def movie_rating_knn(data, query):

#data = user_data
#query = one element of user_data
#M = number of movies 



#Converts movie rating data to a binary format, normalizing ratings between 0 and 1
def convert_binary(data):
    # Convert all values in the matrix to floats and create a new matrix
    for i in range(len(data)):
        data[i][0] = [int(data[i][0])]
        data[i][1] = [float(x) for x in data[i][1]]

        minVal = None
        maxVal = None

        for j in range(len(data[i][1])):
            if minVal is None or data[i][1][j] < minVal:
                minVal = data[i][1][j]
            if maxVal is None or data[i][1][j] > maxVal:
                maxVal = data[i][1][j]
        for j in range(len(data[i][1])):
            data[i][1][j] = (data[i][1][j] - minVal) / (maxVal - minVal)
  
    return data 

#Calculates the mean of a list of points, where each point is represented as a list of dimensions.
def calculate_mean(points):
    num_points = len(points)
    num_dimensions = len(points[0])
    mean = [0] * num_dimensions  #initialize a list for the mean of each dimension

    #sum each dimension separately across all points
    for point in points:
        for i in range(num_dimensions):
            mean[i] += point[i]

    #divide by the number of points to get the average (mean) position
    for i in range(num_dimensions):
        mean[i] /= num_points
    return mean


#Randomly selects k data points from the dataset to serve as initial centroids for clustering
def initialize_centroids(data, k):
    random_indices = random.sample(range(len(data)), k)
    centroids = [data[i] for i in random_indices]       #creates list of indices
    return centroids


#Assigns each data point to the nearest centroid based on the specified distance metric
def assign_clusters(data, centroids, metric):
    clusters = [[] for _ in range(len(centroids))]
    for idx, point in enumerate(data):
        if metric == "euclidean":
            distances = [euclidean(point, centroid) for centroid in centroids]
        elif metric == "cosim":
            distances = [cosim(point, centroid) for centroid in centroids]
        else:
            raise ValueError("Unsupported metric. Use 'euclidean' or 'cosim'.")
        
        nearest_centroid_index = distances.index(min(distances))
        clusters[nearest_centroid_index].append(idx)  # Store indices
    return clusters


#Calculates new centroids as the mean of the data points assigned to each cluster
def update_centroids(clusters, data):
    centroids = []
    for cluster in clusters:        #clusters is a list of lists - each internal list stores all the indices in that cluster
        if len(cluster) > 0:
            num_dimensions = len(data[0])
            new_centroid = [0] * num_dimensions
            for idx in cluster:
                point = data[idx]
                for i in range(num_dimensions):
                    new_centroid[i] += point[i]
            num_points = len(cluster)
            for i in range(num_dimensions):
                new_centroid[i] /= num_points           #takes average of all the points in cluster
        else:
            new_centroid = [0] * len(data[0])  
        centroids.append(new_centroid)
    return centroids                                #the new centroids are all the mean of the previous centroi


#Performs K-Means clustering on the provided data and predicts labels for the given query points.
def kmeans(data, data_labels, query, metric, k=10, max_iterations=100):
    centroids = initialize_centroids(data, k)
    
    for _ in range(max_iterations):
        clusters = assign_clusters(data, centroids, metric)
        
        new_centroids = update_centroids(clusters, data)
        
        if centroids == new_centroids:
            break
        
        centroids = new_centroids
    
    cluster_label_map = {}
    for cluster_idx, cluster in enumerate(clusters):
        if len(cluster) == 0:
            continue
        labels_in_cluster = [data_labels[idx] for idx in cluster]
        label_counts = {}
        for label in labels_in_cluster:
            label_counts[label] = label_counts.get(label, 0) + 1
        most_common_label = max(label_counts, key=label_counts.get)
        cluster_label_map[cluster_idx] = most_common_label
    
    labels = []
    for query_point in query:
        if metric == "euclidean":
            distances = [euclidean(query_point, centroid) for centroid in centroids]
        elif metric == "cosim":
            distances = [cosim(query_point, centroid) for centroid in centroids]
        else:
            raise ValueError("Unsupported metric. Use 'euclidean' or 'cosim'.")
        
        nearest_centroid_index = distances.index(min(distances))
        predicted_label = cluster_label_map.get(nearest_centroid_index, -1)  # Use -1 if cluster not found
        labels.append(predicted_label)
    
    return labels



#Creates a confusion matrix based on actual and predicted labels
def create_confusion_matrix(actual_labels, pred_labels):
    #print(actual_labels, "actual")
    #print(pred_labels, "pred")
    confusion_matrix = [[0 for i in range(62)] for j in range(62)]
    for actual, pred in zip(actual_labels, pred_labels):
        confusion_matrix[actual][pred] += 1
    return confusion_matrix


#Function to display the confusion matrix
def print_confusion_matrix(matrix):
    for row in matrix:
        print("\t".join(map(str, row)))





#Helper function to calculate precision
def precision(tp, fp):
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

#Helper function to calculate recall
def recall(tp, fn):
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

#Helper function to calculate f1 score
def f1_score(precision_score, recall_score):
    if precision_score + recall_score == 0:
        return 0.0
    return 2 * (precision_score * recall_score) / (precision_score + recall_score)

    
    
# reads CSV file
def read_data(file_name):
    count =  0
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            if count == 1000:
                break
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
            count += 1
    return(data_set)
        
# def show(file_name,mode):
    
#     data_set = read_data(file_name)
#     for obs in range(len(data_set)):
#         for idx in range(784):
#             if mode == 'pixels':
#                 if data_set[obs][1][idx] == '0':
#                     print(' ',end='')
#                 else:
#                     print('*',end='')
#             else:
#                 print('%4s ' % data_set[obs][1][idx],end='')
#             if (idx % 28) == 27:
#                 print(' ')
#         print('LABEL: %s' % data_set[obs][0],end='')
#         print(' ')



# Tests and runs the KNN function on different data sets
def evaluate_knn():
    # Read training, validation, and test data
    train_data = read_data('emnist-byclass-train.csv')
    test_data = read_data('emnist-byclass-test.csv')
    data_labels = [label[0] for label in test_data]

    train_data_binary = convert_binary(train_data)
    test_data_binary = convert_binary(test_data)

    # Split test data into features and labels
    test_features = [features for _, features in test_data_binary]
    test_labels = [label[0] for label, _ in test_data_binary]
    
    # # Evaluate on validation set
    # print("Evaluating on validation set...")
    # predictions_euclidean = knn(train_data_binary, test_features, metric="euclidean")
    # predictions_cosim = knn(train_data_binary, test_features, metric="cosim")
    # print("Validation set evaluation completed.")

    # Run k-NN classifier on the test set
    print("Evaluating on test set using Euclidean distance...")
    test_predictions_euclidean = knn(train_data_binary, test_features, metric="euclidean")
    # print("Evaluating on test set using Cosine similarity...")
    # test_predictions_cosim = knn(train_data_binary, test_features, metric="cosim")

    # Create confusion matrices for the test set
    confusion_matrix_euclidean = create_confusion_matrix(test_labels, test_predictions_euclidean)
    total = 0
    correct = 0 
    for r in range(len(confusion_matrix_euclidean)):
        for c in range(len(confusion_matrix_euclidean[0])):
            total += confusion_matrix_euclidean[r][c]
            if r == c:
                correct += confusion_matrix_euclidean[r][c]
    accuracy = correct / total
    
    
    # confusion_matrix_cosim = create_confusion_matrix(test_labels, test_predictions_cosim)

    # Print the confusion matrices
    print("\nConfusion Matrix (Euclidean Distance):")
    print_confusion_matrix(confusion_matrix_euclidean)
    print(f"accuracy: {accuracy}\n total tested: {total}")
    # print("\nConfusion Matrix (Cosine Similarity):")
    # print_confusion_matrix(confusion_matrix_cosim)
    
    
def main():
    print("KNN EVALUATION: ")
    evaluate_knn()    
    

    
if __name__ == "__main__":
    main()

