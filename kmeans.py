'''kmeans.py
Performs K-Means clustering
Logan Marku
CS 251: Data Analysis and Visualization
Spring 2026
'''
import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None

        if data is not None:
            # data: ndarray. shape=(num_samps, num_features)
            self.data = data.copy()
            self.num_samps, self.num_features = data.shape

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        self.data = data.copy()
        self.num_samps, self.num_features = data.shape

    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''
        return self.data.copy()

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE:
        - Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running).
        - Implement the distance formula (see notebook), do not use np.linalg.norm here.
        '''
        difference = pt_1 - pt_2 # difference per pair of points 
        square = difference ** 2 # gets the square
        return np.sqrt(np.sum(square)) # then takes square root for distance

    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE:
        - Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running).
        - Implement the distance formula (see notebook), do not use np.linalg.norm here.
        '''
        difference = centroids - pt # (C, num_features) - (num features,)
        square = difference ** 2
        return np.sqrt(np.sum(square, axis = 1)) # sums to get (C,)
        


    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples and inertia to infinity (i.e. np.inf).

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''
        self.k = k
        randIDX = np.random.choice(self.num_samps, size = k, replace = False) # no replacement here so final replace is false so there is no sampling with replacement
        self.centroids = self.data[randIDX].copy()
        self.inertia = np.inf # so inertia value is really big at start
        return self.centroids


    def cluster(self, k=2, tol=1e-2, max_iter=1000, verbose=False):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the (absolute value of) the absolute difference in the inertia from the
            previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        1. Initialize K-means variables
        2. Do K-means as long as the max number of iterations is not met AND the absolute value of the difference between
        the previous and current inertia is bigger than the tolerance `tol`. K-means should always run for at least 1
        iteration.
        3. Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        4. Print out total number of iterations K-means ran for
        '''
        self.initialize(k)
        diff = np.inf
        iterations = 0

        while iterations < max_iter and np.abs(diff).max() > tol: # creates loop until tolerance is reached
            self.data_centroid_labels = self.update_labels(self.centroids) # assigns point to centroids

            new_centroids, centroid_diff = self.update_centroids( #creation of new centroids
                k, self.data_centroid_labels, self.centroids)
            
            diff = np.abs(centroid_diff)
            self.centroids = new_centroids
            iterations += 1

            if verbose:
                print(f"Iteration {iterations}, max difference is = {diff.max():.3f}")

        self.compute_inertia()
        if verbose:
            print(f"K-Means has finished after {iterations} iterations... the inertia is = {self.inertia():.3f}")
        return self.inertia, iterations



    def cluster_batch(self, k=2, n_iter=1, verbose=False):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.
        '''
        best_inertia = np.inf
        best_centroids = None
        best_labels = None

        for i in range(n_iter):
            inertia, _ = self.cluster(k=k, tol = 1e-2, max_iter = 1000, verbose = False)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = self.centroids.copy()
                best_labels = self.data_centroid_labels.copy()

        self.inertia = best_inertia
        self.centroids = best_centroids
        self.data_centroid_labels = best_labels

        if verbose:
            print(f"The best inertia after {n_iter} runs: {best_inertia:.4f}")
        return self.centroids, self.data_centroid_labels, self.inertia
    

    def update_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,). Holds index of the assigned cluster of each data
            sample. These should be ints (pay attention to/cast your dtypes accordingly).

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''
        diff = self.data[:, None, :] - centroids[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis = 2))

        labels = np.argmin(dists, axis = 1).astype(int)
        return labels

    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values

        NOTE: Your implementation should handle the case when there are no samples assigned to a cluster —
        i.e. `data_centroid_labels` does not have a valid cluster index in it at all.
            For example, if `k`=3 and data_centroid_labels = [0, 1, 0, 0, 1], there are no samples assigned to cluster 2.
        In the case of each cluster without samples assigned to it, you should assign make its centroid a data sample
        randomly selected from the dataset.
        '''
        new_centroids = np.zeros_like(prev_centroids)
        for i in range(k):
            members = self.data[data_centroid_labels == i]
            if members.size == 0:
                randIDX = np.random.randint(0, self.num_samps)
                new_centroids[i] = self.data[randIDX]
            else:
                new_centroids[i] = np.mean(members, axis = 0)
        
        centroid_diff = new_centroids - prev_centroids
        return new_centroids, centroid_diff

    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''
        # sample with assigned cluster centroid 
        assigned = self.centroids[self.data_centroid_labels] 
        sq_distance = np.sum((self.data - assigned) ** 2, axis =1)
        self.inertia = np.mean(sq_distance)
        return self.inertia


    def plot_clusters(self):
        '''Creates a scatter plot of the data color-coded by cluster assignment.

        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). To make sure you change your colors to be clearly differentiable,
        use either the Okabe & Ito or one of the Petroff color palettes: https://github.com/proplot-dev/proplot/issues/424
        Each string in the `colors` list that starts with # is the hexadecimal representation of a color (blue, red, etc.)
        that can be passed into the color `c` keyword argument of plt.plot or plt.scatter.
            Pick one of the palettes with a generous number of colors so that you don't run out if k is large (e.g. >6).
        '''
        colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#999999"]
        k = self.k
        plt.figure(figsize = (7, 6))
        for i in range(k):
            pts = self.data[self.data_centroid_labels == i]
            plt.scatter(
                pts[:,0], pts[:,1],
                c=colors[i % len(colors)], s = 25, label =f"Cluster {i}"
            )

        plt.scatter(
            self.centroids[:,0], self.centroids[:,1],
            c = 'black', s=150, marker = 'x', edgecolors = 'white', linewidths = 1.2, label = "Centroids"
        )

        plt.title(f"K-Means Clustering (k = {k})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(fontsize = 10)
        plt.tight_layout()


    def elbow_plot(self, max_k):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''
        inertias = []
        k_values = range(1, max_k + 1)
        for k in k_values:
            self.cluster(k=k)
            inertias.append(self.inertia)
        
        plt.figure(figsize = (7,5))
        plt.plot(list(k_values), inertias, marker = 'o')
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia (Mean Squared Distance)")
        plt.title("Elbow Plot")
        plt.xticks(list(k_values))
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.show()
        return inertias
            

    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''
        new_color = self.centroids[self.data_centroid_labels]
        self.data = new_color

    def segment_cluster(self, k):
        '''Segments cluster `k` from a COPY of the data. This means setting the entries in all samples NOT assigned to
        cluster `k` to 0.

        Parameters:
        -----------
        k: int.
            The ID of the cluster to segment.

        Returns:
        --------
        ndarray : shape=(num_samples, num_variables).
            COPY of the dataset containing all 0s except for samples that belong to cluster k, which preserve their
            original data.

        Note: Logical indexing can be helpful here.
        '''
        seg_data = self.data.copy()
        seg_data[self.data_centroid_labels != k] = 0
        return seg_data
    
    def zscore_normalize(self):
        '''Z-score normalize self.data column-by-column.

        This is especially important for breast cancer feature data because
        different variables (area, radius, texture, etc.) are on very different scales.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features)
            Normalized version of self.data.
        '''
        means = np.mean(self.data, axis=0)
        stds = np.std(self.data, axis=0, ddof=0)

        # avoid divide-by-zero if any feature is constant
        stds[stds == 0] = 1

        self.data = (self.data - means) / stds
        return self.data


    def assign_new_data(self, new_data):
        '''Assign new samples to the nearest existing centroids.

        Useful after K-means has already been fit.

        Parameters:
        -----------
        new_data: ndarray. shape=(num_new_samps, num_features)

        Returns:
        -----------
        ndarray of ints. shape=(num_new_samps,)
            Cluster assignment for each new sample.
        '''
        diff = new_data[:, None, :] - self.centroids[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        return np.argmin(dists, axis=1).astype(int)


    def cluster_counts(self):
        '''Count how many samples are in each cluster.

        Returns:
        -----------
        ndarray. shape=(self.k,)
            Number of samples assigned to each cluster.
        '''
        counts = np.zeros(self.k, dtype=int)
        for i in range(self.k):
            counts[i] = np.sum(self.data_centroid_labels == i)
        return counts


    def cluster_label_summary(self, true_labels):
        '''Summarize how external class labels are distributed inside each cluster.

        This is useful for comparing unsupervised K-means clusters against
        known breast cancer diagnosis labels such as:
            0 = benign
            1 = malignant

        Parameters:
        -----------
        true_labels: ndarray of ints. shape=(num_samps,)

        Returns:
        -----------
        summary: dict
            Dictionary where each key is a cluster index and each value is
            another dictionary containing:
                'total'
                'class_0'
                'class_1'
                'majority_class'
                'purity'
        '''
        summary = {}

        for i in range(self.k):
            members = (self.data_centroid_labels == i)
            cluster_y = true_labels[members]

            total = cluster_y.size
            class_0 = np.sum(cluster_y == 0)
            class_1 = np.sum(cluster_y == 1)

            if total == 0:
                majority_class = None
                purity = 0.0
            else:
                majority_class = 0 if class_0 >= class_1 else 1
                purity = max(class_0, class_1) / total

            summary[i] = {
                'total': int(total),
                'class_0': int(class_0),
                'class_1': int(class_1),
                'majority_class': majority_class,
                'purity': purity
            }

        return summary


    def majority_label_map(self, true_labels):
        '''Map each cluster to the majority true label inside that cluster.

        Parameters:
        -----------
        true_labels: ndarray of ints. shape=(num_samps,)

        Returns:
        -----------
        ndarray of ints. shape=(self.k,)
            For each cluster, the majority class label.
        '''
        label_map = np.zeros(self.k, dtype=int)

        for i in range(self.k):
            members = true_labels[self.data_centroid_labels == i]

            if members.size == 0:
                label_map[i] = 0
            else:
                count_0 = np.sum(members == 0)
                count_1 = np.sum(members == 1)
                label_map[i] = 0 if count_0 >= count_1 else 1

        return label_map


    def predict_from_clusters(self, true_labels):
        '''Turn cluster assignments into simple class predictions by majority vote.

        This does NOT make K-means a true supervised classifier, but it gives
        a simple way to compare clusters against known labels.

        Parameters:
        -----------
        true_labels: ndarray of ints. shape=(num_samps,)

        Returns:
        -----------
        ndarray of ints. shape=(num_samps,)
            Predicted class for each sample based on its cluster's majority label.
        '''
        label_map = self.majority_label_map(true_labels)
        return label_map[self.data_centroid_labels]


    def simple_accuracy(self, true_labels):
        '''Compute simple accuracy using majority-vote cluster labels.

        Parameters:
        -----------
        true_labels: ndarray of ints. shape=(num_samps,)

        Returns:
        -----------
        float
            Fraction of correctly matched labels.
        '''
        preds = self.predict_from_clusters(true_labels)
        return np.mean(preds == true_labels)


    def confusion_matrix_binary(self, true_labels):
        '''Compute a 2x2 confusion matrix for binary labels using majority-vote clusters.

        Assumes labels are:
            0 = benign / low-risk
            1 = malignant / high-risk

        Returns:
        -----------
        ndarray. shape=(2, 2)
            [[TN, FP],
             [FN, TP]]
        '''
        preds = self.predict_from_clusters(true_labels)

        tn = np.sum((true_labels == 0) & (preds == 0))
        fp = np.sum((true_labels == 0) & (preds == 1))
        fn = np.sum((true_labels == 1) & (preds == 0))
        tp = np.sum((true_labels == 1) & (preds == 1))

        return np.array([[tn, fp],
                         [fn, tp]])


    def plot_feature_clusters(self, x_index, y_index, x_label='Feature 1', y_label='Feature 2'):
        '''Scatter plot of two selected features colored by K-means cluster.

        Parameters:
        -----------
        x_index: int
            Column index of feature on x-axis
        y_index: int
            Column index of feature on y-axis
        x_label: str
        y_label: str
        '''
        colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
                  "#CC79A7", "#56B4E9", "#F0E442", "#999999"]

        plt.figure(figsize=(7, 6))

        for i in range(self.k):
            pts = self.data[self.data_centroid_labels == i]
            plt.scatter(
                pts[:, x_index], pts[:, y_index],
                c=colors[i % len(colors)],
                s=25,
                label=f"Cluster {i}",
                alpha=0.8
            )

        plt.scatter(
            self.centroids[:, x_index], self.centroids[:, y_index],
            c='black', s=150, marker='x', linewidths=2, label='Centroids'
        )

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"K-Means Clusters: {x_label} vs {y_label}")
        plt.legend()
        plt.tight_layout()
        plt.show()


    def plot_cluster_label_bars(self, true_labels, class_names=('Class 0', 'Class 1')):
        '''Bar plot showing class composition inside each cluster.

        Parameters:
        -----------
        true_labels: ndarray of ints. shape=(num_samps,)
        class_names: tuple of str
            Names for class 0 and class 1
        '''
        class_0_counts = []
        class_1_counts = []

        for i in range(self.k):
            members = (self.data_centroid_labels == i)
            cluster_y = true_labels[members]
            class_0_counts.append(np.sum(cluster_y == 0))
            class_1_counts.append(np.sum(cluster_y == 1))

        x = np.arange(self.k)

        plt.figure(figsize=(8, 5))
        plt.bar(x, class_0_counts, label=class_names[0])
        plt.bar(x, class_1_counts, bottom=class_0_counts, label=class_names[1])

        plt.xlabel("Cluster")
        plt.ylabel("Number of Samples")
        plt.title("Class Composition Within Each Cluster")
        plt.xticks(x)
        plt.legend()
        plt.tight_layout()
        plt.show()