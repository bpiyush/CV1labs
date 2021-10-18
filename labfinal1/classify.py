"""Main script to perform BoW based classification."""
import time
from os import makedirs
from os.path import join, exists, isdir, dirname
from typing import Any
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from scipy.special import softmax

from stl10_input import read_all_images, read_labels
from constants import relevant_classes, idx_to_class, DATA_DIR, idx_to_class
from utils import show_many_images, mark_kps_on_image, load_pkl, save_pkl


class STL:
    """STL dataset class containing everything about images in the dataset."""
    def __init__(self, data_path: str, label_path: str, seed=0) -> None:

        assert exists(data_path), f"Images' binary file does not exist at {data_path}."
        self.data_path = data_path

        assert exists(label_path), f"Labels' binary file does not exist at {label_path}."
        self.label_path = label_path

        self.seed = seed

        self.data = self._fill_data(data_path, label_path)
    
    def __len__(self):
        return len(self.data["images"])
    
    @staticmethod
    def _load_data(data_path: str, label_path: str):
        all_images = read_all_images(data_path)
        all_labels = read_labels(label_path)

        print(f"::::::::: Loaded dataset with images ({all_images.shape}) and labels ({all_labels.shape}) ::::::::::")

        assert len(all_images) == len(all_labels)
        assert isinstance(all_images, np.ndarray)
        assert isinstance(all_labels, np.ndarray)

        return all_images, all_labels
    
    def _fill_data(self, data_path: str, label_path: str):
        all_images, all_labels = self._load_data(data_path, label_path)
        indices = np.array(list(range(len(all_images))))
        data = {
            "images": all_images,
            "labels": all_labels,
            "indices": indices,
        }
        return data
    
    def _sample_by_attribute(self, attribute, values: list, num_to_sample: int = None):
        np.random.seed(self.seed)
    
        df = pd.DataFrame(None)

        df["indices"] = self.data["indices"]
        df[attribute] = self.data[attribute]

        _filter = df[attribute].isin(values)
        filtered_indices = df[_filter]["indices"].values

        if num_to_sample is None:
            return filtered_indices
        else:
            sampled_indices = np.random.choice(filtered_indices, num_to_sample, replace=False)
            return sampled_indices
    
    def _get_samples_by_indices(self, attribute, indices):
        samples = [self.data[attribute][x] for x in indices if self.data[attribute][x] is not None]
        return samples
    
    def _show_samples(
            self, attribute, indices: np.ndarray,
            suptitle="Sample images from the STL-10 dataset", save=False, save_path=None,
        ):
        assert len(indices) % 2 == 0, "Number of samples to show should be even."
        assert attribute in ["images", "images_with_kps"], "Can only visualize images for now."

        sampled_images = self.data[attribute][indices]
        sampled_labels = self.data["labels"][indices]
        subtitles = [f"Class: {idx_to_class[l].capitalize()}" for l in sampled_labels]

        if save_path is None:
            save_path = "./results/sample_{attribute}.png"
        
        figsize = ((len(indices) // 2) * 4, (len(indices) // 2) * 2)

        show_many_images(
            sampled_images,
            figsize=figsize,
            subtitles=subtitles,
            suptitle=suptitle,
            save=save,
            save_path=save_path,
        )


def show_topk_and_botk_results(dataset, y_scores, y_indices, classes, k=5):
    """Displays top-K and worst-K results based on predicted probability for each class."""
    for j, c in enumerate(classes):
        y_pred_scores = y_scores[:, j]

        sort_indices = np.array(y_indices)[np.argsort(1 - y_pred_scores)]
        topk_indices = sort_indices[:k]
        botk_indices = sort_indices[len(sort_indices) - k:]

        show_indices = list(topk_indices) + list(botk_indices)
        dataset._show_samples(
            attribute="images",
            indices=show_indices,
            suptitle=f"Top-K (top row) and Worst-K (bottom row) predictions for class {idx_to_class[c]}",
        )


def compute_class_wise_ap(y_true, y_pred, y_scores, classes):
    """Computes AveragePrecision for every class and also the meanAP."""
    mAP = 0.0
    class_wise_ap = dict()

    for j, c in enumerate(classes):

        y_class_gt_binary = (y_true == c).astype(int)
        n_samples_in_class = np.sum(y_class_gt_binary)

        y_class_scores = y_scores[:, j]
        indices = np.argsort(-y_class_scores)

        y_true_class = (y_true[indices] == c).astype(int)
        y_true_class_cumsum = np.cumsum(y_true_class)
        ap = np.multiply(y_true_class, y_true_class_cumsum)
        ap = [x / (i + 1) for i, x in enumerate(ap)]
        ap = np.sum(ap) / n_samples_in_class
        class_wise_ap[c] = ap

        mAP += (ap / len(classes))
    class_wise_ap["mean"] = mAP

    return class_wise_ap


class BoWClassifier:
    """Main classifier class that brings data and model together."""
    def __init__(
            self,
            sift_args=dict(),
            seed=0,
            n_clusters=500,
        ):

        # set seed
        self.seed = seed
        self.n_clusters = n_clusters

        # compute image features for training set
        self.SIFT = cv2.SIFT_create(**sift_args)
    
    def extract_features(self, extractor, data_dict: dict):
        images = data_dict["images"]
        
        image_features = dict(kps=[], des=[], images_with_kps=[])
        desc=f"Extracting image features with {type(extractor).__name__}"
        for img in tqdm(images, desc=desc, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            kp, des = extractor.detectAndCompute(img, None)

            image_features["kps"].append(kp)
            image_features["des"].append(des)
            image_features["images_with_kps"].append(mark_kps_on_image(img, kp))
        image_features["images_with_kps"] = np.array(image_features["images_with_kps"])
        
        data_dict.update(image_features)

        return data_dict
    
    def split_train_data(self, dataset, relevant_classes=relevant_classes, num_sampled_per_class=250):
        """
        Splits training dataset into two subsets:
            (1) one for running K-Means clustering
            (2) other for feature encoding for training SVM
        """
        np.random.seed(self.seed)

        indices = dataset.data["indices"]
        kmc_indices = []
        svm_indices = []

        for label in relevant_classes:
            label_indices = dataset._sample_by_attribute(attribute="labels", values=[label])
            sampled_indices = np.random.choice(label_indices, num_sampled_per_class, replace=False)

            kmc_indices += list(sampled_indices)
            svm_indices += list(set(label_indices) - set(sampled_indices))

        return kmc_indices, svm_indices
    
    def cluster_descriptors(
            self,
            dataset,
            clustering_indices,
            n_clusters,
            save=True,
            path=None,
            ignore_cache=False,
        ):
        # get images and their descriptors to be clustered
        clustering_descriptors = dataset._get_samples_by_indices("des", clustering_indices)

        # perform k-means clustering
        X = np.vstack(clustering_descriptors)

        if path is None:
            path = f"./checkpoints/kmeans_{n_clusters}.pkl"
        
        if not ignore_cache and exists(path):
            print(f"::::: Loading pre-saved k-means clustering model from {path}")
            kmeans = load_pkl(path)
        else:
            print(f"::::: Performing k-means clustering on visual descriptors of size {X.shape}")
            start = time.time()
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(X)
            end = time.time()
            print(f"::::: Finished K-Means on visual descriptors of size {X.shape} in {end - start} secs.")

            if save:
                makedirs(dirname(path), exist_ok=True)
                save_pkl(kmeans, path)

        return kmeans, X
    
    def get_image_features(self, dataset, kmeans, indices=None):
        
        cluster_centers = kmeans.cluster_centers_
        n_clusters = cluster_centers.shape[0]
        nearest_ngbr_predictor = NearestNeighbors(n_neighbors=1).fit(cluster_centers)

        if indices is None:
            # use the entire dataset
            indices = list(range(len(dataset)))

        features = np.zeros((len(indices), n_clusters))
        labels = np.zeros(len(indices))
        for j, idx in enumerate(tqdm(indices, desc="Encoding features", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')):
            image_desc = dataset.data["des"][idx]

            if image_desc is not None:
                _, words = nearest_ngbr_predictor.kneighbors(image_desc)

                for w in words:
                    features[j, w] += 1.0 / len(words)
            
            labels[j] = dataset.data["labels"][idx]

        return features, labels
    
    def fit_svm(self, train_features, train_labels):
        svm = SVC(C=1.0, probability=True)
        ovr = OneVsRestClassifier(estimator=svm)
        ovr.fit(train_features, train_labels)
        return ovr
    
    def fit(self, train_data_path, train_label_path, show_steps=False):
        """Fits kMeans and SVM as described in BoW pipeline on the training set."""

        # load training dataset
        self.train_data = STL(train_data_path, train_label_path)

        # show chosen sample images from the dataset
        if show_steps:
            show_indices = []
            for c in relevant_classes:
                class_indices = self.train_data._sample_by_attribute(
                    attribute="labels", values=[c], num_to_sample=2
                )
                show_indices.extend(class_indices)
            self.train_data._show_samples(attribute="images", indices=show_indices, suptitle="")

        # load SIFT features for training dataset (add to dataset object)
        self.train_data.data = self.extract_features(self.SIFT, self.train_data.data)

        # show results of SIFT on the same chosen images
        if show_steps:
            self.train_data._show_samples(attribute="images_with_kps", indices=show_indices, suptitle="")

        # split training data for (a) k-means clustering (b) training SVM
        kmc_indices, svm_indices = self.split_train_data(self.train_data)

        # perform clustering with given indices only
        kmeans, kmc_descriptors = self.cluster_descriptors(
            self.train_data, kmc_indices, n_clusters=self.n_clusters
        )
        self.kmeans = kmeans

        # show TSNE plots of clusters formed
        if show_steps:
            pass

        # compute image features for training SVM
        svm_features, svm_labels = self.get_image_features(self.train_data, kmeans, svm_indices)

        # show per-class histogram (aggregated across all samples in a class)
        if show_steps:
            pass

        # fit SVM model per class (OneVsRest SVM model)
        ovr_svm = self.fit_svm(svm_features, svm_labels)
        self.ovr_svm = ovr_svm

        # evaluate SVM on the training set (how well has it fit?)
        svm_scores = ovr_svm.decision_function(svm_features)
        svm_probes = softmax(svm_scores, axis=1)
        svm_pred_labels = np.argmax(svm_probes, axis=1)

        # convert 0, 1, 2, .., 4 -> 1, 2, 3, 7, 9
        svm_pred_labels = relevant_classes[svm_pred_labels]

        # part 1: qualitative evaluation
        if show_steps:
            show_topk_and_botk_results(self.train_data, svm_scores, svm_indices, relevant_classes, k=5)
        
        # part 2: quantitative evaluation
        class_wise_ap = compute_class_wise_ap(svm_labels, svm_pred_labels, svm_scores, relevant_classes)
        results = pd.DataFrame(class_wise_ap, index=["Average Precision"])
        results = results.rename(columns={k:idx_to_class[k] for k in relevant_classes})
        print("............... SVM Trained with following results on the training set ...............")
        print(f"..... Model: {ovr_svm}")
        print(f"..... Dataset: kMeans: X {(kmc_descriptors.shape)} SVM: X ({svm_features.shape})")
        print(f"..... Hyperparameters: Number of clusters {(self.n_clusters)}")
        print(results.to_markdown())

        print(f"...... Accuracy: {np.mean(svm_labels == svm_pred_labels)}")
    
    def evaluate(self, test_data_path, test_label_path, show_steps=False):
        """Runs evaluation on the test set."""

        # load test dataset
        self.test_data = STL(test_data_path, test_label_path)

        # load SIFT features for test dataset (add to dataset object)
        self.test_data.data = self.extract_features(self.SIFT, self.test_data.data)
        
        # compute image features for test set
        svm_indices = self.test_data._sample_by_attribute(attribute="labels", values=relevant_classes)
        svm_features, svm_labels = self.get_image_features(self.test_data, self.kmeans, svm_indices)

        assert set(np.unique(svm_labels)) == set(relevant_classes)

        # evaluate SVM on the test set
        svm_scores = self.ovr_svm.decision_function(svm_features)
        svm_probes = softmax(svm_scores, axis=1)
        svm_pred_labels = np.argmax(svm_probes, axis=1)

        # convert 0, 1, 2, 3, 4 -> 1, 2, 3, 7, 9
        svm_pred_labels = relevant_classes[svm_pred_labels]

        # part 1: qualitative evaluation
        if show_steps:
            show_topk_and_botk_results(self.test_data, svm_scores, svm_indices, relevant_classes, k=5)
        
        # part 2: quantitative evaluation
        class_wise_ap = compute_class_wise_ap(svm_labels, svm_pred_labels, svm_scores, relevant_classes)
        print(class_wise_ap)
        results = pd.DataFrame(class_wise_ap, index=["Average Precision"])
        results = results.rename(columns={k:idx_to_class[k] for k in relevant_classes})
        print("............... SVM Trained with following results on the test set ...............")
        print(f"..... Model: {self.ovr_svm}")
        print(f"..... Hyperparameters: Number of clusters {(self.n_clusters)}")
        print(results.to_markdown())

        print(f"...... Accuracy: {np.mean(svm_labels == svm_pred_labels)}")

