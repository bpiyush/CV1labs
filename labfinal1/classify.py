"""Main script to perform BoW based classification."""
import time
from os.path import join, exists, isdir
from typing import Any
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import average_precision_score as AP
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from stl10_input import read_single_image, read_all_images, read_labels, plot_image, keep_relevant_images
from constants import relevant_classes, idx_to_class, DATA_DIR
from utils import show_many_images, show_single_image, mark_kps_on_image


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
            sampled_indices = np.random.choice(filtered_indices, num_to_sample)
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



def compute_map(y_true, y_pred_proba, relevant_classes):
    mAP = 0.0
    per_class_AP = dict()

    for j, i in enumerate(relevant_classes):
        class_ap = AP((y_true == i).astype(int), y_pred_proba[:, j])
        mAP += class_ap / len(relevant_classes)
        per_class_AP[idx_to_class[i]] = class_ap
    
    per_class_AP["average"] = mAP

    return per_class_AP


class BoWClassifier:
    """Main classifier class that brings data and model together."""
    def __init__(self, data_path, label_path, sift_args=dict(), seed=0) -> None:
        
        # set seed
        self.seed = seed

        # load dataset
        self.train_data = STL(data_path, label_path)

        # compute image features for training set
        self.SIFT = cv2.SIFT_create(**sift_args)
        self.train_data.data = self.extract_features(self.SIFT, self.train_data.data)
    
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
            (2) other for feature encoding.
        """
        np.random.seed(self.seed)

        indices = dataset.data["indices"]
        clustering_indices = []
        ftencoding_indices = []

        for label in relevant_classes:
            label_indices = dataset._sample_by_attribute(attribute="labels", values=[label])
            sampled_indices = np.random.choice(label_indices, num_sampled_per_class)

            clustering_indices += list(sampled_indices)
            ftencoding_indices += list(set(label_indices) - set(sampled_indices))

        return clustering_indices, ftencoding_indices
    
    def cluster_descriptors(self, dataset, clustering_indices, n_clusters=500):
        # get images and their descriptors to be clustered
        clustering_images = dataset._get_samples_by_indices("images", clustering_indices)
        clustering_labels = dataset._get_samples_by_indices("labels", clustering_indices)
        clustering_descriptors = dataset._get_samples_by_indices("des", clustering_indices)

        # perform k-means clustering
        X = np.vstack(clustering_descriptors)

        start = time.time()
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(X)
        end = time.time()
        print(f"::::: Finished K-Means on visual descriptors of size {X.shape} in {end - start} secs.")

        return kmeans, X
    
    def get_image_features(self, dataset, kmeans, indices=None):
        
        cluster_centers = kmeans.cluster_centers_
        num_clusters = cluster_centers.shape[0]
        nearest_ngbr_predictor = NearestNeighbors(n_neighbors=1).fit(cluster_centers)

        if indices is None:
            # use the entire dataset
            indices = list(range(len(dataset)))

        features = np.zeros((len(indices), num_clusters))
        for j, i in tqdm(enumerate(indices), desc="Encoding features"):
            image_desc = dataset.data["des"][i]

            if image_desc is not None:
                _, words = nearest_ngbr_predictor.kneighbors(image_desc)

                for w in words:
                    features[j, w] += 1.0 / len(words)

        return features
    
    def classify(self, train_features, train_labels):
        svm = SVC(C=1.0, probability=True)
        ovr = OneVsRestClassifier(estimator=svm)
        ovr.fit(train_features, train_labels)
        return ovr
    
    def predict(self, model, image_features):
        y_pred = model.predict(image_features)
        y_pred_proba = model.predict_proba(image_features)

        ohe = OneHotEncoder()
        y_pred_onehot = ohe.fit_transform(y_pred.reshape((-1, 1))).toarray()

        return y_pred, y_pred_onehot, y_pred_proba
    
    def evaluate(self, y_true, y_pred, y_pred_onehot, y_pred_proba):
        per_class_AP = compute_map(y_true=y_true, y_pred_proba=y_pred_proba)
        accuracy = accuracy_score(y_true, y_pred)

        metrics = {
            "accuracy": accuracy,
            "per_class_average_precision": per_class_AP,
        }

        return metrics
