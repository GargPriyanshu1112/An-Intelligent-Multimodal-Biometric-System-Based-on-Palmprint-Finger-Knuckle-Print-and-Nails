
* Pre-processing of input images (palmar and dorsal hand image) using contrast enhancement and normalization.
* Region of Interest (ROI) segmentation (palmprint from palmar images and FKPs & nails from dorsal images).
* Feature extraction and similarity estimation using a fine-tuned DenseNet201.
* Fusion of similarity estimation results of each modality to obtain the final probability distribution based on which the final id is predicted.
* The model achieved an accuracy of 97.53%
