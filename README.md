# Website classification

Infers the domain name from website screenshots

## How to use

- Clone the repository
- Install requirements.txt
- Run `infer.py`
- Choose strategy to infer with `(SGD, SVM)`
- Insert file path

---

## About

This implementation was trained on the 10 following domains: `'The Guardian', 'Spiegel', 'CNN', 'BBC', 'Amazon', 'Ebay', 'Njuskalo(hr)', 'Google', 'Github', 'Youtube'`

- ## Data collection

  The method is explained in detail [here](https://docs.google.com/presentation/d/1UOnSXGmsaVgv6lchmyeCrehXTzLHveAvsHvIsks-Zrw/edit?usp=sharing)

- ## Data preperation

  - Scraped 2500+ screenshots from 10 given websites using web scrapers [Selenium](https://www.selenium.dev/) (Around 250 samples per website). Samples different parts of the websites. Please see the link above for data collection method
  - Extracted image features using [HOG](https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html#sphx-glr-auto-examples-features-detection-plot-hog-py), after resizing the images.
  - Saved into CSV format for model training.

  Data preparation notebook can be inspected [here](https://github.com/hakanErgin/website-classification/blob/main/notebooks/dataset_creation.ipynb)

  Sample screenshots can be inspected in [assets folder](assets)

- ## Model training

  Implemented 2 different strategies

  1. [SGD(SVM)](https://scikit-learn.org/stable/modules/sgd.html) + [RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html)
  2. [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) + [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

     Both includes:

  - Splitting the data into test and train. Test dataset is kept for model evaluation.
  - Dataset scaling (test and train sets were scaled seperately for preventing data leaks).
  - Reduced dimension (n_features) by unsupervised methods (`RFECV` & `PCA`).
  - Training classification models while finding the best hyperparameters by GridSearch Cross-validation methods.
  - Scalers, dimension reducers and classifiers were serialized as a pipeline for later use.
  - Evaluating model performances with `classification reports` and `confusion matrixes`.

  Model training notebook with `SGD + RFECV` can be inspected [here](https://github.com/hakanErgin/website-classification/blob/main/notebooks/model_training/sgd_rfecv_model_training.ipynb)

  Model training notebook with `SVM + PCA` can be inspected [here](https://github.com/hakanErgin/website-classification/blob/main/notebooks/model_training/svm_pca_model_training.ipynb)

- ## Inference

  Input images are not expected to have scrollbar. There is a scrollbar removal function in the notebooks.

  Inference notebook can be inspected [here](https://github.com/hakanErgin/website-classification/blob/main/notebooks/model_inference.ipynb)

  The program works on `infer.py` and `infer_functions.py`
