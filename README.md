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

Data collection method is explained in detail [here](https://docs.google.com/presentation/d/1UOnSXGmsaVgv6lchmyeCrehXTzLHveAvsHvIsks-Zrw/edit?usp=sharing)

- ## Data preperation

  - Scraped 3000 screenshots from 10 given websites using web scrapers (Around 300 samples per website). Samples include a screenshot with `cookies bar, top part, randomly scrolled down and the bottom part of the website`.
  - Extracted image features using [(HOG)](https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html#sphx-glr-auto-examples-features-detection-plot-hog-py), after resizing the images.
  - Saved into CSV format for model training.

  Data preparation notebook can be found [here](notebooks\dataset_creation.ipynb)

  Sample screenshots can be found in [assets folder](assets)

- ## Model training

  Implemented 2 different strategies (1. [SGD(SVM)](https://scikit-learn.org/stable/modules/sgd.html) + [RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html) and 2. [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) + [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)), both by:

  - Splitting the data into test and train. Test dataset is kept for model evaluation.
  - Scaled the dataset to `0 mean` and `unit variance`.
  - Reduced dimension (n_features) by unsupervised methods.
  - Training classification models while finding the best hyperparameters by GridSearch Cross-validation methods.
  - Scalers, dimension reducers and classifiers were serialized as a pipeline for later use.

  Model training notebook with SGD + RFECV can be found [here](notebooks\model_training\sgd_rfecv_model_training.ipynb)

  Model training notebook with SVM + PCA can be found [here](notebooks\model_training\svm_pca_model_training.ipynb)

- ## Inference

  Input images are not expected to have scrollbar. There is a scrollbar removal function in the notebooks.

  Inference notebook can be found [here](notebooks\model_inference.ipynb)

  The program works on `infer.py` and `infer_functions.py`
