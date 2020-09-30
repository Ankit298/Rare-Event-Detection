# Rare Event Detection

## Introduction:

Most of the datasets for machine learning tasks in the clinical domain are highly imbalanced. In this project, we implement different ways to tackle this problem. Our study was based on two different clinical applications where class imbalance naturally occurs: schizophrenia relapse prediction based on mobile sensing data (using LSTM models) and dermatofibroma lesion detection based on skin images (using CNN models). We evaluated different data resampling approaches and customized loss functions in order to improve model performance when classes are imbalanced.

## Data:

### Dermatafibroma Lesion detection

We use the [HAM1000 Dataset](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6091241/) for the task of Dermatofibroma detection. The dataset contains a large collection of multi-source dermatoscopic images of pigmented lesions. It contains 10015 dermatoscopic images from 7 categories of pigmented lesions. Only 115 of these images are from the dermatofibroma class and rest of the images will be classified as non-dermatofibroma class. Therefore the class imbalance is
high ( 1:100).

### Schizophrenia relapse prediction

The dataset consists of mobile sensing data from 63 schizophrenia patients monitored for about 12 months period. For each of the patients, the dataset consists of various sensor recordings such as accelerometer, activities, audio inferences, call logs, lights, locations collected from their mobile phone. The ground truth about the relapse information is obtained from the clinical notes prepared by medical professional in charge of the patient. The ratio of weeks with relapse to the weeks
without relapse is very low ( 1:500).

## Resampling Techniques:

In data resampling techniques, the underlying data is re-sampled before being made  input to the model for training such that gradients from training examples of both the classes dictate the weight adaptation of deep learning models equally. The two main sampling techniques included where:

### Oversampling

As the name suggests, in oversampling the minority class is oversampled to match
the representation from the majority class. At the end, the training dataset size increases. Although this sampling technique brings back the blance in our dataset, it might not be feasible when the dataset is already large.

### Undersampling

The majority class is sub-sampled to match the representation of minority class and the overall training dataset decreases. Useful when the dataset is already large enough.

## Loss Functions

In our evaluations, we consider the following loss functions:

* Binary Cross Entropy: a standard loss function used in binary clas-
sification tasks. This loss function is defined as: − (ytrue ∗ log(ypred) + (1 − ytrue) ∗ log(1 − ypred)).

* Weighted Binary Cross Entropy: Binary cross entropy can be made sensitive to the class imbalance present by weighing the loss contribution from two classes differently. This can be given by: − (ytrue ∗ log(ypred) ∗ β + (1 − ytrue) ∗ log(1 − ypred)). Here the weight β controls the loss contribution ratio
from the two classes.

* Weighted Loss Function:  The weighted loss function is given by: α ∗ (ytrue − ypred) + β ∗ (ytrue − ypred)<sup>2</sup>. The idea behind the weighted loss function is to penalize false negatives. It is the weighted average of the difference between the ground truth label and the predicted value as well as the absolute difference.

* Focal Loss:  The basic idea is to give higher weights to training examples which are misclassified during an iteration of training. This loss function should prioritize, adaptively ,the samples of minority classes as these samples, being under-represented, are likely to be misclassified during the early phases of training. The focal loss is given by: −((ypred)γ ∗log(1−ypred)+(1−ypred)γ ∗log(ypred).

* F2 Loss:  As the target metric of interest to us is the F2 metric, it might be better to motivate a loss function that is related to this evaluation metric.  The resulting formulation is then given as: 1 − (5∗precision∗recall)/4∗precision+recall.








