# Breast cancer prediction using Support Vector Machine (SVM) #

The input data set consists of 30 features of breast cancer like radius_mean, perimeter_mean and texture_mean including the primary diagnosis whether the cancer is malignant or benign.

Data density plot to get the data distribution patten

![[alt text]](https://github.com/mariyamouseph/breast_cancer_prediction/blob/master/plots/density_plot.png)


Graphical representation attribute correlation and the diagonal suggests that the attributes are correlated with each other
 
![[alt text]](https://github.com/mariyamouseph/breast_cancer_prediction/blob/master/plots/coorelation_plot.png)

 
 Initial performance plot for SVM algorithm
 
 ![[alt text]](https://github.com/mariyamouseph/breast_cancer_prediction/blob/master/plots/performace_initial.png)
 
 Performance plot for SVM after standardizing the dataset
 
 ![[alt text]](https://github.com/mariyamouseph/breast_cancer_prediction/blob/master/plots/performace_final.png)

 
 On standardizing dataset and tuning the algorithm an accuracy of **99.12%** is achieved.
 
 
 Note: For testing the accuracy the data is split in the ration 1:5, thereby 80% of data is trained and 20% of data is validated.
 
 