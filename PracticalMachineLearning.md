-   [Summary](#summary)
-   [Initiate Libraries](#initiate-libraries)
-   [Data](#data)
-   [Data Clean Up](#data-clean-up)
-   [Data Partition](#data-partition)
-   [Model Training and Selection](#model-training-and-selection)
    -   [Confusion Matrix for Recursive Partition](#confusion-matrix-for-recursive-partition)
    -   [Confusion Matrix for Random Forest](#confusion-matrix-for-random-forest)
-   [Predict Activitiy Using Test Data](#predict-activitiy-using-test-data)

Summary
=======

The purpose of this study is to identify the parameters of a set of motions that can be used to identity a particular activity.

The data was provided by <http://groupware.les.inf.puc-rio.br/har>

Initiate Libraries
==================

``` r
#initialize library

library(caret,quietly=TRUE,warn.conflicts=FALSE)
require(caret,quietly=TRUE,warn.conflicts=FALSE)

library(AppliedPredictiveModeling,quietly=TRUE,warn.conflicts=FALSE)
require(AppliedPredictiveModeling,quietly=TRUE,warn.conflicts=FALSE)

library(ElemStatLearn,quietly=TRUE,warn.conflicts=FALSE)
require(ElemStatLearn,quietly=TRUE,warn.conflicts=FALSE)

library(pgmm,quietly=TRUE,warn.conflicts=FALSE)
require(pgmm,quietly=TRUE,warn.conflicts=FALSE)

library(rpart,quietly=TRUE,warn.conflicts=FALSE)
require(rpart,quietly=TRUE,warn.conflicts=FALSE)

library(gbm,quietly=TRUE,warn.conflicts=FALSE)
require(gbm,quietly=TRUE,warn.conflicts=FALSE)

library(lubridate,quietly=TRUE,warn.conflicts=FALSE)
require(lubridate,quietly=TRUE,warn.conflicts=FALSE)

library(forecast,quietly=TRUE,warn.conflicts=FALSE)
require(forecast,quietly=TRUE,warn.conflicts=FALSE)

library(e1071,quietly=TRUE,warn.conflicts=FALSE)
require(e1071,quietly=TRUE,warn.conflicts=FALSE)

library(doBy,quietly=TRUE,warn.conflicts=FALSE)
require(doBy,quietly=TRUE,warn.conflicts=FALSE)
```

Data
====

The data was sourced from the following locations

-   Training Data

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

-   Test Data

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

It was uploaded into the project as follows:

``` r
training_data <- read.csv("pml-training.csv")
test_data     <- read.csv("pml-testing.csv")
```

Data Clean Up
=============

To facilitate a predictive model, the data in the training and testing data sets must be common.

To ensure the above condition is met, fields that are not common in both the training set and the test set as DROPPED!!

``` r
drops <- colnames(test_data[, ( colSums(is.na(test_data)) / nrow(test_data) ) > 0.8 ] )
training_data2 <- training_data[, !(names(training_data) %in%  drops     )]

training_data2 <- training_data2[, !( names(training_data2) ==   colnames(test_data[1]))  ]
training_data2 <- training_data2[, !( names(training_data2) ==   colnames(test_data[3:6]))  ]
```

    ## Warning in names(training_data2) == colnames(test_data[3:6]): longer object
    ## length is not a multiple of shorter object length

If fields are highly correlated in the data set, there is a redundancy in having both fields present. Additionally, this redundancy may have the adverse effect of overfitting our model.

We remove correlated values (cor &gt; 84% ), as follows

``` r
num_fld <- sapply( training_data2 ,  is.numeric )
training_data3 <- training_data2[ , -findCorrelation(   cor(training_data2[,num_fld])  , cutoff=.9  ) ]
```

Data Partition
==============

We partition the training set from approve to 75% training, and 25% testing.

``` r
set.seed(1024)
inTrain = createDataPartition(training_data3$classe , p = 3/4)[[1]]
training = training_data3[ inTrain,]
testing = training_data3[-inTrain,]
```

Model Training and Selection
============================

We train our model using

-   Random Forest
-   Recursive Partition

``` r
set.seed(12345)
# Recursive Partition Tress
modelFitRP <- train( classe ~ ., method = "rpart", data = training )
#Random Forest
modelFitRF <- train( classe ~ ., method = "rf", data = training )
```

    ## Loading required package: randomForest

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

We then test our model based on the data we partitioned

``` r
predRP <- predict( modelFitRP , testing )
predRF <- predict( modelFitRF , testing )
```

Confusion Matrix for Recursive Partition
----------------------------------------

``` r
confusionMatrix( testing$classe , predRP  )
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1143    5   58  186    3
    ##          B  159  474  165  116   35
    ##          C   23   31  682  116    3
    ##          D   50   36  308  342   68
    ##          E   27   24  189   63  598
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.6605         
    ##                  95% CI : (0.647, 0.6737)
    ##     No Information Rate : 0.2859         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.5715         
    ##  Mcnemar's Test P-Value : < 2.2e-16      
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8153  0.83158   0.4864  0.41555   0.8458
    ## Specificity            0.9280  0.89040   0.9506  0.88679   0.9278
    ## Pos Pred Value         0.8194  0.49947   0.7977  0.42537   0.6637
    ## Neg Pred Value         0.9262  0.97573   0.8222  0.88268   0.9728
    ## Prevalence             0.2859  0.11623   0.2859  0.16782   0.1442
    ## Detection Rate         0.2331  0.09666   0.1391  0.06974   0.1219
    ## Detection Prevalence   0.2845  0.19352   0.1743  0.16395   0.1837
    ## Balanced Accuracy      0.8717  0.86099   0.7185  0.65117   0.8868

The recursive partition performs poorly. This is based off the Sensisitivity Values and the confusion matrix.

Confusion Matrix for Random Forest
----------------------------------

``` r
confusionMatrix( testing$classe , predRF  )
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1395    0    0    0    0
    ##          B    1  948    0    0    0
    ##          C    0    3  852    0    0
    ##          D    0    0    0  803    1
    ##          E    0    0    0    0  901
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.999           
    ##                  95% CI : (0.9976, 0.9997)
    ##     No Information Rate : 0.2847          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9987          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9993   0.9968   1.0000   1.0000   0.9989
    ## Specificity            1.0000   0.9997   0.9993   0.9998   1.0000
    ## Pos Pred Value         1.0000   0.9989   0.9965   0.9988   1.0000
    ## Neg Pred Value         0.9997   0.9992   1.0000   1.0000   0.9998
    ## Prevalence             0.2847   0.1939   0.1737   0.1637   0.1839
    ## Detection Rate         0.2845   0.1933   0.1737   0.1637   0.1837
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
    ## Balanced Accuracy      0.9996   0.9983   0.9996   0.9999   0.9994

The Random Forest perform comparatively better than Recursive Partition. This is based off the Sensisitivity Values and the confusion matrix.

WE SELECT RANDOM FOREST AS OUR ALGORITHIM.

Predict Activitiy Using Test Data
=================================

We run the Random Forest model against the data ...

``` r
predRF_test_data <- predict( modelFitRF , test_data )
data.frame( test_data[,1:2], as.data.frame(predRF_test_data) )
```

    ##     X user_name predRF_test_data
    ## 1   1     pedro                B
    ## 2   2    jeremy                A
    ## 3   3    jeremy                B
    ## 4   4    adelmo                A
    ## 5   5    eurico                A
    ## 6   6    jeremy                E
    ## 7   7    jeremy                D
    ## 8   8    jeremy                B
    ## 9   9  carlitos                A
    ## 10 10   charles                A
    ## 11 11  carlitos                B
    ## 12 12    jeremy                C
    ## 13 13    eurico                B
    ## 14 14    jeremy                A
    ## 15 15    jeremy                E
    ## 16 16    eurico                E
    ## 17 17     pedro                A
    ## 18 18  carlitos                B
    ## 19 19     pedro                B
    ## 20 20    eurico                B
