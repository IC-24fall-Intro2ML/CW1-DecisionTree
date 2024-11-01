# CW1-DecisionTree
 Implement a decision tree algorithm and use it to determine one of the indoor locations based on WIFI signal strengths collected from a mobile phone.

Just put the results here for analysing & reminders.
Delete these parts later, before submitting.

## TODO:
1. Visualization: Adjust the parameters for better layout, save the images (two, one complete, one with the first few layers, the latter has not been implemented yet, the code may need to be adjusted), explain in the README how to run the corresponding function, and may need to comment out the corresponding two lines in the main script. 
2. Report
3. README
4. Make sure the code runs on the lab machine.
5. Packaging, submission and double-check.

Cross-validation metrics for clean data:
Average Depth: 6.89
Confusion Matrix:
 [[49.4  0.   0.3  0.3]
 [ 0.  48.   2.   0. ]
 [ 0.2  1.9 47.7  0.2]
 [ 0.5  0.   0.1 49.4]]
Accuracy: 0.9725
Recalls: [0.988, 0.96, 0.954, 0.988]
Precisions: [0.986, 0.9619, 0.9521, 0.99]
F1-Scores: [0.987, 0.961, 0.953, 0.989]

--------------------------------------------------

Cross-validation metrics for noisy data:
Average Depth: 10.87
Confusion Matrix:
 [[37.2  3.6  4.5  3.7]
 [ 3.3 39.5  4.5  2.4]
 [ 3.5  4.  41.4  2.6]
 [ 4.   2.7  3.7 39.4]]
Accuracy: 0.7875
Recalls: [0.7592, 0.7948, 0.8039, 0.7912]
Precisions: [0.775, 0.7932, 0.7652, 0.8191]
F1-Scores: [0.767, 0.794, 0.7841, 0.8049]

--------------------------------------------------

Nested Cross-Validation for Clean Data:
Average Metrics across 10 outer folds:
Confusion Matrix:
 [[49.9         0.          0.1         0.        ]
 [ 0.         47.6         2.4         0.        ]
 [ 0.74444444  2.24444444 46.63333333  0.37777778]
 [ 0.53333333  0.          0.17777778 49.28888889]]
Accuracy: 0.9671
Recalls: [0.998, 0.952, 0.9327, 0.9858]
Precisions: [0.975, 0.955, 0.9457, 0.9924]
F1-Scores: [0.9864, 0.9535, 0.9391, 0.9891]

--------------------------------------------------

Nested Cross-Validation for Noisy Data:
Average Metrics across 10 outer folds:
Confusion Matrix:
 [[44.8         0.9         1.3         2.        ]
 [ 1.8        40.7         6.1         1.1       ]
 [ 2.82222222  2.1        44.8         1.77777778]
 [ 2.2         1.3         1.6        44.7       ]]
Accuracy: 0.875
Recalls: [0.9143, 0.8189, 0.8699, 0.8976]
Precisions: [0.8678, 0.9044, 0.8327, 0.9016]
F1-Scores: [0.8905, 0.8596, 0.8509, 0.8996]


Result analysis after pruning:
Minimal effect on clean data, but significant improvement on noisy data. For clean data, accuracy decreased slightly, from 0.9725 to 0.9671, but the pruned model is simpler and generalizes better. After pruning, the model retains most of its accuracy on clean data, suggesting a slight improvement in generalization with minimal accuracy loss. For noisy data, accuracy improved significantly from 0.7875 to 0.875. Pruning removes noisy features and allows the model to focus on primary characteristics, resulting in better accuracy by reducing overfitting to the noisy data.

Depth analysis:
Reduction in Average Depth: Pruning decreases the average depth on clean data, which reduces overfitting. On noisy data, even after pruning, the depth remains high, indicating that complexity is still needed to handle noisy features.
Relationship between Depth and Accuracy: Greater depth may improve performance on noisy data by capturing complexity but can lead to slight overfitting on clean data.