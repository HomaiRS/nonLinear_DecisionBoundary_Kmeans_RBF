# Clustering and classification using K-means and RBF networks

The goal of this computer project is to design a RBF network with different number of centroids (e.g. 20, 4, and etc.) to solve a binary classification problem. I do not use any existing machine learning library, including libraries for the k-means algorithm, and I implemented the algorithm by my own. We have two classes of  <img src="https://render.githubusercontent.com/render/math?math=C_1={\{x_i : d_i = 1\}}"> and <img src="https://render.githubusercontent.com/render/math?math=C_{-1}={\{x_i : d_i = -1\}}"> that we want to predict the correct labels of these classe's data-points. The points in these classes are separated with a non-linear boundary. We first use the K-means algorithm to cluster the input data (e.g., n=200, 300, etc.) and then apply RBF using the K-means clusters' centroids for classification.


```diff
@ -  See the results in "RESULT.md" file in red @
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```
