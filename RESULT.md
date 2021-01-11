# Clustering via K-means algorithm

In this computer project, the goal is to design an RBF network with different numbers of centroids (e.g. 20, 4, and etc.) to solve a binary classification problem. I do not use any existing machine learning library, including libraries for the k-means algorithm, and I implemented the algorithm by my own. There are two classes of <img src="https://render.githubusercontent.com/render/math?math=C_1={\{x_i : d_i = -1\}}"> and <img src="https://render.githubusercontent.com/render/math?math=C_{-1}={\{x_i : d_i = -1\}}"> that we want to predict the correct labels of these classe's data-points. The region where <img src="https://render.githubusercontent.com/render/math?math=d_i">  is 1 is the union of the region that remains below the mountains and the region that remains inside the sun in the figure below. Sketch the points <img src="https://render.githubusercontent.com/render/math?math=x_i">, indicate (with different colors or markers) their desired class. I drew 100 points <img src="https://render.githubusercontent.com/render/math?math=x_1, ..., x_100"> independently and uniformly at random on [0, 1]. These will be our input patterns, and they live in two-dimensions.

<img width="1028" alt="data_formulation" src="https://user-images.githubusercontent.com/43753085/104145579-28817900-538d-11eb-803f-755ae922b83a.png">

As indicated in above figure, the decision boundary that can classify data points in the 2D plane is highly nonlinear. One way to linearize the input patters is to lift them in higher dimensions in another space and try to classify data in that higher dimension and project (using contour) the obtained boundary in higher dimensions into 2D plane to calculate the boundaries in <img src="https://render.githubusercontent.com/render/math?math=R^2">.

### K-means: 
In the first step, I clustered each class of <img src="https://render.githubusercontent.com/render/math?math=C_1"> and <img src="https://render.githubusercontent.com/render/math?math=C_{-1}"> into ten clusters (20 cluster in total) using the K-means algorithm. I implemented the K-means algorithm, code is attached in the appendix, through the following three steps.

**1. Randominitialization:** Iinitializedthecentroidsofallclustersbypicking20 datapoints randomly.       
**2. Clusteringclosepoints:** Then,allthedatapointsthataretheclosest (similar) to a centroid will create a cluster.     
**3. Movethecentroids:** Inthisstep,Icomputedthemeanofeachcluster,and updated the centroids by this mean value of points per cluster.

By solving the above iterative/numerical optimization by minimizing the sum of squares of distance of each point to the clusters’ centroids, we converge to the solution while the centroids as well as data points per cluster do not change anymore (convergence criteria). The result of K-means clustering is shown below per Class <img src="https://render.githubusercontent.com/render/math?math=C_1"> and <img src="https://render.githubusercontent.com/render/math?math=C_{-1}">  as well as all 20 clusters including all data points.

<img width="1160" alt="allKmeans1" src="https://user-images.githubusercontent.com/43753085/104145783-f9b7d280-538d-11eb-91be-2fadb6635cd6.png">

The following figures show the obtained clusters for class <img src="https://render.githubusercontent.com/render/math?math=C_{-1}">. As it is obvious all clusters are in the regions excluded sun and mountain.

<img width="1150" alt="allKmeans2" src="https://user-images.githubusercontent.com/43753085/104145863-44d1e580-538e-11eb-87ea-20ebfb7d7038.png">

Further, in the following figure all the 20 clusters including 10 clusters per class is demonstrated. As legends shows centroids are shown by black crosses and <img src="https://render.githubusercontent.com/render/math?math=C_{1}"> has indicator marker of circles (each cluster in this class has a different color) whereas <img src="https://render.githubusercontent.com/render/math?math=C_{-1}"> is shown by diamonds (each cluster in this class has a different color).


# Classification via RBF network
