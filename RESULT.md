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

![Kmeans_BothClasses](https://user-images.githubusercontent.com/43753085/104146029-dccfcf00-538e-11eb-9008-402430814510.png)

The number of clusters used in K-means algorithm shows the space that we are going to lift our data to make the input patterns linearly separable there. The number of clusters indicates the length of weight vector that can be learned through Perceptron training algorithm (PTA) in the next steps.

# RBF network

After I got the clusters via K-means, I designed the RBF network by using one input and one output and one hidden layer in which each layer’s activation function has been computed by <img src="https://render.githubusercontent.com/render/math?math=\phi(x)=e^{-\Beta_i||x-c_i||}"> such that <img src="https://render.githubusercontent.com/render/math?math=i"> = 1, 2, ... , 20. The RBFN is designed as follows. To compute the exponent of RBF function showed as <img src="https://render.githubusercontent.com/render/math?math=\Beta_i">, I used the formulas in the following picture that enhances the performance of PTA in the later steps. <img src="https://render.githubusercontent.com/render/math?math=\Beta_i"> and <img src="https://render.githubusercontent.com/render/math?math=c_i"> are the parameters which are specific to each cluster. Per input value, I compute the <img src="https://render.githubusercontent.com/render/math?math=\phi(x_i)"> is computed as it is shown in the following figure. (<img src="https://render.githubusercontent.com/render/math?math=\phi"> is a 20×1 vector). For more details on RBFN, see <https://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/>.

<img width="1052" alt="myHandDrawing" src="https://user-images.githubusercontent.com/43753085/104146508-9ed3aa80-5390-11eb-8140-6a724017baf2.png">

I also used an equal value of Β for all clusters, and I got a perfect classification results using 20 clusters. 

**PTA:** Now data in higher dimensions, by using 20 clusters we are in <img src="https://render.githubusercontent.com/render/math?math=R^20"> dimensions, patterns are hopefully linearly separable. Thus, we can use the iterative algorithm of Perceptron training algorithm (PTA) to compute the weights and bias to construct the decision boundaries that can classify the data into two classes. The PTA equation is as follows.

For I=1 to n
<img src="https://render.githubusercontent.com/render/math?math=W^{new}=\eta\phi(x)(d_i -u(W^T\phi(x)))"> + 𝑤_old

Where the activation function of <img src="https://render.githubusercontent.com/render/math?math=u"> is the signum function. I used online learning to update the weight. The stopping criteria to obtain the optimal solution (optimal weight and bias) is the misclassification error and a threshold on the number of iterations. Then, after obtaining the optimal solution and optimal bias, there are two ways to compute the decision boundary ({𝑥: 𝑔(𝑥) = <img src="https://render.githubusercontent.com/render/math?math=W^T\phi(x)"> + 𝜃 = <img src="https://render.githubusercontent.com/render/math?math=0">}). One is numerically evaluate 1,000/10,000/100,000 points in the 2D plane both in 𝑥 and 𝑦 directions, and then approximate the decision boundaries by accepting the value of <img src="https://render.githubusercontent.com/render/math?math=g(x,y)"> where the <img src="https://render.githubusercontent.com/render/math?math=g=0">. This method is usually time-consuming and not the most efficient way to approximate the decision boundaries. Instead, I computed the <img src="https://render.githubusercontent.com/render/math?math=g(x)=0"> by projecting the contour of this discriminating function in to the 2D plane (as recommended on Piazza). The results of decision boundaries using 20 clusters, are indicated in the following plots that perfectly separates the two classes in four different cases.

<img width="1070" alt="PTA1" src="https://user-images.githubusercontent.com/43753085/104146985-371e5f00-5392-11eb-92f8-2fa503438de2.png">

**Having less clusters (say 4):** In this specific problem that decision boundaries are highly nonlinear, we might not be able to linearly separate the input patterns in 4 dimensions. Thus, based on this, PTA that is a method for optimizing weight and bias for linearly separable input patterns does not yield a good approximation for the highly nonlinear boundaries in 4 dimensions. So, as it is demonstrated in the following plot and according to aforementioned explanation, we are not able to separate the two classes perfectly and we end up having some misclassifications errors (shown in green squares).

<img width="968" alt="PTA2" src="https://user-images.githubusercontent.com/43753085/104147067-83699f00-5392-11eb-8fc6-885750b2890d.png">

The blue line is the projection/ contour of <img src="https://render.githubusercontent.com/render/math?math=g(x)=0">. (the yellow lines are level set equal to 0.5). We have 4 misclassified data points in <img src="https://render.githubusercontent.com/render/math?math=C_1"> and one misclassified point in <img src="https://render.githubusercontent.com/render/math?math=C_{-1}">. This shows we need at least more that 4-dimensional embedding of the data to get a good approximation of decision boundary.

<img width="732" alt="dataStructure2" src="https://user-images.githubusercontent.com/43753085/104147192-f4a95200-5392-11eb-8605-9c1810061460.png">

