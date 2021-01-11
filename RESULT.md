# Clustering via K-means algorithm

In this computer project, the goal is to design an RBF network with different numbers of centroids (e.g. 20, 4, and etc.) to solve a binary classification problem. I do not use any existing machine learning library, including libraries for the k-means algorithm, and I implemented the algorithm by my own. There are two classes of <img src="https://render.githubusercontent.com/render/math?math=C_1={\{x_i : d_i = -1\}}"> and <img src="https://render.githubusercontent.com/render/math?math=C_{-1}={\{x_i : d_i = -1\}}"> that we want to predict the correct labels of these classe's data-points. The region where <img src="https://render.githubusercontent.com/render/math?math=d_i">  is 1 is the union of the region that remains below the mountains and the region that remains inside the sun in the figure below. Sketch the points <img src="https://render.githubusercontent.com/render/math?math=x_i">, indicate (with different colors or markers) their desired class. I drew 100 points <img src="https://render.githubusercontent.com/render/math?math=x_1, ..., x_100"> independently and uniformly at random on [0, 1]. These will be our input patterns, and they live in two-dimensions.




The region where  <img src="https://render.githubusercontent.com/render/math?math=d_i"> is 1 is the union of the region that remains below the mountains and the region that remains inside the sun in the figure below. 


# Classification via RBF network
