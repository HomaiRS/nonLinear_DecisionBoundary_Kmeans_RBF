# nonLinearDecisionBoundary_Kmeans_RBF XXXXXXXXXXXXXXXXXXX

# Classification using SVM

In this computer project, we will design an SVM, and we use an existing library for solving the quadratic optimization problem that is associated with the SVM. Other than that, I do not use any existing machine learning/SVM library and I implemented the algorithm by my own. 

We have two classes of <img src="https://render.githubusercontent.com/render/math?math=C_1={\{x_i : d_i = 1\}}"> and <img src="https://render.githubusercontent.com/render/math?math=C_{-1}={\{x_i : d_i = -1\}}"> in the input data that are separated with a non-linear boundary. 
The region where  <img src="https://render.githubusercontent.com/render/math?math=d_i"> is 1 is the union of the region that remains below the mountains and the region that remains inside the sun in the figure below. 
