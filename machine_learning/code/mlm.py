import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Using matrix method

data_file = "./mlm.csv"
data = pd.read_csv(data_file)
x1_arr = np.array(data.iloc[0:800,0].values)
x2_arr = np.array(data.iloc[0:800,1].values)
y_arr = np.array(data.iloc[0:800,2].values)

x_arr = np.hstack((x1_arr.reshape(800,1),x2_arr.reshape(800,1),np.ones((800,1), dtype = float)))

x_test = data.iloc[800:1001,0]
y_test = data.iloc[800:1001,1]
z_test = data.iloc[800:1001,2]

X_matrix = np.matrix(x_arr)
Y_matrix = np.matrix(y_arr.reshape(800,1))
W_matrix = X_matrix.T.dot(X_matrix).I.dot(X_matrix.T).dot(Y_matrix)

MSE_result = np.sum(pow(np.array(X_matrix.dot(W_matrix) - Y_matrix),2))/800/2
w = np.array(W_matrix)
print(w)
print("Mean Squared Error is:",MSE_result)

ax = plt.axes(projection = "3d")
ax.scatter3D(x_test,y_test,z_test)
draw_x = np.linspace(0,100)
draw_y = np.linspace(0,100)
X_drawing, Y_drawing = np.meshgrid(draw_x,draw_y)
ax.plot_surface(X = X_drawing, Y = Y_drawing, Z = X_drawing*w[0] + Y_drawing*w[1] + w[2],alpha = 0.3)
ax.view_init(elev = 40, azim = 40)
plt.show()