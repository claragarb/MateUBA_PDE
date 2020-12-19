# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 16:01:16 2020

@author: Clara
"""

import dmsh
import numpy as np
import matplotlib.pyplot as plt
import minifemlib as fem
import triangulation

# centro = 4
# forma = dmsh.Union( [
#     dmsh.Polygon([[centro, centro],[6.0+centro, 1.0+centro],[1.0+centro, 1.0+centro],[centro + 1.0, centro + 6.0]]),
#     dmsh.Polygon([[centro + 1.0, centro + 1.0],[centro - 3.8, centro - 0.2],[centro - 0.5, centro - 0.5],[centro - 0.2 , centro - 3.8]]), 
      # ])
      
auxpuente=dmsh.Polygon([[-3.0, 0.0], [-1.0, 0.0], [-1.0, 3.0],
                      [1.0, 3.0],[1.0, 0.0],
                      [4.0,0.0],[4.0,3.0],
                      [6.0,3.0],[6.0,0.0],
                      [9.0,0.0],[9.0,3.0],
                      [11.0,3.0],[11.0,0.0],
                      [14.0,0.0],[14.0,3.0],
                      [16.0,3.0],[16.0,0.0],
                      [18.0,0.0],[18.0,5.7],[-3.0,5.7]])
puente1=dmsh.Difference(auxpuente,dmsh.Circle([0.0, 3.0], 1.2))
puente2=dmsh.Difference(puente1,dmsh.Circle([5.0, 3.0], 1.2))
puente3=dmsh.Difference(puente2,dmsh.Circle([10.0, 3.0], 1.2))
puente=dmsh.Difference(puente3,dmsh.Circle([15.0, 3.0], 1.2))
# Tomamos h=0.25
X, cells = dmsh.generate(puente, 0.5)

#Gráfico de la triangulación
print('Gráfico de la triangulación')
dmsh.helpers.show(X, cells,puente)

####Ejemplo con la fuente de calor cerca de cada cabeza de la llave (ver rhs(x,y))

# Evaluamos la geometría        
T = triangulation.Triangulation(X,cells)
boundary = fem.Boundary(T)

# Definimos un lado derecho
def rhs(x,y):  
	return np.exp(-0.1*(x**2+(y+8.9)**2)) + np.exp(-0.1*(x**2+(y+0.1)**2))

# Construimos matrices de rigidez y vector de cargas
order = 1
A = fem.StiffnessLaplacian(T,'triangular', order)
F = fem.LoadVector(rhs,T,'triangular', order)
B=A
# Agregamos condiciones de borde Dirichlet homogeneas como ecuaciones
for i in boundary:
    A[i,:] = np.zeros(A.shape[1])
    A[i,i] = 1
    F[i] = 0

# Resolvemos el problema lineal
sol_num = np.linalg.solve(A, F)


fig, ax1 = plt.subplots(1, 1)
tpc = ax1.tripcolor(T.points[:,0], T.points[:,1],T.simplices, sol_num, shading='gouraud')

ax1.set_title('Solución numérica')

plt.show()