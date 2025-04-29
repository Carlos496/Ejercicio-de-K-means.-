# Ejercicio-de-K-means.-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generar datos de ejemplo
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Método del Codo para K óptimo')
plt.xlabel('Número de clústeres (k)')
plt.ylabel('SSE (Suma de errores cuadráticos)')
plt.axvline(x=4, color='r', linestyle='--', label='Codo sugerido en k=4')
plt.legend()
plt.grid()
plt.show()






Explicación del Código

1. Generamos datos sintéticos con 4 clústeres usando `make_blobs`.
2. Probamos valores de k desde 1 hasta 10.
3. Para cada k, ejecutamos K-Means y almacenamos el SSE (mediante el atributo `inertia_`).
4. Graficamos SSE vs k para identificar el codo donde la reducción en SSE comienza a disminuir

5. Referencias Bibliográficas

1. Analytics Vidhya. (2019). "Comprehensive Guide to K-Means Clustering". [Enlace](https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/)
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning". Springer.
3. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). "An Introduction to Statistical Learning". Springer.
