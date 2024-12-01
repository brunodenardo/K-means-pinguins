import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Carregar o dataset
caminho_arquivo = 'penguins.csv'
dados_pinguins = pd.read_csv(caminho_arquivo)

# Pré-processamento
dados_pinguins_limpos = dados_pinguins.dropna()
dados_pinguins_limpos.loc[:, 'sexo'] = dados_pinguins_limpos['sex'].map({'FEMALE': 1, 'MALE': 0})
dados_pinguins_limpos = dados_pinguins_limpos.dropna()

# Remover o comprimento da nadadeira
caracteristicas = ['culmen_length_mm', 'culmen_depth_mm', 'body_mass_g', 'sexo']
dados_caracteristicas = dados_pinguins_limpos[caracteristicas]

normalizador = StandardScaler()
dados_normalizados = normalizador.fit_transform(dados_caracteristicas)

# Método do Cotovelo e Índice de Silhueta
wcss = []
silhouette_scores = []
intervalo_clusters = range(2, 11)

for k in intervalo_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(dados_normalizados)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(dados_normalizados, kmeans.labels_))

# Determinar o melhor número de clusters com base no índice de silhueta
melhor_num_clusters = intervalo_clusters[np.argmax(silhouette_scores)]

# Plotar o gráfico do cotovelo
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), [0] + wcss, marker='o', linestyle='--')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.axvline(melhor_num_clusters, color='r', linestyle='--', label=f'Nº ideal: {melhor_num_clusters}')
plt.legend()
plt.grid()
plt.show()

# Plotar o índice de silhueta
plt.figure(figsize=(8, 5))
plt.plot(intervalo_clusters, silhouette_scores, marker='o', linestyle='--')
plt.title('Índice de Silhueta')
plt.xlabel('Número de Clusters')
plt.ylabel('Pontuação Silhueta')
plt.xticks(intervalo_clusters)
plt.axvline(melhor_num_clusters, color='r', linestyle='--', label=f'Nº ideal: {melhor_num_clusters}')
plt.legend()
plt.grid()
plt.show()

# Aplicar o k-means com o número ideal de clusters
modelo_kmeans = KMeans(n_clusters=melhor_num_clusters, random_state=42)
dados_pinguins_limpos['Cluster'] = modelo_kmeans.fit_predict(dados_normalizados)

# Gráfico 2D
plt.figure(figsize=(8, 6))
plt.scatter(dados_normalizados[:, 0], dados_normalizados[:, 1], c=dados_pinguins_limpos['Cluster'], cmap='viridis', s=50)
plt.title('Clusters em 2D (Comprimento vs Profundidade do Culme)')
plt.xlabel('Comprimento do Culme (normalizado)')
plt.ylabel('Profundidade do Culme (normalizado)')
plt.colorbar(label='Cluster')
plt.grid()
plt.show()

# Gráfico 3D
figura_3d = plt.figure(figsize=(10, 7))
ax = figura_3d.add_subplot(111, projection='3d')
scatter = ax.scatter(
    dados_normalizados[:, 0], dados_normalizados[:, 1], dados_normalizados[:, 2],
    c=dados_pinguins_limpos['Cluster'], cmap='viridis', s=50
)
ax.set_title('Clusters em 3D')
ax.set_xlabel('Comprimento do Culme (normalizado)')
ax.set_ylabel('Profundidade do Culme (normalizado)')
ax.set_zlabel('Peso Corporal (normalizado)')
plt.colorbar(scatter, ax=ax, label='Cluster')
plt.show()

# Exibir o melhor número de clusters
print(f"O melhor número de clusters foi determinado como: {melhor_num_clusters}")
