import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
with open(r'C:\Users\xukunpeng\Desktop\Interdisciplinary-Research\Interdisciplinary-Research\data\processed\novelty+group.pkl', 'rb') as file:
    data = pickle.load(file)

data = data.dropna()



def view(df,tit):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df['career_novelty'], df['team_novelty'], df['expedition_novelty'])

    ax.set_xlabel('Career Novelty')
    ax.set_ylabel('Team Novelty')
    ax.set_zlabel('Expedition Novelty')

    # 显示图形
    plt.title(tit)
    plt.show()

def kmeans_visualization(df, tit, n_clusters=2):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_data)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for cluster_id in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster_id]
        ax.scatter(cluster_data['career_novelty'], cluster_data['team_novelty'], cluster_data['cited_num'], label=f'Cluster {cluster_id}', s=5)


    ax.set_xlabel('Career Novelty')
    ax.set_ylabel('Team Novelty')
    ax.set_zlabel('Cited Num')

    ax.set_title(tit + f' KMeans Clustering (n_clusters={n_clusters})')
    t = tit + f' KMeans Clustering (n_clusters={n_clusters})_2'
    
    ax.legend()
    plt.savefig('1.5pic/'+ t + '.png')
    plt.show()

for i in range(1960, 1970, 10):
    df = data[(data['pub_year'] >= i) & (data['pub_year'] <= i + 10) & (data['cited_num'] > 0)]
    #
    #print(len(df))
    kmeans_visualization(df, str(i) + '-' + str(i+10), 3)
    #view(df, str(i) + '-' + str(i+10))
