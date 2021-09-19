# Cryptocurrencies
#### *Cryptos analysis using unsupervised machine learning and Python*

## Overview

The purpose of this project was to analyze a dataset from many alternative cryptocurrencies to spot trends that make a firm or person want to invest in them. The problem with cryptos is that the most common ones, like bitcoin or ethereum, are becoming unaffordable for the common public. That being said, I will be using *unsupervised machine learning* to see if we can spot any trends that result in opportunities of these altcoins. 

## Resources

- Datasets:
  - [crypto_data.csv](https://github.com/nicoserrano/Cryptocurrencies/blob/main/crypto_data.csv)

- Technologies used: 
  - Python
  - Jupyter notebook
  - Sklearn, pandas, and hvplot libraries
  - Unsupervised Machine Learning


## Results

*Follow the code closely in the [crypto_clustering.ipynb](https://github.com/nicoserrano/Cryptocurrencies/blob/main/crypto_clustering.ipynb)*

First, I had to preprocess and transform the data so that unsupervised machine learning could work. This included dropping null values, using only tradaeble and mined cryptocurrencies, numerically encoding categorical columns using the `pandas.get_dummies` method, and scaling the data using the `StandardScaler()` method as well. 

Moreover, I proceeded with the Principal Component Analysis (PCA) to reduce the 98 scaled columns I had, to only 3 principal components. 

<img width="265" alt="Screen Shot 2021-09-19 at 11 44 45 AM" src="https://user-images.githubusercontent.com/83378141/133933791-460045cb-ac4d-422e-88cb-fb3f09d1711d.png">

Then, to see how many clusters (k) I could divide the cryptos in, I created an elbow curve. 

```
inertia = []
k = list(range(1,11))

for i in k:
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(pca_df)
    inertia.append(km.inertia_)
    
Create an elbow curve to find the best value for K.
elbow_data = {'k' : k, 'inertia' : inertia}
elbow_df = pd.DataFrame(elbow_data)
elbow_df.hvplot.line(x='k', y='inertia', title='Elbow Curve', xticks=k)
```

<img width="822" alt="Screen Shot 2021-09-19 at 11 32 29 AM" src="https://user-images.githubusercontent.com/83378141/133933901-53e2f60a-9464-4e41-87ab-e320fa50a4e5.png">

As it can be seen, the optimal result was 4 clusters. So, I then proceeded with the KMeans analysis to fit the pca dataframe and predict the clustering. The product was this `clustered_df` with a 'Class' column that showed the predictions to which group it belonged to. 

<img width="809" alt="Screen Shot 2021-09-19 at 11 32 55 AM" src="https://user-images.githubusercontent.com/83378141/133933982-c9d556d7-60db-47fb-a791-b0b50deb76c2.png">





