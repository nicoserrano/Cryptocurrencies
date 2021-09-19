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

Moreover, I proceeded with the Principal Component Analysis (PCA) to reduce the 98 scaled columns I had to only 3. 
