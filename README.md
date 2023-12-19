# **Price Prediction Model for European Airbnb Listings**

Developed by: Patrick Helcl

Written by: Kai Stern

# Abstract 
We aim to build a model that predicts the price of Airbnbs in Europe using a data set from Kaggle which details the features and pricings of Airbnb’s in various European countries and cities. This model would help people looking to travel quickly determine if an Airbnb is fairly priced and the pricing trends of Airbnbs in general areas. The model would also help owners of Airbnbs determine prices according to their circumstances and attributes. The data used includes various statistics of Airbnb features (e.g number of rooms, price, has superhost, etc). Due to our datasets inclusion of boolean, categorical, and numerical data columns we plan to use a random forest regressor model which we feel will best handle these features. We will be building a model that predicts the price of the Airbnb and compare our model’s results with the price column in our dataset. The specific model will be chosen after feature selection, hyperparameter tuning, and comparison of several algorithms. Algorithms will be compared using appropriate loss functions.  

# Background

Airbnb’s rapid growth over the last 15 years has turned the company into one of the three pillars of the short term rental market, joining traditional renting and the hospitality industry.There are over 5 million Airbnbs worldwide spanning 81,000 cities and 191 countries<a name="barron"></a>[<sup>[1]</sup>](#barronnote). As Airbnb’s popularity first began to rise in the early 2010s, research on the company generally focused on its viability as a competitor in the short term rental sector and what effects the popular new accommodation sharing app would have on jobs and tourism in major cities<a name="Nguyen"></a>[<sup>[4]</sup>](#Nguyennote). Findings correlated increases in the number of Airbnb listings with small decreases in hotel revenue and increases in traditional rent rates <a name="Kerren"></a>[<sup>[2,4]</sup>](#Kerrennote). Due to the popularity of Airbnb as a product, more recent research has focused on exploring how Airbnb pricing can be used as an indicator for the health of the tourism and rental economies in cities<a name="Sans"></a>[<sup>[6]</sup>](#Sansnote). Other research has compared differing spatial distributions of Airbnbs and hotels and how this difference changes crowd size and interaction between locals and tourists in cities<a name="Perez"></a>[<sup>[3]</sup>](#Pereznote). Airbnb has a remarkable ability to capitalize on proximity to tourist attractions relative to hotels<a name="Gutierrez"></a>[<sup>[5]</sup>](#Gutierreznote).This research also developed tools to chart where cities have seen the largest tourist pressure related to Airbnb growth<a name="Gutierrez"></a>[<sup>[5]</sup>](#Gutierreznote). The project we intend to complete focuses on helping hosts and users of Airbnb determine fair pricing for lodging based on features of the homes or apartments.This service would help hosts ensure they aren’t undervaluing or overvaluing their services and give tourists the ability to evaluate whether pricing is fair when looking for an Airbnb to stay in.


# Problem Statement

The problem that we are trying to solve is the difficulty airbnb users have when trying to rent an airbnb. More specifically, there are many factors when trying to figure out which airbnb to book in an area (e.g. policies enforced, how many rooms, reading reviews, etc.). This takes a lot of time and it is very rigorous to figure out whether or not the nightly rate of the airbnb is overpriced or not. Another problem is that airbnb owners may not know what to price their airbnb. They need to make sure the price is high enough to make a good profit but at the same time low enough that people will rent their airbnb. Simply put, the problem is the difficulty and time it takes to accurately value an airbnb. One ML-relevant solution is to use certain features of Airbnbs and try multiple algorithm types to find the best model that would accurately predict the price of a given airbnb.This would allow airbnb owners and renters to get a good estimate of what they should pay or be paid for a specific airbnb. We can measure this problem by looking at the price an airbnb is valued at. This problem can be replicated as there are several Airbnbs in several cities/countries. 

# Data

Link to dataset used on kaggle: [here](https://www.kaggle.com/datasets/dipeshkhemani/airbnb-cleaned-europe-dataset?select=Aemf1.csv)

Our dataset initially has 40k observations, spanning 19 variables (columns): 
Name of the city, Price of Airbnb, If it is weekday or weekend, Room type, Type of Airbnb (e.g. Apartment, house etc.), Private Room, Shared Room,The Person Capacity of Airbnb, If the Airbnb host has Superhost or not, If the Airbnb has multiple rooms, Business (If the owner of the listing has more than 4 listings total), Cleanliness Rating, Guest Satisfaction, Bedrooms (number of bedrooms), Distance to City Center (km), Metro Distance (km) (distance to subway/metro), Attraction Index (score based on proximity to tourist attractions), Normalised Attraction Index,  Restaurant Index (score based on how many/popularity of restaurants nearby), Normalised Restaurant index. 
The initial features are partially changed throughout our machine learning process as we one hot encoded the categorical columns, and dropped “Normalised Attraction Index”, “Normalised Restaurant Index", since the pipeline we plan to use later normalizes all scalars anyways. Some of these variables have many data inputs, and each of these inputs is an observation. Prior to starting our analysis we presumed some particularly critical variables would be the city the airbnb is in, whether it is a weekday or not, the guest satisfaction, and the amount of bedrooms. Categorical variables such as city and weekday/weekend columns will be one-hot encoded and the numerical data will be normalized to avoid scale differences and aid identifying feature importance. As we perform feature selection, feature selected models may drop some data that is unnecessary for testing purposes.


# code block1 goes here 

<iframe src="Assets/plot1.html" width=800 height=600 frameBorder=0></iframe>

# Data

Link to dataset used on kaggle: https://www.kaggle.com/datasets/dipeshkhemani/airbnb-cleaned-europe-dataset?select=Aemf1.csv

Our dataset initially has 40k observations, spanning 19 variables (columns): 
Name of the city, Price of Airbnb, If it is weekday or weekend, Room type, Type of Airbnb (e.g. Apartment, house etc.), Private Room, Shared Room,The Person Capacity of Airbnb, If the Airbnb host has Superhost or not, If the Airbnb has multiple rooms, Business (If the owner of the listing has more than 4 listings total), Cleanliness Rating, Guest Satisfaction, Bedrooms (number of bedrooms), Distance to City Center (km), Metro Distance (km) (distance to subway/metro), Attraction Index (score based on proximity to tourist attractions), Normalised Attraction Index,  Restaurant Index (score based on how many/popularity of restaurants nearby), Normalised Restaurant index. 
The initial features are partially changed throughout our machine learning process as we one hot encoded the categorical columns, and dropped “Normalised Attraction Index”, “Normalised Restaurant Index", since the pipeline we plan to use later normalizes all scalars anyways. Some of these variables have many data inputs, and each of these inputs is an observation. Prior to starting our analysis we presumed some particularly critical variables would be the city the airbnb is in, whether it is a weekday or not, the guest satisfaction, and the amount of bedrooms. Categorical variables such as city and weekday/weekend columns will be one-hot encoded and the numerical data will be normalized to avoid scale differences and aid identifying feature importance. As we perform feature selection, feature selected models may drop some data that is unnecessary for testing purposes.


# Evaluation Metrics

We decided to evaluate our algorithms/models based on two error metrics: r2 score and mean absolute error (MAE). Our model is going to output a continuous value (predicted price) and these error metrics are designed for evaluating models with continuous outputs. The r2 score metric returns a value between 0 and 1 and describes how well the model explains variability in the target variable. A higher r2-score value means the model explains much of the variability in the target variable while a low r2-score value means the opposite. In short an r2 value closer to 1 is more desirable. MAE is a measure of the difference between predicted and actual price values. MAE works by summing the absolute value of distance between its prediction and actual value. Due to the way it is calculated, MAE is more robust to outliers as opposed to error metrics like RMSE. Using both error metrics will give us a solid basis to compare our algorithms and models throughout the process of selecting our final model. While all error metrics are useful, we feel the most important for our specific problem is MAE as housing prices very often have outliers and MAE will give us a more accurate measurement of our models effectiveness.


# Results

Our goal was to create a model that could predict the price listing of an airbnb based on features from our dataset. To this end, we plan on using regression models as our outputs should be a number, not a category (thus we are not using classifier models). The regression models we plan on looking into are decision tree regressors and random forest regressors. This is because decision trees are easy to interpret, thus would make a good base model. Furthermore, Random Forests work similarly to Decision trees so it is fair to use the same features across both models  for comparison. Due to these properties, only one round of feature selection will be used. We expect the random forest regressor to perform better, however if they have similar accuracies and the random forest regressor takes too long to train and test, it may be more beneficial to use a decision tree as our model.

We first fit and trained our models on all the features to look into which model performs better, with all default parameters. This is a preliminary look into the models so we will not do feature selection or hyperparameter tuning yet. Random Forest tends to take a bit longer to train and test, but perform better than the Decision Tree model using the metrics we looked at.


# code blocks  goes here 

