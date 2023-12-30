# **Price Prediction Model for European Airbnb Listings**
This project was initially created for the final assignment in the Supervised Machine Learning Algorithms course (COGS 118A) at UCSD during Spring 2023. It has been refined and expanded upon by Patrick Helcl and Kai Stern since the original submission. Other contributors to the original project, but not the website or additional sections, are credited below along with their contributions. 

# Abstract 
We aim to build a model that predicts the price of Airbnbs in Europe using a data set from Kaggle which details the features and pricings of Airbnb’s in various European countries and cities. This model will help people looking to travel quickly determine if a specific Airbnb is fairly priced as well as the overall pricing trends of Airbnbs in general areas. The model will also help owners of Airbnbs determine prices according to their circumstances and attributes. The data used includes various statistics of Airbnb features (e.g number of rooms, price, has superhost, etc). Due to our datasets inclusion of boolean, categorical, and numerical data columns we plan to use a random forest regressor model which we feel will best handle these features. We will be building a model that predicts the price of the Airbnb and compare our model’s results with the price column in our dataset. The specific model will be chosen after feature selection, hyperparameter tuning, and comparison of several algorithms. Algorithms will be compared using appropriate loss functions.  

# Background

Airbnb’s rapid growth over the last 15 years has turned the company into one of the three pillars of the short term rental market, joining traditional renting and the hospitality industry.There are over 5 million Airbnbs worldwide spanning 81,000 cities and 191 countries[1]. As Airbnb’s popularity first began to rise in the early 2010s, research on the company generally focused on its viability as a competitor in the short-term rental sector and what affects the popular new accommodation sharing app would have on jobs and tourism in major cities[4]. Findings correlated increases in the number of Airbnb listings with small decreases in hotel revenue and increases in traditional rent rates [2,4]. Due to the popularity of Airbnb as a product, more recent research has focused on exploring how Airbnb pricing can be used as an indicator for the health of the tourism and rental economies in cities[6]. Other research compared differing spatial distributions of Airbnbs and hotels and how this difference changes crowd size and interaction between locals and tourists in cities[3]. Airbnb has a remarkable ability to capitalize on proximity to tourist attractions relative to hotels[5].This research also developed tools to chart where cities have seen the largest tourist pressure related to Airbnb growth[5]. The project we intend to complete focuses on helping hosts and users of Airbnb determine fair pricing for lodging based on features of the homes or apartments.This service would help hosts ensure they aren’t undervaluing or overvaluing their services and give tourists the ability to evaluate whether pricing is fair when looking for an Airbnb to stay in.


# Problem Statement

The problem that we are trying to solve is the difficulty airbnb users and owners have when trying to evaluate fair prices for Airbnbs. There are many factors when trying to figure out which airbnb to book in an area (e.g. policies enforced, how many rooms, reading reviews, etc.) and it can be overwhelming for users to determine how fair the price they are paying is given all of the available information. One machine learning relevant solution to this problem is to use certain features of Airbnbs and try multiple algorithm types to find the best model that will accurately predict the price of a given airbnb.This will allow airbnb owners and renters to get a good estimate of what they should pay or be paid for a specific airbnb. We can measure this problem using current airbnb pricing trends. This problem is replicable as there are several Airbnbs in several cities/countries.

# Data

Link to dataset used on kaggle: [here](https://www.kaggle.com/datasets/dipeshkhemani/airbnb-cleaned-europe-dataset?select=Aemf1.csv)

The dataset (before cleaning) has 40,000 observations, spanning 19 variables (columns): - - 
- City: Name of the city 
- Price: Price of Airbnb
- Day: weekday vs weekend 
- Room type: shared vs entire building
- Private Room: whether or not private rooms are available
- Shared Room: Whether or not the rooms are shares
- Person Capacity: max number of inhabitants
- Superhost: If the airbnb host is considered a superhost
- Multiple Rooms: single vs multiple rooms
- Business: If the owner of the listing has more than 4 listings total
- Cleanliness Rating: Guest cleanliness rating
- Guest Satisfaction: Guest satisfaction rating
- Bedrooms: number of bedrooms
- Distance to City Center
- Metro Distance: distance to subway/metro
- Attraction Index: score based on proximity to tourist attractions
- Normalised Attraction Index
- Restaurant Index: score based on how many/popularity of restaurants nearby
- Normalized Restaurant index 

The initial features are partially changed throughout the cleaning and preprocessing of our machine learning process as we one hot encoded the categorical columns and dropped “Normalised Attraction Index”, “Normalised Restaurant Index”. The pipeline we plan to use later normalizes all scalars. Some of these variables have many data inputs, and each of these inputs is an observation. Prior to starting our analysis we presumed some particularly critical variables would be the city the airbnb is in, whether or not it is a weekday, the guest satisfaction, and the amount of bedrooms. Categorical variables such as city and weekday/weekend columns will be one-hot encoded and the numerical data will be normalized to avoid scale differences and aid identifying feature importance. As we perform feature selection, feature selected models may drop some data that is unnecessary for testing purposes.




| City      |   Price | Day     | Room Type    | Shared Room   | Private Room   |   Person Capacity | Superhost   |   Multiple Rooms |   Business |   Cleanliness Rating |   Guest Satisfaction |   Bedrooms |   City Center (km) |   Metro Distance (km) |   Attraction Index |   Restraunt Index |
|:----------|--------:|:--------|:-------------|:--------------|:---------------|------------------:|:------------|-----------------:|-----------:|---------------------:|---------------------:|-----------:|-------------------:|----------------------:|-------------------:|------------------:|
| Amsterdam | 194.034 | Weekday | Private room | False         | True           |                 2 | False       |                1 |          0 |                   10 |                   93 |          1 |           5.02296  |              2.53938  |            78.6904 |           98.2539 |
| Amsterdam | 344.246 | Weekday | Private room | False         | True           |                 4 | False       |                0 |          0 |                    8 |                   85 |          1 |           0.488389 |              0.239404 |           631.176  |          837.281  |
| Amsterdam | 264.101 | Weekday | Private room | False         | True           |                 2 | False       |                0 |          1 |                    9 |                   87 |          1 |           5.74831  |              3.65162  |            75.2759 |           95.387  |
| Amsterdam | 433.529 | Weekday | Private room | False         | True           |                 4 | False       |                0 |          1 |                    9 |                   90 |          2 |           0.384862 |              0.439876 |           493.273  |          875.033  |
| Amsterdam | 485.553 | Weekday | Private room | False         | True           |                 2 | True        |                0 |          0 |                   10 |                   98 |          1 |           0.544738 |              0.318693 |           552.83   |          815.306  |




# Solution

To help alleviate the uncertainty associated with picking and pricing Airbnb’s, we propose a model to meaningfully predict price listings by leveraging features, previously mentioned in the data section above, from our dataset. Our plan is to use untuned decision tree and random forest regressors as our base model, then compare them with our advanced model that is made up of decision tree and random forest regressors using optimized hyperparameters and feature selected columns. Decision tree regressors are built for continuous data and can capture non-linear relationships while remaining interpretable. However decision tree regressors are very sensitive to the training data which can lead to overfitting and high variance. For this reason we aim to build upon the decision tree and use a random forest. Random forests are ensembles made up of decision trees that aggregate the outputs of the many decision trees to help solve the issues with overfitting and high variability. A random forest regressor is a random forest made up of decision tree regressors allowing it to work with continuous data. We expect random forest regressors to perform better on test set data as they are more generalizable. We will perform feature selection using recursive feature elimination cross validation and grid search validation for hyperparameter tuning. Comparison between models will be mainly evaluated using mean absolute error (MAE) with a secondary comparison using r2 score. All of our code is available to provide reproducibility of our results. Reproduced results should be based on data from the nine cities in our dataset: Amsterdam, Athens, Barcelona, Berlin, Budapest, Lisbon, Paris, Rome and Vienna.

# Evaluation Metrics

We decided to evaluate our algorithms/models based on two error metrics: r2 score and mean absolute error (MAE). Our model is going to output a continuous value (predicted price) and these error metrics are designed for evaluating models with continuous outputs. The r2 score metric returns a value between 0 and 1 and describes how well the model explains variability in the target variable. A higher r2-score value means the model explains much of the variability in the target variable while a low r2-score value means the opposite. In short an r2 value closer to 1 is more desirable. MAE is a measure of the difference between predicted and actual price values. MAE works by summing the absolute value of distance between its prediction and actual value. Due to the way it is calculated, MAE is more robust to outliers than error metrics like RMSE. Using both error metrics will give us a solid basis to compare our algorithms and models throughout the process of selecting our final model. While all error metrics are useful, we feel the most important for our specific problem is MAE as housing prices very often have outliers and MAE will give us a more accurate measurement of our models effectiveness.


# Results

Our goal was to create a model that could predict the price listing of an airbnb based on features from our dataset. We first fit and trained our models on all the features to look into which model performs better, with all default parameters. This is a preliminary look into the models so we will not do feature selection or hyperparameter tuning yet. Random Forests tend to take longer to train and test, but perform better than the Decision Tree model using the metrics we looked at.


# Base Model MAE Decision Tree vs. Random Forest Model

The plot below details the MAE of the decision tree and random forest regressors running on all of the features with no hyperparameter tuning. We can see that the random forest regressor has a lower MAE value than the decision tree which means that its predictions were closer to the actual airbnb prices. This is in line with our expectations. 

<iframe src="Assets/plot1.html" width=825 height=600 frameBorder=0></iframe>



# Base Model R-Squared score, Decision Tree vs. Random Forest Model

We can see that the random forest had a slightly higher r2 score (0.33) relative to the decision tree (0.28). This means that the random forest explains slightly more variability in prices than the decision tree. However both models performed relatively poorly as a value closer to 1 is desired for r2 score.

<iframe src="Assets/plot2.html" width=825 height=600 frameBorder=0></iframe>


# Interpretation

The base model decision tree regressor’s r2 score was lower and its MAE was higher than the Random Forest. In other words, our base model random forest performed better than the base model Decision Tree Regressor in both error metrics. We plan on comparing this model with our final model to analyze the performance changes of our final model. The features of the base model will be used for comparison with the features selected in our final model. 

### Generating Train and Test Sets

We decided to utilize the train test split method for training/testing and allocated 70% of our data to training and the remaining 30% to testing. Our data points are randomly placed (with a seed for reproducibility) into the training and test sets. We feel justified in this method for splitting the data as we have over 41,000 data points, thus giving our model approximately 30,000 data points to learn from and 10,000 to test generalizability. Other methods for training and testing models such as k-fold cross validation would add unnecessary computational and time complexity given the large size of our dataset. The train test split method best balances creating representative training and test sets for our model while minimizing complexity and implementation time.


### Feature selection: 

Since the Random Forest Regressor performed better than the decision tree regressor, we decided to use it for our final model. We used Recursive Feature Elimination Cross Validation (RFECV) to select the features used in our final model. The first step of RFECV is to rank the features based on feature importance. Unimportant features are then eliminated. From there the remaining features are trained/tested using cross validation. Features are dropped iteratively and the model is repeatedly tested on the new feature set. The end result is the iteration that performed best based on the metric of choice, which is MAE for this project. This was all implemented and conducted using sci-kit learn’s built in feature selection module and RFECV method. 

```py

# feature selection block

subset = airbnb.sample(n=int(41714/800), random_state=23)

X = subset.drop('Price', axis=1)
y = subset['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

categorical_cols = ['City', 'Day', 'Room Type']

numeric_cols = ['Person Capacity', 'Cleanliness Rating', 'Guest Satisfaction', 'Bedrooms',
                'City Center (km)', 'Metro Distance (km)', 'Attraction Index','Restraunt Index']

boolean_cols = ['Shared Room', 'Private Room', 'Superhost', 'Multiple Rooms', 'Business']

preproc = ColumnTransformer(
    transformers=[
        ('one hot', OneHotEncoder(categories='auto'), categorical_cols),
        ('standardize', StandardScaler(), numeric_cols),
        ('bool as is', 'passthrough', boolean_cols)
    ])

X_train_transformed = preproc.fit_transform(X_train)

X_test_transformed = preproc.transform(X_test)

rf_regressor = RandomForestRegressor()

selector = RFECV(rf_regressor, cv=5, scoring='neg_mean_absolute_error')
selector.fit(X_train_transformed, y_train)

selected_features = preproc.transformers_[0][1].get_feature_names_out(categorical_cols).tolist() + numeric_cols + boolean_cols
selected_features = [selected_features[i] for i, support in enumerate(selector.support_) if support]

print("Selected Features: ")
print(selected_features)
```

After running our code for the feature selection, all columns were selected as important except for the “Shared Room” column. We included a new pair of base models using the updated feature list for another comparison of our model to its baseline. The feature selection suggests that almost all features played an important role in determining the price of an airbnb. Our results below show that both the random forest regressor and decision tree regressor slightly improved their MAE and r2 score after using the new features.


### Hyperparameter Tuning:

Now that we have done feature selection, we can tune the hyperparameters of our model. We did this using grid search cross validation. The hyperparameters that we used grid search on were n_estimators, max_depth, and min_samples_split. These hyperparameters describe the number of trees in the random forest, the max possible depth for each tree, and the minimum number of samples required to split an internal node. The grid search cross validation resulted in recommending the default max_depth and min_samples_split values. The n_estimators value was recommended to be changed to 40.

```py
# hyper parameter selction 

subset = airbnb.sample(n=int(41714/800), random_state=23)

X = subset[['City','Day','Room Type','Person Capacity','Private Room','Cleanliness Rating','Guest Satisfaction','Bedrooms',
            'City Center (km)','Metro Distance (km)','Attraction Index','Restraunt Index','Superhost','Multiple Rooms',
            'Business']]

y = subset['Price'] 

new_cat_cols = ['City', 'Day', 'Room Type']

new_num_cols = ['Person Capacity', 'Cleanliness Rating', 'Guest Satisfaction', 'Bedrooms',
                'City Center (km)', 'Metro Distance (km)', 'Attraction Index','Restraunt Index']

new_bool_cols = ['Private Room', 'Superhost','Multiple Rooms', 'Business']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

preproc3 = ColumnTransformer(
    transformers=[
        ('one hot', OneHotEncoder(categories='auto'), new_cat_cols),
        ('standardize', StandardScaler(), new_num_cols),
        ('bool as is', 'passthrough', new_bool_cols)
    ])

X_train_transformed = preproc3.fit_transform(X_train)

preproc3_copy = preproc3
preproc3_copy.named_transformers_['one hot'].handle_unknown = 'ignore'

X_test_transformed = preproc3_copy.transform(X_test)

rf_regressor = RandomForestRegressor()

param_grid = {
    'n_estimators': [40, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 4, 6],
}

grid_search = GridSearchCV(rf_regressor, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train_transformed, y_train)

best_estimator = grid_search.best_estimator_

print("Best Estimator:")
print(best_estimator)

```

# Discussion

### Interpreting the result

As we anticipated, our main finding from this project is that the featured selected, and hyper parameter tuning random forest performed better on the test data than previous iterations of random forest and decision tree regressors. As noted in the proposed solutions section, random forests are ensembles of decision tree regressors that are less prone to overfit and more generalizable. This helps explain the improvement between our final model and the base model. Our feature selection allowed us to drop the variable “shared room” that was distracting our model from more accurate results. When the model was given the feature selected dataset it performed better because it could focus on the features that are relevant for predicting price. The hyperparameter tuning determined which number of estimators, max depth of each estimator, and number of sample splits from each decision led to the lowest error in predicting price.


The MAE score of the original random forest using all features and un-tuned hyperparameters was 54.04. This means that this model was off by an average of 54.04 from the actual price of the test data. Our final model improved to an MAE of 53.96. There is a 0.08 improvement in MAE. The r2 value of the original random forest was 0.325 and the r2 value of the final model was 0.326. An improvement in r2 scores amounts to the final model explaining the variance in price prediction 0.01% better than the original. This means that the two models explain nearly equal amounts of variability in the independent variable price. The final model achieves a small improvement in MAE and r2 score while being simpler than the original model. The feature selection process led us to drop the “shared rooms” variable giving the model a simpler feature set to train on. The hyperparameter tuning led to the number of estimators in the random forest being dropped from the default value of 100 down to 40. Ensembles with fewer estimators are less likely to overfit to training data so the improvement of our final model on the test set makes sense. The parsimony principle states that when evaluating models the simplest model is preferred. Our final model is both simpler and more accurate than the original random forest.

### Ethics & Privacy

Our project does not raise any glaringly obvious ethical or privacy related concerns. Airbnb is a platform in the public domain and the data set we are using is from Kaggle, another open data set platform. Since we are not gathering any personal data on the individuals that own the properties in the data set or any other personally identifiable data (hosts are anonymous in this data set), informed consent is not needed for our project and is not a concern. Using the aforementioned methods to collect the data we are using for this project, we do not have any concerns about immoral data collection methods or any personally identifiable data being used which ensures privacy of the Airbnb host’s whose listings were collected in the data set. There is potential for bias when predicting the prices of Airbnb in certain locations, due to historical policies in certain communities, which may have correlated to different prices than other similarly featured listings in different areas. If these historical policies play a factor in our models prediction we will make sure to eliminate these biases from our model and algorithms. One possible ethical concern is, if deployed into the real world, how the model could affect the housing and rental market. There is potential for the model to make the market in cities used for data more competitive and potentially expose overpriced airbnb’s. This could result in a loss of business for the hosts of these airbnb and by extension a reduction in income which could affect their standard of living. Beyond this scenario which could only be encountered if the model went into full production, we do not see any other concerns with our project. Overall our project has very few concerns, thus we have followed and considered all factors and guidelines to create an ethical model.

# Conclusion

Our project was designed to predict airbnb prices given data about the Airbnb listings themselves. We used a random forest regressor model and compared it to a decision tree model and found that the random forest model yielded better performance. After selecting features, tuning hyperparameters and comparing models, we found that generally, the random forest model was a better model to predict prices than the decision tree model was. The topic of this project, predicting Airbnb prices based on individual listing attributes, fits into other work in the field of price prediction and cognitive science as a whole. Cognitive Science deals a lot with why people think and behave the way they do, in this case, it could be looked into why certain features affect price more than others, and why people tend to value those features more than others. If research were to continue on this topic, it is possible that researchers could look into the impact that Airbnb prices have on overall tourism and monetary profit for cities. This would further extend the usefulness of investigating Airbnb prices and by extension our model, as well as showing the influence of the industry overall.



## Footnotes

<a name="Barronnote"></a>1.[^](#Barron): Barron, K., Kung, E., & Proserpio, D. (2021, September 17). Research: When airbnb listings in a city increase, so do rent prices. *Harvard Business Review*. https://hbr.org/2019/04/research-when-airbnb-listings-in-a-city-increase-so-do-rent-prices<br> 
<a name="Kerennote"></a>2.[^](#Keren): Keren Horn, Mark Merante, Is home sharing driving up rents? Evidence from Airbnb in Boston, *Journal of Housing Economics*, Volume 38, 2017, Pages 14-24, ISSN 1051-1377, https://doi.org/10.1016/j.jhe.2017.08.002.<br>
<a name="Pereznote"></a>3. [^](#Perez): Perez-Sanchez, V., Serrano-Estrada, L., Marti, P., & Mora-Garcia, R.-T. (2018). The What, Where, and Why of Airbnb Price Determinants. *Sustainability*, 10(12), 4596. https://doi.org/10.3390/su10124596 <br>
<a name="Nguyennote"></a>4.[^](#Nguyen): Nguyen, Quynh, "A Study of Airbnb as a Potential Competitor of the Hotel Industry" (2014). *UNLV Theses, Dissertations, Professional Papers, and Capstones*. 2618. http://dx.doi.org/10.34917/8349601lem <br>
<a name="Gutierreznote"></a>5.[^](#Gutierrez): Gutierrez, J., Carlos, J., Romanillos, G., & Henar, M. (2016). Airbnb in tourist cities: Comparing spatial patterns of hotels and peer-to-peer accommodation. *ArXiv*. https://doi.org/10.1016/j.tourman.2017.05.003 <br>
<a name="Sansnote"></a>6.[^](#Sans): Sans, A. A., & Domínguez, A. Q. (2016, May 3). 13. Unravelling airbnb: Urban perspectives from Barcelona. De Gruyter. https://www.degruyter.com/document/doi/10.21832/9781845415709-015/html <br>

### Other original contributors:

Rio Aguina-Kang: Conducted ANOVA statistical analysis and footnotes section. 

Vignesh Jayananth: Wrote first draft of discussion section.

Christopher Rochez: Found dataset on Kaggle.
