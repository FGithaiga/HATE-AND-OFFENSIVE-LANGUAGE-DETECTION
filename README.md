# HATE AND OFFENSIVE LANGUAGE DETECTION

# PROJECT OVERVIEW

Online hate speech and offensive language are increasing on social media, creating hostile environments and negatively affecting mental health. This project aims to develop a machine learning model that accurately detects hate speech and offensive language in tweets. Using a labeled dataset of 24,783 tweets, natural language processing techniques such as tokenization, stopword removal, and vectorization are applied. Supervised learning algorithms, including Logistic Regression and XGBoost, are trained to classify tweets into hate speech, offensive, or neutral categories. The resulting solution can help social media platforms, researchers, and policymakers reduce harmful content, improve user experience, and promote safer online interactions.

# PROBLEM STATEMENT


Online hate speech and offensive language on social media are increasing, exposing users to harmful content and creating hostile environments. Existing moderation methods, both manual and automated, are often slow or inaccurate, leaving a gap in effective detection. This project seeks to address this problem by developing a machine learning model that can automatically identify hate speech and offensive language in tweets, helping platforms maintain safer online communities.

# **OBJECTIVES**

1. To identify linguistic patterns and characteristics associated with harmful content in tweets, including hate speech and offensive language.

2. To classify tweets as harmful or non-harmful .

3. To analyze linguistic feature and trends that distinguish harmful tweets from non-harmful,supporting automated content moderation. 

4. To assess the frequency and distribution of hate speech across the dataset.


# DATA UNDERSTANDING

**1. Dataset Overview**

The dataset contains 24,783 entries and 6 columns.

Each row represents a single tweet along with its classification counts and category.

The dataset is complete; there are no missing values in any column.


**2. Columns Description**
Column Name	Data Type	Description
count	int64	Total count associated with the tweet (context-specific, usually sum of other counts).

hate_speech_count	int64	Count indicating hate speech instances in the tweet.

offensive_language_count	int64	Count indicating offensive language in the tweet.

neither_count	int64	Count of words that are neither hate speech nor offensive.

class	int64	Label representing the type of tweet: 0 = Neutral, 1 = Offensive, 2 = Hate Speech.

tweet	object	The raw text of the tweet.


**3. Data Types and Memory Usage**

Numeric columns: count, hate_speech_count, offensive_language_count, neither_count, class.

Text column: tweet.

Memory usage is approximately 1.1 MB, which is manageable for analysis in pandas.

**4. Initial Observations**

All columns are non-null, so no imputation is required.

The dataset is imbalanced, as there are likely more neutral tweets than hate speech or offensive tweets (this can be confirmed with a class distribution plot).

The tweet column will require text preprocessing (tokenization, stopword removal, etc.) for NLP tasks.

The counts (hate_speech_count, offensive_language_count, neither_count) could be used for feature engineering or exploratory analysis.


# DATA ANALYSIS
![alt text](Images/image.png)


![alt text](Images/image-1.png)


![alt text](Images/image-2.png)


![alt text](Images/image-3.png)


![alt text](Images/image-4.png)


# DATA MODELING



# CONCLUSIONS



# RECOMMENDATIONS