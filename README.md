# Economic Data Analysis for Future Forecasting

## Overview

This Python project provides a Graphical User Interface (GUI) application for analyzing and visualizing global economy indicators data. It allows users to perform various data manipulation tasks, statistical analysis, and generate insightful visualizations.
This project aims to analyze historical economic data from various countries spanning from 1970 to 2021 to make educated guesses about their future economic trends. By examining key economic indicators such as GDP, GNI, population, exchange rates, and trade data (exports and imports), the project seeks to enhance understanding of past economic performance and provide insights into potential future scenarios.

## Objectives

1. **Graphical and Tabular Representation:** Utilize graphical and tabular methods to visually represent economic data, enhancing accessibility and interpretability of trends and patterns.

2. **Descriptive Statistical Measures:** Employ descriptive statistical measures to quantify and summarize key features of the economic data, providing a comprehensive overview of central tendencies and variability.

3. **Probability Methods/Distribution:** Apply probability methods and distributions to model uncertainty and assess the likelihood of different economic outcomes, contributing to a nuanced understanding of future scenarios.

4. **Regression Modeling and Predictions:** Implement regression modeling techniques to capture relationships between economic variables and make predictions about future GDP values for the selected countries.

5. **Confidence Interval of Descriptive Measures:** Calculate confidence intervals for both descriptive measures and regression estimates, providing a measure of precision and uncertainty around key statistical findings.

## Data Description
The dataset includes essential economic variables for various countries spanning the years 1970 to 2021. Here's a brief overview of the key variables:
- **Country Name:** Categorical identifier for each country.
- **Year:** Temporal context for the data.
- **AMA (Age-Adjusted Mortality Rate):** A crucial health metric indicating mortality rates.
- **IMF Rate:** Exchange rate reported by the International Monetary Fund.
- **Population:** Total residents of the country.
- **GNI (Gross National Income):** Measure of a country's income.
- **GDP (Gross Domestic Product):** Comprehensive economic indicator representing the total value of goods and services produced within a country.
- **Exports and Imports:** Value of goods and services traded internationally.

## Approach
1. **Data Preprocessing:** Clean and preprocess the dataset, handling missing values and outliers.
2. **Exploratory Data Analysis (EDA):** Explore the dataset using graphical and tabular methods to identify trends, patterns, and outliers.
3. **Statistical Analysis:** Calculate descriptive statistics to summarize key features of the data and assess its distribution.
4. **Regression Modeling:** Build regression models to predict future GDP values based on historical data and other economic indicators.
5. **Probability Analysis:** Use probability methods to model uncertainty and assess the likelihood of different economic outcomes.
6. **Visualization:** Visualize the analysis results using graphs and charts to facilitate interpretation and communication of findings.

## Conclusion
By conducting thorough analysis and modeling of historical economic data, this project aims to provide valuable insights into future economic trends and potential scenarios for different countries. The results can inform decision-making processes in various fields, including finance, policymaking, and international trade.

## Data Source
The dataset used for this project can be found on Kaggle: [Link to Dataset](https://www.kaggle.com/code/scratchpad/notebook26ce71c4a9/edit)

---

**Usage:**

To use this application:
1. Ensure that you have Python installed on your system.
2. Install the necessary libraries using `pip install -r requirements.txt`.
3. Run the application by executing the `main.py` file.
4. Explore different functionalities such as graphical representation, descriptive statistics, probability analysis, regression modeling, and confidence interval calculation.

**Note:** Ensure that the dataset file "Global Economy Indicators.csv" is placed in the same directory as the application files.
