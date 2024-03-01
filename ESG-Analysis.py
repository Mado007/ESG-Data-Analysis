import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from fbprophet import Prophet

# Load the CSV files into Pandas DataFrames
esg_country_series_df = pd.read_csv('data/ESGCountry-Series.csv')
esg_country_df = pd.read_csv('data/ESGCountry.csv')
esg_data_df = pd.read_csv('data/ESGData.csv')
esg_footnote_df = pd.read_csv('data/ESGFootNote.csv')
esg_series_time_df = pd.read_csv('data/ESGSeries-Time.csv')
esg_series_df = pd.read_csv('data/ESGSeries.csv')

# Perform data cleaning
# Remove missing values
esg_data_cleaned = esg_data_df.dropna()

# Data exploration and visualization
# Plotting histogram of ESG scores using seaborn
sns.histplot(esg_data_cleaned['ESG Score'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('ESG Score')
plt.ylabel('Frequency')
plt.title('Distribution of ESG Scores')
plt.show()

# Interactive histogram using Plotly
fig = px.histogram(esg_data_cleaned, x='ESG Score', nbins=20, title='Distribution of ESG Scores')
fig.show()

# Statistical analysis
mean_esg_score = esg_data_cleaned['ESG Score'].mean()
median_esg_score = esg_data_cleaned['ESG Score'].median()
print("Mean ESG Score:", mean_esg_score)
print("Median ESG Score:", median_esg_score)

# Correlation analysis using seaborn
correlation_matrix = esg_data_cleaned.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Advanced Analysis and Visualization

# Time Series Analysis
# Assuming ESGSeries-Time.csv contains time-related data
# You can analyze trends and patterns over time

# Example: Plotting time series data
esg_series_time_df['Date'] = pd.to_datetime(esg_series_time_df['Date'])  # Convert Date column to datetime
plt.plot(esg_series_time_df['Date'], esg_series_time_df['ESG Score'], marker='o', color='blue', linestyle='-')
plt.xlabel('Date')
plt.ylabel('ESG Score')
plt.title('ESG Score Over Time')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Further Advanced Analysis
# You can perform more advanced statistical tests, machine learning models, or predictive analytics
# For example, you can use time series forecasting models like ARIMA or Prophet to predict future ESG scores based on historical data

# Example: Time Series Forecasting with Prophet

# Prepare the data for Prophet
esg_prophet_data = esg_series_time_df.rename(columns={'Date': 'ds', 'ESG Score': 'y'})

# Initialize and fit the Prophet model
prophet_model = Prophet()
prophet_model.fit(esg_prophet_data)

# Make future predictions
future_dates = prophet_model.make_future_dataframe(periods=365)
forecast = prophet_model.predict(future_dates)

# Plot the forecast
prophet_model.plot(forecast, xlabel='Date', ylabel='ESG Score')
plt.title('ESG Score Forecast with Prophet')
plt.show()

# Further Customization and Exploration
# You can customize visualizations, explore relationships between different variables, and derive insights based on domain knowledge

# Example: Customizing Visualizations
sns.boxplot(x='Region', y='ESG Score', data=esg_data_cleaned)
plt.title('ESG Score Distribution by Region')
plt.xlabel('Region')
plt.ylabel('ESG Score')
plt.xticks(rotation=45)
plt.show()

# Example: Exploring Relationships
sns.scatterplot(x='Environmental Score', y='Social Score', data=esg_data_cleaned, hue='Governance Score', palette='coolwarm')
plt.title('Relationship between Environmental and Social Scores')
plt.xlabel('Environmental Score')
plt.ylabel('Social Score')
plt.show()