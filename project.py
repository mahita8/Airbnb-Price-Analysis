import base64
import time
import io
import sys
import warnings

# Dash-related imports
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_bio as dashbio

# Plotly and visualization imports
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Data manipulation and analysis
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde, shapiro, normaltest, kstest, boxcox, probplot
from scipy.interpolate import griddata
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.subplots as sp
from sklearn.ensemble import RandomForestClassifier
from statsmodels.graphics.gofplots import qqplot

import seaborn as sns
from prettytable import PrettyTable


# PHASE 1
df1 = pd.read_csv("C:/Users/mohin/OneDrive/Desktop/infoviz/Scripts/finalproject/Airbnb_Data.csv")


np.random.seed(5764)
np.set_printoptions(precision=2)
pd.set_option('display.precision', 2)

# EXPLORING DATA
df1 = pd.read_csv("C:/Users/mohin/OneDrive/Desktop/infoviz/Scripts/finalproject/Airbnb_Data.csv")



# printing table
def display_pretty_table(data, title):
    table = PrettyTable()
    table.title = title
    table.field_names = data.columns.tolist()

    for _, row in data.iterrows():
        row_rounded = [round(val, 2) if isinstance(val, (int, float)) else val for val in row.tolist()]
        table.add_row(row_rounded)

    return table

# First 5 rows
print("First 5 rows:")
first_5_table = display_pretty_table(df1.head(), "First 5 Rows")
print(first_5_table)

# # Last 5 rows
print("\nLast 5 rows:")
last_5_table = display_pretty_table(df1.tail(), "Last 5 Rows")
print(last_5_table)
print("\n")

print("Total number of rows and columns:")
print(df1.shape)
print(f"No of rows:{df1.shape[0]}")
print(f"No of columns:{df1.shape[1]}\n")

#columns list
print("List of columns:")
print(df1.columns)
print("\n")

warnings.filterwarnings("ignore")

#table information
info_data = []
for column in df1.columns:
    non_null_count = df1[column].count()
    data_type = df1[column].dtype
    info_data.append([column, non_null_count, data_type])
info_df1 = pd.DataFrame(info_data, columns=["Column Name", "Non-Null Count", "Data Type"])
info_table = display_pretty_table(info_df1, "Information About the Data")
print(info_table)

#datatypes
dtypes_data = []
for column in df1.columns:
    dtype = df1[column].dtype
    dtypes_data.append([column, dtype])
dtypes_df1 = pd.DataFrame(dtypes_data, columns=["Column Name", "Data Type"])
dtypes_table = display_pretty_table(dtypes_df1, "Data Types of Each Column")
print(dtypes_table)

# data description
describe_df1=df1.describe()
describe_df1.insert(0, 'Statistic', describe_df1.index)
describe_table = display_pretty_table(describe_df1, "Summary Statistics")
print(describe_table)

# DATA CLEANING
# checking for null values
isna_table = PrettyTable()
isna_table.title = "Missing Values (df1.isna())"
isna_table.field_names = df1.columns.tolist()
for _, row in df1.isna().head(10).iterrows():
    row_display = [str(val) if isinstance(val, bool) else val for val in row.tolist()]
    isna_table.add_row(row_display)

print(isna_table)
# sum of missing values
print("\nSum of Missing Values (per column):")
missing_values_sum = df1.isna().sum().reset_index()
missing_values_sum.columns = ['Column Name', 'Missing Count']
missing_values_table = display_pretty_table(missing_values_sum, "Missing Values Count")
print(missing_values_table)

# dropping unnecessary columns
df1.drop(['thumbnail_url','name','description','amenities','latitude','longitude','zipcode'], axis=1, inplace=True)
print("After dropping columns:")
new_table = display_pretty_table(df1.head(), "Preview of new table")
print(new_table)

#final list of columns
print("List of columns:")
print(df1.columns)
print("\n")
print("Number of rows and columns of new table after dropping columns :")
print(df1.shape)
print(f"No of rows:{df1.shape[0]}")
print(f"No of columns:{df1.shape[1]}\n")

#adding extra required columns
df1['price']= np.exp(df1['log_price'])
df1['host_since']=pd.to_datetime(df1['host_since'])
df1['host_since_year']=df1['host_since'].dt.year
df1['property_group'] = df1['property_type'].replace({
    'Apartment': 'Residential', 'House': 'Residential', 'Condominium': 'Residential',
    'Loft': 'Residential', 'Townhouse': 'Residential', 'Guesthouse': 'Residential',
    'Villa': 'Residential', 'Bungalow': 'Residential', 'Dorm': 'Other',
    'Hostel': 'Other', 'Bed & Breakfast': 'Other', 'Guest suite': 'Other',
    'Other': 'Other', 'Camper/RV': 'Other', 'Boutique hotel': 'Hotel', 'Timeshare': 'Hotel',
    'In-law': 'Other', 'Boat': 'Other', 'Serviced apartment': 'Residential', 'Castle': 'Unique',
    'Cabin': 'Residential', 'Treehouse': 'Unique', 'Tipi': 'Unique', 'Vacation home': 'Residential',
    'Tent': 'Other', 'Hut': 'Other', 'Casa particular': 'Other', 'Chalet': 'Residential',
    'Yurt': 'Unique', 'Earth House': 'Unique', 'Parking Space': 'Other', 'Train': 'Unique',
    'Cave': 'Unique', 'Lighthouse': 'Unique', 'Island': 'Unique'
})
df1['first_review']=pd.to_datetime(df1['first_review'])
df1['last_review']=pd.to_datetime(df1['last_review'])
df1['years_between'] = (df1['last_review'] - df1['first_review']).dt.days / 365.25
df1['reviews_per_year'] = df1['number_of_reviews'] / df1['years_between']

# Fill missing rows and dropping rows
df1['bathrooms'] = df1['bathrooms'].fillna(df1['bathrooms'].median())
df1['bedrooms'] = df1['bedrooms'].fillna(df1['bedrooms'].median())
df1['beds'] = df1['beds'].fillna(df1['beds'].median())
df1['review_scores_rating'] = df1['review_scores_rating'].fillna(df1['review_scores_rating'].median())
df1['host_identity_verified']=df1['host_identity_verified'].fillna('f')
df1['host_has_profile_pic']=df1['host_has_profile_pic'].fillna('f')
df1['neighbourhood'] = df1['neighbourhood'].fillna(df1['neighbourhood'].mode()[0])
df1['first_review'] = df1['first_review'].ffill()
df1['last_review'] = df1['last_review'].ffill()
df1['host_response_rate'].fillna("0%", inplace=True)
df1['reviews_per_year'] = df1['reviews_per_year'].fillna(df1['reviews_per_year'].mean())
df1['years_between'] = df1['years_between'].fillna(df1['years_between'].median())

# typecasting
df1['host_response_rate'] = df1['host_response_rate'].str.rstrip('%').astype(float)
df1=df1.dropna(subset=['host_since_year'])
df1['host_since_year']=df1['host_since_year'].astype(int)




#final list of columns
print("Final List of columns:")
print(df1.columns)
print("\n")
print("Total Number of rows and columns of new table:")
print(df1.shape)
print(f"No of rows:{df1.shape[0]}")
print(f"No of columns:{df1.shape[1]}\n")

print("Cleaned dataset: ")
first_5_cleanedtable = display_pretty_table(df1.head(), "First 5 Rows")
print(first_5_cleanedtable)

print("\nSum of Missing Values after filling and dropping missing rows:")
missing_values_sum = df1.isna().sum().reset_index()
missing_values_sum.columns = ['Column Name', 'Missing Count']
missing_values_table = display_pretty_table(missing_values_sum, "Missing Values Count")
print(missing_values_table)

# print("Value counts for each column: \n")
columns= ['log_price', 'property_type', 'room_type', 'accommodates',
       'bathrooms', 'bed_type', 'cancellation_policy', 'cleaning_fee', 'city',
       'first_review', 'host_has_profile_pic', 'host_identity_verified',
       'host_response_rate', 'host_since', 'instant_bookable', 'last_review',
       'neighbourhood', 'number_of_reviews', 'review_scores_rating',
       'bedrooms', 'beds', 'price', 'host_since_year']

# Removing outliers
columns_to_check = [
    'log_price','price'
]

for column in columns_to_check:
    Q1 = df1[column].quantile(0.25)
    Q3 = df1[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_val = Q1 - 1.5 * IQR
    upper_val = Q3 + 1.5 * IQR

    print(f"--- Column: {column} ---")
    print(f"Q1 for {column} is: {Q1:.2f}")
    print(f"Q3 for {column} is: {Q3:.2f}")
    print(f"IQR for {column} is: {IQR:.2f}")
    print(f"Lower bound for {column} is: {lower_val:.2f}")
    print(f"Upper bound for {column} is: {upper_val:.2f}")
    print(f"Any {column} < {lower_val:.2f} or {column} > {upper_val:.2f} is an outlier.")
    print(f"Length of dataset before removing {column} outliers: {df1.shape[0]}")


    df1 = df1[(df1[column] >= lower_val) & (df1[column] <= upper_val)]

    print(f"Length of dataset after removing {column} outliers: {df1.shape[0]}")
    print("\n")

cols_to_check = [
     'accommodates', 'bathrooms', 'number_of_reviews',
     'bedrooms', 'beds'
]

for column in cols_to_check:
    Q1 = df1[column].quantile(0.25)
    Q3 = df1[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_val = Q1 - 1.5 * IQR
    upper_val = Q3 + 1.5 * IQR

    print(f"--- Column: {column} ---")
    print(f"Q1 for {column} is: {Q1:.2f}")
    print(f"Q3 for {column} is: {Q3:.2f}")
    print(f"IQR for {column} is: {IQR:.2f}")
    print(f"Lower bound for {column} is: {lower_val:.2f}")
    print(f"Upper bound for {column} is: {upper_val:.2f}")
    print(f"Any {column} < {lower_val:.2f} or {column} > {upper_val:.2f} is an outlier.")
    print(f"Length of dataset before removing {column} outliers: {df1.shape[0]}")


    df1 = df1[(df1[column] >= lower_val)]

    print(f"Length of dataset after removing {column} outliers: {df1.shape[0]}")
    print("\n")


# normality test
price_s,price_pvalue= shapiro(df1['price'])
price_res= 'Normal Distribution' if price_pvalue> 0.01 else 'Not Normal Distribution'

print(f"Shapiro Test: Statistics = {price_s:.2f}, p-value ={price_pvalue:.2f}"),
print(f"Shapiro Test : Price is {price_res}")

plt.figure(figsize=(8, 6))
df1['price'] = pd.to_numeric(df1['price'], errors='coerce')
stats.probplot(df1['price'], dist="norm", plot=plt)
plt.title('QQ Plot of Price', fontdict={'family': 'serif', 'color': 'blue', 'fontsize': 16})
plt.xlabel('Theoretical Quantiles', fontdict={'family': 'serif', 'color': 'darkred', 'fontsize': 14})
plt.ylabel('Sample Quantiles', fontdict={'family': 'serif', 'color': 'darkred', 'fontsize': 14})
plt.grid()
plt.tight_layout()
plt.show()

#PCA
df1_numeric = df1.select_dtypes(include=[np.number]).dropna()
df1_numeric = df1_numeric[~df1_numeric.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1_scaled = scaler.fit_transform(df1_numeric)
pca = PCA()
pca_components = pca.fit_transform(df1_scaled)                                                                                                                                                                                                                                                     
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
condition_number = np.linalg.cond(df1_scaled)
singular_values = pca.singular_values_

# Observations
print(f"Explained Variance Ratio: {explained_variance}")
print(f"Cumulative Variance: {cumulative_variance}")
print(f"Condition Number: {condition_number}")
print(f"Singular Values: {singular_values}")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Explained Variance Ratio by Component', fontdict={'family': 'serif', 'color': 'blue', 'fontsize': 16})
plt.xlabel('Principal Component', fontdict={'family': 'serif', 'color': 'darkred', 'fontsize': 14})
plt.ylabel('Explained Variance Ratio', fontdict={'family': 'serif', 'color': 'darkred', 'fontsize': 14})

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='orange')
plt.title('Cumulative Explained Variance', fontdict={'family': 'serif', 'color': 'blue', 'fontsize': 16})
plt.xlabel('Number of Components', fontdict={'family': 'serif', 'color': 'darkred', 'fontsize': 14})
plt.ylabel('Cumulative Explained Variance', fontdict={'family': 'serif', 'color': 'darkred', 'fontsize': 14})
plt.tight_layout()
plt.show()

# data transformation
print(df1['log_price'])

# price column distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df1, x='price', kde=True, bins=50, color="skyblue", label='Number of Reviews')
plt.title('Histogram with KDE- Price Distribution', fontdict={'family': 'serif', 'color': 'blue', 'size': 16})
plt.xlabel('Price', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
plt.ylabel('Density', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
plt.grid()
plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.show()

# logprice column distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df1, x='log_price', kde=True, bins=50, color="skyblue", label='Number of Reviews')
plt.title('Histogram with KDE- Log Price Distribution', fontdict={'family': 'serif', 'color': 'blue', 'size': 16})
plt.xlabel('Log price', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
plt.ylabel('Density', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
plt.grid()
plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.show()

# heatmap & pearson correlation coefficient matrix
numdata = df1[['log_price','accommodates', 'bathrooms','beds',
             'bedrooms','number_of_reviews','review_scores_rating','years_between','reviews_per_year']]

df2 = pd.DataFrame(numdata)
correlation_matrix = df2.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Pearson Correlation Coefficient Matrix", fontsize=16)
plt.show()

# statistics

print("heloooooo")
print(df1.isna().sum())
print(df1.head())
describe_df1=df1.describe()
describe_df1.insert(0, 'Statistic', describe_df1.index)
describe_table = display_pretty_table(describe_df1, "Summary Statistics")
print(describe_table)
#
# plt.figure(figsize=(8, 6))
# sns.kdeplot(df1['log_price'], fill=True, color='blue', alpha=0.7)
# plt.title('Kernel Density Estimate of Log-Transformed Prices', fontdict={'family': 'serif', 'color': 'blue', 'fontsize': 16})
# plt.xlabel('Log of Price', fontdict={'family': 'serif', 'color': 'darkred', 'fontsize': 14})
# plt.ylabel('Density', fontdict={'family': 'serif', 'color': 'darkred', 'fontsize': 14})
# plt.grid()
# plt.tight_layout()
# plt.show()

# STATIC PLOTS

# Plot Area Plot
df1_grouped = df1.groupby('host_since_year')[['log_price', 'accommodates', 'number_of_reviews']].mean()
plt.figure(figsize=(10, 6))
# Plotting the area plots for the three columns
plt.fill_between(df1_grouped.index, df1_grouped['log_price'], color='skyblue', alpha=0.4, label='Log Price')
plt.fill_between(df1_grouped.index, df1_grouped['accommodates'], color='orange', alpha=0.4, label='Accommodates')
plt.fill_between(df1_grouped.index, df1_grouped['number_of_reviews'], color='green', alpha=0.4, label='Number of Reviews')
plt.title('Area Plot of Log Price, Accommodates, and Number of Reviews vs Host Since Year', fontdict={'fontname': 'serif', 'fontsize': 16, 'color': 'blue'})
plt.xlabel('Year Host Joined')
plt.ylabel('Average Values')
plt.legend(loc='upper left')
plt.show()

# countplot
plt.figure(figsize=(10, 8))
sns.countplot(data=df1, x='city', palette="Set1", hue='city', order=df1['city'].value_counts().index)
plt.title("Countplot of Number of Airbnbs per City", fontdict={'fontname': 'serif', 'fontsize': 16, 'color': 'blue'})
plt.xlabel("City", fontdict={'fontname': 'serif', 'fontsize': 14, 'color': 'darkred'})
plt.ylabel("Count", fontdict={'fontname': 'serif', 'fontsize': 14, 'color': 'darkred'})
plt.grid()
plt.legend(title='City', loc='upper right', fontsize=12, title_fontsize=14)
plt.tight_layout()
plt.show()

#distplot
plt.figure(figsize=(10, 6))
sns.distplot(df1.log_price, color="dodgerblue", label='Price Distribution')
plt.title(' Distplot of Price Distribution', fontdict={'family': 'serif', 'color': 'blue', 'size': 16})
plt.xlabel('Price', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
plt.ylabel('Count', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
plt.grid()
plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.show()


# pie chart
types = df1["room_type"].value_counts()
plt.figure(figsize=(10, 8))
plt.pie(
    df1['room_type'].value_counts(),
    labels=types.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette("Set3"),
    textprops={'color': 'red'}
)
plt.title("Pie Chart Room Type Composition", fontdict={'family': 'serif', 'color': 'blue', 'size': 16})
plt.legend(title="Room Types", loc='upper right', fontsize=12, title_fontsize=14)
plt.tight_layout()
plt.show()

# histogram with kde
plt.figure(figsize=(10, 6))
sns.histplot(data=df1, x='number_of_reviews', kde=True, bins=50, color="skyblue", label='Number of Reviews')
plt.title('Histogram with KDE- Number of Reviews Distribution', fontdict={'family': 'serif', 'color': 'blue', 'size': 16})
plt.xlabel('Number of Reviews', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
plt.ylabel('Density', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
plt.grid()
plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.show()

#kde with fill
plt.figure(figsize=(8, 6))
sns.kdeplot(data=df1, x="review_scores_rating", fill=True, color="lightcoral", label="Ratings Distribution")
plt.title(' KDE with fill - Ratings Distribution', fontdict={'family': 'serif', 'color': 'blue', 'size': 16})
plt.xlabel('Ratings', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
plt.ylabel('Density', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
plt.grid(True)
plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.show()


#regplot
plt.figure(figsize=(8, 6))
sns.regplot(data=df1, x='accommodates', y='log_price', scatter_kws={'color': 'darkred'}, line_kws={'color': 'blue', 'linewidth': 2})
plt.title('Regplot of Accommodates vs Log Price', fontdict={'family': 'serif', 'color': 'blue', 'size': 16})
plt.xlabel('Accommodates', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
plt.ylabel('Log Price', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
plt.legend(['Regression Line', 'Data Points'], loc='upper left', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

#lmplot
plt.figure(figsize=(8, 6))
sns.lmplot(data=df1, x="beds", y="log_price", col="room_type", height=4, aspect=1, scatter_kws={'color': 'darkred'}, line_kws={'color': 'blue', 'linewidth': 2})
plt.subplots_adjust(top=0.85)
plt.suptitle('Lm plot of Beds vs Log Price by Room Type', fontdict={'family': 'serif', 'color': 'blue', 'fontsize': 16})
for ax in plt.gcf().axes:
    ax.set_xlabel('Beds', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
    ax.set_ylabel('Log Price', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
    ax.grid(True)
plt.gcf().axes[0].legend(['Regression Line', 'Data Points'], loc='upper left', fontsize=12)
plt.tight_layout()
plt.show()

#stripplot
plt.figure(figsize=(8, 6))
sns.stripplot(data=df1, x="room_type", y="log_price", hue="cleaning_fee", jitter=True, dodge=True, palette="Set2")
plt.title('Strip plot og Log Price by Room Type with Cleaning Fee', fontdict={'family': 'serif', 'color': 'blue', 'fontsize': 16})
plt.xlabel('Room Type', fontdict={'family': 'serif', 'color': 'darkred', 'fontsize': 14})
plt.ylabel('Log Price', fontdict={'family': 'serif', 'color': 'darkred', 'fontsize': 14})
plt.grid(True)
plt.legend(title='Cleaning Fee', loc='upper left', fontsize=12)
plt.tight_layout()
plt.show()

#violinplot
plt.figure(figsize=(8,6))
sns.violinplot(data=df1, x='room_type', y='log_price')
plt.title('Violin plot of Price Distribution', fontdict={'family': 'serif', 'color': 'blue', 'size': 16})
plt.xlabel('Room Type', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
plt.ylabel('Price', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
plt.legend(title='Room Type', loc='upper right', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(16,12))

sns.violinplot(data=df1, x='room_type', y='log_price', ax=axes[0, 0], color='green')
axes[0, 0].set_title('Violin plot of Price Distribution of Room Type', fontdict={'family': 'serif', 'color': 'blue', 'size': 16})
axes[0, 0].set_xlabel('Room Type', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
axes[0, 0].set_ylabel('Price', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
axes[0, 0].legend(title='Room Type', loc='upper right', fontsize=12)
axes[0, 0].grid(True)

sns.violinplot(data=df1, x='bed_type', y='log_price', ax=axes[0, 1], color='green')
axes[0, 1].set_title('Violin plot of Price Distribution of Bed Type', fontdict={'family': 'serif', 'color': 'blue', 'size': 16})
axes[0, 1].set_xlabel('Bed Type', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
axes[0, 1].set_ylabel('Price', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
axes[0, 1].legend(title='Bed Type', loc='upper right', fontsize=12)
axes[0, 1].grid(True)

sns.violinplot(data=df1, x='cancellation_policy', y='log_price', ax=axes[1, 0], color='green')
axes[1, 0].set_title('Violin plot of Price Distribution of Cancellation Policy', fontdict={'family': 'serif', 'color': 'blue', 'size': 16})
axes[1, 0].set_xlabel('Cancellation Policy', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
axes[1, 0].set_ylabel('Price', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
axes[1, 0].legend(title='Cancellation Policy', loc='upper right', fontsize=12)
axes[1, 0].grid(True)

sns.violinplot(data=df1, x='city', y='log_price', ax=axes[1, 1], color='green')
axes[1, 1].set_title('Violin plot of Price Distribution of City', fontdict={'family': 'serif', 'color': 'blue', 'size': 16})
axes[1, 1].set_xlabel('City', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
axes[1, 1].set_ylabel('Price', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
axes[1, 1].legend(title='City', loc='upper right', fontsize=12)
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# pairplot
df1_x = df1[['log_price','accommodates','bathrooms','beds','number_of_reviews','review_scores_rating','host_response_rate']]
plt.figure(figsize=(8, 6))
pair_plot = sns.pairplot(data=df1_x)
pair_plot.fig.suptitle('Pairplot of Various Features', fontdict={'family': 'serif', 'color': 'blue', 'fontsize': 16}, y=1.02)
for ax in pair_plot.axes.flatten():
    ax.set_xlabel(ax.get_xlabel(), fontdict={'family': 'serif', 'color': 'darkred', 'fontsize': 14})
    ax.set_ylabel(ax.get_ylabel(), fontdict={'family': 'serif', 'color': 'darkred', 'fontsize': 14})
    ax.grid(True)
plt.title('Pairplot fo Features', fontdict={'family': 'serif', 'color': 'blue', 'fontsize': 16})
plt.tight_layout()
plt.show()


# heatmap with cbar
plt.figure(figsize=(8, 6))
correlation_matrix = df1[['bedrooms', 'beds', 'log_price', 'number_of_reviews', 'review_scores_rating','accommodates']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True)
plt.title('Heatmap of Correlations between Numerical Features', fontdict={'family': 'serif', 'color': 'blue', 'fontsize': 16})
plt.xlabel('Features', fontdict={'family': 'serif', 'color': 'darkred', 'fontsize': 14})
plt.ylabel('Features', fontdict={'family': 'serif', 'color': 'darkred', 'fontsize': 14})
plt.tight_layout()
plt.show()

# jointplot
plt.figure(figsize=(10, 6))
plot = sns.jointplot(
    data=df1,
    x="review_scores_rating",
    y="host_response_rate",
)
plot.ax_joint.grid(True)
plot.set_axis_labels('Review Scores Rating', 'Host Response Rate (%)', fontdict={'family': 'serif', 'color': 'darkred', 'fontsize': 14})
plot.plot_joint(sns.scatterplot, color='violet')
plot.ax_joint.legend(['Data Points'], loc='best')
plot.fig.suptitle('Joint plot of Relationship Between Review Scores and Host Response Rate', fontdict={'family': 'serif', 'color': 'blue', 'fontsize': 16})
plt.tight_layout()
plt.show()

print(df1.columns)
# #qqplot
plt.figure(figsize=(8, 6))
df1['price'] = pd.to_numeric(df1['price'], errors='coerce')
stats.probplot(df1['price'], dist="norm", plot=plt)
plt.title('QQ Plot of Price', fontdict={'family': 'serif', 'color': 'blue', 'fontsize': 16})
plt.xlabel('Theoretical Quantiles', fontdict={'family': 'serif', 'color': 'darkred', 'fontsize': 14})
plt.ylabel('Sample Quantiles', fontdict={'family': 'serif', 'color': 'darkred', 'fontsize': 14})
plt.grid()
plt.tight_layout()
plt.show()

# hexbin
plt.figure(figsize=(8, 6))
joint = sns.jointplot(x=df1['review_scores_rating'], y=df1['log_price'], kind="hex", color="#4CB391")
joint.fig.suptitle('Hexbin Plot of Review Scores vs Log Price', fontdict={'family': 'serif', 'color': 'blue', 'fontsize': 16})
joint.set_axis_labels('Review Scores Rating', 'Log Price', fontdict={'family': 'serif', 'color': 'darkred', 'fontsize': 14})
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


# boxenplot
plt.figure(figsize=(8, 6))
sns.boxenplot(x='log_price', y='property_group', data=df1, palette="Set2")
plt.title('Boxen plot of Log Price vs Property Group', fontdict={'family': 'serif', 'color': 'blue', 'fontsize': 16})
plt.xlabel('Log Price', fontdict={'family': 'serif', 'color': 'darkred', 'fontsize': 14})
plt.ylabel('Property Group', fontdict={'family': 'serif', 'color': 'darkred', 'fontsize': 14})
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

# barplot
# horizontal bar plot
top_property = df1['property_type'].value_counts()[:10]
filtered_df1 = df1[df1['property_type'].isin(top_property.index)]
plt.figure(figsize=(10, 8))
avg_price = filtered_df1.groupby('property_type')['price'].mean()
avg_price = avg_price.sort_values(ascending=False)
plt.barh(avg_price.index, avg_price.values)
plt.xlabel("Average Price", fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
plt.ylabel("Property Type", fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
plt.title("Barplot of Average Price by Property Type", fontdict={'family': 'serif', 'color': 'blue', 'size': 16})
plt.grid()
plt.tight_layout()
plt.show()

# stacked bar plot
plt.figure(figsize=(8, 6))
top_property_types = df1['property_type'].value_counts().nlargest(5).index
df1['property_type_grouped'] = df1['property_type'].apply(
    lambda x: x if x in top_property_types else 'Other'
)
room_type_property = pd.crosstab(df1['room_type'], df1['property_type_grouped'])
room_type_property.plot(kind='bar', stacked=True, figsize=(10, 6), cmap='viridis')
plt.title('Stacked Bar Chart: Room Type by Property Type (Top 5 Types)', fontdict={'family': 'serif', 'color': 'blue', 'fontsize': 16})
plt.xlabel('Room Type', fontdict={'family': 'serif', 'color': 'darkred', 'fontsize': 14})
plt.ylabel('Count of Properties', fontdict={'family': 'serif', 'color': 'darkred', 'fontsize': 14})
plt.xticks(rotation=45)
plt.legend(title='Property Type')
plt.grid()
plt.tight_layout()
plt.show()

# grouped bar chart
plt.figure(figsize=(10, 6))
sns.countplot(x='room_type', hue='cancellation_policy', data=df1, palette='Set2')
plt.title('Grouped bar chart of Number of Reviews by Room Type and Cancellation Policy', fontdict={'family': 'serif', 'color': 'blue', 'size': 16})
plt.xlabel('Room Type', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
plt.ylabel('Number of Reviews', fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
plt.legend(title='Cancellation Policy')
plt.grid()
plt.tight_layout()
plt.show()

# bar subplots
features = ['bathrooms', 'bedrooms', 'beds', 'accommodates']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, feature in enumerate(features):
    avg_price = df1.groupby(feature)['price'].mean().reset_index()
    sns.barplot(data=avg_price, x=feature, y='price', palette='viridis', ax=axes[i])
    axes[i].set_title(f"Average Price vs. {feature.capitalize()}", fontdict={'fontsize': 15, 'color': 'blue'})
    axes[i].set_xlabel(feature.capitalize(), fontdict={'fontsize': 12, 'color': 'darkred'})
    axes[i].set_ylabel("Average Price", fontdict={'fontsize': 12, 'color': 'darkred'})
    axes[i].grid(True)
plt.tight_layout()
plt.suptitle('Bar subplots')
plt.show()

# line plot
plt.figure(figsize=(10,6))
no_of_hosts=df1.groupby('host_since_year').size()
print(no_of_hosts.to_frame())
sns.lineplot(data=no_of_hosts.to_frame(), x=no_of_hosts.index, y=no_of_hosts.values)
plt.title('Line plot of Host Growth Over Time',  fontdict={'family': 'serif', 'color': 'blue', 'size': 16})
plt.xlabel('Host since Year',  fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
plt.ylabel('Number of Hosts',  fontdict={'family': 'serif', 'color': 'darkred', 'size': 14})
plt.tight_layout()
plt.grid()
plt.legend(title='Host Growth', loc='upper left')
plt.show()


# areaplot

df1['host_since_year'] = pd.to_datetime(df1['host_since']).dt.year
price_by_year_property = df1.groupby(['host_since_year', 'property_group'])['log_price'].mean().unstack()
plt.figure(figsize=(14, 10))
price_by_year_property.plot(kind='area', stacked=True, colormap='coolwarm', alpha=0.6)

plt.title('Are Plot of Average Log Price Over Time by Property Type', fontsize=15, fontname='serif', color='blue')
plt.xlabel('Year', fontsize=14, fontname='serif', color='darkred')
plt.ylabel('Average Log Price', fontsize=14, fontname='serif', color='darkred')

plt.gca().set_facecolor('w')
plt.xticks([int(i) for i in plt.xticks()[0]], rotation=45)
plt.yticks([round(i, 2) for i in plt.yticks()[0]])
plt.legend(title="Property Type", bbox_to_anchor=(1.1, 1), loc='upper left', fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()


# rugplot
plt.figure(figsize=(14, 10))
sns.kdeplot(data=df1, x='review_scores_rating', shade=True, color='blue', alpha=0.6, label='KDE Plot')
sns.rugplot(data=df1, x='review_scores_rating', color='darkred', label='Rug Plot')
plt.title('Rug Plot for Review Scores Rating', fontsize=18, fontname='serif', color='blue')
plt.xlabel('Review Scores Rating', fontsize=14, fontname='serif', color='darkred')
plt.ylabel('Density', fontsize=14, fontname='serif', color='darkred')
plt.grid()
plt.yticks([round(i, 2) for i in plt.yticks()[0]])
plt.legend(title='Legend', fontsize=12, loc='upper right')
plt.tight_layout()
plt.show()



# 3d and contour plot
data = df1[['bedrooms', 'bathrooms', 'log_price']].dropna()
x = np.linspace(data['bedrooms'].min(), data['bedrooms'].max(), 100)
y = np.linspace(data['bathrooms'].min(), data['bathrooms'].max(), 100)
X, Y = np.meshgrid(x, y)
Z = griddata(
    (data['bedrooms'], data['bathrooms']),
    data['log_price'],
    (X, Y),
    method='cubic'
)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.contour(X, Y, Z, zdir='x', offset=data['bedrooms'].min() - 1, cmap='viridis', linewidths=0.5)
ax.contour(X, Y, Z, zdir='y', offset=data['bathrooms'].min() - 1, cmap='viridis', linewidths=0.5)
ax.contour(X, Y, Z, zdir='z', offset=data['log_price'].min() - 1, cmap='viridis', linewidths=0.5)
ax.set_xlim(data['bedrooms'].min() - 1, data['bedrooms'].max())
ax.set_ylim(data['bathrooms'].min() - 1, data['bathrooms'].max())
ax.set_zlim(data['log_price'].min() - 1, data['log_price'].max())
ax.set_xlabel('Bedrooms', fontsize=12, family='serif', color='red')
ax.set_ylabel('Bathrooms', fontsize=12, family='serif', color='red')
ax.set_zlabel('Log Price', fontsize=12, family='serif', color='red')
ax.set_title(r'3D Surface Plot of Bedrooms, Bathrooms, and Log Price', fontsize=18, family='serif', color='blue')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()

# clustermap
ncols = [
    'review_scores_rating', 'accommodates', 'bathrooms',
]

df1_clustermap = df1[ncols].dropna()
df1_clustermap_sampled = df1_clustermap.sample(n=4000, random_state=42)

sns.clustermap(
    df1_clustermap_sampled,
    cmap='coolwarm',
    standard_scale=1,
    figsize=(12, 10),
    metric='euclidean',
    method='average',
)

plt.gcf().suptitle(
    'Clustermap of Raw Numerical Data',
    fontsize=20,
    fontfamily='serif',
    color='blue',
    y=1.02
)

plt.xlabel(
    'Features',
    fontsize=14,
    fontfamily='serif',
    color='darkred'
)

plt.ylabel(
    'Observations',
    fontsize=14,
    fontfamily='serif',
    color='darkred'
)

plt.show()

# Swarm Plot

plt.figure(figsize=(20, 15))
sample_df1 = df1.sample(n=2500, random_state=42)
sns.swarmplot(data=sample_df1, x='city', y='review_scores_rating', palette='coolwarm', size=4)

plt.title(
    'Swarmplot of Review Scores by Property Type',
    fontsize=20,
    fontfamily='serif',
    color='blue'
)

plt.xlabel(
    'Property Type',
    fontsize=14,
    fontfamily='serif',
    color='darkred'
)

plt.ylabel(
    'Review Scores Rating',
    fontsize=14,
    fontfamily='serif',
    color='darkred'
)

plt.grid(
    color='gray',
    linestyle='--',
    linewidth=0.5,
    axis='both'
)

plt.xticks(rotation=45)
plt.show()



# PHASE 2
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css','/assets/project.css']
app = Dash(name='project', external_stylesheets=external_stylesheets, suppress_callback_exceptions=True, title="Airbnb Price Analysis")

app.layout = html.Div([
    html.Br(),
    html.H1('Airbnb Price Analysis', id="maintitle"),

    dcc.Tabs(
        id='tabs',
        value='intro',
        children=[
            dcc.Tab(label='Loading Data', value='t1', className='custom-tab'),
            dcc.Tab(label='Data Cleaning', value='t2', className='custom-tab'),
            dcc.Tab(label='Data Transformation', value='t3', className='custom-tab'),
            dcc.Tab(label='Outlier Detection and Removal', value='t4', className='custom-tab'),
            dcc.Tab(label='Dimensionality Reduction [PCA]', value='t5', className='custom-tab'),
            dcc.Tab(label='Normality tests', value='t6', className='custom-tab'),
            dcc.Tab(label='Numerical plots', value='t7', className='custom-tab'),
            dcc.Tab(label='Categorical plots', value='t8', className='custom-tab'),
            dcc.Tab(label='Statistics', value='t9', className='custom-tab')
        ]
    ),


    html.Div(
        id="intro-content",
        children=[

            html.Div(
                id="introimgs",
                children=[
                    html.Img(src="finalproject/assets/tools-categories-XL_(2).jpg"),
                    html.Img(src="finalproject/assets/airbnb.webp")
                ],
                className="maintabs"
            ),

            html.Div(
                children=[
                    html.H2(
                        'Explore insights from Airbnb data, including pricing trends, host growth, and guest preferences, to uncover patterns that shape the short-term rental market.'
                    ),
                    html.H2(
                        'This analysis provides an interactive visualization of key metrics to enhance understanding of the Airbnb ecosystem.'
                    ),
                    html.H4(
                        'By examining geographic trends, seasonal variations, and guest reviews, the study aims to reveal critical factors influencing booking behaviors.'
                    ),
                    html.H5(
'Additionally, the analysis highlights host strategies and their impact on guest satisfaction, offering actionable insights for both hosts and market analysts.'
                    ),

                ]
            ),
        ]
    ),

    html.Div(id='layout')
])

@app.callback(
Output(component_id='intro-content', component_property='style'),
     Input(component_id='tabs', component_property='value')
)

def toggle_intro(tab_value):
    if tab_value == 'intro':
        return {'display': 'block'}
    return {'display': 'none'}

data = pd.read_csv("C:/Users/mohin/OneDrive/Desktop/infoviz/Scripts/finalproject/Airbnb_Data.csv")
df=data.copy()
df['price']= np.exp(df['log_price'])
df['host_response_rate'] = df['host_response_rate'].str.rstrip('%').astype(float)

df['host_since']=pd.to_datetime(df['host_since'])
df['host_since_year']=df['host_since'].dt.year
df=df.dropna(subset=['host_since_year'])
df['host_since_year']=df['host_since_year'].astype(int)

df['property_group'] = df['property_type'].replace({
    'Apartment': 'Residential', 'House': 'Residential', 'Condominium': 'Residential',
    'Loft': 'Residential', 'Townhouse': 'Residential', 'Guesthouse': 'Residential',
    'Villa': 'Residential', 'Bungalow': 'Residential', 'Dorm': 'Other',
    'Hostel': 'Other', 'Bed & Breakfast': 'Other', 'Guest suite': 'Other',
    'Other': 'Other', 'Camper/RV': 'Other', 'Boutique hotel': 'Hotel', 'Timeshare': 'Hotel',
    'In-law': 'Other', 'Boat': 'Other', 'Serviced apartment': 'Residential', 'Castle': 'Unique',
    'Cabin': 'Residential', 'Treehouse': 'Unique', 'Tipi': 'Unique', 'Vacation home': 'Residential',
    'Tent': 'Other', 'Hut': 'Other', 'Casa particular': 'Other', 'Chalet': 'Residential',
    'Yurt': 'Unique', 'Earth House': 'Unique', 'Parking Space': 'Other', 'Train': 'Unique',
    'Cave': 'Unique', 'Lighthouse': 'Unique', 'Island': 'Unique'
})

# LOADING DATA

# dataframe shape
buffer = io.StringIO()
data_shape=data.shape
print(data_shape)
buffer.write(f"Rows: {data_shape[0]}, Columns: {data_shape[1]}")
data_shape_string = buffer.getvalue()

# dataset info
buffer = io.StringIO()
data.info(buf=buffer)
print(data.info)
data_info = buffer.getvalue()

# loading data tab
t1layout = html.Div([
    html.Br(),
    html.H1("Original dataset: Click below to download",  className="heading"),
    html.Button("Download the dataset", id='btn-download'),
    html.Br(),
    dcc.Download(id="download_csv"),
    html.Br(),
    html.H1("About the dataset", className="heading"),
    html.H3("The dataset selected for this analysis contains comprehensive details on Airbnb listings from various cities." ),
    html.Br(),
    html.H1("The dataset contains: ", className="heading"),
    html.Pre(html.H3(data_shape_string)),
    html.Br(),
    html.Br(),
    html.H1("Dataset information: ", className="heading"),
    html.Pre(html.H3(data_info)),
    html.Br(),
    html.H1("Preview of the dataset", className="heading"),
    html.H4("Enter the number of rows you want to see"),
    dcc.Textarea(
        id='textarea-rows',
        value='',
        style={'width': '50%', 'height': 50, 'font-size': 30},
    ),
    html.Br(),
    html.Br(),
    dcc.Loading(
        id="loading-1",
        type="default",
        children=html.Div(id="loading-output-1")
    ),
    html.Br(),
    dash_table.DataTable(
        id="previewtable",
        columns=[{'name': col, 'id': col} for col in data.columns]
    )
], id="t1layout")

@app.callback(
    Output("loading-output-1", "children"),
    Input("textarea-rows", "value"))

def input_triggers_spinner(value):
    time.sleep(1)
    return html.Div([
        html.H4(f"Showing {value} rows")
    ])


@app.callback(
    Output('previewtable', 'data'),
    Input('textarea-rows', 'value')
)

def update_table(rows):
    if rows:
        rows=int(rows.strip())
        return data.head(rows).to_dict('records')
    else:
        return data.head().to_dict('records')


@app.callback(
    Output("download_csv", "data"),
    Input("btn-download", "n_clicks"),
    prevent_initial_call=True
)
def generate_csv(n_clicks):
    return dcc.send_data_frame(data.to_csv, "airbnb_data.csv")


missing_values = data.head(10).copy()
missing_values['id'] = missing_values.index

buffer = io.StringIO()
missing_info = data.isna().sum()
missingdata_info = missing_info.to_string()
buffer.write(missingdata_info)
missingdata_info_result = buffer.getvalue()
print("Missing Data Info String:\n", missingdata_info_result)

missing_cols=['bathrooms','beds','bedrooms','review_scores_rating']
# filling missing values
df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms'].median())
df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].median())
df['beds'] = df['beds'].fillna(df['beds'].median())
df['review_scores_rating'] = df['review_scores_rating'].fillna(df['review_scores_rating'].median())
df['host_identity_verified']=df['host_identity_verified'].fillna('f')
df['host_has_profile_pic']=df['host_has_profile_pic'].fillna('f')
df['neighbourhood'] = df['neighbourhood'].fillna(df['neighbourhood'].mode()[0])
df['first_review'] = df['first_review'].ffill()
df['last_review'] = df['last_review'].ffill()
df['review_scores_rating']=df['review_scores_rating'].fillna(df['review_scores_rating'].median())
df['host_response_rate'].fillna("0%", inplace=True)

buffer = io.StringIO()
filled_info = df.isna().sum()
filleddata_info = filled_info.to_string()
buffer.write(filleddata_info)
filleddata_result = buffer.getvalue()

df.drop(['thumbnail_url','name','description','amenities','zipcode','latitude','longitude'], axis=1, inplace=True)
col_list= df.columns
formatted_col_list = ",\n    ".join([f"'{col}'" for col in col_list])

def calculate_missing_values(data):
    missing_data = {
        "Column": data.columns,
        "Missing Count": data.isnull().sum(),
        "Missing Percentage": (data.isnull().mean() * 100).round(2)
    }
    return pd.DataFrame(missing_data)

clean_df=df.copy()
print(clean_df.columns
      )
clean_df=clean_df.dropna(subset=['host_since'])
t2layout = html.Div([
    html.Br(),
    html.H2("Filtering Irrelevant Data"),
    html.H4("Among all the columns, thumbnail_url, logprice, name, description, amenities, zipcode, latitude, and longitude are not useful for the analysis."),
    html.H4("After dropping the columns, this is the final column list:"),
    html.H4(f"{formatted_col_list}"),
    html.Br(),

    # Before Filling Missing Values
    html.H3("Missing Values Before Filling:"),
    dash_table.DataTable(
        id="missing_before_table",
        data=calculate_missing_values(data).to_dict("records"),
        columns=[{"name": i, "id": i} for i in calculate_missing_values(data).columns],
        style_table={"overflowX": "scroll"},
        style_cell={"textAlign": "center"},
    ),
    html.Br(),
    html.H2("Handling Missing Values"),
    html.H3("Choose a method to fill numerical columns"),
    dcc.RadioItems(
        id="fillnum_method",
        options=[
            {"label": "Fill with Mean", "value": "mean"},
            {"label": "Fill with Median", "value": "median"},
            {"label": "Fill with Mode", "value": "mode"},
        ],
        value="mean",
    ),
    html.Br(),
    html.H3("Choose a method to fill date columns"),
    dcc.RadioItems(
        id="filldate_method",
        options=[
            {"label": "Forward Fill", "value": "ffill"},
            {"label": "Backward Fill", "value": "bfill"},
        ],
        value="ffill",
    ),
    html.Br(),
    html.H3("Choose a method to fill categorical columns"),
    dcc.RadioItems(
        id="fillcat_method",
        options=[
            {"label": "Mode", "value": "cmode"},
            {"label": "Unknown", "value": "unknown"},
        ],
        value="cmode",
    ),
    html.Br(),
    html.H3("Choose a method to fill boolean columns"),
    dcc.RadioItems(
        id="fillbool_method",
        options=[
            {"label": "True", "value": "t"},
            {"label": "False", "value": "f"},
        ],
        value="f",
    ),
    html.Br(),
    html.H3("Cleaned Dataset Preview"),
    html.Div(id="cleaned-data-preview", style={"overflowX": "scroll"}),
    dash_table.DataTable(
        id="previewnewtable",
        data=[],
        columns=[]
    ),
    html.Br(),
    html.H3("Cleaned Dataset Statistics"),
    html.Div(id="cleaned-data-stats", style={"overflowX": "scroll"}),
    dash_table.DataTable(
        id="previewnewtablestats",
        data=[],
        columns=[]
    ),
    html.Br(),
    html.Br(),
    html.H3("Missing Values After Filling:"),
    dash_table.DataTable(
        id="missing_after_table",
        data=[],
        columns=[]
    )
], id="t2layout")

@app.callback(
    [
        Output("previewnewtable", "data"),
        Output("previewnewtable", "columns"),
        Output("previewnewtablestats", "data"),
        Output("previewnewtablestats", "columns"),
        Output("missing_after_table", "data"),
        Output("missing_after_table", "columns"),
    ],
    [
        Input("fillnum_method", "value"),
        Input("filldate_method", "value"),
        Input("fillcat_method", "value"),
        Input("fillbool_method", "value"),
    ]
)
def fill_missing(fillnum, filldate, fillcat, fillbool):
    global clean_df

    clean_df = data.copy()  # Reset clean_df for dynamic updates
    print('Initial Columns:', clean_df.columns)

    # Drop unwanted columns without reassigning to clean_df
    clean_df.drop(['thumbnail_url', 'name', 'description', 'amenities', 'latitude', 'longitude', 'zipcode'], axis=1,
                  inplace=True)
    clean_df = clean_df.dropna(subset=['host_since'])

    print('After Drop Columns:', clean_df.columns)

    # Get numerical columns with missing values
    missing_cols = clean_df.select_dtypes(include=["number"]).columns
    print(f"Numerical columns to fill: {missing_cols}")

    # Numerical columns
    if fillnum == "mean":
        clean_df[missing_cols] = clean_df[missing_cols].fillna(clean_df[missing_cols].mean())
    elif fillnum == "median":
        clean_df[missing_cols] = clean_df[missing_cols].fillna(clean_df[missing_cols].median())
    elif fillnum == "mode":
        clean_df[missing_cols] = clean_df[missing_cols].fillna(clean_df[missing_cols].mode().iloc[0])

    # Date columns
    if filldate == "ffill":
        clean_df[["first_review", "last_review"]] = clean_df[["first_review", "last_review"]].ffill()
    elif filldate == "bfill":
        clean_df[["first_review", "last_review"]] = clean_df[["first_review", "last_review"]].bfill()

    # Categorical columns
    if fillcat == "cmode":
        clean_df["neighbourhood"] = clean_df["neighbourhood"].fillna(clean_df["neighbourhood"].mode()[0])
    elif fillcat == "unknown":
        clean_df["neighbourhood"] = clean_df["neighbourhood"].fillna("Unknown")

    # Boolean columns
    if fillbool == "t":
        clean_df[["host_identity_verified", "host_has_profile_pic"]] = clean_df[
            ["host_identity_verified", "host_has_profile_pic"]].fillna("t")
    elif fillbool == "f":
        clean_df[["host_identity_verified", "host_has_profile_pic"]] = clean_df[
            ["host_identity_verified", "host_has_profile_pic"]].fillna("f")


    clean_df["host_response_rate"] = clean_df["host_response_rate"].fillna("0%")

    # Data for preview table
    preview_data = clean_df.head(10).to_dict("records")
    preview_columns = [{"name": col, "id": col} for col in clean_df.columns]

    # Data for statistics table
    stats_data = clean_df.describe().reset_index().round(2).to_dict("records")
    stats_columns = [{"name": col, "id": col} for col in ["index"] + list(clean_df.describe().columns)]

    # Missing values after filling
    missing_after = calculate_missing_values(clean_df).to_dict("records")
    missing_after_columns = [{"name": col, "id": col} for col in calculate_missing_values(clean_df).columns]

    return preview_data, preview_columns, stats_data, stats_columns, missing_after, missing_after_columns


# DATA TRANSFORMATION

t3layout= html.Div([
    html.Br(),
    html.H1("Transformation of the data"),
    html.H5("Data transformation involves converting data into a suitable format or scale to improve its usability and analytical performance."),
    html.H2("Choose a Method:"),
    dcc.RadioItems(
        id="transform_method",
        options=[
            {"label": "Box-Cox Transformation", "value": "boxcox"},
            {"label": "Log Transform", "value": "log"},
        ],
        value="boxcox",
    ),

    html.Br(),
    html.Div(id="transform_output"),


],  id="t3layout")

@app.callback(
    Output("transform_output", "children"),
    Input("transform_method", "value"),

)

def data_transform(method):

    global df
    if method == "boxcox":
        df['price_boxcox'], fitted_lambda = boxcox(df['price'])

        print(f"Lambda used for Box-Cox transformation: {fitted_lambda}")
        print(df['price_boxcox'])

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=('Original Price Distribution', 'Box-Cox Transformed Price Distribution'))

        # Original Data Histogram
        fig.add_trace(go.Histogram(x=df['price'], nbinsx=15, name='Original Price',
                                   marker=dict(color='blue', opacity=0.7)), row=1, col=1)

        # Transformed Data Histogram
        fig.add_trace(go.Histogram(x=df['price_boxcox'], nbinsx=15, name='Box-Cox Transformed Price',
                                   marker=dict(color='green', opacity=0.7)), row=1, col=2)


        fig.update_layout(
            title_text='Price Distribution Before and After Box-Cox Transformation',
            showlegend=False,
            height=600,
            width=1200,
            template='plotly_white'
        )

        return html.Div([
            html.H3(f"Boxcox transformation: "),
            dcc.Graph(figure=fig)
        ])

    elif method == "log":
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=('Original Price Distribution', 'Log Transformed Price Distribution'))

        fig.add_trace(go.Histogram(x=df['price'], nbinsx=15, name='Original Price',
                                   marker=dict(color='blue', opacity=0.7)), row=1, col=1)
        fig.add_trace(go.Histogram(x=df['log_price'], nbinsx=15, name='Log Transformed Price',
                                   marker=dict(color='orange', opacity=0.7)), row=1, col=2)

        fig.update_layout(
            title_text='Price Distribution Before and After Log Transformation',
            showlegend=False,
            height=600,
            width=1200,
            template='plotly_white'
        )

        return html.Div([
            html.H3(f"Log transformation: "),
            dcc.Graph(figure=fig)
        ])


df['price_boxcox'], fitted_lambda = boxcox(df['price'])
print(f"Lambda used for Box-Cox transformation: {fitted_lambda}")
print(df['price_boxcox'])




# # normality test again after data transformations
# stats.probplot(df['log_price'], dist="norm", plot=plt)
# # Show the plot
# plt.title('QQ Plot')
# plt.show()
#
# stats.probplot(df['price_boxcox'], dist="norm", plot=plt)
# # Show the plot
# plt.title('QQ Plot')
# plt.show()
#
# price_s,price_pvalue= shapiro(df['log_price'])
# price_res= 'Normal Distribution' if price_pvalue> 0.01 else 'Not Normal Distribution'
# print(f"Shapiro Test: Statistics = {price_s:.2f}, p-value ={price_pvalue:.2f}"),
# print(f"Shapiro Test : Log Price is {price_res}")
#
# price_s,price_pvalue= shapiro(df['price_boxcox'])
# price_res= 'Normal Distribution' if price_pvalue> 0.01 else 'Not Normal Distribution'
# print(f"Shapiro Test: Statistics = {price_s:.2f}, p-value ={price_pvalue:.2f}"),
# print(f"Shapiro Test : Log Price is {price_res}")


t4layout = html.Div([
    html.Br(),
    html.H2("Detecting Outliers and Removing Them for Price", style={"textAlign": "center"}),
    html.H6("Removing outliers involves identifying and eliminating data points that deviate significantly from the datasets overall pattern, ensuring a more accurate analysis."),
    html.H3("Choose a Method:"),
    dcc.RadioItems(
        id="outlier_detection_method",
        options=[
            {"label": "Z-scores", "value": "zscore"},
            {"label": "IQR (Interquartile Range)", "value": "iqr"}
        ],
        value="zscore",
    ),
    html.Br(),

    html.Div(id="outlier_output"),

],  id="t4layout")

@app.callback(
    Output("outlier_output", "children"),
    Input("outlier_detection_method", "value"),

)

def outlier_detection(method):
    global df

    if method == "zscore":
        z_scores = (df['price'] - df['price'].mean()) / df['price'].std()
        outliers = np.abs(z_scores) > 3
        filtered_df = df[~outliers]

        df_table = dash_table.DataTable(
            id='previewtable',
            data=filtered_df.head(10).to_dict('records'),
            columns=[{'name': col, 'id': col} for col in filtered_df.columns]
        )

        fig = px.box(df, y='price')
        fig.update_layout(
            title="Before Removing Outliers (Z-Score Method)",
            title_font=dict(family='serif', color='blue', size=16),
            xaxis_title="Distribution",
            yaxis_title="Price",
            xaxis=dict(
                title_font=dict(family='serif', color='darkred', size=14),
                showgrid=True, gridcolor='lightgray', gridwidth=1
            ),
            yaxis=dict(
                title_font=dict(family='serif', color='darkred', size=14),
                showgrid=True, gridcolor='lightgray', gridwidth=1
            ),
            width=1000,
            height=800,
            template="simple_white"
        )

        fig.update_traces(
            hovertemplate='<b>Price: %{y}</b>'

        )
        fig2 = px.box(filtered_df, y='price')
        fig2.update_layout(
            title="After Removing Outliers (Z-Score Method)",
            title_font=dict(family='serif', color='blue', size=16),
            xaxis_title="Distribution",
            yaxis_title="Price",
            xaxis=dict(
                title_font=dict(family='serif', color='darkred', size=14),
                showgrid=True, gridcolor='lightgray', gridwidth=1
            ),
            yaxis=dict(
                title_font=dict(family='serif', color='darkred', size=14),
                showgrid=True, gridcolor='lightgray', gridwidth=1
            ),
            width=1000,
            height=800,
            template="simple_white"
        )

        fig2.update_traces(
            hovertemplate='<b>Price: %{y}</b>'

        )

        return html.Div([
            html.H4(f"Outlier detection using Z-score:"),
            html.H5(f"Length before removing outliers: {df.shape[0]}"),
            html.H5(f"Length after removing outliers: {filtered_df.shape[0]}"),
            df_table,
            html.H4("Before Removing the Outliers"),
            dcc.Graph(figure=fig),
            html.H4("After Removing the Outliers"),
            dcc.Graph(figure=fig2)
        ])

    elif method == "iqr":
        len_original = df.shape[0]
        Q1 = df['price'].quantile(0.25)
        Q3 = df['price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_val = Q1 - 1.5 * IQR
        upper_val = Q3 + 1.5 * IQR
        filtered_df = df[(df['price'] >= lower_val) & (df['price'] <= upper_val)]

        fig = px.box(df, y='price')
        fig.update_layout(
            title="Before Removing Outliers (IQR Method)",
            title_font=dict(family='serif', color='blue', size=16),
            xaxis_title="Distribution",
            yaxis_title="Price",
            xaxis=dict(
                title_font=dict(family='serif', color='darkred', size=14),
                showgrid=True, gridcolor='lightgray', gridwidth=1
            ),
            yaxis=dict(
                title_font=dict(family='serif', color='darkred', size=14),
                showgrid=True, gridcolor='lightgray', gridwidth=1
            ),
            width=1000,
            height=800,
            template="simple_white"
        )

        fig2 = px.box(filtered_df, y='price')
        fig2.update_layout(
            title="After Removing Outliers (IQR Method)",
            title_font=dict(family='serif', color='blue', size=16),
            xaxis_title="Distribution",
            yaxis_title="Price",
            xaxis=dict(
                title_font=dict(family='serif', color='darkred', size=14),
                showgrid=True, gridcolor='lightgray', gridwidth=1
            ),
            yaxis=dict(
                title_font=dict(family='serif', color='darkred', size=14),
                showgrid=True, gridcolor='lightgray', gridwidth=1
            ),
            width=1000,
            height=800,
            template="simple_white"
        )

        return html.Div([
            html.H4(f"Filtered Data (IQR Method):"),
            html.H5(f"Q1 for price is: {Q1:.2f}, Q3 for price is: {Q3:.2f}"),
            html.H5(f"IQR is: {IQR:.2f}"),
            html.H5(f"Any price < {lower_val:.2f} or > {upper_val:.2f} is considered an outlier."),
            html.H5(f"Length of original dataset with outliers: {len_original}"),
            html.H5(f"Length of dataset after removing outliers: {filtered_df.shape[0]}"),

            dash_table.DataTable(
                id="previewtable",
                data=filtered_df.head(10).to_dict('records'),
                columns=[{'name': col, 'id': col} for col in filtered_df.columns]
            ),

            html.H4("Before Removing the Outliers"),
            dcc.Graph(figure=fig),
            html.H4("After Removing the Outliers"),
            dcc.Graph(figure=fig2)
        ])
# PCA

t5layout = html.Div([
    html.Br(),
    html.H2("Dimensionality Reduction (PCA)", style={"textAlign": "center"}),

    html.H3("Choose a Method:"),
    dcc.RadioItems(
        id="pca",
        options=[
            {"label": "Z-scores", "value": "zscore"},
            {"label": "IQR (Interquartile Range)", "value": "iqr"}
        ],
        value="zscore",
    ),
    html.Br(),

    html.Div(id="pca_output"),
], id="t5layout")


@ app.callback(Output("pca_output", "children"),
    Input("pca", "value"),
)

def perform_pca(method):
    global df

    # Select numerical columns
    num_columns = df.select_dtypes(include=[np.number])

    # Preprocess the data
    if method == "zscore":
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(num_columns)
    elif method == "iqr":
        Q1 = num_columns.quantile(0.25)
        Q3 = num_columns.quantile(0.75)
        IQR = Q3 - Q1
        data_scaled = ((num_columns - Q1) / IQR).fillna(0).values

    # Handle invalid values in scaled data
    data_scaled = np.nan_to_num(data_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Optional: Clip extreme values
    data_scaled = np.clip(data_scaled, -1e6, 1e6)

    # Perform PCA
    pca = PCA()
    principal_components = pca.fit_transform(data_scaled)
    explained_variance = pca.explained_variance_ratio_

    # Prepare data for scatter plot
    pca_df = pd.DataFrame(data=principal_components, columns=[f"PC{i + 1}" for i in range(len(explained_variance))])

    # Create scatter plot for the first two principal components
    scatter_plot = px.scatter(
        pca_df, x="PC1", y="PC2",
        title="PCA Scatter Plot (PC1 vs PC2)",
        labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"}
    )
    scatter_plot.update_layout(
        title="PCA Scatter Plot (PC1 vs PC2)",
        title_font=dict(family='serif', color='blue', size=16),
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        xaxis=dict(
            title_font=dict(family='serif', color='darkred', size=14),
            showgrid=True, gridcolor='lightgray', gridwidth=1
        ),
        yaxis=dict(
            title_font=dict(family='serif', color='darkred', size=14),
            showgrid=True, gridcolor='lightgray', gridwidth=1
        ),
        width=1000,
        height=800,
        template="simple_white"
    )

    # Create bar chart for explained variance
    explained_variance_plot = px.bar(
        x=[f"PC{i + 1}" for i in range(len(explained_variance))],
        y=explained_variance,
        title="Explained Variance by Principal Components",
        labels={"x": "Principal Components", "y": "Explained Variance Ratio"}
    )
    explained_variance_plot.update_layout(
        title="Explained Variance by Principal Components",
        title_font=dict(family='serif', color='blue', size=16),
        xaxis_title="Principal Components",
        yaxis_title="Explained Variance Ratio",
        xaxis=dict(
            title_font=dict(family='serif', color='darkred', size=14),
            showgrid=True, gridcolor='lightgray', gridwidth=1
        ),
        yaxis=dict(
            title_font=dict(family='serif', color='darkred', size=14),
            showgrid=True, gridcolor='lightgray', gridwidth=1
        ),
        width=1000,
        height=800,
        template="simple_white"
    )

    # Return the plots
    return html.Div([
        html.H4("PCA Results", style={"textAlign": "center", "fontFamily": "Arial", "color": "black"}),
        dcc.Graph(figure=scatter_plot),
        dcc.Graph(figure=explained_variance_plot)
    ])







# Normality tests
t6layout = html.Div([
    html.Br(),
    html.H2("Normality tests for Price", style={"textAlign": "center"}),
    html.H5("A normality test evaluates whether a dataset follows a normal distribution, which is essential for many statistical analyses."),
    html.H3("Choose a Method:"),
    html.H2("Statistical and Graphical methods"),
    dcc.RadioItems(
        id="normality_test_method",
        options=[
            {"label": "Shapiro-Wilk Test", "value": "shapiro"},
            {"label": "Kolmogorov-Smirnov (K-S) Test", "value": "ks"},
            {"label": "D'Agostino-Pearson Test", "value": "pearson"},
            {"label": "Histogram Plot", "value": "histplot"},
            {"label": "Quantile-Quantile Plot", "value": "qqplot"}

        ],
        value="shapiro",
    ),

    html.Br(),

    html.Div(id="normality_output"),

],  id="t6layout")

@app.callback(
    Output("normality_output", "children"),
  Input("normality_test_method", "value")

)

def normality_tests(method):
    price_s, price_pvalue, price_res = None, None, None
    if method == "shapiro":
        price_s,price_pvalue= shapiro(df['price'])
        price_res= 'Normal Distribution' if price_pvalue> 0.01 else 'Not Normal Distribution'

        return html.Div([
        html.H3(f"Shapiro Test: Statistics = {price_s:.2f}, p-value ={price_pvalue:.2f}"),
        html.H3(f"Shapiro Test : Price is {price_res}")

    ])
    if method =="pearson":
        price_s, price_pvalue = normaltest(df['price'])
        price_res = 'Normal Distribution' if price_pvalue > 0.01 else 'Not Normal Distribution'

        return html.Div([
            html.H3(f"D'Agostino-Pearson Test: Statistics = {price_s:.2f}, p-value ={price_pvalue:.2f}"),
            html.H3(f"D'Agostino-Pearson Test : Log Price is {price_res}")
        ])
    if method == "ks":
        price_s, price_pvalue = kstest(df['price'],'norm')
        price_res = 'Normal Distribution' if price_pvalue > 0.01 else 'Not Normal Distribution'

        return html.Div([
            html.H3(f"D'Agostino-Pearson Test: Statistics = {price_s:.2f}, p-value ={price_pvalue:.2f}"),
            html.H3(f"D'Agostino-Pearson Test : Log Price is {price_res}")
        ])
    if method == "histplot":
        fig = px.histogram(df, x='price')
        fig.update_layout(
            title={
                'text': "Histogram of Price",
                'font': {'family': 'serif', 'color': 'blue', 'size': 30},
                'x': 0.5
            },
            xaxis_title={
                'text': "Price",
                'font': {'family': 'serif', 'color': 'darkred', 'size': 20}
            },
            yaxis_title={
                'text': "Frequency",
                'font': {'family': 'serif', 'color': 'darkred', 'size': 20}
            },
            showlegend=False,
            width=1000,
            height=800,
            template='plotly_white',
            xaxis=dict(
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1
            ), margin=dict(l=100, r=100, t=100, b=100)
        )

        return html.Div([
            html.H3(
                "The data appears heavily concentrated around certain values, with a clear peak. This distribution is not symmetric and shows irregular spikes, which suggests that the data doesn't follow a normal distribution.",
                style={'fontFamily': 'serif', 'fontSize': 25}
            ),
            dcc.Graph(figure=fig)
        ])

    if method == "qqplot":
        res = probplot(df['price'], dist="norm", plot=None)

        x = res[0][0]
        y = res[0][1]
        line = res[1][0] * x + res[1][1]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            name='Data Points',
            marker=dict(color='blue')
        ))


        fig.add_trace(go.Scatter(
            x=x,
            y=line,
            mode='lines',
            name='Normality Line',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title={
                'text': "Q-Q Plot of Price",
                'font': {'family': 'serif', 'color': 'blue', 'size': 20},
                'x': 0.5
            },
            xaxis_title={
                'text': "Theoretical Quantiles",
                'font': {'family': 'serif', 'color': 'darkred', 'size': 16}
            },
            yaxis_title={
                'text': "Sample Quantiles",
                'font': {'family': 'serif', 'color': 'darkred', 'size': 16}
            },
            showlegend=True,
            legend=dict(
                title="Legend",
                title_font=dict(family='serif', color='darkred', size=14),
                font=dict(family='serif', color='darkblue', size=12),
                bgcolor='lightyellow',
                bordercolor='black',
                borderwidth=1
            ),
            width=1000,
            height=800,
            template='plotly_white',
            xaxis=dict(
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1
            )
        )

        return html.Div([
            html.H3(
                "The points deviate significantly from the red dashed line (representing the theoretical quantiles for a normal distribution), especially at the tails. This deviation indicates that the data does not follow a normal distribution, particularly in the extremes."
            ),
            html.Pre(dcc.Graph(figure=fig))
        ])

    else:
        return html.Div([
            html.H3("No method selected or invalid method")
        ])


t7layout= html.Div([html.H2("Choose a numerical graph to plot"),
                   dcc.Dropdown(
                   id='numcategory_dropdown',
                  options=[
                    {"label": "Line Plot", "value": "lineplot"},
                    {"label": "Histogram with KDE", "value": "histogram"},
                    {"label": "Dist plot", "value": "distplot"},
                    {"label": "KDE with fill", "value": "kdewithfillplot"},
                    {"label": "Boxen Plot", "value": "boxenplot"},
                    {"label": "Violin Plot", "value": "violinplot"},
                    {"label": "Lm Plot", "value": "lmplot"},
                    {"label": "Pair Plot", "value": "pairplot"},
                    {"label": "Joint Plot", "value": "jointplot"},
                    {"label": "Hexbin Plot", "value": "hexbinplot"},
                    {"label": "Rug Plot", "value": "rugplot"},
                    {"label": "Area Plot", "value": "areaplot"},
                    {"label": "Cluster Map", "value": "clustermap"},
                    {"label": "QQ Plot", "value": "qqplot"},
                    {"label": "Heatmap with cbar", "value": "heatmap"},
                    {"label": "3D and Contour Plot", "value": "3dc"},
                ],
                       value="histogram"
                   ),
                   html.Div(id="numgraph_inputs"),
],  id="t7layout"
                   )

@app.callback(
Output(component_id='numgraph_inputs', component_property='children'),
     Input(component_id='numcategory_dropdown', component_property='value')
)

# updating numerical graph inputs
def update_numgraphinput(graph_type):
    if graph_type == "pairplot":
        return pairplotinputs
    elif graph_type == "jointplot":
        return jointinputs
    elif graph_type == "hexbinplot":
        return hexbininputs
    elif graph_type == "violinplot":
        return violininputs
    elif graph_type == "distplot":
        return distinputs
    elif graph_type == "kdewithfillplot":
        return kdewithfillinputs
    elif graph_type == "rugplot":
        return rugplotinputs
    elif graph_type == "lineplot":
        return lineinputs
    elif graph_type == "lmplot":
        return lminputs
    elif graph_type == "histogram":
        return histograminputs
    elif graph_type == "boxenplot":
        return boxeninputs
    elif graph_type == "areaplot":
        return areainputs
    elif graph_type == "clustermap":
        return clustermapinputs
    elif graph_type == "qqplot":
        return qqinputs
    elif graph_type == "3dc":
        return dcinputs
    elif graph_type == "heatmap":
        return heatmapinputs

# heatmap with cbar
heatmapinputs = html.Div([
    html.H3("Choose variables for Heatmap with cbar:"),
    dcc.Checklist(
        id="heatmap_options",
        options=[
            {'label': 'Log Price', 'value': 'log_price'},
            {'label': 'Accommodates', 'value': 'accommodates'},
            {'label': 'Bathrooms', 'value': 'bathrooms'},
            {'label': 'Bedrooms', 'value': 'bedrooms'},
            {'label': 'Beds', 'value': 'beds'},
            {'label': 'Number of Reviews', 'value': 'number_of_reviews'},
            {'label': 'Review Scores Rating', 'value': 'review_scores_rating'},
            {'label': 'Host Response Rate', 'value': 'host_response_rate'}
        ],
        value=['log_price', 'bedrooms', 'bathrooms'],
        inline=True
    ),
    dcc.Graph(id="heatmap_plot")
])

# Callback to update the heatmap
@app.callback(
    Output('heatmap_plot', 'figure'),
    Input('heatmap_options', 'value')
)
def plotheatmap(selected_columns):
    # Ensure that the selected columns are in the dataframe
    selected_df = df[selected_columns].corr()

    # Create the heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=selected_df.values,
        x=selected_df.columns,
        y=selected_df.columns,
        colorscale='RdBu',
        zmin=-1, zmax=1,  # Color scale range
        colorbar=dict(title='Correlation'),
        hoverongaps=False,
        text=selected_df.round(2).values,
        hovertemplate="%{x} - %{y}: %{text}<extra></extra>"
    ))

    # Update layout with titles and font styling
    fig.update_layout(
        title="Heatmap of Correlations",
        title_font=dict(family="serif", color="blue", size=16),
        xaxis=dict(
            title="Features",
            title_font=dict(family="serif", color="darkred", size=14),
            tickfont=dict(family="serif", color="black", size=12)
        ),
        yaxis=dict(
            title="Features",
            title_font=dict(family="serif", color="darkred", size=14),
            tickfont=dict(family="serif", color="black", size=12)
        ),
        plot_bgcolor="white",
        width=1000,
        height=800
    )
    # Show grid lines
    fig.update_xaxes(showgrid=True, gridcolor="lightgrey", gridwidth=0.5)
    fig.update_yaxes(showgrid=True, gridcolor="lightgrey", gridwidth=0.5)

    return fig


# 3D CONTOUR Plot
dcinputs = html.Div([
    html.H3("Choose variables for 3D contour plot:"),
    dcc.Checklist(
        id="3dc_options",
        options=[
            {'label': 'Log Price', 'value': 'log_price'},
            {'label': 'Accommodates', 'value': 'accommodates'},
            {'label': 'Bathrooms', 'value': 'bathrooms'},
            {'label': 'Bedrooms', 'value': 'bedrooms'},
            {'label': 'Beds', 'value': 'beds'},
            {'label': 'Number of Reviews', 'value': 'number_of_reviews'},
            {'label': 'Review Scores Rating', 'value': 'review_scores_rating'},
            {'label': 'Host Response Rate', 'value': 'host_response_rate'}
        ],
        value=['log_price', 'bedrooms', 'bathrooms'],
        inline=True
    ),
    dcc.Graph(id="3dc_plot")
])

@app.callback(
    Output(component_id='3dc_plot', component_property='figure'),
    Input(component_id='3dc_options', component_property='value')
)
def plot3dc(selected_columns):
    if len(selected_columns) < 3:
        return go.Figure().update_layout(
            title="Please select exactly three variables for the 3D contour plot."
        )

    z_col = selected_columns[0]
    x_col = selected_columns[1]
    y_col = selected_columns[2]

    data = df[[x_col, y_col, z_col]].dropna()

    x = np.linspace(data[x_col].min(), data[x_col].max(), 100)
    y = np.linspace(data[y_col].min(), data[y_col].max(), 100)
    X, Y = np.meshgrid(x, y)
    Z = griddata(
        (data[x_col], data[y_col]),
        data[z_col],
        (X, Y),
        method='cubic'
    )

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='Viridis',
        opacity=0.8
    ))

    fig.add_trace(go.Contour(
        z=Z.mean(axis=0),
        x=x,
        y=y,
        contours=dict(showlabels=True),
        colorscale='Viridis',
        line_width=1,
        showscale=False
    ))

    fig.update_layout(
        title=f"3D Contour Plot of {z_col}, {x_col}, and {y_col}",
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
            xaxis=dict(title_font=dict(family='serif', color='red', size=12)),
            yaxis=dict(title_font=dict(family='serif', color='red', size=12)),
            zaxis=dict(title_font=dict(family='serif', color='red', size=12))
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=50)
    )

    return fig

# CLUSTERMAP

clustermapinputs = html.Div([
    html.H3("Choose variables for Clustermap:"),
    dcc.Checklist(
        id="clustermap_options",
        options=[
            {'label': 'Log Price', 'value': 'log_price'},
            {'label': 'Accommodates', 'value': 'accommodates'},
            {'label': 'Bathrooms', 'value': 'bathrooms'},
            {'label': 'Bedrooms', 'value': 'bedrooms'},
            {'label': 'Beds', 'value': 'beds'},
            {'label': 'Number of Reviews', 'value': 'number_of_reviews'},
            {'label': 'Review Scores Rating', 'value': 'review_scores_rating'},
            {'label': 'Host Response Rate', 'value': 'host_response_rate'}
        ],
        value=['review_scores_rating', 'accommodates','bathrooms'],
        inline=True
    ),
    dcc.Graph(id="clustermap_plot")
])

@app.callback(
    Output(component_id='clustermap_plot', component_property='figure'),
    Input(component_id='clustermap_options', component_property='value')
)

def plotclustermap(selected_columns):
    df_sampled = df[selected_columns].dropna().sample(n=1000, random_state=42)

    fig = dashbio.Clustergram(
        data=df_sampled.values,
        column_labels=selected_columns,
        row_labels=df_sampled.index.astype(str).tolist(),
        height=800,
        width=700,
        color_threshold={"row": 150, "col": 150},
        color_map="Cividis",
    )
    fig.update_layout(
        title="Clustermap of Selected Variables",
        title_font=dict(size=20, color='blue', family='serif'),
        xaxis_title="Features",
        yaxis_title="Observations",
        xaxis=dict(tickangle=45),  # Optional: rotate x-axis labels
        yaxis=dict(tickangle=45),  # Optional: rotate y-axis labels
        plot_bgcolor="black",  # Set background color of the plot area
        paper_bgcolor="white",  # Set background color outside the plot
        font=dict(family="serif", color="black", size=12),  # General font settings
        showlegend=False  # Disable legend
    )

    return fig
    # df_sampled = df[selected_columns].dropna().sample(n=1000, random_state=42)
    #
    # data_values = df_sampled.values
    # fig = dash_bio.Clustergram(
    #     data=data_values,
    #     column_labels=selected_columns,
    #     row_labels=list(df_sampled.index),
    #     height=800,
    #     width=700
    # )
    #
    # return fig




# QQPLOT
qqinputs = html.Div([
    html.H3("Choose variables for QQ Plot:"),
    dcc.Checklist(
        id="qq_options",
        options=[
            {'label': 'Price', 'value': 'price'},
            {'label': 'Number of Reviews', 'value': 'number_of_reviews'},
            {'label': 'Review Scores Rating', 'value': 'review_scores_rating'},
            {'label': 'Host Response Rate', 'value': 'host_response_rate'}
        ],
        value=['price']
    ),
    dcc.Graph(id="qq_plot")
])


@app.callback(
    Output(component_id='qq_plot', component_property='figure'),
    Input(component_id='qq_options', component_property='value')
)
def plotqq(col):
    col = col[0]
    res = probplot(df[col], dist="norm", plot=None)
    x = res[0][0]
    y = res[0][1]
    line = res[1][0] * x + res[1][1]

    # Create the plotly figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='Data Points',
        marker=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=line,
        mode='lines',
        name='Normality Line',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title={
            'text': f"Q-Q Plot of {col.replace('_', ' ').title()}",
            'font': {'family': 'serif', 'color': 'blue', 'size': 20},
            'x': 0.5
        },
        xaxis_title={
            'text': "Theoretical Quantiles",
            'font': {'family': 'serif', 'color': 'darkred', 'size': 16}
        },
        yaxis_title={
            'text': "Sample Quantiles",
            'font': {'family': 'serif', 'color': 'darkred', 'size': 16}
        },
        showlegend=True,
        legend=dict(
            title="Legend",
            title_font=dict(family='serif', color='darkred', size=14),
            font=dict(family='serif', color='darkblue', size=12),
            bgcolor='lightyellow',
            bordercolor='black',
            borderwidth=1
        ),
        width=1000,
        height=800,
        template='plotly_white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1
        )
    )

    return fig


# AREAPLOT
areainputs = html.Div([
    html.H3("Choose variables for Area Plot:"),
    dcc.Checklist(
        id="area_options",
        options=[
            {'label': 'Log Price', 'value': 'log_price'},
            {'label': 'Accommodates', 'value': 'accommodates'},
            {'label': 'Bathrooms', 'value': 'bathrooms'},
            {'label': 'Bedrooms', 'value': 'bedrooms'},
            {'label': 'Beds', 'value': 'beds'},
            {'label': 'Number of Reviews', 'value': 'number_of_reviews'},
            {'label': 'Review Scores Rating', 'value': 'review_scores_rating'},
            {'label': 'Host Response Rate', 'value': 'host_response_rate'}
        ],
        value=['log_price', 'accommodates'],  # Default selected values
        inline=True
    ),
    dcc.Graph(id="area_plot")
])


@app.callback(
    Output(component_id='area_plot', component_property='figure'),
    Input(component_id='area_options', component_property='value')
)
def plotarea(selected_columns):
    # Ensure the dataset includes the 'host_since_year' column
    if 'host_since_year' not in df.columns:
        df['host_since_year'] = pd.to_datetime(df['host_since'], errors='coerce').dt.year

    # Check if selected_columns is not empty
    if not selected_columns:
        return px.area(title="No Data Selected", labels={'host_since_year': 'Year', 'Value': 'Average Value'})

    # Filter the columns to include only numeric ones
    numeric_columns = [col for col in selected_columns if pd.api.types.is_numeric_dtype(df[col])]

    if not numeric_columns:
        return px.area(title="No Numeric Data Selected", labels={'host_since_year': 'Year', 'Value': 'Average Value'})

    # Prepare the data for plotting
    data = df.groupby(['host_since_year'])[numeric_columns].mean(numeric_only=True).reset_index()

    # Reshape the data into a long format for Plotly
    data_long = data.melt(
        id_vars='host_since_year',
        var_name='Variable',
        value_name='Average Value'
    )

    # Create the area plot with Plotly Express
    fig = px.area(
        data_long,
        x='host_since_year',
        y='Average Value',
        color='Variable',
        title='Trends of Average Values Over Time',
        labels={'host_since_year': 'Year', 'Average Value': 'Value'},

    )

    # Add layout customizations for a polished look
    fig.update_layout(
        title=dict(
            text='Trends of Average Values Over Time',
            font=dict(family='serif', size=20, color='darkblue'),
            x=0.5  # Center align the title
        ),
        xaxis=dict(
            title='Year',
            title_font=dict(family='serif', size=14, color='darkred'),
            tickangle=45,
            tickformat=".0f",  # Ensure years are integers
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Average Value',
            title_font=dict(family='serif', size=14, color='darkred'),
            tickformat=".2f",  # Limit y-axis values to 2 decimal places
            showgrid=True,
            gridcolor='lightgray'
        ),
        legend=dict(
            title='Variables',
            font=dict(size=12),
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='lightgray',
            borderwidth=1,
            orientation='h',  # Horizontal legend
            x=0.5,  # Center the legend below the plot
            xanchor='center',
            y=-0.2
        ),
        template='simple_white',
        width=1000,
        height=600
    )

    # Add gridlines and enhance interactivity
    fig.update_traces(opacity=0.8)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

    return fig


# PAIRPLOT
pairplotinputs=html.Div([
        html.H3("Choose variables for pairplot: "),
        dcc.Checklist( id="pair_options",
            options=[
                {'label': 'Log Price', 'value': 'log_price'},
                {'label': 'Accommodates', 'value': 'accommodates'},
                {'label': 'Bathrooms', 'value': 'bathrooms'},
                {'label': 'Bedrooms', 'value': 'bedrooms'},
                {'label': 'Beds', 'value': 'beds'},
                {'label': 'Number of reviews', 'value': 'number_of_reviews'},
                {'label': 'Review Scores Rating', 'value': 'review_scores_rating'},
                {'label': 'Host response rate', 'value': 'host_response_rate'},

            ],
                       value=['log_price', 'accommodates'],  # Default selected values
                       inline=True
        )
        , dcc.Graph(id="pair_plot")]
    )

@app.callback(
Output(component_id='pair_plot', component_property='figure'),
     Input(component_id='pair_options', component_property='value')

)

# plotting pairplot
def plotpair(val):
    if len(val) < 2:
        return px.scatter_matrix(
            df,
            dimensions=[],
            title="Please select at least two variables",
        )

    fig = px.scatter_matrix(
        df,
        dimensions=val,
        title="Pair Plot of Selected Features",
    )
    fig.update_layout(
        title=dict(
            text="Pair Plot of Selected Features",
            font=dict(size=20, family="Serif", color="blue"),
        ),
        template="plotly_white",
        width=1000,
        height=800,
    )
    fig.update_traces(
        marker=dict(size=5, color="dodgerblue", opacity=0.7)
    )

    # Update axis labels to red for all selected columns
    for col in val:
        fig.update_xaxes(title_font=dict(size=14, family="Serif", color="red"), matches='x')
        fig.update_yaxes(title_font=dict(size=14, family="Serif", color="red"), matches='y')

    return fig


# JOINTPLOT
jointinputs=html.Div([
        html.H3("Choose x variable for jointplot: "),
    dcc.RadioItems(
        id="jointx_radio",
        options=[
            {"label": "Number of reviews", "value": "number_of_reviews"},
            {"label": "Log price", "value": "log_price"},
            {"label": "Review scores rating", "value": "review_scores_rating"},
            {"label": "Host Response Rate", "value": "host_response_rate"}
        ],
        value="review_scores_rating"
    ),
html.H3("Choose y variable for jointplot: "),
    dcc.RadioItems(
        id="jointy_radio",
        options=[
            {"label": "Number of reviews", "value": "number_of_reviews"},
            {"label": "Log price", "value": "log_price"},
            {"label": "Review scores rating", "value": "review_scores_rating"},
            {"label": "Host Response Rate", "value": "host_response_rate"}
        ],
        value="host_response_rate"
    ),
    html.H3("Choose hue variable for jointplot: "),
    dcc.RadioItems(
        id="jointhue_radio",
        options=[
            {"label": "Cleaning fee", "value": "cleaning_fee"},
            {"label": "Host has profile pic", "value": "host_has_profile_pic"},
            {"label": "Host Identity verified", "value": "host_identity_verified"},
            {"label": "Instant Bookable", "value": "instant_bookable"}
        ],
        value="host_identity_verified"
    )
        , dcc.Graph(id="joint_plot")]
    )

@app.callback(
Output(component_id='joint_plot', component_property='figure'),
    [ Input(component_id='jointx_radio', component_property='value'),
Input(component_id='jointy_radio', component_property='value'),
Input(component_id='jointhue_radio', component_property='value')]

)

def plotjoint(valx, valy, valhue):
    fig = px.scatter(df, x=valx, y=valy, marginal_x="histogram", marginal_y="histogram", color=valhue)

    fig.update_layout(
        title=f'Relationship between {valx} and {valy}',
        title_font=dict(family='serif', color='blue', size=20),
        xaxis_title=valx,
        yaxis_title=valy,
        xaxis=dict(title_font=dict(family='serif', color='darkred', size=18)),
        yaxis=dict(title_font=dict(family='serif', color='darkred', size=18)),
        width=1200,  # Set width
        height=1000,  # Set height
        template="simple_white",  # Clean background
    )

    # Add grid to the plot
    fig.update_xaxes(
        showgrid=True,       # Enable gridlines for x-axis
        gridcolor='lightgray',  # Set gridline color to light gray
        gridwidth=0.5        # Set gridline width
    )
    fig.update_yaxes(
        showgrid=True,       # Enable gridlines for y-axis
        gridcolor='lightgray',  # Set gridline color to light gray
        gridwidth=0.5        # Set gridline width
    )

    return fig


# HEXBINPLOT

hexbininputs=html.Div([
        html.H3("Choose variables for hexbinplot: "),
    dcc.RadioItems(
        id="hexbin_radio",
        options=[
            {"label": "Number of reviews", "value": "number_of_reviews"},
            {"label": "Review scores rating", "value": "review_scores_rating"},
            {"label": "Host Response Rate", "value": "host_response_rate"}
        ],
        value="review_scores_rating"
    )
        , dcc.Graph(id="hexbin_plot")]
    )

@app.callback(
Output(component_id='hexbin_plot', component_property='figure'),
    Input(component_id='hexbin_radio', component_property='value'),
)

# plotting hexbinplot
def plothexbin(val):
    fig = go.Figure(go.Histogram2d(
        x=df[val],
        y=df['log_price'],
        showscale=True,
        nbinsx=30,
        nbinsy=30,
    ))

    # Customize layout
    fig.update_layout(
        title=f"Hexbin Plot of {val} vs Log Price",
        title_font=dict(family='serif', color='blue', size=16),
        xaxis_title=f"{val}",
        yaxis_title="Log Price",
        xaxis=dict(
            title_font=dict(family='serif', color='darkred', size=14),
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title_font=dict(family='serif', color='darkred', size=14),
            showgrid=True,
            zeroline=False
        ),
        template="plotly_white",
        showlegend=False
    )

    # Show the plot
    return fig


#VIOLIN PLOT

violininputs=html.Div([
        html.H3("Choose x variable for violinplot: "),
    dcc.RadioItems(
        id="violin_radio",
        options=[
            {"label": "City", "value": "city"},
            {"label": "Room type", "value": "room_type"},
            {"label": "Bed type", "value": "bed_type"},
            {"label": "Cancellation policy", "value": "cancellation_policy"}
        ],
        value="city"
    )
        , dcc.Graph(id="violin_plot")]
    )

@app.callback(
Output(component_id='violin_plot', component_property='figure'),
    Input(component_id='violin_radio', component_property='value'),
)

# plotting violinplot
def plotviolin(val):
    fig = px.violin(
        df,
        x=val,
        y="log_price",
        box=True,
        color_discrete_sequence=["lightcoral"],
        title=f"Price Distribution by {val}",
    )
    fig.update_layout(
        title=dict(
            text=f"Price Distribution by {val}",
            font=dict(size=18, family="Serif", color="blue"),
        ),
        xaxis=dict(
            title=dict(text=val, font=dict(size=14, family="Serif", color="darkred")),
            showgrid=True,  # Enable grid for x-axis
            gridcolor="lightgray",  # Optional: set gridline color
            gridwidth=0.5,  # Optional: set gridline width
        ),
        yaxis=dict(
            title=dict(text="Log Price", font=dict(size=14, family="Serif", color="darkred")),
            showgrid=True,  # Enable grid for y-axis
            gridcolor="lightgray",  # Optional: set gridline color
            gridwidth=0.5,  # Optional: set gridline width
        ),
        template="plotly_white",
        width=1000,
        height=800,
    )
    return fig

# LMPLOT
lminputs=html.Div([
        html.H3("Choose variables for lmplot: "),
    dcc.RadioItems(
        id="lm_radio",
        options=[
            {'label': 'Accommodates', 'value': 'accommodates'},
            {'label': 'Bathrooms', 'value': 'bathrooms'},
            {'label': 'Bedrooms', 'value': 'bedrooms'},
            {'label': 'Beds', 'value': 'beds'}
        ],
        value="bathrooms"
    ), html.H3("Choose variables for lmplot col: "),
    dcc.RadioItems(
        id="lmcol_radio",
        options=[
            {'label': 'Room type', 'value': 'room_type'},
            {'label': 'Bed type', 'value': 'bed_type'},

        ],
        value="room_type"
    )
        , dcc.Graph(id="lm_plot")]
    )

@app.callback(
Output(component_id='lm_plot', component_property='figure'),
    [Input(component_id='lm_radio', component_property='value'),
Input(component_id='lmcol_radio', component_property='value')],
)

# plotting lmplot
def plotlm(val, col):
    fig = px.scatter(
        df,
        x=val,
        y="log_price",
        facet_col=col,
        trendline="ols",
        title=f"Lm plot of {val} vs Log Price by {col}",
        hover_data=["city", "room_type", "bed_type"]
    )

    fig.update_layout(
        title=dict(
            text=f"Lm plot of {val} vs Log Price by {col}",
            font=dict(family="serif", color="blue", size=16)
        ),
        xaxis=dict(
            title=dict(text=val, font=dict(family="serif", color="darkred", size=14)),
            showgrid=True,  # Adds gridlines
            gridcolor="lightgray",
            gridwidth=0.5,
        ),
        yaxis=dict(
            title=dict(text="Log Price", font=dict(family="serif", color="darkred", size=14)),
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=0.5,
        ),
        width=1000,
        height=800,
        template="simple_white",
    )

    fig.for_each_annotation(lambda a: a.update(font=dict(family="serif", color="darkblue", size=12)))

    fig.update_traces(marker=dict(color="darkred", size=6, opacity=0.7))
    fig.update_traces(line=dict(color="blue", width=2), selector=dict(mode="lines"))

    return fig




# LINEPLOT
lineinputs=html.Div([
        html.H3("Choose variables for lineplot: "),
    dcc.RadioItems(
        id="line_radio",
        options=[
            {"label": "Number of Hosts", "value": "host_since_year"}
        ],
        value="host_since_year"
    )
        , dcc.Graph(id="line_plot")]
    )

@app.callback(
Output(component_id='line_plot', component_property='figure'),
    Input(component_id='line_radio', component_property='value'),
)

# plotting lineplot
def plotline(val):
    # Group data to calculate the number of hosts for each year
    df.columns = df.columns.str.strip()
    no_of_hosts = df.groupby(val).size().reset_index(name="number_of_hosts")

    fig = px.line(
        no_of_hosts,
        x=val,
        y="number_of_hosts",
        title=f"Host Growth Over Time ({val})",
        markers=True,
    )

    # Update layout for better styling and readability
    fig.update_layout(
        title=dict(
            text=f"Host Growth Over Time ({val})",
            font=dict(family="serif", color="blue", size=16),
        ),
        xaxis=dict(
            title=dict(
                text="Year",
                font=dict(family="serif", color="darkred", size=14),
            ),
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=0.5,
        ),
        yaxis=dict(
            title=dict(
                text="Number of Hosts",
                font=dict(family="serif", color="darkred", size=14),
            ),
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=0.5,
        ),
        width=1000,
        height=800,
        template="simple_white",
    )

    # Style the line and markers
    fig.update_traces(
        line=dict(color="blue", width=2),
        marker=dict(color="darkred", size=8),
    )

    return fig




# DISTPLOT
distinputs=html.Div([
        html.H3("Choose variables for distplot: "),
    dcc.RadioItems(
        id="dist_radio",
        options=[
            {"label": "Log Price", "value": "log_price"},
            {"label": "Price", "value": "price"},
            {"label": "Number of reviews", "value": "number_of_reviews"},

        ],
        value="log_price"
    )
        , dcc.Graph(id="dist_plot")]
    )

@app.callback(
Output(component_id='dist_plot', component_property='figure'),
    Input(component_id='dist_radio', component_property='value'),
)

# plotting distplot
def plotdist(val):
    # Ensure the column exists
    if val not in df.columns:
        raise KeyError(f"The column '{val}' does not exist in the DataFrame. Available columns: {list(df.columns)}")

    # Create the distribution plot
    fig = ff.create_distplot(
        [df[val]],
        group_labels=[val],
        curve_type='kde',
        show_hist=True,
        show_rug=False,
        bin_size=0.3
    )

    # Update layout for better styling
    fig.update_layout(
        title=dict(
            text=f"Distplot of Distribution of {val}",
            font=dict(size=20, family="Serif", color="blue")
        ),
        xaxis=dict(
            title=dict(text=val, font=dict(size=18, family="Serif", color="darkred")),
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=0.5
        ),
        yaxis=dict(
            title=dict(text="Density", font=dict(size=18, family="Serif", color="darkred")),
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=0.5
        ),
        width=1000,
        height=800,
        template="simple_white"
    )

    # Adjust line properties for the KDE curve (density line)
    for trace in fig.data:
        if "kde" in trace.name:  # KDE curve trace
            trace.update(line=dict(color="blue", width=2))
        elif "histogram" in trace.name:  # Histogram trace
            trace.update(marker=dict(color="dodgerblue", opacity=0.7))

    return fig

# KDE WITH FILL

kdewithfillinputs=html.Div([
        html.H3("Choose variables for kde with fill plot: "),
    dcc.RadioItems(
        id="kdewithfill_radio",
        options=[
            {"label": "Log Price", "value": "log_price"},
            {"label": "Price", "value": "price"},
            {"label": "Number of reviews", "value": "number_of_reviews"},

        ],
        value="log_price"
    )
        , dcc.Graph(id="kdewithfill_plot")]
    )

@app.callback(
Output(component_id='kdewithfill_plot', component_property='figure'),
    Input(component_id='kdewithfill_radio', component_property='value'),
)

def plot_kde_fill(val):
    kde = stats.gaussian_kde(df['review_scores_rating'].dropna(), bw_method='scott')

    x_values = np.linspace(df['review_scores_rating'].min(), df['review_scores_rating'].max(), 1000)

    y_values = kde(x_values)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(255, 99, 71, 0.5)',
        line=dict(color='lightcoral', width=2),
        name='Ratings Distribution'
    ))

    # Layout customizations
    fig.update_layout(
        title="Ratings Distribution",
        title_font=dict(family="serif", color="blue", size=16),
        xaxis_title=f"{val}",
        xaxis=dict(title_font=dict(family="serif", color="darkred", size=14)),
        yaxis_title="Density",
        yaxis=dict(title_font=dict(family="serif", color="darkred", size=14)),
        legend=dict(title=f"{val} Distribution", font=dict(size=12), x=0.8, y=0.9),
        template="plotly_white",
        width=800,
        height=600
    )

    # Show grid lines
    fig.update_xaxes(showgrid=True, gridcolor="lightgrey", gridwidth=0.5)
    fig.update_yaxes(showgrid=True, gridcolor="lightgrey", gridwidth=0.5)

    # Show the plot
    return fig


# BOXENPLOT
boxeninputs=html.Div([
        html.H3("Choose variables for boxenplot: "),

    dcc.RadioItems(
        id="boxen_radio",
        options=[
            {"label": "property group", "value": "property_group"},
            {"label": "city", "value": "city"},
            {"label": "bed type", "value": "bed_type"},

        ], value= 'property_group')
        , dcc.Graph(id="boxen_plot")]
    )

@app.callback(
Output(component_id='boxen_plot', component_property='figure'),
    Input(component_id='boxen_radio', component_property='value'),
)


def plotboxen(val):
    fig = px.box(
        df,
        x='log_price',
        y=val,  # Use the value directly, not inside a set
        title=f'Boxen Plot of Distribution of Log Price by {val}',  # Title dynamically based on `val`
        color=val  # Color by the same variable `val`
    )

    fig.update_layout(
        title=dict(
            text=f"Boxen Plot of Log Price vs {val}",  # Title dynamically updated
            font=dict(family='serif', color='blue', size=20)
        ),
        xaxis_title=dict(
            text="Log Price",
            font=dict(family='serif', color='darkred', size=18)
        ),
        yaxis_title=dict(
            text=val,  # Dynamic y-axis label
            font=dict(family='serif', color='darkred', size=18)
        ),
        template='simple_white',  # Clean, white template for the plot
        height=800,  # Set height of the plot
        width=1000,  # Set width of the plot
        showlegend=True,  # Show legend
    )

    # Add grid to the plot
    fig.update_xaxes(showgrid=True, gridcolor='lightgray', gridwidth=0.5)
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=0.5)

    return fig



# HISTOGRAM + KDE
histograminputs=html.Div([
    html.H3("Choose x variable for histogram: "),
    dcc.RadioItems(
        id="hist_radio",
        options=[
            {"label": "Number of reviews", "value": "number_of_reviews"},
            {"label": "Log price", "value": "log_price"},
            {"label": "Review scores rating", "value": "review_scores_rating"},
        ],
        value="number_of_reviews"
    ),

    # Add a slider for adjusting bin size
    html.Div([
        html.Label("Choose bin size for histogram:"),
        dcc.Slider(
            id="bin_size_slider",
            min=5,
            max=50,
            step=5,
            value=10,  # default bin size
            marks={i: str(i) for i in range(5, 51, 5)},
        )
    ]),

    # Histogram plot
    dcc.Graph(id="histogram_plot")
])

@app.callback(
    Output(component_id='histogram_plot', component_property='figure'),
    [
        Input(component_id='hist_radio', component_property='value'),
        Input(component_id='bin_size_slider', component_property='value')
    ]
)
def plothistogram(val, bin_size):
    data = df[val].tolist()

    # Create the histogram with KDE and adjust the bin size based on the slider
    fig = ff.create_distplot(
        [data],  # Data as a list
        group_labels=[val],  # Label for the legend
        bin_size=bin_size,
        show_rug=False
    )

    # Customize layout
    fig.update_layout(
        title=f'Histogram with KDE - {val} Distribution',
        title_font=dict(family='serif', color='blue', size=16),
        xaxis_title=val,
        yaxis_title='Density',
        xaxis=dict(title_font=dict(family='serif', color='darkred', size=14)),
        yaxis=dict(title_font=dict(family='serif', color='darkred', size=14)),
        width=800,  # Set width
        height=600,  # Set height
        template="simple_white"  # Clean background
    )

    # Add grid to the plot
    fig.update_xaxes(showgrid=True, gridcolor='lightgray', gridwidth=0.5)
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=0.5)

    return fig


# RUGPLOT
rugplotinputs=html.Div([
    html.H3("Choose x variable for rugplot: "),
    dcc.RadioItems(
                    id="rug_radio",
                    options=[
                        {"label": "Number of reviews", "value": "number_of_reviews"},

                        {"label": "Review scores rating", "value": "review_scores_rating"},
                        {"label": "Price", "value": "price"},

                    ],
                    value="price"
                ), dcc.Graph(id="rug_plot")]
)

@app.callback(
Output(component_id='rug_plot', component_property='figure'),
     Input(component_id='rug_radio', component_property='value')

)



def plotrug(val):
    fig = ff.create_distplot([df[val]], group_labels=[val],
                             curve_type='kde', show_hist=False, bin_size=0.3)

    fig.update_layout(
        title=f'{val} Distribution',
        title_font=dict(family='serif', color='blue', size=16),
        xaxis_title=val,
        yaxis_title='Density',
        xaxis=dict(title_font=dict(family='serif', color='darkred', size=14)),
        yaxis=dict(title_font=dict(family='serif', color='darkred', size=14)),
        width=1000,  # Set width
        height=800,  # Set height
        template="simple_white"  # Clean background
    )

    # Add grid to the plot
    fig.update_xaxes(showgrid=True, gridcolor='lightgray', gridwidth=0.5)
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=0.5)

    return fig


# categorical graphs tab layout t8
t8layout= html.Div([html.H2("Choose a categorical bar to plot"),
                   dcc.Dropdown(
                   id='category_dropdown',
                  options=[
                    {"label": "Count Plot", "value": "countplot"},
                    {"label": "Bar Plot", "value": "barplot"},
                    {"label": "Bar Subplots", "value": "barsubplots"},
                    {"label": "Pie Chart", "value": "piechart"},
                      {"label": "Strip Plot", "value": "stripplot"},
                      {"label": "Swarm Plot", "value": "swarmplot"}
                ],
                       value="countplot"
                   ),
                   html.Div(id="graph_inputs"),
],  id="t8layout"
                   )

@app.callback(
Output(component_id='graph_inputs', component_property='children'),
     Input(component_id='category_dropdown', component_property='value')
)

# updating graph inputs
def update_graphinput(graph_type):
    if graph_type == "countplot":
        return countplotinputs
    elif graph_type == "piechart":
        return piechartinputs
    elif graph_type == "barplot":
        return barplotinputs
    elif graph_type == "barsubplots":
        return barsubplotinputs
    elif graph_type == "stripplot":
        return stripplotinputs
    elif graph_type == "swarmplot":
        return swarmplotinputs


# swarmplot inputs
swarmplotinputs=html.Div([
    html.H2("Choose x variable for swarmplot: "),
    dcc.RadioItems(
                    id="swarm_radio",
                    options=[
                        {"label": "City", "value": "city"},
                        {"label": "Room type", "value": "room_type"},
                        {"label": "Bed type", "value": "bed_type"},
                        {"label": "Cancellation policy", "value": "cancellation_policy"}
                    ],
                    value="city"
                ), dcc.Graph(id="swarm_plot")]
)


@app.callback(
Output(component_id='swarm_plot', component_property='figure'),
     Input(component_id='swarm_radio', component_property='value')

)

# plotting swarmplot
def plotswarm(val):
    sample_df = df.sample(n=2500, random_state=42)
    fig = px.strip(
        sample_df,
        x=val,
        y="review_scores_rating",
        color=val,
        hover_data=sample_df.columns,
        title=f'Swarmplot of Review Scores by {val.capitalize()}'
    )

    fig.update_layout(
        title=dict(
            text=f'Swarmplot of Review Scores by {val.capitalize()}',
            font=dict(size=20, family="serif", color="blue")
        ),
        xaxis=dict(
            title=dict(text=f'{val.capitalize()}', font=dict(size=14, family="serif", color="darkred"))
        ),
        yaxis=dict(
            title=dict(text='Review Scores Rating', font=dict(size=14, family="serif", color="darkred"))
        ),
        plot_bgcolor="white",
        xaxis_tickangle=45,
        width=1000,
        height=800
    )

    # Add grid lines
    fig.update_xaxes(showgrid=True, gridcolor='lightgray', gridwidth=0.5)
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=0.5)

    return fig




# countplot inputs

countplotinputs = html.Div([
    html.H2("Customize Countplot:"),
    html.H4("For x axis enter one of these options - 'city' or 'bed_type'or  'room_type' or 'cancellation_policy'"),
    html.Div([
        html.Label("Global Title:"),
        dcc.Input(id="gtitle", type="text", placeholder="Enter plot title", value="Count Plot"),
    ]),
    html.Div([
        html.Label("X-Axis Label:"),
        dcc.Input(id="xlabel", type="text", placeholder="city", value="city"),
    ]),
    html.Div([
        html.Label("Y-Axis Label:"),
        dcc.Input(id="ylabel", type="text", placeholder="Enter y-axis label", value="Count"),
    ]),
    dcc.Graph(id="count_plot"),
])


@app.callback(
    Output(component_id="count_plot", component_property="figure"),
    [
        Input(component_id="gtitle", component_property="value"),
        Input(component_id="xlabel", component_property="value"),
        Input(component_id="ylabel", component_property="value"),
    ]
)
def plotcount(gtitle, xlabel, ylabel):
    if xlabel not in df.columns:
        xlabel = df.columns[0]  # Use the first column if input column is invalid

    fig = px.histogram(
        df,
        x=xlabel,
        title=gtitle,
        template="plotly",
        category_orders={xlabel: df[xlabel].value_counts().index.tolist()},
    )

    # Update the figure layout and styling
    fig.update_layout(
        title=dict(
            text=gtitle,
            font=dict(family="serif", size=25, color="blue"),
            x=0.5,  # Center align the title
        ),
        xaxis=dict(
            title=dict(text=xlabel, font=dict(family="serif", size=20, color="darkred")),
            tickfont=dict(family="serif", size=18, color="black"),
        ),
        yaxis=dict(
            title=dict(text=ylabel, font=dict(family="serif", size=20, color="darkred")),
            tickfont=dict(family="serif", size=18, color="black"),
        ),
        plot_bgcolor="white",
        width=1000,
        height=800,
    )

    fig.update_xaxes(showgrid=True, gridcolor="lightgrey", gridwidth=0.5)
    fig.update_yaxes(showgrid=True, gridcolor="lightgrey", gridwidth=0.5)

    return fig


# piechart inputs
piechartinputs=html.Div([
    html.H2("Choose column for piechart: "),
    dcc.RadioItems(
                    id="pie_radio",
                    options=[
                        {"label": "City", "value": "city"},
                        {"label": "Room type", "value": "room_type"},
                        {"label": "Bed type", "value": "bed_type"},
                        {"label": "Cancellation policy", "value": "cancellation_policy"},
                        {"label": "Host identity verified", "value": "host_identity_verified"},
                        {"label": "Host has profile picture", "value": "host_has_profile_pic"}
                    ],
                    value="city"
                ), dcc.Graph(id="pie_plot")]
)

@app.callback(
Output(component_id='pie_plot', component_property='figure'),
     Input(component_id='pie_radio', component_property='value')

)

# plotting pieplot
def plotpie(val):
    data_counts = df[val].value_counts()
    selected_label = next(
        (option["label"] for option in piechartinputs.children[1].options if option["value"] == val),
        val
    )
    fig = px.pie(
        df,
        names=data_counts.index,
        values=data_counts.values,
        title=f"{selected_label} Composition",
    )
    fig.update_traces(textinfo='percent+label',
                      hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent:.2f}%')
    fig.update_layout(
        title=dict(
            text=f"{selected_label} Composition",
            font=dict(family="serif", size=25, color="blue"),
            x=0.5,
        ),
        legend=dict(
            title=dict(text=val.capitalize(), font=dict(size=14)),
            font=dict(size=15),
            orientation="v",
            x=1,
            y=1,
        ),
        plot_bgcolor="white",
        width=1000,
        height=800,
    )
    return fig


barplotinputs = html.Div([
    html.H2("Choose column for barplot: "),
    dcc.RadioItems(
        id="bar_radio",
        options=[
            {"label": "City", "value": "city"},
            {"label": "Room type", "value": "room_type"},
            {"label": "Bed type", "value": "bed_type"},
            {"label": "Property type", "value": "property_type"}
        ],
        value="city"
    ),
    html.Br(),

    # Select plot type
    html.H2("Choose bar plot type: "),
    dcc.RadioItems(
        id="bar_plot_type",
        options=[
            {"label": "Normal Bar Plot", "value": "normal"},
            {"label": "Stacked Bar Plot", "value": "stacked"},
            {"label": "Grouped Bar Plot", "value": "grouped"}
        ],
        value="normal"
    ),

    # Inputs for stacked and grouped bar plot
    html.Div(id='stack_group_inputs', children=[
        # For stacked and grouped, input for 'x' and 'grouping' columns will appear.
        dcc.Dropdown(
            id='stack_group_x',
            options=[
                {"label": "City", "value": "city"},
                {"label": "Room type", "value": "room_type"},
                {"label": "Bed type", "value": "bed_type"},
                {"label": "Property type", "value": "property_type"}
            ],
            value="city",
            placeholder="Select x-axis column"
        ),
        dcc.Dropdown(
            id='stack_group_group',
            options=[
                {"label": "Cancellation policy", "value": "cancellation_policy"},
                {"label": "Room type", "value": "room_type"},
                {"label": "Bed type", "value": "bed_type"},
                {"label": "Property type", "value": "property_type"}
            ],
            value="room_type",
            placeholder="Select grouping column"
        ),
    ]),

    dcc.Graph(id="bar_plot")
])


@app.callback(
    Output(component_id='bar_plot', component_property='figure'),
    [Input(component_id='bar_radio', component_property='value'),
     Input(component_id='bar_plot_type', component_property='value'),
     Input(component_id='stack_group_x', component_property='value'),
     Input(component_id='stack_group_group', component_property='value')]
)
def plotbar(val, plot_type, stack_group_x, stack_group_group):
    # Get the top categories based on the selected column
    top_categories = df[val].value_counts()[:10]
    filtered_df = df[df[val].isin(top_categories.index)]
    avg_price = filtered_df.groupby(val)['price'].mean().sort_values(ascending=False)

    # Normal Bar Plot
    if plot_type == "normal":
        fig = px.bar(
            avg_price,
            x=avg_price.values,
            y=avg_price.index,
            orientation='h',
            labels={"x": "Average Price", "y": val.capitalize()},
        )

    # Stacked Bar Plot (showing multiple categories stacked)
    elif plot_type == "stacked":
        grouped_data = df.groupby([stack_group_x, stack_group_group])[val].count().unstack(fill_value=0)
        fig = go.Figure()

        # Add a trace for each property type
        for property_type in grouped_data.columns:
            fig.add_trace(go.Bar(
                x=grouped_data.index,
                y=grouped_data[property_type],
                name=property_type,
                opacity=0.7
            ))

        fig.update_layout(
            barmode='stack',
            title=f"Stacked Bar Chart: {stack_group_x.capitalize()} by {stack_group_group.capitalize()}",
            title_font=dict(family='serif', size=25, color='blue'),
            xaxis_title=stack_group_x.capitalize(),
            yaxis_title=f"Count of {stack_group_group.capitalize()}",
            template='plotly_white',
            width=1000,
            height=800
        )


    elif plot_type == "grouped":
        fig = px.bar(
            df,
            x=stack_group_x,
            color=stack_group_group,
            title=f"Grouped Bar Chart: {stack_group_x.capitalize()} by {stack_group_group.capitalize()}",
            labels={stack_group_x: stack_group_x.capitalize()},
            barmode='group'
        )


    fig.update_layout(
        title=dict(
            text=f"Average Price by {val.capitalize()}",
            font=dict(family="serif", size=25, color="blue"),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(text="Average Price", font=dict(family="serif", size=20, color="darkred")),
            tickfont=dict(family="serif", size=18, color="black"),
        ),
        yaxis=dict(
            title=dict(text=val.capitalize(), font=dict(family="serif", size=20, color="darkred")),
            tickfont=dict(family="serif", size=18, color="black"),
        ),
        plot_bgcolor="white",
        width=1000,
        height=800,
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgrey", gridwidth=0.5)
    fig.update_yaxes(showgrid=True, gridcolor="lightgrey", gridwidth=0.5)

    return fig

# bar subplots
barsubplotinputs = html.Div([
    html.H2("Choose column for subplots: "),
    dcc.Checklist(
        id="barsub_radio",
        options=[
            {"label": "City", "value": "city"},
            {"label": "Room type", "value": "room_type"},
            {"label": "Bed type", "value": "beds"},
            {"label": "Cancellation policy", "value": "bedrooms"},
            {"label": "Host identity verified", "value": "accommodates"},
            {"label": "Host has profile picture", "value": "bathrooms"}
        ],
        value=['beds','bathrooms','bedrooms','accommodates']
    ),
    dcc.Graph(id="barsub_plot")
])


@app.callback(
    Output(component_id='barsub_plot', component_property='figure'),
    Input(component_id='barsub_radio', component_property='value')
)
def plotbarsubplots(selected_features):
    # If no features are selected, return an empty figure
    if not selected_features:
        return go.Figure()

    # Determine the number of rows and columns based on selected features
    num_features = len(selected_features)
    rows = (num_features + 1) // 2  # Round up to handle odd numbers
    cols = 2 if num_features > 1 else 1  # Ensure at least 1 column

    # Create subplots layout dynamically
    fig = sp.make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"Average Price vs. {feature.capitalize()}" for feature in selected_features],
        vertical_spacing=0.1, horizontal_spacing=0.1
    )

    # Loop through each selected feature and create subplots
    for i, feature in enumerate(selected_features):
        # Calculate the average price for the feature
        avg_price = df.groupby(feature)['price'].mean().reset_index()

        # Create bar plot for the feature
        bar = go.Bar(
            x=avg_price[feature],
            y=avg_price['price'],
            name=feature,
            marker=dict(
                colorscale='Viridis',
                color=avg_price['price'],  # Color bars based on price
                showscale=True
            ),
            text=avg_price['price'],
            textposition='auto'
        )

        # Add bar plot to the appropriate subplot
        fig.add_trace(bar, row=(i // 2) + 1, col=(i % 2) + 1)

    # Update layout
    fig.update_layout(
        title="Average Price by Categories",
        title_font=dict(family="serif", color="blue", size=16),
        showlegend=False,
        plot_bgcolor="white",
        height=600 * rows,
        width=1200
    )

    # Update axes labels for all subplots
    fig.update_xaxes(title_text="Category", title_font=dict(family="serif", color="darkred", size=12))
    fig.update_yaxes(title_text="Average Price", title_font=dict(family="serif", color="darkred", size=12))

    return fig




# stripplot inputs
stripplotinputs=html.Div([
    html.H3("Choose column for stripplot: "),
    dcc.RadioItems(
                    id="strip_radio",
                    options=[
                        {"label": "City", "value": "city"},
                        {"label": "Room type", "value": "room_type"},
                        {"label": "Bed type", "value": "bed_type"},
                        {"label": "Property type", "value": "property_type"},
                        {"label": "Cancellation policy", "value": "cancellation_policy"},

                    ],
                    value="city"
                ),html.H3("Choose column for hue: "),
    dcc.RadioItems(
                    id="striphue_radio",
                    options=[
                        {"label": "Cleaning fee", "value": "cleaning_fee"},
                        {"label": "Host has profile pic", "value": "host_has_profile_pic"},
                        {"label": "Host Identity verified", "value": "host_identity_verified"},
                        {"label": "Instant Bookable", "value": "instant_bookable"}

                    ],
                    value="cleaning_fee"
                ), dcc.Graph(id="strip_plot")]
)

@app.callback(
Output(component_id='strip_plot', component_property='figure'),
     [Input(component_id='strip_radio', component_property='value'),
Input(component_id='striphue_radio', component_property='value')]

)

# plotting stripplot
def plotstripplot(val, hue):
    fig = px.strip(
        df,
        x=val,
        y=df.log_price,
        color=hue,
    )
    fig.update_layout(
        title=dict(
            text=f"Strip Plot for {val.capitalize()} vs. Log Price",
            font=dict(family="serif", size=25, color="blue"),
            x=0.5,  # Center align the title
        ),
        xaxis=dict(
            title=dict(text=val.capitalize(), font=dict(family="serif", size=20, color="darkred")),
            tickfont=dict(family="serif", size=18, color="black"),
        ),
        yaxis=dict(
            title=dict(text="Log Price", font=dict(family="serif", size=20, color="darkred")),
            tickfont=dict(family="serif", size=18, color="black"),
        ),
        legend=dict(
            title=dict(text=hue.capitalize(), font=dict(size=14)),
            font=dict(size=15),
            orientation="v",
            x=1,
            y=1,
        ),
        plot_bgcolor="white",
        width=1500,
        height=1000,
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgrey", gridwidth=0.5)
    fig.update_yaxes(showgrid=True, gridcolor="lightgrey", gridwidth=0.5)
    return fig

t9layout = html.Div([
    html.Br(),
    html.H2("Statistics Dashboard", style={"textAlign": "center"}),

    # Statistic Selection
    html.H3("Choose a statistic:"),
    dcc.RadioItems(
        id="statistic_method",
        options=[
            {"label": "Descriptive Statistics", "value": "des_stat"},
            {"label": "Distribution Analysis", "value": "dis_stat"},
            {"label": "Correlation Analysis", "value": "cor_stat"},
            {"label": "Multivariate Kernel Density Estimation", "value": "multikde"},
        ],
        value="des_stat",
    ),
    html.Br(),
    html.Div(id="statistics_output"),
], id="t9layout")

num_columns=['accommodates', 'bathrooms','beds',
             'bedrooms','number_of_reviews','review_scores_rating',
             'price','price_boxcox']

filter_df=df[num_columns].describe().reset_index()
filter_df= filter_df.round(2)

summary_stats = []
for col in num_columns:
    stats = {
        "Feature": col,
        "Mean": df[col].mean(),
        "Median": df[col].median(),
        "Mode": df[col].mode()[0] if not df[col].mode().empty else None,
        "Standard Deviation": df[col].std(),
        "Variance": df[col].var(),

    }
    summary_stats.append(stats)

summary_stats_df = pd.DataFrame(summary_stats)
summary_stats_df= summary_stats_df.round(2)

@app.callback(
    Output("statistics_output", "children"),
    Input("statistic_method", "value"),

)

def statistics(method):
    if method == "des_stat":
        desc_table = dash_table.DataTable(
            data=filter_df.to_dict("records"),
            columns=[{"name": i, "id": i} for i in filter_df.columns],
            style_table={"overflowX": "scroll"},
            id='desctable'
        )
        statscomp_table = dash_table.DataTable(
            data=summary_stats_df.to_dict("records"),
            columns=[{"name": i, "id": i} for i in summary_stats_df.columns],
            style_table={"overflowX": "scroll"},
            id='statscomptable'
        )
        return html.Div([
            html.H4("Descriptive Statistics Table:"),
            desc_table,
            html.Br(),
            html.Br(),
            html.H4("Overall Statistics Table:"),
            statscomp_table,
            html.Br(),
            html.H2("Choose an individual attribute for it's complete statistics:"),
            dcc.RadioItems(
                id="attribute_selector",
                options=[{"label": col, "value": col} for col in num_columns],
                value='accommodates',
            ),
            html.Div(id='attribute_stats_output')
        ])

    if method == "dis_stat":
        return html.Div([
            html.H2("Select an attribute for distribution analysis:"),
            html.Br(),
            dcc.Dropdown(
                id="distribution_attribute",
                options=[{"label": col, "value": col} for col in num_columns],
                value='accommodates',
            ),
            html.Br(),
            html.Br(),
            dcc.Graph(id="distribution_graphs"),
        ])
    #
    elif method == "cor_stat":
        corr = df[num_columns].corr()

        # Heatmap with custom width and height
        heatmap = px.imshow(
            corr,
            text_auto=".2f",
            title="Correlation Heatmap",
            color_continuous_scale="Viridis",
            width=800,
            height=600
        )


        scatter_matrix = px.scatter_matrix(
            df,
            dimensions=num_columns,
            title="Scatter Matrix",
            labels={col: col.replace("_", " ").title() for col in num_columns},
            width=1000,
            height=800
        )

        return html.Div([
            html.H4("Correlation Analysis"),
            dcc.Graph(figure=heatmap),
            dcc.Graph(figure=scatter_matrix)
        ])
    #
    elif method == "multikde":
        return html.Div([
            html.H4("Select two attributes for Multivariate KDE:"),
            dcc.Dropdown(
                id="multikde_attribute_x",
                options=[{"label": col, "value": col} for col in df.select_dtypes(include=[np.number]).columns],
                value=df.select_dtypes(include=[np.number]).columns[0],
            ),
            dcc.Dropdown(
                id="multikde_attribute_y",
                options=[{"label": col, "value": col} for col in df.select_dtypes(include=[np.number]).columns],
                value=df.select_dtypes(include=[np.number]).columns[1],
            ),
            dcc.Graph(id="multikde_graph"),
        ])

# Callback for descriptive stats attributes
@app.callback(
Output("attribute_stats_output", "children"),
Input("attribute_selector", "value")
)
def attribute_stats(attribute):
    if attribute:
        mean = df[attribute].mean()
        median = df[attribute].median()
        mode = df[attribute].mode().iloc[0]
        variance = df[attribute].var()
        std_dev = df[attribute].std()
        return html.Div([
            html.H3(f"Statistics for {attribute}:"),
            html.H4(f"Mean: {mean:.2f}"),
            html.H4(f"Median: {median:.2f}"),
            html.H4(f"Mode: {mode:.2f}"),
            html.H4(f"Variance: {variance:.2f}"),
            html.H4(f"Standard Deviation: {std_dev:.2f}"),
        ])

# Callback for distribution analysis
@app.callback(
    Output("distribution_graphs", "figure"),
    Input("distribution_attribute", "value")
)
def distribution_analysis(attribute):

    fig = px.histogram(df, x=attribute, title=f"Distribution of {attribute}",
                       labels={attribute: f'{attribute} Value'},
                       color_discrete_sequence=['blue'],
                       nbins=20)
    fig.update_layout(
        xaxis_title=f'{attribute} Value',
        yaxis_title='Count',
        title_font=dict(size=25, family='serif', color='blue'),
        xaxis=dict(
            title_font=dict(size=20, family='serif', color='darkred'),
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title_font=dict(size=20, family='serif', color='darkred'),
            showgrid=True,
            gridcolor='lightgray'
        ),
        template='plotly_white',
        width=1000,
        height=800

    )

    return fig

# Callback for multivariate KDE
@app.callback(
    Output("multikde_graph", "figure"),
    [Input("multikde_attribute_x", "value"), Input("multikde_attribute_y", "value")]
)
def multivariate_kde(x, y):
    if x and y:
        kde = gaussian_kde(df[[x, y]].dropna().T)
        xi, yi = np.mgrid[df[x].min():df[x].max():100j, df[y].min():df[y].max():100j]
        zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
        fig = px.density_heatmap(
            x=xi.flatten(), y=yi.flatten(), z=zi,
            labels={"x": x, "y": y}, title=f"Multivariate KDE: {x} vs {y}"
        )
        return fig





# rendering the main layout
@app.callback(
     Output(component_id='layout', component_property='children'),
     Input(component_id='tabs', component_property='value')
 )

# updating layouts
def update_layout(tab):
    if tab == 't8':
        return t8layout
    if tab == 't7':
        return t7layout
    if tab == 't1':
        return t1layout
    if tab == 't3':
        return t3layout
    if tab == 't4':
        return t4layout
    if tab == 't6':
        return t6layout
    if tab == 't9':
        return t9layout
    if tab == 't2':
        return t2layout
    if tab == 't5':
        return t5layout

if __name__ == '__main__':
    app.run_server(debug=True
)