# import libraries & clean data 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from itertools import combinations

# 1. Load the CSV file
df = pd.read_csv("foodhub_order.csv")

# 2. Clean missing values and duplicates
df = df.dropna(how='all')
df = df.drop_duplicates()

# 3. Convert numeric columns safely
# For 'rating', replace non-numeric like 'Not given' with NaN, then convert
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Convert other columns safely
df['order_id'] = pd.to_numeric(df['order_id'], errors='coerce').astype('Int64')
df['customer_id'] = pd.to_numeric(df['customer_id'], errors='coerce').astype('Int64')
df['cost_of_the_order'] = pd.to_numeric(df['cost_of_the_order'], errors='coerce')
df['food_preparation_time'] = pd.to_numeric(df['food_preparation_time'], errors='coerce').astype('Int64')
df['delivery_time'] = pd.to_numeric(df['delivery_time'], errors='coerce').astype('Int64')

# 4. Final check
print("Cleaned Data Info:")
print(df.info())
print("Sample Cleaned Data:")
print(df.head())


# 2. Feature Engineering mandatory 
# Total order time 
df['total_order_time'] = df['food_preparation_time'] + df['delivery_time']

# Cost category (Low, Medium, High)
df['cost_category'] = pd.cut(df['cost_of_the_order'],
                             bins=[0, 200, 500, float('inf')],
                             labels=['Low', 'Medium', 'High'])

# Delivery speed category 
df['delivery_speed'] = pd.cut(df['delivery_time'],
                              bins=[0, 20, 40, float('inf')],
                              labels=['Fast', 'Normal', 'Slow'])

# Rating category 
df['rating_Category'] = pd.cut(df['rating'],
                               bins=[0, 2, 4, 5],
                               labels=['Poor', 'Average', 'Excellent'])

# Dat type (Weekend vs Weekday)
df['day_type'] = df['day_of_the_week'].apply(
    lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday'
)

# Final Check 
print("Feature Engineered Data Info:")
print(df.info())
print("Sample Feature Engineered Data:")
print(df.head())


# 3. Core Analysis 
# Total Sales 
total_Sales = df['cost_of_the_order'].sum()

# Profit 
df['profit'] = df['cost_of_the_order'] * 0.20
total_profit = df['profit'].sum()

# Average order value 
avg_order_value = df['cost_of_the_order'].mean()

# Monthly Trend 
if 'order_date' in df.columns:
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    monthly_trend = df.groupby(df['order_date'].dt.to_period('M'))['cost_of_the_order'].sum()
else:
    monthly_trend = "No 'order_date' column found in dataset"


# Print results 
print("=== Core Business Analysis ===")
print(f"Total sales: {total_Sales}")
print(f"Total profit (20% margin): {total_profit}")
print(f"Average Order Value: {avg_order_value}")
print("Monthly Trend:")
print(monthly_trend)


# 4. Basket Analysis Lite
# Group by order_id and collect items
basket = df.groupby('order_id')['restaurant_name'].apply(list)

# Find frequently bought together pairs
pair_counts = {}
for items in basket:
    if len(items) > 1:  # only if more than 1 item in order
        for combo in combinations(set(items), 2):  # unique pairs
            pair_counts[combo] = pair_counts.get(combo, 0) + 1

# Convert to DataFrame safely
if pair_counts:
    basket_analysis = pd.DataFrame([
        {'item_1': k[0], 'item_2': k[1], 'pair_count': v}
        for k, v in pair_counts.items()
    ])
    basket_analysis = basket_analysis.sort_values(by='pair_count', ascending=False)
else:
    basket_analysis = pd.DataFrame(columns=['item_1', 'item_2', 'pair_count'])

# 3. Profitability Analysis
# Assume profit margin = 20% of cost
df['profit'] = df['cost_of_the_order'] * 0.20

# High sales but low profit products
high_sales_low_profit = df.groupby('restaurant_name').agg({
    'cost_of_the_order': 'sum',
    'profit': 'sum'
}).reset_index()

high_sales_low_profit = high_sales_low_profit[
    (high_sales_low_profit['cost_of_the_order'] > high_sales_low_profit['cost_of_the_order'].mean()) &
    (high_sales_low_profit['profit'] < high_sales_low_profit['profit'].mean())
]

# High margin products (profit per order > average)
df['profit_margin'] = df['profit'] / df['cost_of_the_order']
high_margin_products = df.groupby('restaurant_name')['profit_margin'].mean().reset_index()
high_margin_products = high_margin_products.sort_values(by='profit_margin', ascending=False)

# 4. Print Results
print("=== Basket Analysis Lite ===")
print(basket_analysis.head())

print("\n=== High Sales but Low Profit Products ===")
print(high_sales_low_profit)

print("\n=== High Margin Products ===")
print(high_margin_products.head())


'''# 5. Visulization 
# 1. Sales Trend in LINE chart 
if 'order_date' in df.columns:
    sales_trend = df.groupby(df['order_date'].dt.to_period('M'))['cost_of_the_order'].sum()
    sales_trend.plot(kind='line', marker='o', title="Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    plt.show()


# 2. Top products Bar chart 
top_products = df.groupby('restaurant_name')['cost_of_the_order'].sum().nlargest(10)
top_products.plot(kind='bar', title="Top 10 Products by seles")
plt.xlabel("Restaurant Name")
plt.ylabel("Total Sales")
plt.show()

# 3. Hourly sales Heatmap
if 'order_time' in df.columns:
    df['order_time'] = pd.to_datetime(df['order_time'], errors='coerce')
    df['hour'] = df['order_time'].dt.hour
    hourly_sales = df.groupby(['day_of_the_week', 'hour'])['cost_of_the_order'].sum()
    sns.heatmap(hourly_sales, cmap="YlGnBu")
    plt.title("Hourly Sales Heatmap")
    plt.show()

# 4. Profit distribution Boxplot
df['profit'] = df['cost_of_the_order'] * 0.20
sns.boxplot(x=df['profit'])
plt.title("Profit Distribution")
plt.show()

# 5. Payment method pie chart 
if 'payment_method' in df.columns:
    payment_counts = df['payment_method'].value_counts()
    payment_counts.plot(kind='pie', autopct='%1.1f%%', title="Payment Method Distribution")
    plt.ylabel("")
    plt.show()

# 6. Correlation Heatmap 
numeric_cols = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
'''

# 2. Check if order_date exists
if 'order_date' in df.columns:
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    # Aggregate daily sales
    daily_sales = df.groupby(df['order_date'].dt.date)['cost_of_the_order'].sum().reset_index()
    daily_sales.rename(columns={'order_date':'date','cost_of_the_order':'sales'}, inplace=True)

    # Moving Average Forecast
    daily_sales['moving_avg'] = daily_sales['sales'].rolling(window=7).mean()
    next_week_sales_ma = daily_sales['moving_avg'].iloc[-1] * 7

    # Linear Regression Forecast
    daily_sales['day_num'] = np.arange(len(daily_sales))
    X = daily_sales[['day_num']]
    y = daily_sales['sales']

    model = LinearRegression()
    model.fit(X, y)

    future_days = np.arange(len(daily_sales), len(daily_sales)+7).reshape(-1,1)
    future_sales_lr = model.predict(future_days)
    next_week_sales_lr = future_sales_lr.sum()

    # Results
    print("=== Demand Forecasting ===")
    print(f"Next Week Sales (Moving Average): {next_week_sales_ma:.2f}")
    print(f"Next Week Sales (Linear Regression): {next_week_sales_lr:.2f}")

    # Visualization
    plt.figure(figsize=(10,5))
    plt.plot(daily_sales['date'], daily_sales['sales'], label='Actual Sales')
    plt.plot(daily_sales['date'], daily_sales['moving_avg'], label='7-day Moving Avg', color='orange')
    plt.plot(pd.date_range(daily_sales['date'].iloc[-1], periods=8, freq='D')[1:], future_sales_lr, 
             label='Linear Regression Forecast', color='red')
    plt.legend()
    plt.title("Demand Forecasting - Next Week Sales")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.show()

else:
    print("Error: 'order_date' column not found in dataset. Demand forecasting requires a date column.")
