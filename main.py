from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, stddev, lag, to_date
from pyspark.sql.window import Window
from math import sqrt


# Preparing the Data
spark = SparkSession.builder.appName("StocksAnalysis").getOrCreate()
df = spark.read.csv("stocks_data.csv", header=True, inferSchema=True)
df = df.withColumn('Date', to_date(col('Date'), 'yyyy-MM-dd'))
windowSpec = Window.partitionBy('ticker').orderBy('Date')
df = df.withColumn('prev_close', lag('close', default=None).over(windowSpec))
df = df.withColumn('daily_return', (col('close') - col('prev_close')) / col('prev_close'))


# Objective 1: Compute Daily Returns and Average Daily Return for All Stocks for Every Date
print("Objective 1: Compute daily returns and average daily return for all stocks for every date")
average_daily_return_df = df.groupBy('Date').agg(avg('daily_return').alias('average_return'))
average_daily_return_df.show()


# Objective 2: Which Stock Was Traded with the Highest Worth on Average
print("\nObjective 2: Which stock was traded with the highest worth (closing price * volume) on average")
df = df.withColumn('daily_worth', col('close') * col('volume'))
average_worth_df = df.groupBy('ticker').agg(avg('daily_worth').alias('average_worth'))
max_worth_df = average_worth_df.orderBy(col('average_worth').desc()).limit(1)
max_worth_df.show()


# Objective 3: Which Stock Was the Most Volatile as Measured by Annualized Std Dev of Daily Returns
print("\nObjective 3: Which stock was the most volatile as measured by annualized std dev of daily returns")
stddev_df = df.groupBy('ticker').agg(stddev('daily_return').alias('stddev_daily_return'))
annualization_factor = sqrt(252)
stddev_df = stddev_df.withColumn('annualized_stddev', col('stddev_daily_return') * annualization_factor)
most_volatile_df = stddev_df.orderBy(col('annualized_stddev').desc()).limit(1)
most_volatile_df.show()


# Objective 4: Top Three 30-Day Return Dates
print("\nObjective 4: Top three 30-day return dates")
df = df.withColumn('close_30d_prior', lag('close', 30).over(windowSpec))
df = df.withColumn('30d_return', (col('close') - col('close_30d_prior')) / col('close_30d_prior'))
df_30d = df.select('ticker', 'Date', '30d_return')
df_30d = df_30d.filter(col('30d_return').isNotNull())
top_three_df = df_30d.orderBy(col('30d_return').desc()).select('ticker', 'Date', '30d_return').limit(3)
top_three_df.show()
