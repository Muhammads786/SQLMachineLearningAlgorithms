# Databricks notebook source
import numpy as np
import pandas as pd
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import * 
#import pyspark.pandas as ps

# COMMAND ----------

from pyspark.ml.linalg import Vector
from pyspark.ml.feature  import VectorAssembler
from pyspark.ml.regression import LinearRegression

# COMMAND ----------

#dbutils.fs.mkdirs('/FileStore/tables/regressions')

# COMMAND ----------

dbutils.fs.ls('/FileStore/tables/')

# COMMAND ----------

# File location and type
file_location_Auto_Data = "/FileStore/tables/regressions/auto_mpg.csv"


file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.


# COMMAND ----------

dfAutodata = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location_Auto_Data)


# COMMAND ----------

dfAutodata.show(100)

# COMMAND ----------

dfAutodata.printSchema()

# COMMAND ----------

dfAutodata.createOrReplaceTempView("VW_AUTO_MPG")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM VW_AUTO_MPG

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT *) Total_Unique_Auto_Mpg_Records FROM VW_AUTO_MPG

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) Total_NULL 
# MAGIC FROM VW_AUTO_MPG 
# MAGIC WHERE cylinders  IS NULL

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) Total_NULL 
# MAGIC FROM VW_AUTO_MPG 
# MAGIC WHERE displacements  IS NULL

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) Total_NULL 
# MAGIC FROM VW_AUTO_MPG 
# MAGIC WHERE horsepower  IS NULL

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) Total_NULL 
# MAGIC FROM VW_AUTO_MPG 
# MAGIC WHERE weight IS NULL

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) Total_NULL 
# MAGIC FROM VW_AUTO_MPG 
# MAGIC WHERE accelaration  IS NULL

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) Total_NULL 
# MAGIC FROM VW_AUTO_MPG 
# MAGIC WHERE model_year  IS NULL

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) Total_NULL 
# MAGIC FROM VW_AUTO_MPG 
# MAGIC WHERE origin  IS NULL

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) Total_NULL 
# MAGIC FROM VW_AUTO_MPG 
# MAGIC WHERE car_name  IS NULL

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC cast(mpg as double),
# MAGIC cast(cylinders as double),
# MAGIC cast(displacements as double),
# MAGIC cast(horsepower as double),
# MAGIC cast(weight as double),
# MAGIC cast(accelaration as double),
# MAGIC cast(model_year as double),
# MAGIC cast(origin as double),
# MAGIC car_name
# MAGIC FROM VW_AUTO_MPG
# MAGIC WHERE horsepower='?'

# COMMAND ----------

dfAutoDataPrepared = spark.sql("SELECT cast(mpg as double), cast(cylinders as double), cast(displacements as double), cast(horsepower as double), cast(weight as double), \
cast(accelaration as double), cast(model_year as double), cast(origin as double)  FROM VW_AUTO_MPG WHERE horsepower<>'?' and cylinders IN ('4','6','8')" )

# COMMAND ----------

display(dfAutoDataPrepared)

# COMMAND ----------

dfAutoDataPrepared.describe().show()

# COMMAND ----------

#correlation between features and output
dfAutoDataPrepared.select(corr('cylinders','mpg')).show()

# COMMAND ----------

dfAutoDataPrepared.select(corr('accelaration','mpg')).show()

# COMMAND ----------

dfAutoDataPrepared.select(corr('horsepower','mpg')).show()

# COMMAND ----------

dfAutoDataPrepared.select(corr('origin','mpg')).show()

# COMMAND ----------

dfAutoDataPrepared.select(corr('model_year','mpg')).show()

# COMMAND ----------

dfAutoDataPrepared.select(corr('weight','mpg')).show()

# COMMAND ----------

display(dfAutoDataPrepared.select("weight","mpg"))

# COMMAND ----------

#display(dfAutoDataPrepared.select("displacements","mpg"))
dfAutoDataPrepared.select(corr('weight','cylinders')).show()

# COMMAND ----------

dfAutoDataPrepared.select(corr('weight','displacements')).show()

# COMMAND ----------

dfAutoDataPrepared.select(corr('cylinders','horsepower')).show()

# COMMAND ----------

dfAutoDataPrepared.select(corr('cylinders','displacements')).show()

# COMMAND ----------

dfAutoDataPrepared.select(corr('displacements','model_year')).show()

# COMMAND ----------

# mpg,cylinders,displacements,horsepower,weight,accelaration,model_year,origin,car_name
#,'horsepower',,'accelaration','model_year','origin'
vec_assembler  = VectorAssembler(inputCols=['cylinders','displacements','weight','horsepower','accelaration','model_year','origin'],outputCol='features')
# vec_assembler  = VectorAssembler(inputCols=['cylinders','model_year','origin'],outputCol='features')
#vec_assembler  = VectorAssembler(inputCols=['cylinders','displacements','weight','model_year','origin','accelaration'],outputCol='features')

# COMMAND ----------

features_df = vec_assembler.transform(dfAutoDataPrepared)

# COMMAND ----------

features_df.select("*").show()

# COMMAND ----------

linear_model_df = features_df.select("features","mpg")

# COMMAND ----------

display(linear_model_df)

# COMMAND ----------

train_df,test_df,validation_df = linear_model_df.randomSplit([0.6,0.2,0.2])

# COMMAND ----------

print(train_df.count(),len(train_df.columns))

# COMMAND ----------

linear_reg = LinearRegression(labelCol='mpg')

# COMMAND ----------

lr_model = linear_reg.fit(train_df)

# COMMAND ----------

print(lr_model.coefficients)

# COMMAND ----------

print(lr_model.intercept)

# COMMAND ----------

training_prediction = lr_model.evaluate(train_df)

# COMMAND ----------

print(training_prediction.r2)

# COMMAND ----------

testing_prediction = lr_model.evaluate(test_df)

# COMMAND ----------

print(testing_prediction.r2)

# COMMAND ----------

display(test_df.select("features","mpg")) 

# COMMAND ----------

validation_prediction = lr_model.evaluate(validation_df)

# COMMAND ----------

print(validation_prediction.r2)

# COMMAND ----------

print(testing_prediction.meanSquaredError)

# COMMAND ----------

display(validation_df.select("features","mpg"))

# COMMAND ----------


