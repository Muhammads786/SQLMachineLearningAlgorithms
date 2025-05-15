# Databricks notebook source
import numpy as np
import pandas as pd
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import * 

# COMMAND ----------

from pyspark.ml.linalg import Vector
from pyspark.ml.feature  import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.classification import LogisticRegression

# COMMAND ----------

#/FileStore/tables/classification/adult.csv

# COMMAND ----------

dbutils.fs.ls('/FileStore/tables/classification/')

# COMMAND ----------

# File location and type
file_location_Auto_Data = "/FileStore/tables/classification/adult.csv"


file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.

# COMMAND ----------

dfAdultdata = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location_Auto_Data)

# COMMAND ----------

dfAdultdata.printSchema()

# COMMAND ----------

dfAdultdata.show(1000)

# COMMAND ----------

dfAdultdata.createOrReplaceTempView("VW_ADULTS")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM VW_ADULTS LIMIT 1000;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT class FROM VW_ADULTS

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) Total, class, gender 
# MAGIC FROM VW_ADULTS
# MAGIC GROUP BY Class, gender

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) Total, class, age
# MAGIC FROM VW_ADULTS
# MAGIC GROUP BY Class, age

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) Total, class, workclass
# MAGIC FROM VW_ADULTS
# MAGIC GROUP BY Class, workclass

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) Total, class, marital_status 
# MAGIC FROM VW_ADULTS
# MAGIC GROUP BY Class, marital_status

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) Total, class, occupation 
# MAGIC FROM VW_ADULTS
# MAGIC GROUP BY Class, occupation

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) Total, class, race 
# MAGIC FROM VW_ADULTS
# MAGIC GROUP BY Class, race

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) Total, class, hours_per_week 
# MAGIC FROM VW_ADULTS
# MAGIC GROUP BY Class, hours_per_week

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) Total, class, fnlwgt 
# MAGIC FROM VW_ADULTS
# MAGIC GROUP BY Class, fnlwgt

# COMMAND ----------

dfAdultdata_indexer = StringIndexer(inputCol="workclass",outputCol="workclass_num").fit(dfAdultdata)
dfAdultdataVector = dfAdultdata_indexer.transform(dfAdultdata)
dfAdultdataVector.show(10)

# COMMAND ----------

dfAdultdataVector.select("workclass","workclass_num").distinct().orderBy("workclass_num").show()

# COMMAND ----------

dfAdultdata_encoder = OneHotEncoder(inputCol="workclass_num",outputCol="workclass_vector")
dfAdultWithWorClass_fit_encoder=dfAdultdata_encoder.fit(dfAdultdataVector)
dfAdultdataVector=dfAdultWithWorClass_fit_encoder.transform(dfAdultdataVector)

# COMMAND ----------

dfAdultdataVector.show(10)
#workclass_vector is showing oneHotEncoding in terms of (total_values,position_of_the_value,actual_value(0,1))

# COMMAND ----------

dfAdultdata_indexer = StringIndexer(inputCol="marital_status",outputCol="marital_status_num").fit(dfAdultdataVector)
dfAdultdataVector = dfAdultdata_indexer.transform(dfAdultdataVector)
dfAdultdataVector.show(10)

# COMMAND ----------

dfAdultdataVector.select("marital_status","marital_status_num").distinct().orderBy("marital_status_num").show()

# COMMAND ----------

dfAdultdata_encoder = OneHotEncoder(inputCol="marital_status_num",outputCol="marital_status_vector")
dfAdult_marital_status_num_fit_encoder=dfAdultdata_encoder.fit(dfAdultdataVector)
dfAdultdataVector=dfAdult_marital_status_num_fit_encoder.transform(dfAdultdataVector)

# COMMAND ----------

dfAdultdataVector.show(10)

# COMMAND ----------

dfAdultdata_indexer = StringIndexer(inputCol="occupation",outputCol="occupation_num").fit(dfAdultdataVector)
dfAdultdataVector = dfAdultdata_indexer.transform(dfAdultdataVector)
dfAdultdataVector.show(10)


# COMMAND ----------

dfAdultdataVector.select("occupation","occupation_num").distinct().orderBy("occupation_num").show()

# COMMAND ----------

dfAdultdata_encoder = OneHotEncoder(inputCol="occupation_num",outputCol="occupation_vector")
dfAdult_occupation_fit_encoder=dfAdultdata_encoder.fit(dfAdultdataVector)
dfAdultdataVector=dfAdult_occupation_fit_encoder.transform(dfAdultdataVector)


# COMMAND ----------

dfAdultdata_indexer = StringIndexer(inputCol="relationship",outputCol="relationship_num").fit(dfAdultdataVector)
dfAdultdataVector = dfAdultdata_indexer.transform(dfAdultdataVector)
dfAdultdataVector.show(10)


# COMMAND ----------

dfAdultdataVector.select("relationship","relationship_num").distinct().orderBy("relationship_num").show()

# COMMAND ----------

dfAdultdata_encoder = OneHotEncoder(inputCol="relationship_num",outputCol="relationship_vector")
dfAdult_relationship_fit_encoder=dfAdultdata_encoder.fit(dfAdultdataVector)
dfAdultdataVector=dfAdult_relationship_fit_encoder.transform(dfAdultdataVector)
dfAdultdataVector.show(10)

# COMMAND ----------

dfAdultdata_indexer = StringIndexer(inputCol="race",outputCol="race_num").fit(dfAdultdataVector)
dfAdultdataVector = dfAdultdata_indexer.transform(dfAdultdataVector)
dfAdultdataVector.show(10)

# COMMAND ----------

dfAdultdataVector.select("race","race_num").distinct().orderBy("race_num").show()

# COMMAND ----------

dfAdultdata_encoder = OneHotEncoder(inputCol="race_num",outputCol="race_vector")
dfAdult_race_fit_encoder=dfAdultdata_encoder.fit(dfAdultdataVector)
dfAdultdataVector=dfAdult_race_fit_encoder.transform(dfAdultdataVector)
dfAdultdataVector.show(10)

# COMMAND ----------

dfAdultdata_indexer = StringIndexer(inputCol="gender",outputCol="gender_num").fit(dfAdultdataVector)
dfAdultdataVector = dfAdultdata_indexer.transform(dfAdultdataVector)
dfAdultdataVector.show(10)

# COMMAND ----------

dfAdultdataVector.select("gender","gender_num").distinct().orderBy("gender_num").show()

# COMMAND ----------

dfAdultdata_encoder = OneHotEncoder(inputCol="gender_num",outputCol="gender_vector")
dfAdult_gender_fit_encoder=dfAdultdata_encoder.fit(dfAdultdataVector)
dfAdultdataVector=dfAdult_gender_fit_encoder.transform(dfAdultdataVector)
dfAdultdataVector.show(10)

# COMMAND ----------

dfAdultdata_indexer = StringIndexer(inputCol="native_country",outputCol="native_country_num").fit(dfAdultdataVector)
dfAdultdataVector = dfAdultdata_indexer.transform(dfAdultdataVector)
dfAdultdataVector.show(10)

# COMMAND ----------

dfAdultdataVector.select("native_country","native_country_num").distinct().orderBy("native_country_num").show()

# COMMAND ----------

dfAdultdata_encoder = OneHotEncoder(inputCol="native_country_num",outputCol="native_country_vector")
dfAdult_native_country_fit_encoder=dfAdultdata_encoder.fit(dfAdultdataVector)
dfAdultdataVector=dfAdult_native_country_fit_encoder.transform(dfAdultdataVector)
dfAdultdataVector.show(10)

# COMMAND ----------

display(dfAdultdataVector.select("workclass_vector","marital_status_vector","occupation_vector","relationship_vector","race_vector","gender_vector","native_country_num"))

# COMMAND ----------

dfAdultdataVector.printSchema()

# COMMAND ----------

dfAdultdataVector=dfAdultdataVector.drop("class_output")
dfAdultdataVector.createOrReplaceTempView("VW_ADULTS_VECTOR") 

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC select if('>50K'!='>50K',2,0)

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC --SELECT COUNT(*),class_output
# MAGIC --FROM(
# MAGIC SELECT 
# MAGIC Cast(age as double) age,
# MAGIC education,
# MAGIC cast(education_num as double) education_num,
# MAGIC cast(hours_per_week as double) hours_per_week,
# MAGIC cast(Replace(Replace(class,'>50K','1'),'<=50K','0') as double) class_output,
# MAGIC workclass_num,
# MAGIC workclass_vector,
# MAGIC marital_status_num,
# MAGIC marital_status_vector,
# MAGIC occupation_num,
# MAGIC occupation_vector,
# MAGIC relationship_num,
# MAGIC relationship_vector
# MAGIC race_num,
# MAGIC race_vector,
# MAGIC gender_num,
# MAGIC gender_vector,
# MAGIC native_country_num,
# MAGIC native_country_vector
# MAGIC FROM VW_ADULTS_VECTOR
# MAGIC --)x
# MAGIC --group by class_output

# COMMAND ----------

dfAdultdataVectorPrepared = spark.sql("SELECT Cast(age as double) age,education, cast(education_num as double) education_num,cast(hours_per_week as double) hours_per_week,cast(Replace(Replace(class,'>50K','1'),'<=50K','0') as double) class_output,workclass_num, workclass_vector, marital_status_num, marital_status_vector,occupation_num, occupation_vector, relationship_num, relationship_vector,race_num, race_vector, gender_num, gender_vector, native_country_num, native_country_vector,class FROM VW_ADULTS_VECTOR")

# COMMAND ----------

dfAdultdataVectorPrepared.printSchema()

# COMMAND ----------

display(dfAdultdataVectorPrepared)

# COMMAND ----------

dfAdultdataVectorPrepared.groupBy("class").count().show()

# COMMAND ----------

dfAdultdataVectorPrepared.groupBy("class_output").count().show()

# COMMAND ----------

dfAdultdataVectorAssembler = VectorAssembler(inputCols=['age','education_num','hours_per_week','workclass_vector','marital_status_vector','occupation_vector','relationship_vector','race_vector','gender_vector','native_country_vector'],outputCol='features')

# COMMAND ----------

dfAdultdataVectorPrepared = dfAdultdataVectorAssembler.transform(dfAdultdataVectorPrepared)

# COMMAND ----------

display(dfAdultdataVectorPrepared.select("features","class_output"))

# COMMAND ----------

lgr_model_df = dfAdultdataVectorPrepared.select(['features','class_output'])

# COMMAND ----------

lgr_model_df.show(100)

# COMMAND ----------

traning_df,testing_df,validation_df = lgr_model_df.randomSplit([0.6,0.2,0.2])

# COMMAND ----------

print(traning_df.count(),len(traning_df.columns))

# COMMAND ----------

traning_df.groupBy("class_output").count().show()

# COMMAND ----------

log_reg = LogisticRegression(labelCol='class_output').fit(traning_df)

# COMMAND ----------

training_results = log_reg.evaluate(traning_df).predictions

# COMMAND ----------

display(training_results.select("class_output","prediction","probability").where("prediction==1.0 and class_output==1.0"))

# COMMAND ----------

display(training_results.select("class_output","prediction","probability").where("prediction==1.0 and class_output==0.0"))

# COMMAND ----------

display(training_results.select("class_output","prediction","probability").where("prediction==1.0 and class_output==1.0").count())

# COMMAND ----------

display(training_results.select("class_output","prediction","probability").where("prediction==0.0 and class_output==0.0").count())

# COMMAND ----------

tp = training_results.select("class_output","prediction","probability").where("prediction==1.0 and class_output==1.0").count()
tn = training_results.select("class_output","prediction","probability").where("prediction==0.0 and class_output==0.0").count()
fp = training_results.select("class_output","prediction","probability").where("prediction==1.0 and class_output==0.0").count()
fn = training_results.select("class_output","prediction","probability").where("prediction==0.0 and class_output==1.0").count()

# COMMAND ----------

#correct prediction % from total dataset
accuracy = float((tp+tn)/(training_results.count()))
print(accuracy)

# COMMAND ----------

#recall % of total positive from total positive observations
recall = float(tp/(tp+fn))
print(recall)

# COMMAND ----------

#precision % of total negative from total positive observations
precision = float(tn/(tn+fp))
print(precision)

# COMMAND ----------

#Evaluating Logistic Regression Model on Testing Data

# COMMAND ----------

testing_df.groupBy("class_output").count().show()

# COMMAND ----------

testing_results = log_reg.evaluate(testing_df).predictions

# COMMAND ----------

#true positive
display(testing_results.select("class_output","prediction","probability").where("prediction==1.0 and class_output==1.0"))

# COMMAND ----------

#false positive
display(testing_results.select("class_output","prediction","probability").where("prediction==1.0 and class_output==0.0"))

# COMMAND ----------

#true negative
display(testing_results.select("class_output","prediction","probability").where("prediction==0.0 and class_output==0.0"))

# COMMAND ----------

#false negative
display(testing_results.select("class_output","prediction","probability").where("prediction==0.0 and class_output==1.0"))

# COMMAND ----------

ttp = testing_results.select("class_output","prediction","probability").where("prediction==1.0 and class_output==1.0").count()
ttn = testing_results.select("class_output","prediction","probability").where("prediction==0.0 and class_output==0.0").count()
tfp = testing_results.select("class_output","prediction","probability").where("prediction==1.0 and class_output==0.0").count()
tfn = testing_results.select("class_output","prediction","probability").where("prediction==0.0 and class_output==1.0").count()

# COMMAND ----------

#correct prediction % from total dataset
t_accuracy = float((ttp+ttn)/(testing_results.count()))
print(t_accuracy)

# COMMAND ----------

#recall % of total positive from total positive observations
t_recall = float(ttp/(ttp+tfn))
print(t_recall)

# COMMAND ----------

#precision % of total negative from total positive observations
t_precision = float(ttn/(ttn+tfp))
print(t_precision)

# COMMAND ----------


