// Databricks notebook source
import org.apache.spark.sql.SparkSession 
import org.apache.spark.sql.functions._ 
import org.apache.spark.sql.types._ 
import scala.collection.JavaConverters._ 
import java.util.Date
import java.time.LocalDate
import java.text.SimpleDateFormat
import spark.implicits._ 
import java.util.{ArrayList, List, UUID}
// CDQ Imports 
//import com.owl.core.Owl 
//import com.owl.common.options._
//import com.owl.common.domain2._
//import com.owl.core.util.OwlUtils
spark.catalog.clearCache

// COMMAND ----------

// MAGIC %python
// MAGIC # File location and type
// MAGIC file_location_Persons = "/FileStore/tables/HumanResource/Persons.csv"
// MAGIC #file_location_Employees = "/FileStore/tables/HumanResource/Employees.csv"
// MAGIC #file_location_Departments = "/FileStore/tables/HumanResource/Departments.csv"
// MAGIC #file_location_EmployeeDepartmentHistory = "/FileStore/tables/HumanResource/EmployeeDepartmentHistory.csv"
// MAGIC #file_location_Shifts = "/FileStore/tables/HumanResource/Shifts.csv"
// MAGIC 
// MAGIC file_type = "csv"
// MAGIC 
// MAGIC # CSV options
// MAGIC infer_schema = "false"
// MAGIC first_row_is_header = "true"
// MAGIC delimiter = ","
// MAGIC 
// MAGIC # The applied options are for CSV files. For other file types, these will be ignored.

// COMMAND ----------

// MAGIC %python
// MAGIC dfPersons = spark.read.format(file_type) \
// MAGIC   .option("inferSchema", infer_schema) \
// MAGIC   .option("header", first_row_is_header) \
// MAGIC   .option("sep", delimiter) \
// MAGIC   .load(file_location_Persons)

// COMMAND ----------

val file_location_Employees = "/FileStore/tables/HumanResource/Employees.csv"
val dfEmployees = (spark.read.format("csv").option("header", true).option("delimiter", ",").load(file_location_Employees))
dfEmployees.show()

// COMMAND ----------

dfEmployees.printSchema()

// COMMAND ----------

val file_location_Persons = "/FileStore/tables/HumanResource/Persons.csv"
val df = (spark.read.format("csv").option("header", true).option("delimiter", ",").load(file_location_Persons))
df.show()

// COMMAND ----------

df.printSchema()

// COMMAND ----------

//Variables to set up Collibra DQ Metastore database location
val pgHost = "xxxx.amazonaws.com" 
val pgDatabase = "postgres" 
val pgSchema = "public"
val pgUser = "???????" 
val pgPass = "????"
val pgPort = "0000"

// COMMAND ----------

// val opt = new OwlOptions
//--- Owl Metastore ---//
//opt.host = s"$owlHost"
//opt.port = s"5432/postgres?currentSchema=public"
//opt.pgUser = s"$owlUser"
//opt.pgPassword = s"$owlPass"
//--- Run Options ---//
//opt.dataset = "owl_test.nyse"
//opt.runId = "2018-01-10"
//opt.datasetSafeOff = true

//val owl = OwlUtils.OwlContext(jdbcDF, opt)
print("Calling OwlOptions library,,,,,,,")

// COMMAND ----------

val connProps = Map(
"driver" -> "org.postgresql.Driver", 
"user" -> "your-username", 
"password" -> "your-password",
"url" -> "jdbc:postgresql://abc:1234/postgres",
"dbtable" -> "public.example_data") 
//--- Load Spark DataFrame ---//
//val df = spark.read.format("jdbc").options(connProps).load display(df)
//display(df) // view your data

// COMMAND ----------

val dataset = "cdq_notebook_db_rules"
var date = "2018-01-11"

// Options
//val opt = new OwlOptions()
//opt.dataset = dataset
//opt.runId = date
//opt.host = pgHost
//opt.port = pgPort
//opt.pgUser = pgUser
//opt.pgPassword = pgPass

//opt.setDatasetSafeOff(false) // to enable historical overwrite of dataset

// COMMAND ----------

// Create a simple rule and assign it to dataset
//val simpleRule = OwlUtils.createRule(opt.dataset)
  //    simpleRule.setRuleNm("nyse-stocks-symbol")
  //    simpleRule.setRuleValue("symbol == 'BHK'")
  //    simpleRule.setRuleType("SQLG")
  //    simpleRule.setPerc(1.0)
  //    simpleRule.setPoints(1)
  //    simpleRule.setIsActive(1)
  //    simpleRule.setUserNm("admin")
  //    simpleRule.setPreviewLimit(8)
print("Creating Simple Rule and Assign it.....")

// COMMAND ----------

// Create a rule from generic rules that are created from UI:
//val genericRule = OwlUtils.createRule(opt.dataset)
  //  genericRule.setRuleNm("exchangeRule") // this could be any name
  //  genericRule.setRuleType("CUSTOM")
  //  genericRule.setPoints(1)
  //  genericRule.setIsActive(1)
  //  genericRule.setUserNm("admin")
  //  genericRule.setRuleRepo("exchangeCheckRule"); // Validate the generic rule name //from UI
   // genericRule.setRuleValue("EXCH") // COLUMN associate with the rule
print("Creating rule from generic Rules that are create from UI.....")

// COMMAND ----------


