import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

object HousePriceAnalysis extends App {
  Logger.getLogger("org.apache").setLevel(Level.WARN)

  val spark = SparkSession.builder().appName("liner_Reg").master("local[*]").getOrCreate()

  //read data
  val data = spark.read.option("header", true)
    .option("inferschema",true)
    .csv("src/main/resources/kc_house_data.csv")

//  data.printSchema()
//  data.show()

  val vectorAssembler = new VectorAssembler()
    .setInputCols(Array("bedrooms","bathrooms","bathrooms","sqft_lot","floors","grade"))
    .setOutputCol("features")

  val trans_data = vectorAssembler.transform(data)
//
//  trans_data.printSchema()
//
//  trans_data.show()

  val model_data = trans_data.select("price","features")
    .withColumnRenamed("price","label")

//  model_data.printSchema()
//  model_data.show()

  val split_dataset = model_data.randomSplit(Array(0.8,0.2))

  val training_data = split_dataset(0)

  val test_data = split_dataset(1)

  val model = new LinearRegression().fit(training_data)

  println("the training data r2 value is "+ model.summary.r2 + "and rmse is "+ model.summary.rootMeanSquaredError)
  println("the Test data r2 value is "+ model.evaluate(test_data).r2 + "and rmse is "+ model.evaluate(test_data).rootMeanSquaredError)

  model.transform(test_data).show()




}
