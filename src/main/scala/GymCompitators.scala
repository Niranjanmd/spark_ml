import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

object GymCompitators extends App {

  Logger.getLogger("org.apache").setLevel(Level.WARN)

  val spark = SparkSession.builder().appName("liner_Reg").master("local[*]").getOrCreate()

  //read data
  val data = spark.read.option("header", true)
    .option("inferschema",true)
    .csv("src/main/resources/GymCompetition.csv")

//  data.printSchema()
//
//  data.show()


  //Create Vector passing the features
  val vectorAssembler = new VectorAssembler()
  vectorAssembler.setInputCols(Array("Age","Height","Weight"))
  vectorAssembler.setOutputCol("features")

  //Creates data set with features column
  val data_vector = vectorAssembler.transform(data)

  // take only label and features
 val model_input = data_vector.select("NoOfReps","features")
      .withColumnRenamed("NoOfReps","label")

//  model_input.show()

  val linearRegression = new LinearRegression

  val model=linearRegression.fit(model_input)

  println("The model has intercept " + model.intercept)
  println("The model has coefficients " + model.coefficients)

  model.transform(model_input).show()

}
