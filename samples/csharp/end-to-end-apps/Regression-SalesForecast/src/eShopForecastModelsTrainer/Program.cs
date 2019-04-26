using Microsoft.ML;
using System;
using System.IO;
using System.Threading.Tasks;
using Common;
using static eShopForecastModelsTrainer.ConsoleHelpers;

namespace eShopForecastModelsTrainer
{
    class Program
    {
        private static readonly string BaseDatasetsRelativePath = @"../../../Data";
        private static readonly string ProductDataRealtivePath = $"{BaseDatasetsRelativePath}/products.stats.csv";

        private static readonly string ProductDataPath = GetAbsolutePath(ProductDataRealtivePath);

        static void Main(string[] args)
        {
            
            try
            {
                MLContext mlContext = new MLContext(seed: 1);  //Seed set to any number so you have a deterministic environment


                ///////////////////////////////
                /// 1. Get data ///////////////
                /// ///////////////////////////
                string dataPath = GetAbsolutePath(@"../../../Data/products.stats.csv");
                var trainingDataView = mlContext.Data.LoadFromTextFile<ProductData>(dataPath, hasHeader: true, separatorChar: ',');


                /////////////////////////////////////////////////
                /// 2. Run the algorithm to identify patterns ///
                /// /////////////////////////////////////////////
                ConsoleWriteHeader("Running algo over training data");

                var algorithm = mlContext.Regression.Trainers.FastTreeTweedie(labelColumnName: "Label", featureColumnName: "Features");

                var trainingPipeline = mlContext.Transforms.Concatenate(outputColumnName: "NumFeatures", nameof(ProductData.year),
                    nameof(ProductData.month), nameof(ProductData.units), nameof(ProductData.avg), nameof(ProductData.count),
                    nameof(ProductData.max), nameof(ProductData.min), nameof(ProductData.prev))
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "CatFeatures", inputColumnName: nameof(ProductData.productId)))
                    .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features", "NumFeatures", "CatFeatures"))
                    .Append(mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(ProductData.next)))
                    .Append(algorithm);

                // Run Cross-Validation on training data to assess the accuracy of our model
                Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
                var crossValidationResults = mlContext.Regression.CrossValidate(data: trainingDataView, estimator: trainingPipeline, numberOfFolds: 6, labelColumnName: "Label");
                ConsoleHelper.PrintRegressionFoldsAverageMetrics(algorithm.ToString(), crossValidationResults);

                //////////////////////////////////////////////////////////////
                /// 3. Save the patterns into some structure - the "model" ///
                //////////////////////////////////////////////////////////////
                // Train the model
                var model = trainingPipeline.Fit(trainingDataView);

                // Save the model for later comsumption from end-user apps
                mlContext.Model.Save(model, trainingDataView.Schema, "product_month_fastTreeTweedie.zip");

            }
            catch (Exception ex)
            {
                ConsoleWriteException(ex.Message);
            }
            ConsolePressAnyKey();
        }

        /// <summary>
        /// Train and save model for predicting next month country unit sales
        /// </summary>
        /// <param name="dataPath">Input training file path</param>
        /// <param name="outputModelPath">Trained model path</param>
        public static void TrainAndSaveModel(MLContext mlContext, string dataPath, string outputModelPath = "product_month_fastTreeTweedie.zip")
        {
            //if (File.Exists(outputModelPath))
            //{
            //    File.Delete(outputModelPath);
            //}

            CreateProductModelUsingPipeline(mlContext, dataPath, outputModelPath);
        }


        /// <summary>
        /// Build model for predicting next month country unit sales using Learning Pipelines API
        /// </summary>
        /// <param name="dataPath">Input training file path</param>
        /// <returns></returns>
        private static void CreateProductModelUsingPipeline(MLContext mlContext, string dataPath, string outputModelPath)
        {
            ConsoleWriteHeader("Training product forecasting");

            var trainingDataView = mlContext.Data.LoadFromTextFile<ProductData>(dataPath, hasHeader: true, separatorChar: ',');

            var trainer = mlContext.Regression.Trainers.FastTreeTweedie(labelColumnName: "Label", featureColumnName: "Features");

            var trainingPipeline = mlContext.Transforms.Concatenate(outputColumnName: "NumFeatures", nameof(ProductData.year), nameof(ProductData.month), nameof(ProductData.units), nameof(ProductData.avg), nameof(ProductData.count),
                nameof(ProductData.max), nameof(ProductData.min), nameof(ProductData.prev))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "CatFeatures", inputColumnName: nameof(ProductData.productId)))
                .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features", "NumFeatures", "CatFeatures"))
                .Append(mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(ProductData.next)))
                .Append(trainer);

            // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics
            Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
            var crossValidationResults = mlContext.Regression.CrossValidate(data: trainingDataView, estimator: trainingPipeline, numberOfFolds: 6, labelColumnName: "Label");
            ConsoleHelper.PrintRegressionFoldsAverageMetrics(trainer.ToString(), crossValidationResults);

            // Train the model
            var model = trainingPipeline.Fit(trainingDataView);

            // Save the model for later comsumption from end-user apps
            mlContext.Model.Save(model, trainingDataView.Schema, outputModelPath);
        }

        public static string GetAbsolutePath(string relativeDatasetPath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativeDatasetPath);

            return fullPath;
        }
    }
}
