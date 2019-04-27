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
        static void Main(string[] args)
        {       
            try
            {
                // Initialize ML.NET assemblies and set random seed for consistent test runs
                MLContext mlContext = new MLContext(seed: 1);


                ////////////////////////////////////////////////////////////////
                /// 1. Get data                                              ///
                /// ////////////////////////////////////////////////////////////
                string dataPath = GetAbsolutePath(@"../../../Data/products.stats.csv");
                var trainingDataView = mlContext.Data.LoadFromTextFile<ProductData>(dataPath, hasHeader: true, separatorChar: ',');


                ////////////////////////////////////////////////////////////////
                /// 2. Run passes over data to ID patterns - the "algorithm" ///
                /// ////////////////////////////////////////////////////////////
                ConsoleWriteHeader("Running algorithm over training data...");

                var algorithm = mlContext.Regression.Trainers.FastTree();
                //var algorithm = mlContext.Regression.Trainers.FastTreeTweedie();
                //var algorithm = mlContext.Regression.Trainers.Sdca();
                //var algorithm = mlContext.Regression.Trainers.FastForest();

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
                var model = trainingPipeline.Fit(trainingDataView);
                mlContext.Model.Save(model, trainingDataView.Schema, "sales_forecast_model.zip");
                Console.WriteLine("Model successfully saved to file.");


                //////////////////////////////////////////////////////////////
                /// 4. Send new data through to make predictions /////////////
                /// //////////////////////////////////////////////////////////
                //ProductModelHelper.TestPrediction(mlContext, "sales_forecast_model.zip");

            }
            catch (Exception ex)
            {
                ConsoleWriteException(ex.Message);
            }
            ConsolePressAnyKey();
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
