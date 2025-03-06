using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace column_rad_app
{
    class Program
    {
        static void Main(string[] args)
        {
            // create InferenceSessions for the two models.
            var modelPath = System.IO.Path.Combine(AppContext.BaseDirectory, "column_model_pipeline.onnx");
            var onnxSession = new InferenceSession(modelPath);

            var inversePath = System.IO.Path.Combine(AppContext.BaseDirectory, "inverse_target_scaler.onnx");
            var inverseSession = new InferenceSession(inversePath);

            // example input values.
            string columnStyleInput = "Simple Tube";
            double height = 300;
            double width = 500;
            int columns = 3;

            // map categorical value to a number.
            int encodedColumnStyle = (columnStyleInput == "Flat") ? 0 : (columnStyleInput == "Simple Tube" ? 1 : 0);

            // create separate tensors for each input feature.
            // each tensor is created with shape [1, 1] (one sample, one feature)
            var tensorColumnStyle = new DenseTensor<float>(new float[] { encodedColumnStyle }, new int[] { 1, 1 });
            var tensorHeight = new DenseTensor<float>(new float[] { (float)height }, new int[] { 1, 1 });
            var tensorWidth = new DenseTensor<float>(new float[] { (float)width }, new int[] { 1, 1 });
            var tensorCols = new DenseTensor<float>(new float[] { (float)columns }, new int[] { 1, 1 });

            // build the list of NamedOnnxValue objects with keys matching the model's input names.
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("Column_Style", tensorColumnStyle),
                NamedOnnxValue.CreateFromTensor("Height", tensorHeight),
                NamedOnnxValue.CreateFromTensor("Width", tensorWidth),
                NamedOnnxValue.CreateFromTensor("Cols", tensorCols)
            };

            // run the model. we get scaled prediction here.
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = onnxSession.Run(inputs);
            var predictionTensor = results.First().AsTensor<float>();
            float scaledPrediction = predictionTensor[0];

            // prepare input for inverse scaling
            var reverseInput = new DenseTensor<float>(new float[] { scaledPrediction }, new int[] { 1, 1 });

            var revInputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", reverseInput),
            };

            // apply inverse scaling model.
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> resultsInverse = inverseSession.Run(revInputs);
            var predictionTensorInverse = resultsInverse.First().AsTensor<float>();
            float Prediction = predictionTensorInverse[0];
            
            Console.WriteLine($"\nPrediction : {Prediction}");


        }
    }
}