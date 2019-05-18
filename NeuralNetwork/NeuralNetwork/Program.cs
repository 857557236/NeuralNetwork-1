using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace NeuralNetwork
{
    class Program
    {
        //The CSV files from which the data is read
        static string filename = "mnist_train_100.csv";

        static void Main(string[] args)
        {
            //Create an instance of streamreader to read from the file
            TextReader reader = new StreamReader(filename);
            string list = reader.ReadToEnd();
            TextWriter writer = new StreamWriter("answers.txt");

            //Split the file to get the numbers
            string[] allValues = list.Split(new char[] { ',',' ','\n','\t','.' });
            int imageCounter = 0;
            int targetValue = 0;

            int input_Nodes = 784;
            int hidden_Nodes = 100;
            int output_Nodes = 10;
            double learning_Rate = 0.1;

            NeuralNetwork n = new NeuralNetwork(input_Nodes, hidden_Nodes, output_Nodes, learning_Rate);

            writer.WriteLine("Training....");
            int numberLength = 100;
            int[] scorecard = new int[numberLength];
            for (int i = 0; i < numberLength; i++)
            {
                //Create an image array to hold the coordinates of each number
                int[,] imageArray = new int[28, 28];
                double[] input = new double[784];
                int newCounter = 0;

                for (int x = 0; x < imageArray.GetLength(0); x++)
                {
                    for (int y = 0; y < imageArray.GetLength(1); y++)
                    {
                        imageCounter++;

                        imageArray[x, y] = int.Parse(allValues[imageCounter]);
                        input[newCounter] = (imageArray[x, y] / 255.0 * 0.99) + 0.01;
                        newCounter++;

                        if (imageArray[x, y] == 0)
                        {
                            writer.Write(" ");
                        }
                        else
                        {
                            writer.Write("*");
                        }
                    }
                   writer.WriteLine();
                }
                imageCounter++;

                double[] targets = new double[output_Nodes];
                for (int j = 0; j < targets.Length; j++)
                {
                    targets[j] = 0.01;
                }
                writer.WriteLine("{0} - correct label",allValues[targetValue]);
                targets[int.Parse((allValues[targetValue]))] = 0.99;

                for (int k = 0; k < 10; k++)
                {
                    n.Train(input, targets);
                }

                double[] result = n.Query(input);
                int highestValue = 0;
               
                for (int l = 0; l < result.Length; l++)
                {
                    if (result[l] == result.Max())
                    {
                        highestValue = l;
                    }
                }
                writer.WriteLine("{0} - network's answer",highestValue);
                int value = int.Parse(allValues[targetValue]);
                writer.WriteLine();
                //Go to the next number
                targetValue += 785;

                if (value == highestValue)
                {
                    scorecard[i] = 1;
                }
                else
                {
                    scorecard[i] = 0;
                }
            }
            double correctCounter = 0;
            writer.Write("Scorecard: [");
            for (int l = 0; l < scorecard.Length; l++)
            {
                if (scorecard[l] == 1)
                {
                    correctCounter++;
                }
                Console.Write(" {0} ", scorecard[l]);
            }
            writer.Write("]");
            writer.WriteLine("Your accuracy was {0}%", (correctCounter / numberLength) * 100);
        }

    }

    public class NeuralNetwork
    {
        public int InputNodes { get; set; }
        public int HiddenNodes { get; set; }
        public int OutputNodes { get; set; }
        public double LearningRate { get; set; }
        public double[,] WInputHidden { get; set; }
        public double[,] WHiddenOutput { get; set; }

        private static Random rnd=new Random();

        private void Randomize(double[,] matrix)
        {
            for(int row = 0; row < matrix.GetLength(0); row++)
            {
                for (int col = 0; col < matrix.GetLength(1); col++)
                {
                    matrix[row, col] = rnd.NextDouble() - 0.5;
                }
            }
        }

        public NeuralNetwork()
        {
            InputNodes = 3;
            HiddenNodes = 3;
            OutputNodes = 3;
            LearningRate = 0.3;
            WInputHidden = new double[HiddenNodes, InputNodes];
            WHiddenOutput = new double[OutputNodes, HiddenNodes];
            Randomize(WInputHidden);
            Randomize(WHiddenOutput);
        }

        public NeuralNetwork(int inputNodes,int hiddenNodes,int outputNodes,double learningRate)
        {
            InputNodes = inputNodes;
            HiddenNodes = hiddenNodes;
            OutputNodes = outputNodes;
            LearningRate = learningRate;
            WInputHidden = new double[HiddenNodes, InputNodes];
            WHiddenOutput = new double[OutputNodes, HiddenNodes];
            Randomize(WInputHidden);
            Randomize(WHiddenOutput);
        }

        public void Train(double[] inputs,double[] targets) 
        {
            double[] hiddenInputs=GetInputs(inputs,WInputHidden.Transpose());
            double[] hiddenOutputs = ActivationFunction(hiddenInputs);

            double[] finalInputs = GetInputs(hiddenOutputs, WHiddenOutput.Transpose());
            double[] finalOutputs = ActivationFunction(finalInputs);

            double[] outputError = new double[finalOutputs.Length];
            for(int i = 0; i < outputError.Length; i++)
            {
                outputError[i] = targets[i] - finalOutputs[i];
            }
            double[] hiddenError = GetInputs(outputError, WHiddenOutput);

            for(int j = 0; j < WHiddenOutput.GetLength(0); j++)
            {
                for (int k = 0; k < WHiddenOutput.GetLength(1); k++)
                {
                    double weightDelta = -LearningRate * outputError[j] * finalOutputs[j] * (1 - finalOutputs[j]) * hiddenOutputs[j];
                    WHiddenOutput[j, k] -= weightDelta;
                }
            }

            for (int j = 0; j < WInputHidden.GetLength(0); j++)
            {
                for (int k = 0; k < WInputHidden.GetLength(1); k++)
                {
                    double weightDelta = -LearningRate * hiddenError[j] * hiddenOutputs[j] * (1 - hiddenOutputs[j]) * hiddenInputs[j];
                    WInputHidden[j, k] -= weightDelta;
                }
            }
        }

        public double[] Query(double[] inputs)
        {
            double[] hiddenInputs = GetInputs(inputs, WInputHidden.Transpose());
            double[] hiddenOutputs = ActivationFunction(hiddenInputs);

            double[] finalInputs = GetInputs(hiddenOutputs, WHiddenOutput.Transpose());
            double[] finalOutputs = ActivationFunction(finalInputs);

            return finalOutputs;
        }

        public double[] GetInputs(double[] matrix1, double[,] matrix2)
        {
            double[] result = new double[matrix2.GetLength(1)];

            for (int x = 0; x < matrix2.GetLength(1); x++)
            {
                for (int y = 0; y < matrix2.GetLength(0); y++)
                {
                    result[x] += (matrix2[y,x]) * matrix1[y];
                }
            }
            return result;
        }

        public double[] ActivationFunction(double[] matrix)
        {
            for(int x = 0; x < matrix.Length; x++) 
            {
                double value = matrix[x];
                matrix[x] = (1 / (1 + Math.Pow(Math.E, -value)));
            }
            return matrix;
        }
        
    }

    public static class ArrayExtensions
    {
        public static double[,] Transpose(this double[,] matrix)
        {
            double[,] tranposedMatrix = new double[matrix.GetLength(1), matrix.GetLength(0)];
            for(int x = 0; x < matrix.GetLength(0); x++)
            {
                for (int y = 0; y < matrix.GetLength(1); y++)
                {
                    tranposedMatrix[y, x] = matrix[x, y];
                }
            }  
            return tranposedMatrix;
        }

    }
}
