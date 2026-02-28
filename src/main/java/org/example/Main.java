package org.example;

import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.*;
import java.util.*;

/**
 * Main class that runs the neural network training and evulation
 */
public class Main {

    public static void main(String[] args) throws IOException {
        String configPath = "config.txt";
        // load parameters from the config file
        Configuration config = ConfigLoader.loadConfig(configPath);


        String resultsDir = "result";
        new File(resultsDir).mkdir();

        // print settings to the console
        System.out.println("=== UKRYTE NEURONY = " + config.getHiddenNeurons());
        System.out.println("=== FUNKCJA AKTYWACJI = " + config.getActivation().toString().toUpperCase());
        System.out.println("=== EPOKI = " + config.getEpochs());
        System.out.println("=== WSPÓŁCZYNNIK UCZENIA = " + config.getLearningRate());
        System.out.println("=== CIERPLIWOŚĆ = " + config.getPatience());
        System.out.println("=== BATCH SIZE = " + config.getBatchSize());

        // Load data from csv folders
        List<String> trainingFolders = Arrays.asList(
                "data/f8/stat",
                "data/f10/stat"
        );
        List<String> testingFolders = Arrays.asList(
                "data/f8/dyn",
                "data/f10/dyn"
        );

        DataSet trainData = FileLoader.loadDataFromFolders(trainingFolders);
        DataSet testData = FileLoader.loadDataFromFolders(testingFolders);

        // Normalize data (Scale numbers to similar range )
        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fitLabel(true);
        normalizer.fit(trainData);
        normalizer.transform(trainData);
        normalizer.transform(testData);

        DataSetIterator trainIter = new ListDataSetIterator<>(trainData.asList(), config.getBatchSize());
        DataSetIterator testIter = new ListDataSetIterator<>(testData.asList(), config.getBatchSize());

        // Build the neural network architector
        MultiLayerConfiguration networkConfig = new NeuralNetConfiguration.Builder()
                .seed(123456)
                .updater(new Adam(config.getLearningRate()))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(2)
                        .nOut(config.getHiddenNeurons())
                        .weightInit(WeightInit.XAVIER)
                        .activation(config.getActivation())
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(config.getHiddenNeurons())
                        .nOut(2)
                        .activation(org.nd4j.linalg.activations.Activation.IDENTITY)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(networkConfig);
        model.init();

        List<Double> trainMSEList = new ArrayList<>();
        List<Double> testMSEList = new ArrayList<>();


        // training loop with early stopping
        int noImprovementCount = 0;
        double bestTestScore = Double.MAX_VALUE;


        for (int epoch = 0; epoch < config.getEpochs(); epoch++) {
            trainIter.reset();
            model.fit(trainIter); // train te model


            double trainScore = model.score(trainData);
            double testScore = model.score(testData);

            trainMSEList.add(trainScore);
            testMSEList.add(testScore);

            System.out.printf("Epoka %d | Train MSE: %.6f | Test MSE: %.6f%n", epoch + 1, trainScore, testScore);

            // early stopping logic: stop if the error does not decrease
            if (testScore < bestTestScore) {
                bestTestScore = testScore;
                noImprovementCount = 0;
            } else {
                noImprovementCount++;
            }

            if (noImprovementCount >= config.getPatience()) {
                System.out.println("Early stopping triggered");
                break;
            }
        }

        // Save results to csv (reverting normalization to see real coordinates
        try (PrintWriter mseWriter = new PrintWriter(resultsDir + "/mse_per_epoch_" + config.getActivation() + ".csv")) {
            mseWriter.println("Epoch;TrainMSE;TestMSE");
            for (int i = 0; i < trainMSEList.size(); i++) {
                mseWriter.println((i + 1) + ";" + trainMSEList.get(i) + ";" + testMSEList.get(i));
            }
        }

        // TESTOWANIE I ZAPIS WYNIKÓW (W ORYGINALNEJ SKALI)
        testIter.reset();
        String resultFilename = resultsDir + "/results_" + config.getActivation().toString() + ".csv";
        try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(resultFilename)))) {
            writer.println("Example;PredX;PredY;ActualX;ActualY;ErrorX;ErrorY;Distance;MeasX;MeasY");

            System.out.println("\nTest predictions (original scale):");
            int example = 1;
            while (testIter.hasNext()) {
                DataSet ds = testIter.next();

                INDArray predicted = model.output(ds.getFeatures(), false);
                INDArray actual = ds.getLabels();
                INDArray measured = ds.getFeatures();


                INDArray predictedUnscaled = predicted.dup().castTo(DataType.DOUBLE);
                INDArray actualUnscaled = actual.dup().castTo(DataType.DOUBLE);
                INDArray measuredUnscaled = measured.dup().castTo(DataType.DOUBLE);

                normalizer.revertLabels(predictedUnscaled);
                normalizer.revertLabels(actualUnscaled);
                normalizer.revertFeatures(measuredUnscaled);

                for (int i = 0; i < predictedUnscaled.rows(); i++) {
                    double predX = predictedUnscaled.getDouble(i, 0);
                    double predY = predictedUnscaled.getDouble(i, 1);
                    double actualX = actualUnscaled.getDouble(i, 0);
                    double actualY = actualUnscaled.getDouble(i, 1);

                    double errorX = predX - actualX;
                    double errorY = predY - actualY;
                    double distance = Math.sqrt(errorX * errorX + errorY * errorY);

                    double measX = measuredUnscaled.getDouble(i, 0);
                    double measY = measuredUnscaled.getDouble(i, 1);

                    writer.printf("%d;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f%n",
                            example, predX, predY, actualX, actualY, errorX, errorY, distance, measX, measY);

                    example++;
                }
            }
        }

    }

}
