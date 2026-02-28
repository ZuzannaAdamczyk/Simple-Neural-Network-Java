package org.example;

import org.nd4j.linalg.activations.Activation;


/**
 * This class stores the neural network configuration
 * It uses encaplulation to keep data safe and immutable
 */
public class Configuration {
    private final int hiddenNeurons;
    private final Activation activation; // "sigmoid", "tanh", "relu"
    private final double learningRate;
    private final int epochs;
    private final int patience;
    private final int batchSize;

    /**
     * Constructor, initializes all neural networ parameters
     */
    public Configuration(int hiddenNeurons, String activation, double learningRate, int epochs, int patience, int batchSize) {
        this.hiddenNeurons = hiddenNeurons;
        this.activation = getActivationFromString(activation);
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.patience = patience;
        this.batchSize = batchSize;
    }

    //GETTERS
    public int getHiddenNeurons() {
        return hiddenNeurons;
    }

    public Activation getActivation() {
        return activation;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public int getEpochs() {
        return epochs;
    }

    public int getPatience() {
        return patience;
    }

    public int getBatchSize() {
        return batchSize;
    }


    /**
     * Returns a deafult configuration
     */
    public static Configuration defaultConfig() {
        return new Configuration(10, "relu", 0.01, 100, 10, 32);
    }

    /**
     *Converts a String form the text file into an Activation object for the DL4J library
     */
    private static Activation getActivationFromString(String act) {
        return switch (act.toLowerCase()) {
            case "relu" -> Activation.RELU;
            case "tanh" -> Activation.TANH;
            case "sigmoid" -> Activation.SIGMOID;
            default -> {
                System.err.println("Unknown activation: " + act);
                yield Activation.RELU;
            }
        };
    }
}
