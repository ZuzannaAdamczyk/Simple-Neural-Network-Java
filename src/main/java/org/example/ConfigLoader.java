package org.example;

import java.io.FileInputStream;
import java.util.Properties;

/**
 * This class reads settings from configuration file.
 */
public class ConfigLoader {
    /**
     * Loads the config file. If there is a error, it returns default values.
     */
    public static Configuration loadConfig(String filename) {
        Properties prop = new Properties(); // Properties is a special java class to store key=value pairs

        /**
         * Try open and read a file
         */
        try (FileInputStream input = new FileInputStream(filename)) {
            prop.load(input); // read all lines from the file and parses the key=value lines automatically

        } catch (Exception e) {
            System.err.println("File not found");
            return Configuration.defaultConfig();
        }


        /**
         * Get vaalues from the file and convert them to numbers
         * In text files everything is a String, so we must convert it
         * prop.getPropertt(key, defaultValue)
         */
        try {
            int neurons = Integer.parseInt(prop.getProperty("hidden_neurons",  "10"));
            String act = prop.getProperty("activation",  "tanh").toLowerCase();
            double lr = Double.parseDouble(prop.getProperty("learning_rate",  "0.01"));
            int ep = Integer.parseInt(prop.getProperty("epochs",  "100"));
            int pat = Integer.parseInt(prop.getProperty("patience",  "10"));
            int batch = Integer.parseInt(prop.getProperty("batch_size",  "32"));

            // create and return a new Configuration object
            return new Configuration(neurons, act, lr, ep, pat, batch);


        } catch (Exception e) {
            System.err.println("Error in file data");
            return Configuration.defaultConfig();
        }
    }
}

