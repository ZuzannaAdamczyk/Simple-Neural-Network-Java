# Simple Neural Network Java
A simple java project that uses a neural network to fix coordinate errors.
It uses the DeepLearning4j library to train model on CSV data.

##What does this project do? 
This program read measured coordinates and learn how to predict the actual coordinates.
It acts like a digital corrector for sensor or GPS errors using a Multi-Layer Perceptron (MLP).

## Key Features
- **Configurable**: Change neurons, epochs, and learning rate in config.txt.
- **Encapsulation**: Clean Java code using private fields and getters for safety.
- **Early Stopping**: The training stops automatically when the model is ready to prevent overfitting.
- **Normalization**: Automatically scales data to improve training efficiency.

##How to run it?
1. Ensure you have **Java17+** and **Maven** installed.
2. Put your csv data in te data/ folder
3. Adjust setting in config.txt
4. Run Main.java
5. Check te result/ folder for MSE logs and prediction results

##Technologies
- Java 17
- Maven
- DeepLearning4j 
