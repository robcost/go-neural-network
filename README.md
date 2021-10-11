# go-neural-network

This repo contains some basic ML constructs in golang. This is a learning repo, definitely not for use in a real application. 

## 1 - Basic
Single node network showing predictions with weights.

## 2 - MultiInput


## 3 - MultiOutput


## 4 - MultiInputOutput


## 5 - HoldColdLearning
Simple loop demonstrating stepping up/down towards a an optimal weight.

## 6 - Derivative
Find a derivative (weight delta) that represents the signal we should follow, sending the weights up/down and by how much in subsequent itereations.

## 7 - Back Propatation
Implement loop to carry weight updates back through network. It seems that gonum precision is higher than numpy, meaning it takes longer for the layer 2 error to converge, meaning more iterations required for same result.

## 8 - Drop Out
*not working*
Adding early stop with drop out, randomly turning off nodes in layer 1, telling the network to train in smaller groups so it doesn't overfit to noise. apply the dropout on forward pass at layer 1, then also apply to layer_1_delta so it's considered as part of back propagation

## 9 - Batch Gradient Descent

*not implemented*

## 10 - Sigmoid

Add Sigmoid Activation and derivative functions.

## 11 - Tanh

Add Tanh Activation and derivative functions.

## 12 - Softmax

*not implemented*

## 13 - Convolution


