package main

import (
	"fmt"
	"math"
)

var weight = 0.0
var input = 1.1
var goal_prediction = 0.8

func main() {
	for iteration := 0; iteration <= 4; iteration++ {
		var prediction = input * weight
		var err = math.Pow(prediction-goal_prediction, 2) // squared to make non-zero, error is our measure of accuracy on a prediction.
		var delta = prediction - goal_prediction          // how far off we were in absolute terms
		var weight_delta = delta * input                  // This is the signal, it gives us a direction and amount to adjust the weight for this node.
		weight = weight - weight_delta                    // adjust the weight for the next iteration on this node.

		fmt.Printf("Error: %f. Prediction: %f", err, prediction)
		fmt.Println()
	}
}
