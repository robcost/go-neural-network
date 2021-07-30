package main

import (
	"fmt"
	"math"
)

var weight = 0.5
var input = 0.5
var goal_prediction = 0.8

var step_amount = 0.001

func main() {
	for iteration := 0; iteration <= 1101; iteration++ {

		var prediction = input * weight
		var err = math.Pow(prediction-goal_prediction, 2)

		fmt.Printf("Error: %f. Prediction: %f", err, prediction)
		fmt.Println()

		var up_prediction = input * (weight + step_amount)
		var up_error = math.Pow(goal_prediction-up_prediction, 2)

		var down_prediction = input * (weight - step_amount)
		var down_error = math.Pow(goal_prediction-down_prediction, 2)

		if down_error < up_error {
			weight = weight - step_amount
		}
		if down_error > up_error {
			weight = weight + step_amount
		}
	}
}
