package main

import "fmt"

var weight float64 = 0.1
var number_of_toes = [...]float64{8.5, 9.5, 10, 9}

func neural_network(input, weight float64) float64 {
	prediction := input * weight
	return prediction
}

func main() {
	var input = number_of_toes[0]
	pred := neural_network(input, weight)
	fmt.Println(pred)
}
