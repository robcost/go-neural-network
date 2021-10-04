package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

var alpha = 0.2

var streetlights = mat.NewDense(4, 3, []float64{1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1})
var walk_vs_stop = mat.NewVecDense(4, []float64{1, 1, 0, 0}).T()

var weights_0_1 = mat.NewDense(3, 4, []float64{-0.16595599, 0.44064899, -0.99977125, -0.39533485, -0.70648822, -0.81532281, -0.62747958, -0.30887855, -0.20646505, 0.07763347, -0.16161097, 0.370439})

var weights_1_2 = mat.NewVecDense(4, []float64{-0.5910955, 0.75623487, -0.94522481, 0.34093502})

func bool2float64(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

func relu(x mat.Dense) *mat.VecDense {
	rawOj := x.RawMatrix().Data
	out := make([]float64, len(rawOj))
	for j := range rawOj {
		// Little bit of magic
		out[j] = math.Abs(bool2float64(rawOj[j] > 0) * rawOj[j])
	}
	var outMat = mat.NewVecDense(4, out)
	return outMat
}

func main() {

	var layer_0 = mat.NewDense(1, 3, streetlights.RawRowView(0))
	fmt.Printf("Layer 0: %v.", layer_0)
	fmt.Println()

	var layer_1 mat.Dense

	layer_1.Mul(layer_0, weights_0_1)

	fmt.Printf("Layer 1: %v.", relu(layer_1))
	fmt.Println()

	layer_2 := mat.Dot(relu(layer_1), weights_1_2)

	fmt.Printf("Layer 2: %v.", layer_2)
	fmt.Println()

	layer_2_delta := walk_vs_stop.At(0, 0) - layer_2

	fmt.Printf("Layer 2 Delta: %v.", layer_2_delta)
	fmt.Println()

	var layer_1_delta_input = mat.NewVecDense(1, []float64{layer_2_delta})

	var layer_1_delta mat.VecDense
	layer_1_delta.MulVec(weights_1_2, layer_1_delta_input)

	fmt.Printf("Layer 1 Delta: %v.", layer_1_delta)
	fmt.Println()

	// update Weights 1_2
	var weights_1_2_input mat.Dense
	weights_1_2_input.Apply(func(i int, j int, v float64) float64 { return layer_2_delta * v }, relu(layer_1).T())

	for i := 0; i < weights_1_2.Len(); i++ {
		weights_1_2.SetVec(i, weights_1_2.At(i, 0)+(alpha*weights_1_2_input.At(0, i)))
	}

	fmt.Printf("Weights 1_2: %v.", weights_1_2)
	fmt.Println()

	var weights_0_1_input mat.Dense
	weights_0_1_input.Mul(layer_0.T(), layer_1_delta.T())
	weights_0_1_input.Apply(func(i int, j int, v float64) float64 { return v * alpha }, &weights_0_1_input)
	weights_0_1.Add(weights_0_1, &weights_0_1_input)

	fmt.Printf("Weights 0_1: %v.", weights_0_1)
	fmt.Println()

}
