package main

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

var alpha = 0.5
var hidden_size = 4 // hidden layer with 4 nodes
var input_size = 3  // number of input values at layer 0
var num_inputs = 4

var streetlights = mat.NewDense(num_inputs, input_size, []float64{1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1})
var walk_vs_stop = mat.NewVecDense(num_inputs, []float64{1, 1, 0, 0})

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
	var outMat = mat.NewVecDense(hidden_size, out)
	return outMat
}

func relu2deriv(x mat.VecDense) *mat.VecDense {
	rawOj := x.RawVector().Data
	out := make([]float64, len(rawOj))
	for j := range rawOj {
		// Little bit of magic
		out[j] = math.Abs(bool2float64(rawOj[j] > 0))
	}
	var outMat = mat.NewVecDense(len(rawOj), out)
	return outMat
}

func sigmoid(x mat.Dense) *mat.VecDense {
	rawOj := x.RawMatrix().Data
	out := make([]float64, len(rawOj))
	for j := range rawOj {
		// Little bit of magic
		out[j] = 1.0 / (1.0 + math.Exp(-rawOj[j]))
	}
	var outMat = mat.NewVecDense(hidden_size, out)
	return outMat
}

func sigmoid2deriv(x mat.VecDense) *mat.VecDense {
	rawOj := x.RawVector().Data
	out := make([]float64, len(rawOj))
	for j := range rawOj {
		// Little bit of magic
		out[j] = rawOj[j] * (1 - rawOj[j])
	}
	var outMat = mat.NewVecDense(len(rawOj), out)
	return outMat
}

func tanh(x mat.Dense) *mat.VecDense {
	rawOj := x.RawMatrix().Data
	out := make([]float64, len(rawOj))
	for j := range rawOj {
		// Little bit of magic
		out[j] = math.Tanh(rawOj[j])
	}
	var outMat = mat.NewVecDense(hidden_size, out)
	return outMat
}

func tanh2deriv(x mat.VecDense) *mat.VecDense {
	rawOj := x.RawVector().Data
	out := make([]float64, len(rawOj))
	for j := range rawOj {
		// Little bit of magic
		out[j] = 1 - math.Pow(rawOj[j], 2)
	}
	var outMat = mat.NewVecDense(len(rawOj), out)
	return outMat
}

func main() {

	// Create random starting weights for layer 0_1
	var data_0_1 = make([]float64, hidden_size*input_size) // weight matrix needs to be num input layer nodes times hidden layer nodes
	for i := range data_0_1 {
		data_0_1[i] = rand.NormFloat64() / 5
	}
	var weights_0_1 = *mat.NewDense(input_size, hidden_size, data_0_1)

	// Create random starting weights for layer 1_2
	var data_1_2 = make([]float64, hidden_size*1) // output weight matrix needs to be hidden layer nodes times 1 (aiming for single output prediction)
	for i := range data_1_2 {
		data_1_2[i] = rand.NormFloat64() / 5
	}
	var weights_1_2 = *mat.NewVecDense(hidden_size, data_1_2)

	w01 := &weights_0_1
	w12 := &weights_1_2

	for iteration := 1; iteration <= 1000; iteration++ {

		fmt.Printf("Iteration: %v", iteration)
		fmt.Println()

		var layer_2_err float64 = 0
		for j := 0; j <= num_inputs-1; j++ {

			var layer_0 = mat.NewDense(1, 3, streetlights.RawRowView(j))

			var layer_1_input mat.Dense
			layer_1_input.Mul(layer_0, w01) // Number of Cols in 'a' must match number of Rows in 'b'.

			// run relu on the layer to zero-out any negative values
			var layer_1 = tanh(layer_1_input)

			// run dot-product of layer_1 with associated weights
			layer_2 := mat.Dot(layer_1, w12) // Number of Cols in 'a' must match number of Rows in 'b'.

			// calculate layer_2 delta = prediction minus real goal prediction
			var layer_2_delta = walk_vs_stop.At(j, 0) - layer_2

			// calculate layer_2 error

			layer_2_err += math.Pow((layer_2 - walk_vs_stop.At(j, 0)), 2)

			// calculate layer_1 delta
			var layer_1_delta_input = mat.NewVecDense(1, []float64{layer_2_delta})

			var layer_1_delta_dot mat.VecDense
			layer_1_delta_dot.MulVec(w12, layer_1_delta_input)

			layer_1_deriv := tanh2deriv(*layer_1)

			var layer_1_delta mat.VecDense
			layer_1_delta.MulElemVec(layer_1_delta_dot.TVec(), layer_1_deriv)

			// update values for weights_1_2
			var weights_1_2_input mat.Dense

			weights_1_2_input.Apply(func(i int, j int, v float64) float64 { return layer_2_delta * v }, layer_1.T())

			for i := 0; i < w12.Len()-1; i++ {
				w12.SetVec(i, w12.At(i, 0)+(alpha*weights_1_2_input.At(0, i)))
			}

			// update values for weights_0_1
			var weights_0_1_input mat.Dense
			weights_0_1_input.Mul(layer_0.T(), layer_1_delta.T())

			weights_0_1_input.Apply(func(i int, j int, v float64) float64 { return v * alpha }, &weights_0_1_input)

			w01.Add(w01, &weights_0_1_input)

			fmt.Printf("Error: %v", layer_2_err)
			fmt.Println()
		}
		fmt.Print("======================================")
		fmt.Println()
	}
}
