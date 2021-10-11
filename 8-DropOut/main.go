package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

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

type Pair struct {
	Key   int
	Value float64
}

type PairList []Pair

func (p PairList) Len() int           { return len(p) }
func (p PairList) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
func (p PairList) Less(i, j int) bool { return p[i].Value < p[j].Value }

func newDropOutMask(hidden_size int) mat.VecDense {

	// create mask array
	m := map[int]float64{}

	// populate with random floats
	for i := 0; i < hidden_size; i++ {
		m[i] = rand.Float64()
	}

	p := make(PairList, len(m))

	i := 0
	for k, v := range m {
		p[i] = Pair{k, v}
		i++
	}

	sort.Sort(p)

	out := make(PairList, len(m))

	j := 0
	for _, k := range p {
		// fmt.Printf("%v\t%v\n", k.Key, k.Value)
		if k.Key < hidden_size/2 {
			out[j] = Pair{k.Key, 1}
		} else {
			out[j] = Pair{k.Key, 0}
		}
		j++
	}

	data_mask := make([]float64, 4)

	for i := 0; i < hidden_size; i++ {
		data_mask[i] = out[i].Value
	}

	/* fmt.Printf("Data Mask: %v", data_mask)
	fmt.Println() */

	// create vecdense for return
	var dropout_mask = *mat.NewVecDense(hidden_size, data_mask)
	return dropout_mask
}

func main() {

	rand.Seed(42)

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

	for iteration := 1; iteration <= 5000; iteration++ {

		fmt.Printf("Iteration: %v", iteration)
		fmt.Println()

		var layer_2_err float64 = 0
		for j := 0; j <= num_inputs-1; j++ {

			var dropout_mask = newDropOutMask(hidden_size)

			var layer_0 = mat.NewDense(1, 3, streetlights.RawRowView(j))

			var layer_1_input mat.Dense
			layer_1_input.Mul(layer_0, w01) // Number of Cols in 'a' must match number of Rows in 'b'.

			// run relu on the layer to zero-out any negative values
			var layer_1_relu = relu(layer_1_input)

			/* fmt.Printf("Layer 1 relu: %v", layer_1_relu)
			fmt.Println() */

			/* fmt.Printf("Dropout Mask: %v", dropout_mask)
			fmt.Println() */

			// multiply layer 1 by the dropout mask of random 0/1's
			var layer_1_dropout mat.VecDense
			layer_1_dropout.MulElemVec(layer_1_relu, &dropout_mask)

			/* fmt.Printf("Layer 1 Dropout: %v", layer_1_dropout)
			fmt.Println() */

			// multiple each element by 2 now that we have dropped 50% of the values, so we don't skew the signal downstream
			var layer_1 mat.VecDense

			layer_1.MulVec(&layer_1_dropout, mat.NewVecDense(1, []float64{2}))

			/* fmt.Printf("Layer 1: %v", layer_1)
			fmt.Println() */

			// run dot-product of layer_1 with associated weights
			layer_2 := mat.Dot(&layer_1, w12) // Number of Cols in 'a' must match number of Rows in 'b'.

			// calculate layer_2 delta = prediction minus real goal prediction
			var layer_2_delta = walk_vs_stop.At(j, 0) - layer_2

			// calculate layer_2 error

			layer_2_err += math.Pow((layer_2 - walk_vs_stop.At(j, 0)), 2)

			// calculate layer_1 delta
			var layer_1_delta_input = mat.NewVecDense(1, []float64{layer_2_delta})

			var layer_1_delta_dot mat.VecDense
			layer_1_delta_dot.MulVec(w12, layer_1_delta_input)

			layer_1_deriv := relu2deriv(mat.VecDense(layer_1))

			var layer_1_delta_deriv mat.VecDense
			layer_1_delta_deriv.MulElemVec(layer_1_delta_dot.TVec(), layer_1_deriv)

			// apply dropout to layer 1 delta so we take the dropout into account during back-propagation.
			var layer_1_delta mat.VecDense
			layer_1_delta.MulElemVec(&layer_1_delta_deriv, &dropout_mask)

			/* fmt.Printf("Layer 1 delta with dropout: %v", layer_1_delta)
			fmt.Println() */

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
