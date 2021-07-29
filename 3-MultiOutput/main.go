package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

var weights = mat.NewVecDense(3, []float64{0.1, 0.2, 0})

var toes = mat.NewVecDense(4, []float64{8.5, 9.5, 9.9, 9.0})
var wlrec = mat.NewVecDense(4, []float64{0.65, 0.8, 0.8, 0.9})
var nfans = mat.NewVecDense(4, []float64{1.2, 1.3, 0.5, 1.0})

func ele_mul(number float64, vector *mat.VecDense) mat.VecDense {
	output := mat.NewVecDense(3, []float64{0, 0, 0})

	for i, val := range vector.RawVector().Data {
		//fmt.Printf("At position %d, the value %s is present\n", i, val)
		output.SetVec(i, number*val)
	}
	return *output
}

func neural_network(input, weights *mat.VecDense) mat.VecDense {
	pred_ele_mul := ele_mul(input.At(0, 0), weights)
	return pred_ele_mul
}

func main() {
	var input = mat.NewVecDense(3, []float64{toes.At(0, 0), wlrec.At(0, 0), nfans.At(0, 0)})
	pred := neural_network(input, weights)
	fmt.Println(pred)
}
