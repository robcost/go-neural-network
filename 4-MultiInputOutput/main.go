package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

var weights = mat.NewDense(3, 3, []float64{0.1, 0.1, -0.3, 0.1, 0.2, 0.0, 0.0, 1.3, 0.1})

var toes = mat.NewVecDense(4, []float64{8.5, 9.5, 9.9, 9.0})
var wlrec = mat.NewVecDense(4, []float64{0.65, 0.8, 0.8, 0.9})
var nfans = mat.NewVecDense(4, []float64{1.2, 1.3, 0.5, 1.0})

func w_sum(a *mat.VecDense, b mat.Vector) float64 {
	var output float64 = 0

	for i := range a.RawVector().Data {
		output += (a.At(i, 0) * b.At(i, 0))
	}

	return output
}

func vect_mat_mul(vect *mat.VecDense, matrix *mat.Dense) *mat.VecDense {
	output := mat.NewVecDense(3, []float64{0, 0, 0})

	for i := range vect.RawVector().Data {
		output.SetVec(i, w_sum(vect, matrix.RowView(i)))
	}

	return output
}

func neural_network(input *mat.VecDense, weights *mat.Dense) *mat.VecDense {
	pred := vect_mat_mul(input, weights)
	return pred
}

func main() {
	var input = mat.NewVecDense(3, []float64{toes.At(0, 0), wlrec.At(0, 0), nfans.At(0, 0)})
	pred := neural_network(input, weights)
	fmt.Println(pred)
}
