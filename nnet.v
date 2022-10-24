module nnet

import arrays
import math

pub fn name() string {
    return 'neural networks'
}

type FLoatFunc = fn (f64) f64

pub struct NeuralNetModel {
mut:
	layers []int
	activation  fn (f64) f64
	error       fn ([]f64, []f64) f64
	backprop    bool
}

fn relu(input f64) f64 {
    return arrays.max([0.0, input]) or { 0 }
}

fn mse(target []f64, actual []f64) f64 {
    mut sqerr := []f64{}
    for i in 0 .. actual.len {
        sqerr << math.pow(target[i] - actual[i], 2.0)
    }
    return arrays.sum(sqerr) or { 0 } / sqerr.len
}

pub fn (mut m NeuralNetModel) train<T>(inputs [][]T, output []T, iterations int) []NeuralNetModel {
	return [NeuralNetModel{}]
}


pub fn (mut m KMeansModel) predict<T>(data [][]T) ![]f64 {
	return [0.0]
}

pub fn demo() ![]NeuralNetModel {
	return [NeuralNetModel{
        [3,3,3]
        relu
        mse
        false
    }]
}