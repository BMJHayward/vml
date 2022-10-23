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
	error       fn (f64) f64
	backprop    bool
}

fn relu(input f64) f64 {
    return arrays.max([0.0, input]) or { 0 }
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
        relu
        false
    }]
}