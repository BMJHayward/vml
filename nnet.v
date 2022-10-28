module nnet

import arrays
import math

pub fn name() string {
    // TODO: single layer perception
    // TODO: multilayer perception
    // TODO: backpropagation, stochastic gradient descent
    // TODO: other activate functs e.g. sigmoid, softmax
    return 'basic feed forward neural network'
}

type FLoatFunc = fn (f64) f64

pub struct NeuralNetModel {
mut:
	shape []int
	layers []string
	activation  fn ([]f64) []f64
	error       fn ([]f64, []f64) []f64
	backprop    bool
}

// Activation Functions
fn relu(input []f64) []f64 {
    return input.map(arrays.max([0.0, it]) or { 0 })
}

fn tanh(x []f64) []f64 {
    return x.map(math.tanh(it))
}

fn d_tanh(x []f64) []f64 {
    return (x.map(math.tanh(it))).map(math.pow(it, 2)).map(1 - it)
}

fn sigmoid(x []f64) []f64 {
    return x.map(1/(1 + math.exp(-it)))
}

fn d_sigmoid(x []f64) []f64 {
    mut sg := sigmoid(x)
    return sg.map((1 - it) * it)
}

// Loss Functions 
fn logloss(y []f64, a []f64) []f64 {
    // asdf
    ya_range := math.min(y.len, a.len)
    alog := a.map(math.log(it))
    alogit := a.map(math.log(1-it))
    mut lgl := []f64{}
    for i in 0 .. ya_range {
        lgl << -1 * (y[i] * alog[i] + (1 - y[i]) * alogit[i])
    }
    return lgl
}

fn d_logloss(y []f64, a []f64) []f64 {
    mut ay := []f64{len: math.min(y.len, a.len)}
    for i in 0 .. int(math.min(y.len, a.len)) {
        ay << a[i] - y[i]
    }
    mut ll := a.map(it*(1 - it))
    return arrays.group<f64>(ay, ll).map(it[0] / it[1])
}

fn mse(target []f64, actual []f64) []f64 {
    mut sqerr := []f64{}
    for i in 0 .. actual.len {
        sqerr << math.pow(target[i] - actual[i], 2.0)
    }
    return [arrays.sum(sqerr) or { 0 } / sqerr.len]
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
        ['relu', 'relu', 'relu']
        relu
        mse
        false
    }]
}
