module nnet

import arrays
import math
import rand

pub fn name() string {
    // TODO: single layer perception
    // TODO: multilayer perception
    // TODO: backpropagation, stochastic gradient descent
    // TODO: other activate functs e.g. sigmoid, softmax
    return 'basic feed forward neural network'
}

type FLoatFunc = fn (f64) f64
type ActFunc = fn ([]f64) []f64
type ErrorFunc = fn ([]f64, []f64) []f64

pub struct NeuralNetModel {
mut:
	shape []int
	layers []string
	activation  ActFunc
	error       ErrorFunc
    backprop    bool
}

fn get_acts(fun string) (ActFunc, ActFunc) {
    act_funcs := {
        'tanh': [tanh, d_tanh]
        'sig': [sigmoid, d_sigmoid]
    }
    rv := act_funcs[fun]
    return rv[0], rv[1]
}

pub struct Layer {
    mut:
        prev_layer [][]f64
    	z [][]f64 // the dot prod of weights and prev layer Z
        a [][]f64 // result of activation func on a
        act ActFunc
        act_d ActFunc
        w [][]f64
        learn_rate f64
        b []f64
}

fn (mut l Layer) init_layer(neurons int, inputs int, func string) {
    l.act, l.act_d = get_acts(func)
    for n in 0 .. neurons {
        l.w << []f64{}
        l.b << 0.0
        for _ in 0 .. inputs {
            l.w[n] << rand.f64()
        }
    }
    println('layer initialised')
}

fn dot_prod(a []f64, b []f64) f64 {
    return arrays.sum(arrays.group<f64>(a,b).map(it[0]*it[1])) or { 0.0 }
}

fn mat_mul(a [][]f64, b [][]f64) [][]f64 {
    if a[0].len != b.len {
        println(error)
    }

	mut res := [][]f64{}

    for row in 0 .. a.len {
        for col in 0 .. b.len {
            for bcol in 0 .. b[col].len {
                res[row][col] = a[row][bcol] * b[bcol][col]
            }
        }
    }
    return res
}

fn transpose(a [][]f64) [][]f64 {
    mut b := [][]f64{len: a.len, init: []f64{init: a[0].len}}
    for r in 0 .. a.len {
        for c in 0 .. a[r].len {
            b[r][c] = a[c][r]
        }
    }
    return b
}
// feed_fwd takes input from previous layer, computes dot product and passes to next layer
fn (mut l Layer) feed_fwd(prev_layer [][]f64) [][]f64 {
    l.prev_layer = prev_layer
    zmm := mat_mul(l.w, l.prev_layer)
    zmmgroup := zmm.map(arrays.group<f64>(it, l.b))
    l.z = zmmgroup.map(it.map(it[0] + it[1]))
    l.a = l.z.map(l.act(it)) 
    return l.a
}

fn (mut l Layer) back_prop(da [][]f64) [][]f64 {
    // apply derivative of activation and multiply
    // element wise with da - differentiated A
    act_d_z := l.z.map(l.act_d(it))
    mut dz := [][]f64{}
    for i in 0 .. act_d_z.len {
        for j in 0 .. da.len {
            dz[i][j] = act_d_z[i][j] * da[i][j]
        }
    }
    prev_da := mat_mul(transpose(l.w), dz)
    // 1/dz.len * mat_mul(dz, transpose(l.prev_layer))
    mut dw := mat_mul(dz, transpose(l.prev_layer))
    for p in 0 .. dw.len {
        for q in 0 .. dw[0].len {
            dw[p][q] *= 1/dz.len
        }
    }
    // 1/dz.len * sum_reduce(dz)
    mut db := dz.map(arrays.sum(it) or { 0.0 })
    for r in 0 .. db.len {
        db[r] *= 1/dz.len
    }
    // l.w -= l.learn_rate * dw
    for x in 0 .. l.w.len {
        for y in 0 .. l.w[0].len {
            l.w[x][y] -= l.learn_rate * dw[x][y]
        }
    }
    // l.b -= l.learn_rate * db
    for x in 0 .. l.b.len {
            l.b[x] -= l.learn_rate * db[x]
        }

    return prev_da
}

// Activation Functions
fn relu(input []f64) []f64 {
    return input.map(arrays.max([0.0, it]) or { 0 })
}

fn tanh(x []f64) []f64 {
    return x.map(math.tanh(it))
}

// d_tanh is derivative of tanh function
fn d_tanh(x []f64) []f64 {
    return (x.map(math.tanh(it))).map(math.pow(it, 2)).map(1 - it)
}

fn sigmoid(x []f64) []f64 {
    return x.map(1/(1 + math.exp(-it)))
}

// d_sigmoid is derivative of sigmoid function
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
