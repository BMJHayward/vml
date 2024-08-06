module nnet

import arrays
import math
import rand

pub fn name() string {
	return 'basic feed forward neural network'
}

type FLoatFunc = fn (f64) f64

type ActFunc = fn ([]f64) []f64

type ErrorFunc = fn ([]f64, []f64) []f64

pub struct NeuralNetModel {
mut:
	shape      []int
	layers     []string
	activation ActFunc @[required]
	error      ErrorFunc @[required]
	backprop   bool
}

fn get_acts(fun string) (ActFunc, ActFunc) {
	act_funcs := {
		'tanh': [tanh, d_tanh]
		'sig':  [sigmoid, d_sigmoid]
		'relu': [relu, d_relu]
	}
	rv := act_funcs[fun]
	return rv[0], rv[1]
}

pub struct Layer {
	func string
mut:
	inputs     int
	neurons    int
	learn_rate f64
	act        ActFunc @[required]
	act_d      ActFunc @[required]
	prev_layer [][]f64
	z          [][]f64 // the dot prod of weights and prev layer Z
	a          [][]f64 // result of activation func on a
	w          [][]f64
	b          []f64
}

fn init_layer(neurons int, inputs int, func string) Layer {
	act, act_d := get_acts(func)
	learn_rate := 0.1
	mut w := [][]f64{}
	mut b := []f64{}
	mut z := [][]f64{}
	mut a := [][]f64{}
	mut prev_layer := [][]f64{}
	for n in 0 .. neurons {
		w << []f64{}
		b << 0.0
		z << []f64{}
		a << []f64{}
		prev_layer << []f64{}
		for _ in 0 .. inputs {
			w[n] << rand.f64()
		}
	}
	new_layer := Layer{
		neurons: neurons
		inputs: inputs
		func: func
		act: act
		act_d: act_d
		learn_rate: learn_rate
		w: w
		b: b
		a: a
		z: z
		prev_layer: prev_layer
	}
	return new_layer
}

fn dot_prod(a []f64, b []f64) f64 {
	return arrays.sum(arrays.group<f64>(a, b).map(it[0] * it[1])) or { 0.0 }
}

fn mat_mul(a [][]f64, b [][]f64) [][]f64 {
	if a[0].len != b.len {
		println('a x b')
		println(a)
		println(b)
		panic('matrix dimensions different. a x b: ${a.len}x${a[0].len} x ${b.len}x${b[0].len}')
	}
	mut res := [][]f64{len: a.len, init: []f64{len: b[0].len}}
	for row in 0 .. a.len {
		for col in 0 .. b[0].len {
			res[row][col] = 0.0
			for bcol in 0 .. b.len {
				res[row][col] += a[row][bcol] * b[bcol][col]
			}
		}
	}
	return res
}

fn transpose(a [][]f64) [][]f64 {
	mut b := [][]f64{len: a[0].len, init: []f64{len: a.len}}
	for r in 0 .. a.len {
		for c in 0 .. a[0].len {
			b[c][r] = a[r][c]
		}
	}
	return b
}

// feed_fwd takes input from previous layer, computes dot product and passes to next layer
fn (mut l Layer) feed_fwd(prev_layer [][]f64) [][]f64 {
	l.prev_layer = prev_layer.map(it.clone())
	// TODO: add l.b to mat_mul result in zmm
	zmm := mat_mul(prev_layer, l.w)
	l.z = zmm
	l.a = zmm.map(l.act(it))
	return zmm.map(l.act(it))
}

fn (mut l Layer) back_prop(da [][]f64) [][]f64 {
	// apply derivative of activation and multiply
	// element wise with da - differentiated A
	act_d_z := l.z.map(l.act_d(it))
	mut dz := act_d_z.map(it.clone())
	for i in 0 .. act_d_z.len {
		for j in 0 .. act_d_z[0].len {
			// dz[i][j] = act_d_z[i][j] * da[i][j]
			if act_d_z.len == da[0].len && act_d_z[0].len == da.len {
				dz[i][j] = act_d_z[i][j] * da[j][i]
			} else {
				dz[i][j] = act_d_z[i][j] * da[i][j]
			}
		}
	}
	prev_da := mat_mul(l.w, transpose(dz))
	// 1/dz.len * mat_mul(dz, transpose(l.prev_layer))
	mut dw := mat_mul(transpose(dz), l.prev_layer)
	for p in 0 .. dw.len {
		for q in 0 .. dw[0].len {
			dw[p][q] *= 1 / dz.len
		}
	}
	// 1/dz.len * sum_reduce(dz)
	mut db := dz.map(arrays.sum(it) or { 0.0 })
	for r in 0 .. db.len {
		db[r] *= 1 / dz.len
	}
	// l.w -= l.learn_rate * dw
	// dw should be transposed dimensions to l.w, so we are using reversed indices here
	for x in 0 .. l.w.len {
		for y in 0 .. l.w[0].len {
			l.w[x][y] -= l.learn_rate * dw[y][x]
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

fn d_relu(x []f64) []f64 {
	return x.map(fn (w f64) f64 {
		if w < 0.0 {
			return 0.0
		} else {
			return 1.0
		}
	})
}

fn tanh(x []f64) []f64 {
	return x.map(math.tanh(it))
}

// d_tanh is derivative of tanh function
fn d_tanh(x []f64) []f64 {
	return (x.map(math.tanh(it))).map(math.pow(it, 2)).map(1 - it)
}

fn sigmoid(x []f64) []f64 {
	return x.map(1 / (1 + math.exp(-it)))
}

// d_sigmoid is derivative of sigmoid function
fn d_sigmoid(x []f64) []f64 {
	mut sg := sigmoid(x)
	return sg.map((1 - it) * it)
}

// Loss Functions
fn logloss(y [][]f64, a [][]f64) [][]f64 {
	// alog := a.map(math.log(it))
	// alogit := a.map(math.log(1-it))
	mut alog := [][]f64{len: a.len, init: []f64{len: a[0].len}}
	mut alogit := [][]f64{len: a.len, init: []f64{len: a[0].len}}
	for i in 0 .. a.len {
		for j in 0 .. a[0].len {
			alog[i][j] = math.log(a[i][j])
			alogit[i][j] = math.log(1 - a[i][j])
		}
	}
	mut one_sub_y := y.map(it.map(1 - it))
	mut osy_t := transpose(one_sub_y)
	// mut lgla := alog ⊙ one_sub_y.T
	mut lgla := alog.map(it.clone())
	for i in 0 .. lgla.len {
		for j in 0 .. lgla[0].len {
			lgla[i][j] *= osy_t[i][j]
		}
	}
	// mut lglb := alogit ⊙ one_sub_y.T
	mut lglb := alogit.map(it.clone())
	for i in 0 .. lglb.len {
		for j in 0 .. lglb[0].len {
			lglb[i][j] *= osy_t[i][j]
		}
	}

	// -1 * (alog + alogit)
	// for k in 0 .. a.len {
	// lgl << -1 * (lgla[k] + lglb[k])
	// }
	mut lgl := lgla.map(it.clone())
	for i in 0 .. lgla.len {
		for j in 0 .. lgla[0].len {
			lgl[i][j] = -1 * (lgla[i][j] + lglb[i][j])
		}
	}
	return lgl
}

// d_logloss is an element-wise operation of form
// (a - y)/(a*(1 - a))
fn d_logloss(y [][]f64, a [][]f64) [][]f64 {
	mut bot_a := a.map(it.clone())
	for i in 0 .. a.len {
		for j in 0 .. a[0].len {
			bot_a[i][j] = a[i][j] * (1 - a[i][j])
		}
	}
	// a.map(it - y)
	mut top_a := a.map(it.clone())
	yt := transpose(y)
	for k in 0 .. top_a.len {
		for l in 0 .. top_a[0].len {
			// use transpose if dimensions are reversed
			if top_a.len == y[0].len && top_a[0].len == y.len {
				top_a[k][l] -= yt[k][l]
			} else {
				top_a[k][l] -= y[k][l]
			}
			top_a[k][l] /= bot_a[k][l]
		}
	}
	return top_a
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

pub fn demo() []NeuralNetModel {
	x_train := [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]]
	y_train := [[0.0, 1.0, 0.0, 0.0]]
	m := 4
	epochs := 10
	mut layers := [
		init_layer(2, 3, 'tanh'),
		init_layer(3, 1, 'sig'),
	]
	mut costs := []f64{}

	for _ in 0 .. epochs {
		// train by feedforward
		mut a := transpose(x_train.map(it.clone()))
		a = layers[0].feed_fwd(a)
		a = layers[1].feed_fwd(a)
		// keep track of costs to plot
		epoch_loss := logloss(y_train, a)
		costs << 1 / m * arrays.sum(arrays.flatten(epoch_loss)) or { 0.0 }

		// perform backpropagation
		mut da := d_logloss(y_train, a)
		// for mut l in layers.reverse() {
		// da = l.back_prop(da)
		// }
		da = layers[1].back_prop(da)
		da = layers[0].back_prop(da)
	}

	println('NEURAL NET PREDICTION')
	mut demo_train := [[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]]
	println('DEMO DATA INPUT:\n$demo_train')
	//    for mut lyr in layers {
	//        demo_train = lyr.feed_fwd(transpose(demo_train))
	//    }
	demo_train = layers[0].feed_fwd(transpose(demo_train))
	println(layers[0])
	demo_train = layers[1].feed_fwd(demo_train)
	println(layers[1])
	println('DEMO DATA PREDICTED OUTPUT:\n$demo_train')

	return [
		NeuralNetModel{[3, 3, 3], ['relu', 'relu', 'relu'], relu, mse, false},
	]
}
