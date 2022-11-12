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
        'relu': [relu, d_relu]
    }
    rv := act_funcs[fun]
    return rv[0], rv[1]
}

pub struct Layer {
    mut:
        inputs int
        neurons int
        learn_rate f64
        act ActFunc
        act_d ActFunc
        prev_layer [][]f64
    	z [][]f64 // the dot prod of weights and prev layer Z
        a [][]f64 // result of activation func on a
        w [][]f64
        b []f64
}

fn init_layer(neurons int, inputs int, func string) Layer {
    act, act_d := get_acts(func)
    learn_rate := 0.1
    mut w := [][]f64{}
    mut b := []f64 {}
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
        act: act
        act_d: act_d
        learn_rate: learn_rate
        w: w
        b: b
        a: a
        z: z
        prev_layer: prev_layer
    }
    println('new layer initialised')
    return new_layer
}

fn dot_prod(a []f64, b []f64) f64 {
    return arrays.sum(arrays.group<f64>(a,b).map(it[0]*it[1])) or { 0.0 }
}

fn mat_mul(a [][]f64, b [][]f64) [][]f64 {
    println('mat mul a')
    println(a)
    println('mat mul b')
    println(b)
    if a[0].len != b.len {
        println('args have incompatible dimensions')
        println('arg a has ${a.len} rows and ${a[0].len} columns, b has ${b.len} rows and ${b[0].len} columns')
        println('pass in matrices with matching a-col b-row dimensions')
        println(error)
    } else if a.len == 0 || b.len == 0 {
        println('matrices a and b cannot be 0 length')
        println('length of a: ${a.len}')
        println('length of b: ${b.len}')
        println(error)
    }

	mut res := [][]f64{}

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
    println('prev layer')
    println(prev_layer)
    println('l.w')
    println(l.w)
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

fn d_relu(x []f64) []f64 {
    return x.map(fn (w f64) f64 {
        if w < 0.0 {
            return 0.0
        }
        else {
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
    return x.map(1/(1 + math.exp(-it)))
}

// d_sigmoid is derivative of sigmoid function
fn d_sigmoid(x []f64) []f64 {
    mut sg := sigmoid(x)
    return sg.map((1 - it) * it)
}

// Loss Functions 
fn logloss(y []f64, a [][]f64) []f64 {
    // alog := a.map(math.log(it))
    // alogit := a.map(math.log(1-it))
    mut alog := [][]f64{}
    mut alogit := [][]f64{}
    for i in 0 .. a.len {
        for j in 0 .. a[0].len {
            alog[i][j] = math.log(a[i][j])
            alogit[i][j] = math.log(1 - a[i][j])
        }
    }
    mut one_sub_y := y.map(1-it)
    mut lgl := []f64{}
    mut lgla := alog.map(dot_prod(it, y))
    mut lglb := alogit.map(dot_prod(it, one_sub_y))
    for k in 0 .. y.len {
        lgl << -1 * (lgla[k] + lglb[k])
    }
    return lgl
}

// d_logloss is an element-wise operation of form
// (a - y)/(a*(1 - a))
fn d_logloss(y []f64, a [][]f64) [][]f64 {
    mut bot_a := a.map(it.clone())
    for i in 0 .. a.len {
        for j in 0 .. a[0].len {
            bot_a[i][j] = a[i][j] * (1 - a[i][j])
        }
    }
    // a.map(it - y)
    mut top_a := a.map(it.clone())
    for k in 0 .. a.len {
        for l in 0 .. y.len {
            top_a[k][l] -= y[l]
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

pub fn demo() ![]NeuralNetModel {
    x_train := [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]]
    y_train := [0.0, 1.0, 0.0, 0.0]
    m := 4
    epochs := 10
    mut layers := [
        init_layer(2, 2, 'relu'),
        init_layer(2, 3, 'tanh'),
        init_layer(3, 3, 'sig')
    ]
    mut costs := []f64{}

    for _ in 0 .. epochs {
      // train by feedforward
      mut a := x_train.map(it.clone())
      println('A')
      println(a)
      for mut l in layers {
        a = l.feed_fwd(a)
      }
      println('A')
      println(a)
      // keep track of costs to plot
      costs << 1/m * arrays.sum(logloss(y_train, a)) or { 1.0 }

      // perform backpropagation
      mut da := d_logloss(y_train, a)
      for mut l in layers.reverse() {
        da = l.back_prop(da)
      }
    }

    mut demo_train := [[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]]
    for mut lyr in layers {
        demo_train = lyr.feed_fwd(demo_train)
    }
    println('neural net prediction')
    println(demo_train)

	return [NeuralNetModel{
        [3,3,3]
        ['relu', 'relu', 'relu']
        relu
        mse
        false
    }]
}
