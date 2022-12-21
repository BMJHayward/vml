module linreg

import math
import math.stats
import rand
// import vsl.plot

pub fn name() string {
	return 'linear regression'
}

fn sum<T>(a ...T) T {
	mut total := T(0)
	for x in a {
		total += x
	}
	return total
}

fn vecdot<T>(uu []T, vv []T) []T {
	mut ww := []T{}
	for i := 0; i < uu.len; i++ {
		ww << uu[i] * vv[i]
	}
	return ww
}

fn estimate_coefficients<T>(input []T, output []T) []f64 {
	// need to use floats for mean, division and dot products
	x := input.map(f64(it))
	y := output.map(f64(it))

	// number of observations/points
	n := x.len

	// mean of x and y vector
	xmean := stats.mean(x.map(f64(it)))
	ymean := stats.mean(y.map(f64(it)))

	// calculating cross-deviation and deviation about x
	ss_xy := sum(...vecdot(y, x)) - n * ymean * xmean
	ss_xx := sum(...vecdot(x, x)) - n * xmean * xmean

	// calculating regression coefficients
	b_1 := ss_xy / ss_xx
	b_0 := ymean - b_1 * xmean
	rv := [b_0, b_1]
	return rv
}

struct LinearModel {
	coeffs []f64
}

/*
Train and predict will make a common API amongst as many model types as possible
*/
pub fn (m LinearModel) train<T>(inputs [][]T, output []T) []LinearModel {
	mut lm := []LinearModel{len: inputs.len}
	for inp in inputs {
		lm << LinearModel{
			coeffs: estimate_coefficients(inp, output)
		}
	}
	return lm
}

pub fn (m LinearModel) predict<T>(data []T) []T {
	println('DATA: $data')
}

pub fn demo() []LinearModel {
	mut test_x1 := []f64{}
	mut test_x2 := []f64{}
	mut test_y := []f64{}
	for i := 0; i < 100; i++ {
		test_x1 << f64(i)
		test_x2 << f64(2 * i)
		test_y << 3 * (i + rand.int_in_range(-1, 1) or { 0 })
	}
	// test_reg_coeffs := estimate_coefficients(test_x1, test_y)
	lm_runner := LinearModel{
		coeffs: []
	}
	linear_model := lm_runner.train([test_x1, test_x2], test_y)
	return linear_model
}
