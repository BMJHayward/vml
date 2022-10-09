module kmean

import rand

pub fn name() string {
    return 'kmeans clustering'
}

struct KMeansModel {
    optimum_clusters i8
}

/*
Train and predict will make a common API amongst as many model types as possible
K-Means clustering
TODO: implement elbow function to automate optimum K
TODO: implement convervgence measure, gradient descent etc to stop training
1. choose number of clusters (K), default 4
2. place centroids c1, c2, ... ck randomly, or parition the space evenly
3. repeat (4) and (5) until convergence or maximum iterations reached
4. foreach datum x_i
  - find nearest centroid (c1, c2, ... ck)
  - assign point to that centroid
5. foreach cluster j1, j2, ... jk:
  - update centroid -> mean of all points currently in cluster
6. done
*/
pub fn (m KMeansModel) train<T>(inputs [][]T, output []T) []KMeansModel{
	mut lm := []KMeansModel{len: inputs.len}
	for inp in inputs {
		lm << KMeansModel{ optimum_clusters: i8(inp.len / 4) }
	}
	return lm
}

pub fn (m KMeansModel) predict<T>(data []T) []T {
	println('DATA: $data')
}

pub fn run() []KMeansModel {
	mut test_x1 := []f64{}
	mut test_x2 := []f64{}
	mut test_y := []f64{}
	for i := 0; i < 100; i++ {
		test_x1 << f64(i)
		test_x2 << f64(2*i)
		test_y << 3 * (i + rand.int_in_range(-1,1) or { 0 })
	}
	// test_reg_coeffs := estimate_coefficients(test_x1, test_y)
	km_runner := KMeansModel{optimum_clusters: 1}{}
	km_model := km_runner.train([test_x1, test_x2], test_y)
	return km_model
}
