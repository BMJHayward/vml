module kmean

import arrays
import math
import math.stats
import rand
import rand.config

pub fn name() string {
	return 'kmeans clustering'
}

pub struct KMeansModel {
mut:
	optimum_clusters int
	centroids        [][]f64
	distances        []f64
	point_count      []int
}

// point_distance calculates euclidian distance between 2 points of [x,y] and [x,y], returns f64
fn point_distance<T>(pta []T, ptb []T) f64 {
	dx := pta[0] - ptb[0]
	dy := pta[1] - ptb[1]
	pd := math.sqrt(dx * dx + dy * dy)
	return pd
}

// random_point gives a random [x, y] in the given range
fn random_point(min_x f64, min_y f64, max_x f64, max_y f64) []f64 {
	rand_x := rand.f64_in_range(min_x, max_x) or { 0 }
	rand_y := rand.f64_in_range(min_y, max_y) or { 0 }
	return [rand_x, rand_y]
}

fn calc_opt_clusters() int {
	return 4
}

/*
train and predict will make a common API amongst as many model types as possible
K-Means clustering
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
This function will work on a new model or an existing, already trained model.
If training an existing model, the centroids and point counts will be updated, where as the cluster size and count will be replaced.
If you give this model new data which is clustered much the same as the data it was trained with, it will remain "in line" or equivalent.
// TODO: generalise this function to create new model or update existing model i.e. update `(mut m KMeansModel)`
*/
pub fn (mut m KMeansModel) train<T>(inp []T, output []T, iterations int, clusters int) KMeansModel {
	if clusters < 1 {
		panic('argument clusters must be > 0, you gave $clusters')
	}
	m.optimum_clusters = clusters
	mut km := KMeansModel{}
	omax := arrays.max(output) or { 0 }
	omin := arrays.min(output) or { 0 }

	imax := arrays.max(inp) or { 0 }
	imin := arrays.min(inp) or { 0 }
	mut diameters := []f64{}
	mut pairs := [][][]f64{} // add extra element at end for junk
	mut point_counts := []int{}
	mut centroids := [][]f64{}
	match m.centroids.len {
        0 {
			centroids = [][]f64{}
			for pt in 0 .. m.optimum_clusters {
				if pt < 0 {
					panic('cant iterate negative numbers. fix arg <iterations> in kmean.train')
				}
				centroids << random_point(imin, omin, imax, omax)
			}
		}
		else {
			centroids = m.centroids.clone()
		}
	}
	for phase in 0 .. iterations {
		diameters.clear()
		pairs.clear()
		for i := 0; i < centroids.len; i++ {
			pairs << [][]f64{}
		}
		if phase < 0 {
			panic('cant iterate negative numbers. fix arg <iterations> in kmean.train')
		}
		for idx, iv in inp {
			mut pr := []f64{len: 2}
			pr = [iv, output[idx]]
			mut dts := centroids.map(point_distance(it, pr))
			midx := arrays.idx_min(dts) or { pairs.len - 1 } // drop in junk if failed
			pairs[midx].prepend(pr)
		}
		for cdx, cntr in centroids {
			// get average of the cluster and update the centroid
			mut newcntr := arrays.reduce(pairs[cdx], fn (acc_pt []f64, next_pt []f64) []f64 {
				mut tt := []f64{len: 2}
				tt = [(acc_pt[0] + next_pt[0]) / 2, (acc_pt[1] + next_pt[1]) / 2]
				return tt
			}) or { cntr }
			centroids[cdx] = newcntr
		}
		// calculate distortion for each cluster
		for pdx in 0 .. centroids.len {
			mut dists := pairs[pdx].map(point_distance(it, centroids[pdx]))
			diameters << arrays.reduce(dists, fn <T>(cur T, nex T) T {
				return math.sqrt(cur * cur + nex * nex)
			}) or { 0 }
		}
	}
	point_counts << pairs.map(it.len)
	km = KMeansModel{
		optimum_clusters: int(clusters)
		centroids: centroids
		distances: diameters
		point_count: m.point_count + point_counts
	}
	/*
	TODO: incorporate this logic _nicely_
		if update {
			m.point_count[closest_cluster.last()]++
			m.centroids[closest_cluster.last()] = ((
				f64(m.point_count[closest_cluster.last()]) * m.centroids[closest_cluster.last()] + d)
				/ m.point_count[closest_cluster.last()])
		}*/
	return km
}

// predict takes a point as a 2-list and returns the index of the centroid it belongs to
// TODO: set whether this func updates the model or not
// if it does update the model, need to keep the count of clusters assigned to it and update
// using arithmetic mean
pub fn (mut m KMeansModel) predict<T>(data [][]T) ![]int {
	mut closest_cluster := []int{}
	for d in data {
		closest_cluster << arrays.idx_min(m.centroids.map(point_distance(it, d)))!
	}
	return closest_cluster
}

fn decile<T>(num T) T {
	return math.ceil(num / 10.0) * 10
}

pub fn demo() ![]KMeansModel {
	mut test_x1 := []f64{}
	mut test_x2 := []f64{}
	mut test_y := []f64{}
	mut valx := []f64{}
	mut valy := []f64{}

	for i := 0; i < 100; i++ {
		tx0 := rand.normal(config.NormalConfigStruct{ mu: 50, sigma: 1.0 }) or { 50.0 }
		tx1, ty1 := rand.normal_pair(config.NormalConfigStruct{ mu: 25, sigma: 1.0 }) or {
			25.0, 25.0
		}
		tx2, ty2 := rand.normal_pair(config.NormalConfigStruct{ mu: 75, sigma: 1.0 }) or {
			75.0, 75.0
		}
		vx, vy := rand.normal_pair(config.NormalConfigStruct{ mu: decile(f64(i)), sigma: 1.0 }) or {
			50.0, 1.0
		}
		test_x1 << tx0
		test_x2 << tx1
		test_x2 << tx2
		test_y << ty1
		test_y << ty2
		valx << vx
		valy << vy
	}

	mut km_runner := KMeansModel{
		optimum_clusters: 1
	}

	mut ds := []f64{}
	for cl in 1 .. 11 {
		mut opt_model := km_runner.train(test_x2, test_y, 100, cl)
		ds << stats.mean(opt_model.distances)
	}
	// calculate secondDerivative[i] = x[i+1] + x[i-1] - 2 * x[i]
	mut d2 := []f64{}
	for d := 1; d < ds.len - 1; d++ {
		d2 << ds[d + 1] + ds[d - 1] - 2 * ds[d]
	}
	d2.prepend(0) // prepend so that clusters is at least 1, not 0
	d2.prepend(0) // prepend so that clusters is at least 1, not 0
	opt_clusters := arrays.idx_max(d2.map(math.abs(it))) or { 1 }

	mut km_model := []KMeansModel{}
	km_model << km_runner.train(test_x1, test_y, 100, opt_clusters)
	km_model << km_runner.train(test_x2, test_y, 100, opt_clusters)
	mut validations := [][]f64{}
	for i in 0 .. valx.len {
		validations << [valx[i], valy[i]]
	}
	println('KMEANS PREDICTIONS:')
	println(km_model[0].predict(validations) or { [-1] })
	println(km_model[1].predict(validations) or { [-1] })
	return km_model
}
