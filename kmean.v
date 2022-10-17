module kmean

import arrays
import math
import rand
import rand.config

pub fn name() string {
	return 'kmeans clustering'
}

struct KMeansModel {
	optimum_clusters i8
	centroids        [][]f64
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
	println('calculate clusters')
	return 4
}

/*
train and predict will make a common API amongst as many model types as possible
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
// TODO: elbow method vs silhouette score vs hierarchical clustering to determine optimum clusters
*/
pub fn (m KMeansModel) train<T>(inputs [][]T, output []T, iterations int, clusters int) []KMeansModel {
	mut km := []KMeansModel{len: inputs.len}
	omax := arrays.max(output) or { 0 }
	omin := arrays.min(output) or { 0 }

	for inp in inputs {
		imax := arrays.max(inp) or { 0 }
		imin := arrays.min(inp) or { 0 }
		mut centroids := [][]f64{}
		for pt in 0 .. clusters {
			if pt < 0 {
				panic('cant iterate negative numbers. fix arg <iterations> in kmean.train')
			}
			centroids << random_point(imin, omin, imax, omax)
		}
		println('PHASES TO RUN: $iterations')
		for phase in 0 .. iterations {
			if phase < 0 {
			panic('cant iterate negative numbers. fix arg <iterations> in kmean.train')
			}
			mut pairs := [][][]f64{len: centroids.len + 1} // add extra element at end for junk
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
		}
		println('PHASES RUN: $iterations')
		km << KMeansModel{
			optimum_clusters: i8(clusters)
			centroids: centroids
		}
	}
	return km
}

// predict takes a point as a 2-list and returns the index of the centroid it belongs to
// TODO: set whether this func updates the model or not
// if it does update the model, need to keep the count of clusters assigned to it and update
// using arithmetic mean
pub fn (m KMeansModel) predict<T>(data [][]T) []T {
	mut closest_cluster := []T{}
	for d in data {
		closest_cluster << arrays.idx_min(m.centroids.map(point_distance(it, d)))
	}
	return closest_cluster
}

fn decile<T>( num T) T {
    return math.ceil(num / 10.0)
}

pub fn demo() []KMeansModel {
	mut test_x1 := []f64{}
	mut test_x2 := []f64{}
	mut test_y := []f64{}

	for i := 0; i < 100; i++ {
		test_x1 << f64(i)
		tx2, ty := rand.normal_pair(config.NormalConfigStruct{mu: decile(f64(i)), sigma: 2.0}) or { 50.0, 1.0 }
		test_x2 << tx2
		test_y  << ty
	}

	km_runner := KMeansModel{
		optimum_clusters: 1
	}

	km_model := km_runner.train([test_x1, test_x2], test_y, 100, 4)
	return km_model
}
