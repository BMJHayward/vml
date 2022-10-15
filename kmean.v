module kmean

import arrays
import math
import rand

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
pub fn (m KMeansModel) train<T>(inputs [][]T, output []T, iterations int) []KMeansModel {
	mut km := []KMeansModel{len: inputs.len}
	default_clusters := 4
	omax := arrays.max(output) or { 0 }
	omin := arrays.min(output) or { 0 }
	orange := omax - omin
	ostep := orange / default_clusters
	o1 := omin + ostep
	o2 := omin + 3 * ostep

	for inp in inputs {
		imax := arrays.max(inp) or { 0 }
		imin := arrays.min(inp) or { 0 }
		irange := imax - imin
		istep := irange / default_clusters
		i1 := imin + istep
		i2 := imin + 3 * istep
		mut centroids := [[i1, o1], [i1, o2], [i2, o1], [i2, o2]]
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
			optimum_clusters: i8(inp.len / default_clusters)
			centroids: centroids
		}
	}
	return km
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
		test_x2 << f64(2 * i)
		test_y << 3 * (i + rand.int_in_range(-1, 1) or { 0 })
	}
	km_runner := KMeansModel{
		optimum_clusters: 1
	}
	{
	}
	km_model := km_runner.train([test_x1, test_x2], test_y, 1000)
	return km_model
}
