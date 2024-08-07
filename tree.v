module tree

import arrays
import datagen
import math
import rand

pub fn name() string {
	return 'decision tree'
}

pub fn most_common<T>(y []T) T {
	mut max_count := 0
	mut most_frequent := T(0)
	for i in 0 .. y.len {
		mut count := 0
		for j in 0 .. y.len {
			if y[i] == y[j] {
				count += 1
			}
		}
		if count > max_count {
			max_count = count
			most_frequent = y[i]
		}
	}
	return most_frequent
}

fn entropy<T>(y []T) f64 {
	mut hist := map[T]int{}
	for i in 0 .. y.len {
		hist[y[i]] += 1
	}
	mut probs := hist.values().map(it / f64(y.len))
	mut logits := probs.filter(it > 0).map(-1 * it * math.log2(it))
	return arrays.sum(logits) or { panic('failed to sum array') }
}

pub fn accuracy(y_true []f64, y_pred []f64) f64 {
	mut acc := 0.0
	for t in 0 .. math.min(y_true.len, y_pred.len) {
		if y_true[t] == y_pred[t] {
			acc += 1
		}
	}
	return acc / y_true.len
}

type Feature = []f64 | []int | []string
pub type Tree = Empty | Node

pub struct Empty {}

pub struct Node {
mut:
	feature   int
	threshold f64
	left      Tree
	right     Tree
	value     f64
}

fn init_node(feature int, threshold f64, left Node, right Node, value f64) Tree {
	return Node{mut feature, threshold, left, right, value}
}

fn (n Node) is_leaf() bool {
	return n.value > 0
}

pub struct DecisionTree {
mut:
	min_samples_split int
	max_depth         int
	n_feats           int
	root              Tree
}

pub fn init_tree(min_samples_split int, max_depth int, n_feats int) DecisionTree {
	return DecisionTree{min_samples_split, max_depth, n_feats, Empty{}}
}

pub fn (mut dt DecisionTree) fit(x [][]f64, y []f64) ? {
	if dt.n_feats > 0 {
		dt.n_feats = math.min(dt.n_feats, x.len)
	} else {
		dt.n_feats = x[0].len
	}
	dt.root = dt.grow_tree(x, y, 0)
}

pub fn (mut dt DecisionTree) predict(x [][]f64) []f64 {
	mut predictions := []f64{}
	for datum in x {
		predictions << traverse(datum, dt.root)
	}
	return predictions
}

fn traverse(x []f64, node Tree) f64 {
	match node {
		Empty {
			return -1.0
		}
		Node {
			if node.is_leaf() {
				return node.value
			}

			if x[node.feature] <= node.threshold {
				return traverse(x, node.left)
			}
			return traverse(x, node.right)
		}
	}
}

fn (dt DecisionTree) grow_tree(x [][]f64, y []f64, depth int) Node {
	n_samples := x.len
	n_features := match n_samples {
		0 { 0 }
		1 { x[0].len }
		else { x[0].len }
	}
	mut yuniq := map[f64]f64{}
	for yq in y {
		yuniq[yq] = yq
	}
	n_labels := yuniq.len

	// stopping criteria
	if depth >= dt.max_depth || n_labels == 1 || n_samples < dt.min_samples_split {
		leaf_value := most_common(y)
		return Node{0, 0.0, Empty{}, Empty{}, leaf_value}
	}

	mut n_feat_array := []int{}
	for n in 0 .. n_features {
		n_feat_array << n
	}
	feature_indices := rand.choose<int>(n_feat_array, dt.n_feats) or {
		panic('failed to create feat indices')
	}

	// greedily select the best split according to information gain
	best_feat, best_thresh := best_criteria<f64>(x, y, feature_indices)
	// grow the children that result from the split
	left_idxs, right_idxs := split<f64>(x[..][best_feat], best_thresh)
	xlix := left_idxs.map(x[it])
	xrix := right_idxs.map(x[it])
	ylix := left_idxs.map(y[it])
	yrix := right_idxs.map(y[it])
	left := dt.grow_tree(xlix, ylix, depth + 1)
	right := dt.grow_tree(xrix, yrix, depth + 1)
	return Node{best_feat, best_thresh, left, right, 0}
}

fn split<T>(x_column []T, split_thresh T) ([]int, []int) {
	mut left_idxs := []int{}
	mut right_idxs := []int{}
	for i in 0 .. x_column.len {
		if x_column[i] <= split_thresh {
			left_idxs << i
		} else {
			right_idxs << i
		}
	}

	return left_idxs, right_idxs
}

fn unique<T>(all []T) []T {
	mut s := map[T]T{}
	for a in all {
		s[a] = a
	}
	return s.keys()
}

fn best_criteria<T>(x [][]T, y []T, feat_idxs []int) (int, T) {
	mut best_gain := -1.0
	mut split_idx := 0
	mut split_thresh := 0.0
	for feat_idx in feat_idxs {
		mut x_column := []T{}
		for xc in 0 .. x.len {
			x_column << x[xc][feat_idx]
		}
		thresholds := unique(x_column) // TODO make a unique function
		for threshold in thresholds {
			// gain := feat_idxs.len / y.len
			gain := info_gain(y, x_column, threshold)
			if gain > best_gain {
				best_gain = gain
				split_idx = feat_idx
				split_thresh = threshold
			}
		}
	}
	return split_idx, split_thresh
}

fn info_gain<T>(y []T, xcol []T, threshold T) f64 {
	parent_entropy := entropy(y)
	l, r := split(xcol, threshold)
	if l.len == 0.0 {
		return 0.0
	}
	if r.len == 0.0 {
		return 0.0
	}

	ylen := y.len
	rlen := r.len
	llen := l.len
	left_idxs := l.map(y[it])
	right_idxs := r.map(y[it])
	lentropy := entropy(left_idxs)
	rentropy := entropy(right_idxs)
	child_entropy := (llen / ylen) * lentropy + (rlen / ylen) * rentropy

	return parent_entropy - child_entropy
}

pub fn demo() DecisionTree {
	// x = data.data
	/*
	tx0 := rand.normal(config.NormalConfigStruct{ mu: 50, sigma: 1.0 }) or { 50.0 }
    tx1 := rand.exponential(2)
    tx2 := rand.binomial(2, 0.65)
	*/
	num_features := 12
	mut src, mut target := datagen.generate_data()
	mut clf := init_tree(10, 5, num_features)
	clf.fit(src[200..], target[200..]) or { panic('fit failed') }
	mut y_pred := clf.predict(src[200..])
	mut acc := accuracy(target[200..], y_pred)
	println(acc)
	y_pred = clf.predict(src[..200])
	acc = accuracy(target[..200], y_pred)
	println(acc)
	return clf
}
