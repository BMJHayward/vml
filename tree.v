module tree

import arrays
import math
import rand

pub fn name() string {
    return 'decision tree'
}

fn most_common<T>(y []T) T {
    mut max_count := 0
    mut most_frequent := 0
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
    mut probs := hist.values().map(it / y.len)
    return arrays.sum(probs.filter(it > 0).map(it * math.log2(it))) or { panic('failed to sum array') }
}

fn accuracy(y_true []f64, y_pred []f64) f64 {
  mut acc := 0.0
  for t in 0 .. y_true.len {
    if y_true[t] == y_pred[t] {
        acc += 1
    }
  }
  return acc / y_true.len
}

type Feature = []f64 | []int | []string
type Tree = Empty | Node

struct Empty {}

struct Node {
    mut: 
    feature int
    threshold f64
    left Tree
    right Tree
    value f64
}

fn init_node(feature int, threshold f64, left Node, right Node, value f64) Tree {
    return Node {
      mut
        feature
        threshold
        left
        right
        value
    }
}

fn (n Node) is_leaf() bool {
    return n.value > 0
}

struct DecisionTree {
  mut:
    min_samples_split int
    max_depth int
    n_feats int
    root Tree
}

fn init_tree(min_samples_split int, max_depth int, n_feats int) DecisionTree {
    return DecisionTree {
        min_samples_split
        max_depth
        n_feats
        Empty{}
    }
}

fn (mut dt DecisionTree) fit(x []f64, y []f64) ? {
    // dt.n_feats = X.shape[1] if not dt.n_feats else min(dt.n_feats, x.shape[1])
    // dt.root = dt.grow_tree(X, y)
    if dt.n_feats > 0 {
        dt.n_feats = math.min(dt.n_feats, x.len)
    } else {
        dt.n_feats = x.len
    }
    dt.root = Empty{} // should be grow tree function here
}

fn (dt DecisionTree) grow_tree(x [][]f64, y []f64, depth int) Node {
    n_samples := x.len
    n_features := x[0].len
    // n_labels := len(np.unique(y))
    mut yuniq := map[f64]f64{}
    for yq in y {
        yuniq[yq] = yq
    }
    n_labels := yuniq.len

    // stopping criteria
    if depth >= dt.max_depth
        || n_labels == 1
        || n_samples < dt.min_samples_split {
        leaf_value := most_common(y)
        return Node{0, 0.0, Empty{}, Empty{}, leaf_value}
    }

    mut n_feat_array := []int{}
	for n in 0 .. n_features {
        n_feat_array << n
    }
    feature_indices := rand.choose<int>(n_feat_array, dt.n_feats) or { panic('failed to create feat indices') }

    // greedily select the best split according to information gain
    best_feat, best_thresh := best_criteria<f64>(x, y, feature_indices)
    // grow the children that result from the split
    left_idxs, right_idxs := split<f64>(x[..][best_feat], best_thresh)
    xlix := left_idxs.map(x[it])
    xrix := right_idxs.map(x[it])
    ylix := left_idxs.map(y[it])
    yrix := right_idxs.map(y[it])
    left := dt.grow_tree(xlix, ylix, depth+1)
    right := dt.grow_tree(xrix, yrix, depth+1)
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

fn best_criteria<T>(x [][]T, y []T, feat_idxs []int) (int, T) {
    mut best_gain := -1
    mut split_idx, split_thresh := 0, 0
    for feat_idx in feat_idxs {
        mut x_column := []int{}
        for xc in 0 .. x.len {
            x_column << x[xc][feat_idx]
        }
        // thresholds = unique(x_column)  // TODO make a unique function
        mut tuniq := map[f64]f64{}
        for t in x_column {
            tuniq[t] = t
        }
        for _, v in tuniq {
            thresholds << v
        }
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
    child_entropy := ( llen / ylen ) * lentropy + (rlen / ylen) * rentropy

    return parent_entropy - child_entropy
}