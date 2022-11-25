module tree

import arrays
import math

pub fn name() string {
    return 'decision tree'
}

fn entropy(y []u8) f64 {
    ymax := arrays.max(y) or { panic('max y failed')}
    mut hist := []u8{len: int(ymax) + 1, init: 0} 
    for i in 0 .. y.len {
        hist[y[i]] += 1
    }
    mut probs := hist.map(it / y.len)
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
    feature Feature
    threshold f64
    left Tree
    right Tree
    value f64
}

fn init_node(feature Feature, threshold f64, left Node, right Node, value f64) Tree {
    return Node {
        feature
        threshold
        left
        right
        value
    }
}

struct DecisionTree {
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