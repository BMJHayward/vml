module tree

pub fn name() string {
    return 'decision tree'
}

fn entropy(y []u8) f64 {
    mut hist := []u8{len: arrays.max(y) + 1, init: 0} 
    for i in 0 .. y.len {
        hist[y[i]] += 1
    }
    mut probs := hist.map(it / y.len)
    return arrays.sum(probs.filter(it > 0).map(it * log2(it)))
}

fn accuracy(y_true []f64, y_pred []f64) f64 {
  mut acc = 0.0
  for t in 0 .. y_true.len {
    if y_true[t] == y_pred[t] {
        acc += 1
    }
  }
  return acc / y_true.len
}

type featureType = []f64 | []int | []string
struct Node {
    feature featureType
    threshold f64
    left Node
    right Node
    value f64
}

fn _initNode(feature featureType, threshold f64 | string, left Node, right Node, value f64) Node {
    return Node {
        feature
        threshold
        left
        right
        value
    }
}

struct DecisionTree:
        min_samples_split int
        max_depth int
        n_feats int
        root Node

fn (dt DecisionTree) _init_tree(min_samples_split=2, max_depth=100, n_feats=0) {
    return DecisionTree {
        min_samples_split
        max_depth
        n_feats
        _init_Node()
    }
}