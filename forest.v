module forest

import rand
import rand.config
import tree

pub fn name() string {
	return 'random forest'
}

pub struct RandomForest {
mut:
	n_trees           int
	trees             []tree.DecisionTree
	min_samples_split int
	max_depth         int
	n_feats           int
}

pub fn init_forest(n_trees int, trees []tree.DecisionTree, min_samples_split int, max_depth int, n_feats int) RandomForest {
	return RandomForest{n_trees, trees, min_samples_split, max_depth, n_feats}
}

fn (mut rf RandomForest) fit(x [][]f64, y []f64) {
    n_samples := x.len
	mut sample_list := []int{}
	for s in 0 .. n_samples {
		sample_list << s
	}
	for _ in 0 .. rf.n_trees {
		mut tree := tree.init_tree(rf.min_samples_split, rf.max_depth, rf.n_feats)
		// sample x and y
		mut idxs := rand.choose(sample_list, n_samples) or {
			panic('could not choose random sample')
		}
		tree.fit(idxs.map(x[it]), idxs.map(y[it])) or { panic('failed to fit tree') }
		rf.trees << tree
	}
}

fn (mut rf RandomForest) predict(x [][]f64) []f64 {
    mut tpreds := [][]f64{}
    for t in 0 .. rf.trees.len {
        tpreds << rf.trees[t].predict(x)
    }
    mut ypreds := []f64{}
    ypreds << tpreds.map(tree.most_common(it))
	return []f64{}
}

pub fn demo() RandomForest {
    mut clf := RandomForest{
        3,
        []tree.DecisionTree{len: 3},
        20,
        10,
        12
        }
    num_samples := 1000
	num_features := 12
	mut src := [][]f64{len: num_samples}
	mut target := []f64{}
	for s in 0 .. num_samples {
		mut class := rand.int_in_range(1, 6) or { panic('rand int failed') }
		match class {
			0 {
				for _ in 0 .. num_features {
					src[s] << rand.normal(config.NormalConfigStruct{ mu: 10.0, sigma: 1.0 }) or {
						10.0
					}
				}
				target << f64(class)
			}
			1 {
				for _ in 0 .. num_features {
					// mut tx1 := rand.exponential(2)
					src[s] << rand.normal(config.NormalConfigStruct{ mu: 50.0, sigma: 1.0 }) or {
						50.0
					}
				}
				target << f64(class)
			}
			2 {
				for _ in 0 .. num_features {
					// mut tx2 := rand.binomial(2, 0.65) or { 2 }
					src[s] << rand.normal(config.NormalConfigStruct{ mu: 90.0, sigma: 1.0 }) or {
						90.0
					}
				}
				target << f64(class)
			}
			3 {
				for _ in 0 .. num_features {
					// mut tx2 := rand.binomial(2, 0.65) or { 2 }
					src[s] << rand.normal(config.NormalConfigStruct{ mu: 30.0, sigma: 1.0 }) or {
						30.0
					}
				}
				target << f64(class)
			}
			4 {
				for _ in 0 .. num_features {
					// mut tx2 := rand.binomial(2, 0.65) or { 2 }
					src[s] << rand.normal(config.NormalConfigStruct{ mu: 10.0, sigma: 1.0 }) or {
						10.0
					}
				}
				target << f64(class)
			}
			5 {
				for _ in 0 .. num_features {
					// mut tx2 := rand.binomial(2, 0.65) or { 2 }
					src[s] << rand.normal(config.NormalConfigStruct{ mu: 70.0, sigma: 1.0 }) or {
						70.0
					}
				}
				target << f64(class)
			}
			else {
				for _ in 0 .. num_features {
					src[s] << 9.0
				}
				target << 9.0
			}
		}
	}
    clf.fit(src[200..], target[200..])
	mut y_pred := clf.predict(src[200..])
	mut acc := tree.accuracy(target[200..], y_pred)
	println(acc)
	y_pred = clf.predict(src[..200])
	acc = tree.accuracy(target[..200], y_pred)
	println(acc)
	return clf
}
