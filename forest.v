module forest

import rand
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

fn (mut rf RandomForest) fit(x [][]f64, y []f64, n_samples int, bootstrap bool) {
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
	return clf
}
