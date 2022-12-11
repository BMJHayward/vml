import rand
import tree


pub struct RandomForest {
mut:
    n_trees int
    trees []tree.DecisionTree
	min_samples_split int
	max_depth         int
	n_feats           int
}

fn (mut rf RandomForest) fit(x [][]f64, y []f64, n_samples int, bootstrap bool) {
    mut sample_list := []int{}
    for s in 0 .. n_samples {
        sample_list << s
    }
	for _ in 0 .. rf.n_trees {
        mut tree := tree.init_tree(rf.min_samples_split, rf.max_depth, rf.n_feats)
        // sample x and y
        mut idxs := rand.choose(sample_list, n_samples) or { panic('could not choose random sample')}
        tree.fit(idxs.map(x[it]), idxs.map(y[it])) or { panic('failed to fit tree') }
        rf.trees << tree
    }}

fn (rf RandomForest) predict(x [][]f64) []f64 {
    return []f64{}
}
