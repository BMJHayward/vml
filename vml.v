module main

// import knn
// import logreg
// import reinf
import forest
import kmean
import linreg
import nnet
import tree
import os
import flag

fn main() {
	mut fp := flag.new_flag_parser(os.args)
	fp.application('vml')
	fp.version('0.0.1')
	fp.description('simple machine learning for the V programming language')
	fp.skip_executable()

	model := fp.string('model', `m`, 'none', 'The model to analyse your data.').to_lower()
	demo := fp.string('demo', `e`, 'none', 'Demonstrate model training and output on fake data.').to_lower()
	data := fp.string('data', `d`, 'none', 'Input data as csv file.').to_lower()
	additional_args := fp.finalize() !

	if additional_args.len > 0 {
		println('Unprocessed arguments:\n$additional_args.join_lines()')
	}

	println(model)
	println(demo)
	println(data[0..data.len-1])

	println('Hello VML!')
	println('Loaded: $forest.name()')
	println('Loaded: $kmean.name()')
	// println('Loaded: $knn.name()')
	println('Loaded: $linreg.name()')
	// println('Loaded: $logreg.name()')
	println('Loaded: $nnet.name()')
	// println('Loaded: $reinf.name()')
	println('Loaded: $tree.name()')

	lrr := linreg.demo()
	println('RUN: linreg: $lrr')
	kmr := kmean.demo() or { []kmean.KMeansModel{} }
	println('RUN: kmeans: $kmr')
	nnd := nnet.demo()
	println('RUN: neural net: $nnd')
	dtd := tree.demo()
	println('RUN: decision tree: $dtd')
	rfd := forest.demo()
	println('RUN: random forest: $rfd')
	println('DONE')
}
