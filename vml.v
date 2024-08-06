module main

// import knn
// import logreg
// import reinf
import forest
import kmean
import linreg
import nnet
import tree

fn main() {
	println('Hello VML!')
	println('Loaded: $forest.name()')
	println('Loaded: $kmean.name()')
	// println('Loaded: $knn.name()')
	println('Loaded: $linreg.name()')
	//   println('Loaded: $logreg.name()')
	println('Loaded: $nnet.name()')
	//    println('Loaded: $reinf.name()')
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
