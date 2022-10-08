module main

import forest
import kmean
import knn
import linreg
import logreg
import nnet
import reinf
import tree

fn main() {
	println('Hello VML!')
    println('Loaded: $forest.name()')
    println('Loaded: $kmean.name()')
    println('Loaded: $knn.name()')
    println('Loaded: $linreg.name()')
    println('Loaded: $logreg.name()')
    println('Loaded: $nnet.name()')
    println('Loaded: $reinf.name()')
    println('Loaded: $tree.name()')

    trc := linreg.run()
    println('RUN: linreg: $trc')
}
