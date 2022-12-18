module datagen

import rand
import rand.config

pub fn generate_data() ([][]f64, []f64) {
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
return src, target
}
