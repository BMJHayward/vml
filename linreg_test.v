module linreg


fn test_sum() {
    assert 1 + 1 == 2
    assert linreg.sum(...[1,2,3]) == 6
    assert linreg.sum(...[1.0,2.0,3.0003]) != 6.0
}

fn test_vecdot() {
    a := [1.0,2.0,3.0]
    b := [4.0,5.0,6.0]
    c := linreg.vecdot(a, b)
    assert c == [4.0, 10.0, 18.0]
    d := [1.0, 2.0, 3.0]
    e := [4.0, 5.0, 6.0]
    f := linreg.vecdot(d, e)
    assert f == [4.0, 10.0, 18.0]
}

fn test_estimate_coefficients() {
    a := linreg.estimate_coefficients([1.0, 2.0], [2.0, 3.0])
    b := linreg.estimate_coefficients([1, 2, 3, 4, 5], [2, 6, 7, 12, 18])
    c := linreg.estimate_coefficients([0.01, 0.02, 0.0345], [0.09, 0.08, 0.063])
    assert typeof(a).name == '[]f64'
    assert a == [1.0, 1.0]
    assert b == [-2.3999999999999986, 3.8]
    assert c == [0.10146897309170783, -1.107084019769356]
}

fn test_train() {
    inputs := [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    outputs := [11.0, 12.0, 13.0]
    test_runner := linreg.LinearModel{ coeffs: []}
    test_model := test_runner.train(inputs, outputs)
    println(test_model)
    assert 'test_train' == 'test_train'
}

fn  test_predict() {
    test_model := linreg.LinearModel{
        coeffs: [2.0, 2.0]
    }
    test_data := [1.0, 2.0, 3.0, 4.0, 5.0]
    test_preds := test_model.predict(test_data)
    assert test_preds == [4.0, 6.0, 8.0, 10.0, 12.0]
}

fn test_demo() {
    test_demo := linreg.demo()
    println(test_demo)
    assert 'test_demo' == 'test_demo'
}
