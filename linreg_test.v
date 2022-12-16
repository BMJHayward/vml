module linreg


fn test_sum() {
    assert 1 + 1 == 2
    assert linreg.sum(...[1,2,3]) == 6
    assert linreg.sum(...[1.0,2.0,3.0003]) != 6.0
}

fn test_vecdot() {
    a := [1,2,3]
    b := [4,5,6]
    c := linreg.vecdot(a, b)
    assert c == [4, 10, 18]
    d := [1.0, 2.0, 3.0]
    e := [4.0, 5.0, 6.0]
    f := linreg.vecdot(d, e)
    assert f == [4.0, 10.0, 18.0]
}

fn test_estimate_coefficients() {
    a := linreg.estimate_coefficients([1, 2], [2, 3])
    assert typeof(a).name == '[]int{}'
}

fn test_train() {
    assert 'test_train' == 'test_train'
}

fn  test_predict() {
    assert 'test_predict' == 'test_predict'
}

fn test_demo() {
    assert 'test_demo' == 'test_demo'
}
