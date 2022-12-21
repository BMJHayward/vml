module kmean

fn test_point_distance() {
    pt1 := kmean.point_distance([0,0], [3,4])
    assert pt1 == 5
    pt2 := kmean.point_distance([0,0], [-3,4])
    assert pt2 == 5
    pt3 := kmean.point_distance([0,0], [-3,-4])
    assert pt3 == 5
    pt4 := kmean.point_distance([0,0], [3,-4])
    assert pt4 == 5
}

fn test_random_point() {
    mut minxy := [0,0]
    mut maxxy := [1,1]
    mut rxy := kmean.random_point(minxy[0], minxy[1], maxxy[0], maxxy[1])
    assert minxy[0] <= rxy[0] && rxy[0] <= maxxy[0]
    assert minxy[1] <= rxy[1] && rxy[1] <= maxxy[1]

    minxy = [-10,-10]
    maxxy = [10,10]
    rxy = kmean.random_point(minxy[0], minxy[1], maxxy[0], maxxy[1])
    assert minxy[0] <= rxy[0] && rxy[0] <= maxxy[0]
    assert minxy[1] <= rxy[1] && rxy[1] <= maxxy[1]
}

fn test_calc_opt_clusters() {
    assert kmean.calc_opt_clusters()==4
}

fn test_decile() {
    for i in 1 .. 11 {
        assert kmean.decile(f64(i))== 10 * (i-1)
    }
}


fn test_train() {}
fn test_predict() {}
fn test_demo() {}
