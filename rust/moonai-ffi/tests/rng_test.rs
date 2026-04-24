use moonai_ffi::Random;

#[test]
fn test_rng_basic() {
    let mut rng = Random::new(42);
    let val = rng.next_int(0, 100);
    assert!(val >= 0 && val <= 100);
}

#[test]
fn test_rng_sequence_deterministic() {
    let mut rng1 = Random::new(42);
    let mut rng2 = Random::new(42);

    for _ in 0..100 {
        assert_eq!(rng1.next_int(0, 1000), rng2.next_int(0, 1000));
    }
}

#[test]
fn test_next_float() {
    let mut rng = Random::new(42);
    for _ in 0..100 {
        let val = rng.next_float(0.0, 1.0);
        assert!(val >= 0.0 && val <= 1.0);
    }
}

#[test]
fn test_next_bool() {
    let mut rng = Random::new(42);
    let mut true_count = 0;
    for _ in 0..1000 {
        if rng.next_bool(0.5) {
            true_count += 1;
        }
    }
    assert!(true_count > 400 && true_count < 600);
}

#[test]
fn test_weighted_select() {
    let mut rng = Random::new(42);
    let weights = vec![1.0, 2.0, 3.0];
    for _ in 0..100 {
        let idx = rng.weighted_select(&weights);
        assert!(idx >= 0 && idx < 3);
    }
}

#[test]
fn test_weighted_select_empty() {
    let mut rng = Random::new(42);
    let weights: Vec<f32> = vec![];
    let idx = rng.weighted_select(&weights);
    assert_eq!(idx, -1);
}
