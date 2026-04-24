use moonai_ffi::Random;

fn main() {
    let mut rng = Random::new(42);

    println!("Comparing RNG sequences...\n");

    println!("First 20 int values [0, 1000]:");
    for i in 0..20 {
        let val = rng.next_int(0, 1000);
        println!("int {}: {}", i, val);
    }

    println!("\nFirst 10 float values [0.0, 1.0]:");
    for i in 0..10 {
        let val = rng.next_float(0.0, 1.0);
        println!("float {}: {:.6}", i, val);
    }

    println!("\nFirst 10 bool values [probability=0.5]:");
    for i in 0..10 {
        let val = rng.next_bool(0.5);
        println!("bool {}: {}", i, val);
    }

    println!("\nFirst 10 weighted select values [1,2,3,4,5]:");
    let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    for i in 0..10 {
        let val = rng.weighted_select(&weights);
        println!("weighted {}: {}", i, val);
    }
}
