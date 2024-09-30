pub mod matrix;

fn main() {
    let vec = matrix::new_vector(vec![1.0, 2.0, 3.0]);
    let mat = matrix::new_matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    println!("Vector: {vec:?}, Matrix: {mat:?}");
}
