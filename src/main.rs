pub mod matrix;

fn main() {
    let vec = matrix::Matrix::new([[1.0, 2.0, 3.0]]);
    let mat = matrix::Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    println!("Vector: {vec:?}, Matrix: {mat:?}");
}
