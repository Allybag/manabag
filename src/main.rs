#[derive(Debug)]
struct Matrix {
    data: Vec<f32>,
    dimensions: Vec<usize>,
}

fn new_matrix(data: Vec<f32>, dimensions: Vec<usize>) -> Matrix {
    let size = dimensions.iter().fold(1, |acc, num| { acc * num});
    assert_eq!(size, data.len());
    Matrix {
        data,
        dimensions
    }
}

fn new_vector(data: Vec<f32>) -> Matrix {
    let size = data.len();
    Matrix {
        data,
        dimensions: vec![size],
    }
}

fn main() {
    let vec = new_vector(vec![1.0, 2.0, 3.0]);
    let mat = new_matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    println!("Vector: {vec:?}, Matrix: {mat:?}");
}
