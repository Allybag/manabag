#[derive(Debug)]
pub struct Matrix {
    data: Vec<f32>,
    dimensions: Vec<usize>,
}

pub fn new_matrix(data: Vec<f32>, dimensions: Vec<usize>) -> Matrix {
    let size = dimensions.iter().fold(1, |acc, num| { acc * num});
    assert_eq!(size, data.len());
    Matrix {
        data,
        dimensions
    }
}

pub fn new_vector(data: Vec<f32>) -> Matrix {
    let size = data.len();
    Matrix {
        data,
        dimensions: vec![size],
    }
}
