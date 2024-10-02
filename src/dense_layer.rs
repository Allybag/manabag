use crate::matrix::Matrix;

#[derive(Debug)]
pub struct DenseLayer<const INPUTS: usize, const NEURONS: usize> {
    weights: Matrix<f32, INPUTS, NEURONS>,
    biases: Matrix<f32, 1, NEURONS>,
}

pub fn relu_activate<const C: usize, const R: usize>(matrix: &mut Matrix<f32, R, C>) {
    matrix.transform(| val | { if val > 0.0 { val } else { 0.0 } });
}

impl<const INPUTS: usize, const NEURONS: usize> DenseLayer<INPUTS, NEURONS> {
    pub fn new() -> Self {
        DenseLayer {
            weights: Matrix::new([[f32::default(); NEURONS]; INPUTS]),
            biases: Matrix::new([[f32::default(); NEURONS]]),
        }
    }

    pub fn set_weights(&mut self, weights: [[f32; NEURONS]; INPUTS]) {
        self.weights = Matrix::new(weights);
    }

    pub fn forward<const ROWS: usize>(self, inputs: [[f32; INPUTS]; ROWS]) -> Matrix<f32, ROWS, NEURONS> {
        self.weights * Matrix::new(inputs) + self.biases
    }

}