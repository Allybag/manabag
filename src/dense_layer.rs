use crate::matrix::Matrix;

#[derive(Debug)]
pub enum Activation
{
    Relu,
    Softmax,
}

#[derive(Debug)]
pub struct DenseLayer<const INPUTS: usize, const NEURONS: usize> {
    weights: Matrix<f32, INPUTS, NEURONS>,
    biases: Matrix<f32, 1, NEURONS>,
    activation: Activation,
}

pub fn relu_activate<const C: usize, const R: usize>(matrix: &mut Matrix<f32, R, C>) {
    matrix.transform(| val | { if val > 0.0 { val } else { 0.0 } });
}

pub fn softmax_activate<const C: usize, const R: usize>(matrix: &mut Matrix<f32, R, C>) {
    matrix.row_transform(| row | {
        let exponents = row.map(| val | { val.exp() });
        let exponent_sum : f32 = exponents.iter().sum();
        exponents.map(| val | { val / exponent_sum })
    });
}

impl<const INPUTS: usize, const NEURONS: usize> DenseLayer<INPUTS, NEURONS> {
    pub fn new() -> Self {
        DenseLayer {
            weights: Matrix::new([[f32::default(); NEURONS]; INPUTS]),
            biases: Matrix::new([[f32::default(); NEURONS]]),
            activation: Activation::Relu,
        }
    }

    pub fn set_weights(&mut self, weights: [[f32; NEURONS]; INPUTS]) {
        self.weights = Matrix::new(weights);
    }

    pub fn set_activation(&mut self, activation: Activation) {
        self.activation = activation;
    }

    pub fn forward<const ROWS: usize>(&self, inputs: Matrix<f32, ROWS, INPUTS>) -> Matrix<f32, ROWS, NEURONS> {
        self.weights * inputs + self.biases
    }

    pub fn activate<const ROWS: usize>(&self, matrix: &mut Matrix<f32, ROWS, NEURONS>) {
        match (self.activation)
        {
            Activation::Relu => relu_activate(matrix),
            Activation::Softmax => softmax_activate(matrix),
        }
    }

    pub fn output<const ROWS: usize>(self, inputs: Matrix<f32, ROWS, INPUTS>)  -> Matrix<f32, ROWS, NEURONS> {
        let mut mat = self.forward(inputs);
        self.activate(&mut mat);
        mat
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax() {
        let mut mat = Matrix::new([[4.8, 1.21, 2.385]]);
        softmax_activate(&mut mat);

        assert_eq!(mat, Matrix::new([[0.89528266, 0.02470831, 0.08000903]]));
    }
}
