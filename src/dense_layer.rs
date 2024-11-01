use crate::simple_matrix::Matrix;

#[derive(Debug)]
pub enum Activation
{
    Relu,
    Softmax,
}

#[derive(Debug)]
pub struct DenseLayer {
    weights: Matrix,
    biases: Matrix,
    inputs: Matrix,
    pub dweights: Matrix,
    pub dbiases: Matrix,
    pub dinputs: Matrix,
}

pub fn relu_activate(matrix: &mut Matrix) {
    matrix.transform(| val | { if val > 0.0 { val } else { 0.0 } });
}

pub fn softmax_activate(matrix: &mut Matrix) {
    matrix.row_transform(| values, start_index, end_index | {
        let mut exponent_sum = 0.0;
        for index in start_index..end_index {
            values[index] = values[index].exp();
            exponent_sum += values[index];
        }

        for index in start_index..end_index {
            values[index] = values[index] / exponent_sum;
        }
    });
}

fn clip(value: f32) -> f32 {
    assert!(value >= 0.0 && value <= 1.0);
    let epsilon: f32 = 1e-7;

    value.clamp(epsilon, 1.0 - epsilon)
}

pub fn categorical_loss(matrix: &Matrix, labels: &[usize]) -> f32 {
    let rows = matrix.rows;
    let mut confidences = Vec::<f32>::with_capacity(rows);
    confidences.resize(rows, 0.0);
    for row in 0..rows {
        confidences[row] = clip(matrix.at(row, labels[row]))
    }

    let negative_log_likelihoods: Vec<f32> = confidences.iter().map(|val| -1.0 * val.ln()).collect();
    let loss: f32 = negative_log_likelihoods.iter().sum::<f32>() / labels.len() as f32;
    loss
}

fn col_sum(matrix: &Matrix) -> Matrix {
    let cols = matrix.cols;
    let mut values = Vec::<f32>::with_capacity(cols);
    values.resize(cols, 0.0);

    for row in 0..matrix.rows {
        for col in 0..cols {
            values[col] += matrix.at(row, col);
        }
    }

    Matrix { rows: 1, cols: cols, values }
}

pub fn get_predictions(matrix: &Matrix) -> Vec<usize> {
    let rows = matrix.rows;
    let mut predictions = Vec::<usize>::with_capacity(rows);
    predictions.resize(rows, matrix.cols + 1);
    for row in 0..rows {
        // Each value should be between 0.0 and 1.0
        let mut max = 0.0;
        for col in 0..matrix.cols {
            let val = matrix.at(row, col);
            assert!(val >= 0.0 && val <= 1.0);
            if val >= max {
                predictions[row] = col;
                max = val;
            }
        }
    }

    predictions
}

pub fn calc_accuracy(predictions: &Vec<usize>, labels: &[usize]) -> f32 {
    assert!(predictions.len() == labels.len());
    let mut correct_count = 0;
    for index in 0..predictions.len() {
        correct_count += if predictions[index] == labels[index] { 1 } else { 0 };
    }

    correct_count as f32 / predictions.len() as f32
}

impl DenseLayer {
    pub fn new() -> Self {
        DenseLayer {
            weights: Matrix::new([[]]),
            biases: Matrix::new([[]]),
            inputs: Matrix::new([[]]),
            dweights: Matrix::new([[]]),
            dbiases: Matrix::new([[]]),
            dinputs: Matrix::new([[]]),
        }
    }

    pub fn set_weights<const NEURONS: usize, const INPUTS: usize>(&mut self, weights: [[f32; NEURONS]; INPUTS]) {
        self.weights = Matrix::new(weights);
        self.biases = Matrix::new([[f32::default(); NEURONS]])
    }

    pub fn forward(&mut self, inputs: Matrix) -> Matrix {
        self.inputs = inputs.clone();
        inputs * self.weights.clone() + self.biases.clone()
    }

    pub fn backward(&mut self, dvalues: Matrix) {
        self.dweights = self.inputs.clone().transpose() * dvalues.clone();
        self.dbiases = col_sum(&dvalues);
        self.dinputs = dvalues * self.weights.clone().transpose();
    }
}

pub struct ActivationLayer {
    activation: Activation,
    inputs: Matrix,
    pub dinputs: Matrix,
}

impl ActivationLayer {
    pub fn new(activation: Activation) -> Self {
        ActivationLayer {
            activation: activation,
            inputs: Matrix::new([[]]),
            dinputs: Matrix::new([[]]),
        }
    }

    pub fn forward(&mut self, matrix: &mut Matrix) {
        self.inputs = matrix.clone();
        match self.activation
        {
            Activation::Relu => relu_activate(matrix),
            Activation::Softmax => softmax_activate(matrix),
        }
    }

    pub fn backward(&mut self, matrix: Matrix, optional_labels: Option<&[usize]>) {
        self.dinputs = matrix.clone();
        match self.activation
        {
            Activation::Relu => {
                assert!(optional_labels.is_none());
                for row in 0..self.dinputs.rows {
                    for col in 0..self.dinputs.cols {
                        if self.inputs.at(row, col) < 0.0 {
                            self.dinputs.set(0.0, row, col);
                        }
                    }
                }
            }
            Activation::Softmax => {
                match optional_labels
                {
                    None => panic!("Softmax activation expects labels"),
                    Some(labels) => {
                        let rows = self.dinputs.rows;
                        assert!(labels.len() == rows);
                        for row in 0..rows {
                            let col = labels[row];
                            let val = self.dinputs.at(row, col);
                            self.dinputs.set(val - 1.0, row, col);
                        }

                        // TODO: I have deliberately been ignoring cloning
                        // and references and that, but this is particularly mad
                        self.dinputs = self.dinputs.clone() * (1.0 / rows as f32);
                    }
                }
            }
        }
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
