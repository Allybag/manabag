#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
use crate::simple_matrix::Matrix;
use dense_layer::{categorical_loss, get_predictions, calc_accuracy, Activation, DenseLayer, ActivationLayer};
use spiral_data::{*};

pub mod simple_matrix;
pub mod dense_layer;
pub mod spiral_data;

fn main() {
    let mut dense_one = DenseLayer::new();
    let mut relu = ActivationLayer::new(Activation::Relu);
    dense_one.set_weights(LAYER_ONE_WEIGHTS);

    let mut dense_two = DenseLayer::new();
    let mut softmax = ActivationLayer::new(Activation::Softmax);
    dense_two.set_weights(LAYER_TWO_WEIGHTS);

    for epoch in 0..10001 {
        // Forward pass
        let inputs = Matrix::new(X);
        let mut first = dense_one.forward(inputs);
        relu.forward(&mut first);
        let mut output = dense_two.forward(first);
        softmax.forward(&mut output);

        let loss = categorical_loss(&output, &Y);
        let predictions = get_predictions(&output);
        let accuracy = calc_accuracy(&predictions, &Y);

        // Backward pass
        softmax.backward(output, Some(&Y));
        dense_two.backward(softmax.dinputs.clone());
        relu.backward(dense_two.dinputs.clone(), None);
        dense_one.backward(relu.dinputs.clone());

        if epoch == 1 || epoch % 1000 == 0 {
            println!("Epoch {epoch}: Accuracy: {accuracy}, Loss: {loss}");
        }

        let learning_rate = -1.0;
        dense_two.weights = dense_two.weights + dense_two.dweights.clone() * learning_rate;
        dense_two.biases = dense_two.biases + dense_two.dbiases.clone() * learning_rate;
        dense_one.weights = dense_one.weights + dense_one.dweights.clone() * learning_rate;
        dense_one.biases = dense_one.biases + dense_one.dbiases.clone() * learning_rate;

    }
}
