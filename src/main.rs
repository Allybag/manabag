#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
use crate::simple_matrix::Matrix;
use dense_layer::{categorical_loss, Activation, DenseLayer, ActivationLayer};
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

    // Forward pass
    let inputs = Matrix::new(X);
    let mut first = dense_one.forward(inputs);
    relu.forward(&mut first);
    let mut output = dense_two.forward(first);
    softmax.forward(&mut output);

    let loss = categorical_loss(&output, &Y);

    // Backward pass
    softmax.backward(output, Some(&Y));
    dense_two.backward(softmax.dinputs);
    relu.backward(dense_two.dinputs, None);
    dense_one.backward(relu.dinputs);

    println!("Loss: {loss}");
}
