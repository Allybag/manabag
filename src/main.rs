#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
use crate::matrix::Matrix;
use dense_layer::{categorical_loss, Activation, DenseLayer};
use spiral_data::{*};

pub mod matrix;
pub mod dense_layer;
pub mod spiral_data;

fn main() {
    let mut dense_one = DenseLayer::<2, 3>::new();
    dense_one.set_weights(LAYER_ONE_WEIGHTS);

    let mut dense_two = DenseLayer::<3, 3>::new();
    dense_two.set_weights(LAYER_TWO_WEIGHTS);
    dense_two.set_activation(Activation::Softmax);

    let inputs = Matrix::new(X);
    let output = dense_two.output(dense_one.output(inputs));

    // output.print_head();
    let loss = categorical_loss(&output, Y);
    println!("Loss: {loss}");
}
