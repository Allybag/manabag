#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
use dense_layer::{relu_activate, DenseLayer};
use spiral_data::{weights, X};

pub mod matrix;
pub mod dense_layer;
pub mod spiral_data;

fn main() {
    let mut dense = DenseLayer::<2, 3>::new();
    dense.set_weights(weights);
    let mut output = dense.forward(X);
    relu_activate(&mut output);

    println!("Output: {output:?}")
}
