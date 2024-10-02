#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
use dense_layer::DenseLayer;
use spiral_data::{weights, X};

pub mod matrix;
pub mod dense_layer;
pub mod spiral_data;

fn main() {
    let mut dense = DenseLayer::<2, 3>::new();
    dense.set_weights(weights);
    let output = dense.forward(X);

    println!("Output: {output:?}")
}
