use std::ops::{Add, Mul};

pub trait Value: Add<Output = Self> + Mul<Output = Self> + Copy {
}

impl <T: Add<Output = T> + Mul<Output = T> + Copy> Value for T {
}

#[derive(Debug)]
pub struct Matrix<T: Value, const R: usize, const C: usize> {
    data: [[T; R]; C],
}

impl<T: Value, const R: usize, const C: usize> Matrix<T, R, C> {
    pub fn new(data: [[T; R]; C]) -> Self {
        Matrix {
            data
        }
    }
}
