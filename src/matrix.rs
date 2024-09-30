use std::ops::{Add, Mul};

pub trait Value: Add<Output = Self> + Mul<Output = Self> + Copy + Default {
}

impl <T: Add<Output = T> + Mul<Output = T> + Copy + Default> Value for T {
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

impl <T: Value, const R: usize, const C: usize> Add<T> for Matrix<T, R, C> {
    type Output = Matrix<T, R, C>;

    fn add(self, other: T) -> Matrix<T, R, C> {
        let mut data = [[T::default(); R]; C];

        for col in 0..C {
            for row in 0..R {
                data[col][row] = self.data[col][row] + other;            }
        }

        Matrix { data }
    }
}
