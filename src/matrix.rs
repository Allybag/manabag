use std::ops::{Add, Mul};

pub trait Value: Add<Output = Self> + Mul<Output = Self> + Copy + Default {
}

impl <T: Add<Output = T> + Mul<Output = T> + Copy + Default> Value for T {
}

#[derive(Debug)]
pub struct Matrix<T: Value, const R: usize, const C: usize> {
    values: [[T; C]; R],
}

impl<T: Value, const R: usize, const C: usize> Matrix<T, R, C> {
    pub fn new(values: [[T; C]; R]) -> Self {
        Matrix {
            values
        }
    }

    pub fn transpose(&self) -> Matrix<T, C, R> {
        let mut values = [[T::default(); R]; C];

        for col in 0..C {
            for row in 0..R {
                values[col][row] = self.values[row][col];
            }
        }

        Matrix { values }
    }
 }

impl <T: Value, const R: usize, const C: usize> Add<T> for Matrix<T, R, C> {
    type Output = Matrix<T, R, C>;

    fn add(self, other: T) -> Self::Output {
        let mut values = [[T::default(); C]; R];

        for col in 0..C {
            for row in 0..R {
                values[row][col] = self.values[row][col] + other;
            }
        }

        Matrix { values }
    }
}

impl <T: Value, const R: usize, const C: usize> Mul<T> for Matrix<T, R, C> {
    type Output = Matrix<T, R, C>;

    fn mul(self, other: T) -> Self::Output {
        let mut values = [[T::default(); C]; R];

        for col in 0..C {
            for row in 0..R {
                values[row][col] = self.values[row][col] * other;            }
        }

        Matrix { values }
    }
}

impl <T: Value, const R: usize, const C: usize> Add<Matrix<T, R, C>> for Matrix<T, R, C> {
    type Output = Matrix<T, R, C>;

    fn add(self, other: Matrix<T, R, C>) -> Self::Output {
        let mut values = [[T::default(); C]; R];

        for col in 0..C {
            for row in 0..R {
                values[row][col] = self.values[row][col] + other.values[row][col];            }
        }

        Matrix { values }
    }
}

impl <T: Value, const R: usize, const C: usize, const K: usize> Mul<Matrix<T, R, K>> for Matrix<T, K, C> {
    type Output = Matrix<T, R, C>;

    fn mul(self, other: Matrix<T, R, K>) -> Self::Output {
        let mut values = [[T::default(); C]; R];

        for col in 0..C {
            for row in 0..R {
                for i in 0..K {
                    values[col][row] = values[col][row] + self.values[row][i] + other.values[i][col];
                }
            }
        }

        Matrix { values }
    }
}
