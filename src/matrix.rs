use std::ops::{Add, Mul};

pub enum Assert<const CHECK: bool> {}
pub trait IsTrue {}
impl IsTrue for Assert<true> {}

pub trait Numeric: Add<Output = Self> + Mul<Output = Self> + Sized + std::fmt::Debug {
}

impl <T> Numeric for T where T: Add<Output = T> + Mul<Output = T> + Sized + std::fmt::Debug {
}

pub trait Value: Numeric + Copy + Default + std::cmp::PartialOrd {
}

impl <T: Numeric + Copy + Default + std::cmp::PartialOrd> Value for T {
}

// TODO: We definitely should not be regularly copying matrices
#[derive(Debug, Clone, Copy)]
pub struct Matrix<T: Value, const R: usize, const C: usize> {
    values: [[T; C]; R],
}

pub trait Feq {
    fn feq(self, other: Self) -> bool;
}

impl Feq for u32 {
    fn feq(self, other: u32) -> bool {
        return self == other;
    }
}

impl Feq for f32 {
    fn feq(self, other: f32) -> bool {
        return (self - other).abs() <= 1e-6;
    }
}

impl<T: Value + Feq, const R: usize, const C: usize> PartialEq<Matrix<T, R, C>> for Matrix<T, R, C> {
    fn eq(&self, other: &Matrix<T, R, C>) -> bool {
        for (rows, other_rows) in self.values.iter().zip(other.values.iter()) {
            for (val, other) in rows.iter().zip(other_rows.iter()) {
                if !(*val).feq(*other) {
                    return false
                }
            }
        }

        true
    }
}

impl<T: Value, const R: usize, const C: usize> Matrix<T, R, C> {
    pub fn new(values: [[T; C]; R]) -> Self {
        Matrix {
            values
        }
    }

    pub fn transpose(self) -> Matrix<T, C, R> {
        let mut values = [[T::default(); R]; C];

        for col in 0..C {
            for row in 0..R {
                values[col][row] = self.values[row][col];
            }
        }

        Matrix { values }
    }

    pub fn transform(&mut self, f: fn(T) -> T) {
        for row in 0..R {
            for col in 0..C {
                self.values[row][col] = f(self.values[row][col]);
            }
        }
    }

    pub fn row_transform(&mut self, f: fn([T; C]) -> [T; C]) {
        for row in 0..R {
            self.values[row] = f(self.values[row]);
        }
    }

    pub fn print_head(self) {
        let head = &self.values[0..5];
        dbg!(head);
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

impl <T: Value, const R: usize, const C: usize> Add<Matrix<T, 1, C>> for Matrix<T, R, C> {
    type Output = Matrix<T, R, C>;

    fn add(self, other: Matrix<T, 1, C>) -> Self::Output {
        let mut values = [[T::default(); C]; R];

        for col in 0..C {
            for row in 0..R {
                values[row][col] = self.values[row][col] + other.values[0][col];            }
        }

        Matrix { values }
    }
}

impl <T: Value, const R: usize, const C: usize> Add<Matrix<T, R, C>> for Matrix<T, R, C> where
    Assert<{R != 1}> : IsTrue,
{
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
                    values[row][col] = values[row][col] + other.values[row][i] * self.values[i][col];
                }
            }
        }

        Matrix { values }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_addition() {
        let mat = Matrix::new([[1, 2, 3], [4, 5, 6]]);
        let result = mat + 2;

        assert_eq!(result, Matrix::new([[3, 4, 5], [6, 7, 8]]));
    }

    #[test]
    fn scalar_multiplication() {
        let mat = Matrix::new([[1, 2, 3], [4, 5, 6]]);
        let result = mat * 2;

        assert_eq!(result, Matrix::new([[2, 4, 6], [8, 10, 12]]));
    }

    #[test]
    fn matrix_addition() {
        let mat = Matrix::new([[1, 2, 3], [4, 5, 6]]);
        let other = Matrix::new([[1, 1, 1], [2, 2, 2]]);
        let result = mat + other;

        assert_eq!(result, Matrix::new([[2, 3, 4], [6, 7, 8]]));
    }

    #[test]
    fn matrix_multiplication() {
        let mat = Matrix::new([[1.0, 2.0, 3.0, 2.5]]).transpose();
        let other = Matrix::new([
            [0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]);
        let result = mat * other;

        assert_eq!(result, Matrix::new([[2.8, -1.79, 1.885]]).transpose());
    }
}
