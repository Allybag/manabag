use std::ops::{Add, Mul};

// TODO: We definitely should not be regularly copying matrices
#[derive(Debug, Clone)]
pub struct SimpleMatrix {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
}

impl SimpleMatrix {
    pub fn new<const R: usize, const C: usize>(values: [[f32; C]; R]) -> Self {
        SimpleMatrix {
            rows: R,
            cols: C,
            values: {
                let mut vec = Vec::<f32>::with_capacity(R * C);
                vec.resize(R * C, 0.0);
                for row in 0..R {
                    for col in 0..C {
                        vec[row * C + col] = values[row][col];
                    }
                }
                vec
            }
        }
    }

    pub fn transpose(self) -> SimpleMatrix {
        let mut values = self.values.clone();

        for row in 0..self.rows {
            for col in 0..self.cols {
                let index = self.index(row, col);
                let transposed_index = col * self.rows + row;
                values[transposed_index] = self.values[index];
            }
        }

        SimpleMatrix { rows: self.cols, cols: self.rows, values }
    }

    pub fn transform(&mut self, f: fn(f32) -> f32) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                let index = self.index(row, col);
                self.values[index] = f(self.values[index]);
            }
        }
    }

    pub fn row_transform(&mut self, f: fn(&[f32]) -> Vec<f32>) {
        for row in 0..self.rows {
            let mut index = self.index(row, 0);
            let result = f(&self.values[index..index + self.cols]);
            for val in result.iter() {
                self.values[index] = *val;
                index += 1;
            }
        }
    }

    pub fn print_head(self) {
        let head = &self.values[0..5];
        dbg!(head);
    }

    fn index(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }
 }

impl Add<f32> for SimpleMatrix {
    type Output = SimpleMatrix;

    fn add(self, other: f32) -> Self::Output {
        let mut values = self.values.clone();

        for row in 0..self.rows {
            for col in 0..self.cols {
                let index = self.index(row, col);
                values[index] = self.values[index] + other;
            }
        }

        SimpleMatrix { rows: self.rows, cols: self.cols, values }
    }
}

impl Mul<f32> for SimpleMatrix {
    type Output = SimpleMatrix;

    fn mul(self, other: f32) -> SimpleMatrix {
        let mut values = self.values.clone();

        for col in 0..self.cols {
            for row in 0..self.rows {
                let index = self.index(row, col);
                values[index] = self.values[index] * other;
            }
        }

        SimpleMatrix { rows: self.rows, cols: self.cols, values }
    }
}

// TODO: Add vector to matrix
impl Add<SimpleMatrix> for SimpleMatrix
{
    type Output = SimpleMatrix;

    fn add(self, other: SimpleMatrix) -> SimpleMatrix {
        if self.cols != other.cols || self.rows != other.rows {
            panic!("SimpleMatrix::Add self {0},{1} vs other {2},{3}", self.cols, self.rows, other.cols, other.rows);
        }

        let mut values = self.values.clone();

        for col in 0..self.cols {
            for row in 0..self.rows {
                let index = self.index(row, col);
                values[index] = self.values[index] + other.values[index];
            }
        }

        SimpleMatrix { rows: self.rows, cols: self.cols, values }
    }
}

impl Mul<SimpleMatrix> for SimpleMatrix {
    type Output = SimpleMatrix;

    fn mul(self, other: SimpleMatrix) -> Self::Output {
        if self.cols != other.rows {
            panic!("SimpleMatrix::Mul self {0},{1} vs other {2},{3}", self.cols, self.rows, other.cols, other.rows);
        }

        let mut values = Vec::<f32>::with_capacity(self.rows * other.cols);
        values.resize(self.rows * other.cols, 0.0);

        for row in 0..self.rows {
            for col in 0..other.cols {
                let index = row * other.cols + col;
                for i in 0..self.cols {
                    let result = self.values[self.index(row, i)] * other.values[other.index(i, col)];
                    values[index] = values[index] + result;
                    dbg!("{}, {}", result, values[index]);
                }
            }
        }

        SimpleMatrix { rows: self.rows, cols: other.cols, values }
    }
}

pub trait Feq {
    fn feq(self, other: Self) -> bool;
}

impl Feq for f32 {
    fn feq(self, other: f32) -> bool {
        return (self - other).abs() <= 1e-6;
    }
}

impl PartialEq<SimpleMatrix> for SimpleMatrix {
    fn eq(&self, other: &SimpleMatrix) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false
        }

        for (val, other_val) in self.values.iter().zip(other.values.iter()) {
            if !(*val).feq(*other_val) {
                return false
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_addition() {
        let mat = SimpleMatrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let result = mat + 2.0;

        assert_eq!(result, SimpleMatrix::new([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]));
    }

    #[test]
    fn scalar_multiplication() {
        let mat = SimpleMatrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let result = mat * 2.0;

        assert_eq!(result, SimpleMatrix::new([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]));
    }

    #[test]
    fn matrix_addition() {
        let mat = SimpleMatrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let other = SimpleMatrix::new([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]);
        let result = mat + other;

        assert_eq!(result, SimpleMatrix::new([[2.0, 3.0, 4.0], [6.0, 7.0, 8.0]]));
    }

    #[test]
    fn matrix_multiplication() {
        let mat = SimpleMatrix::new([[1.0, 2.0, 3.0, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]);
        let other = SimpleMatrix::new([
            [0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]).transpose();
        let result = mat * other;

        assert_eq!(result, SimpleMatrix::new([[2.8, -1.79, 1.885], [6.9, -4.81, -0.3], [-0.59, -1.949, -0.474]]));
    }
}
