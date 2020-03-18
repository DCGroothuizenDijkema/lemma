
#![allow(dead_code,unused_imports)]

use std::boxed::Box;

use std::clone::Clone;
use std::default::Default;
use std::marker::Copy;

use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Index;
use std::ops::IndexMut;

trait Scalar:
  // impl Copy means impl Clone
  Copy + Default + AddAssign
{}

impl Scalar for f64 {}
impl Scalar for f32 {}

struct Tensor<T: Scalar, const N: usize>
{
  data: Box<[T]>,
  dim: [usize;N],
}

impl<T,const N: usize> Tensor<T,N>
where T: Scalar
{
  fn new(dim: [usize; N]) -> Tensor<T,N>
  {
    let size: usize=dim.iter().fold(1,|a, &b| a*b);
    let data: Box<[T]>=vec![T::default();size].into_boxed_slice();
    Tensor{data:data,dim:dim}
  }
}

impl<T,const N: usize> Index<[usize;N]> for Tensor<T,N>
where T: Scalar
{
  type Output=T;
  fn index(&self, index: [usize;N]) -> &Self::Output
  {
    let nd: usize=self.dim.len();
    let mut ind: usize=0;
    for itr in 0..nd
    {
      let mut prod: usize=1;
      for jtr in (itr+1)..nd
      {
        prod*=self.dim[jtr];
      }
      ind+=prod*index[itr]
    }
    &self.data[ind]
  }
}

impl<T,const N: usize> IndexMut<[usize;N]> for Tensor<T,N>
where T: Scalar
{
  fn index_mut(&mut self, index: [usize;N]) -> &mut Self::Output
  {
    let nd: usize=self.dim.len();
    let mut ind: usize=0;
    for itr in 0..nd
    {
      let mut prod: usize=1;
      for jtr in (itr+1)..nd
      {
        prod*=self.dim[jtr];
      }
      ind+=prod*index[itr]
    }
    &mut self.data[ind]
  }
}

impl<T,const N: usize> Clone for Tensor<T,N>
where T: Scalar
{
  fn clone(&self) -> Tensor<T,N>
  {
    let mut t: Tensor<T,N>=Tensor::<T,N>::new(self.dim);

    for itr in 0..self.data.len()
    {
      t.data[itr]=self.data[itr];
    }
    t
  }
}

// impl<T> AddAssign for Tensor<T>
// where T: Scalar
// {
//   fn add_assign(&mut self, rhs: Self)
//   {
//     if self.dim.len()!=rhs.dim.len() { panic!("Tensors must be of the same dimension to add them."); }
//     for (dim1,dim2) in self.dim.iter().zip(&rhs.dim)
//     {
//       if dim1!=dim2 { panic!("All dimensions of two tensors must be of the same size to add them.")}
//     }

//     for (this,other) in self.data.iter_mut().zip(rhs.data.iter())
//     {
//       *this+=*other;
//     }
//   }
// }

// impl<T> AddAssign<T> for Tensor<T>
// where T: Scalar
// {
//   fn add_assign(&mut self, rhs: T)
//   {
//     for this in self.data.iter_mut() { *this+=rhs; }
//   }
// }

// impl<T> Add<T> for Tensor<T>
// where T: Scalar
// {
//   type Output=Self;
//   fn add(mut self, rhs: T) -> Self::Output
//   {
//     self+=rhs;
//     return self;
//   }
// }

// impl<T> Add for Tensor<T>
// where T: Scalar
// {
//   type Output=Self;
//   fn add(mut self, rhs: Self) -> Self::Output
//   {
//     self+=rhs;
//     self
//   }
// }

//
// Tests
//

#[cfg(test)]
mod tensor_tests
{
  use super::{Tensor};

  #[test]
  fn tensor_test_new()
  {
    let t: Tensor<f64,1>=Tensor::<f64,1>::new([5]);
    assert!(t.dim==[5]);
    assert!(t.data.len()==5);
    for elem in t.data.iter()
    {
      assert!(elem==&0f64);
    }

    let t: Tensor<f32,2>=Tensor::<f32,2>::new([3,2]);
    assert!(t.dim==[3,2]);
    assert!(t.data.len()==6);
    for elem in t.data.iter()
    {
      assert!(elem==&0f32);
    }
  }

  #[test]
  fn tensor_test_index()
  {
    let t: Tensor<f64,3>=Tensor::<f64,3>::new([2,4,3]);
    for itr in 0..2
    {
      for jtr in 0..4
      {
        for ktr in 0..3
        {
          assert!(t[[itr,jtr,ktr]]==0f64);
        }
      }
    }
  }

  #[test]
  fn tensor_test_index_mut()
  {
    let mut t: Tensor<f64,1>=Tensor::<f64,1>::new([5]);
    t[[1]]=3.14;
    assert!(t[[1]]==3.14);
    t[[4]]=1.618;
    assert!(t[[4]]==1.618);
    t[[0]]=2.718;
    assert!(t[[0]]==2.718);

    let mut t: Tensor<f64,2>=Tensor::<f64,2>::new([2,4]);
    t[[1,3]]=3.14;
    assert!(t[[1,3]]==3.14);
    t[[0,0]]=1.618;
    assert!(t[[0,0]]==1.618);
    t[[0,2]]=2.718;
    assert!(t[[0,2]]==2.718);
  }

  // // #[test]
  // // #[should_panic(expected="Tensors must be of the same dimension to add them.")]
  // // fn tensor_test_add_assign_tensor_1()
  // // {

  // // }

  // #[test]
  // #[should_panic(expected="All dimensions of two tensors must be of the same size to add them.")]
  // fn tensor_test_add_assign_tensor_2()
  // {
  //   let mut t1: Tensor<f64>=Tensor::<f64>::new(5);
  //   let t2: Tensor<f64>=Tensor::<f64>::new(4);

  //   t1+=t2;
  // }

  // #[test]
  // fn tensor_test_add_assign_tensor_3()
  // {
  //   let mut t1: Tensor<f64>=Tensor::<f64>::new(3);
  //   let mut t2: Tensor<f64>=Tensor::<f64>::new(3);

  //   t1[0]=1.3;
  //   t1[1]=2.2;
  //   t1[2]=3.1;

  //   t2[0]=7.9;
  //   t2[1]=8.8;
  //   t2[2]=9.7;

  //   t1+=t2;

  //   assert!(t1[0]==1.3+7.9);
  //   assert!(t1[1]==2.2+8.8);
  //   assert!(t1[2]==3.1+9.7);
  // }

  // #[test]
  // fn tensor_test_add_assign_scalar()
  // {
  //   let mut t: Tensor<f64>=Tensor::<f64>::new(4);
  //   t[0]=3.14;
  //   t[1]=1.618;
  //   t[2]=2.71;
  //   t[3]=1.414;

  //   let s: f64=1.202;

  //   t+=s;
  //   assert!(t[0]==3.14+s);
  //   assert!(t[1]==1.618+s);
  //   assert!(t[2]==2.71+s);
  //   assert!(t[3]==1.414+s);
  // }

  // #[test]
  // fn tensor_test_add_tensor()
  // {
  //   let mut t1: Tensor<f64>=Tensor::<f64>::new(3);
  //   let mut t2: Tensor<f64>=Tensor::<f64>::new(3);

  //   t1[0]=1.3;
  //   t1[1]=2.2;
  //   t1[2]=3.1;

  //   t2[0]=7.9;
  //   t2[1]=8.8;
  //   t2[2]=9.7;

  //   let t3: Tensor<f64>=t1.clone()+t2;

  //   assert!(t3[0]==1.3+7.9);
  //   assert!(t3[1]==2.2+8.8);
  //   assert!(t3[2]==3.1+9.7);
  // }

  // #[test]
  // fn tensor_test_add_scalar()
  // {
  //   let mut t1: Tensor<f64>=Tensor::<f64>::new(3);
  //   t1[0]=1.3;
  //   t1[1]=2.2;
  //   t1[2]=3.1;

  //   let t2: Tensor<f64>=t1+3.14;

  //   assert!(t2[0]==1.3+3.14);
  //   assert!(t2[1]==2.2+3.14);
  //   assert!(t2[2]==3.1+3.14);
  // }

  #[test]
  fn tensor_test_clone()
  {
    let mut t1: Tensor<f64,1>=Tensor::<f64,1>::new([3]);
    t1[[0]]=1.3;
    t1[[1]]=2.2;
    t1[[2]]=3.1;

    let t2: Tensor<f64,1>=t1.clone();
    assert!(t2[[0]]==1.3);
    assert!(t2[[1]]==2.2);
    assert!(t2[[2]]==3.1);
  }
}
