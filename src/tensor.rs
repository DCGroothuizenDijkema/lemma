
#![allow(dead_code,unused_imports)]

use std::boxed::Box;

use std::clone::Clone;
use std::default::Default;
use std::marker::Copy;

use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Index;
use std::ops::IndexMut;

type Idx=usize;
type Dim<const N: Idx>=[Idx;N];

trait Operand: Clone {}
trait Scalar: Operand + Default + AddAssign {}

trait Dimension<D>
{
  fn index(self, ind: D) -> Idx;
  fn size(self) -> Idx;
}

impl<const N: Idx> Dimension<Dim<N>> for Dim<N>
{
  fn index(self, ind: Dim<N>) -> Idx
  {
    ind.iter()
      .enumerate()
      .fold(0,|sum,d| {
        let itr: usize=d.0+1;
        let prod: usize=self[itr..].iter()
          .fold(1,|prod,d| prod*d);
        sum+prod*d.1
      })
  }

  fn size(self) -> Idx
  {
    self.iter()
      .fold(1,|prod,d| prod*d)
  }
}

impl Operand for f32 {}
impl Operand for f64 {}
impl Operand for &f32 {}
impl Operand for &f64 {}

impl Scalar for f32 {}
impl Scalar for f64 {}

struct Tensor<T: Scalar, const N: Idx>
{
  data: Box<[T]>,
  dim: Dim<N>,
}

impl<T,const N: Idx> Tensor<T,N>
where T: Scalar
{
  fn new(dim: Dim<N>) -> Tensor<T,N>
  {
    let size: usize=dim.size();
    let data: Box<[T]>=vec![T::default();size].into_boxed_slice();
    Tensor{data:data,dim:dim}
  }
}

impl<T,const N: Idx> Index<Dim<N>> for Tensor<T,N>
where T: Scalar
{
  type Output=T;
  fn index(&self, ind: Dim<N>) -> &Self::Output
  {
    &self.data[self.dim.index(ind)]
  }
}

impl<T> Index<Idx> for Tensor<T,1>
where T: Scalar
{
  type Output=T;
  fn index(&self, ind: Idx) -> &Self::Output
  {
    &self.data[ind]
  }
}

impl<T,const N: Idx> IndexMut<Dim<N>> for Tensor<T,N>
where T: Scalar
{
  fn index_mut(&mut self, ind: Dim<N>) -> &mut Self::Output
  {
    &mut self.data[self.dim.index(ind)]
  }
}

impl<T> IndexMut<Idx> for Tensor<T,1>
where T: Scalar
{
  fn index_mut(&mut self, ind: Idx) -> &mut Self::Output
  {
    &mut self.data[ind]
  }
}

impl<T,const N: Idx> Clone for Tensor<T,N>
where T: Scalar
{
  fn clone(&self) -> Tensor<T,N>
  {
    let mut t: Tensor<T,N>=Tensor::<T,N>::new(self.dim);
    t.data=self.data.clone();
    t
  }
}

impl<T,const N: Idx> AddAssign for Tensor<T,N>
where T: Scalar
{
  fn add_assign(&mut self, rhs: Self)
  {
    for (dim1,dim2) in self.dim.iter().zip(rhs.dim.iter())
    {
      if dim1!=dim2 { panic!("All dimensions of two tensors must be of the same size to add them.")}
    }

    for (this,other) in self.data.iter_mut().zip(rhs.data.iter())
    {
      *this+=other.clone();
    }
  }
}

impl<T,const N: Idx> AddAssign<&Tensor<T,N>> for Tensor<T,N>
where T: Scalar
{
  fn add_assign(&mut self, rhs: &Self)
  {
    for (dim1,dim2) in self.dim.iter().zip(rhs.dim.iter() )
    {
      if dim1!=dim2 { panic!("All dimensions of two tensors must be of the same size to add them.")}
    }

    for (this,other) in self.data.iter_mut().zip(rhs.data.iter())
    {
      *this+=other.clone();
    }
  }
}

impl<T,U,const N: Idx> AddAssign<U> for Tensor<T,N>
where T: Scalar + AddAssign<U>, U: Operand
{
  fn add_assign(&mut self, rhs: U)
  {
    self.data.iter_mut().for_each(|this| *this+=rhs.clone());
  }
}

impl<T,const N: Idx> Add<T> for Tensor<T,N>
where T: Scalar
{
  type Output=Self;
  fn add(mut self, rhs: T) -> Self::Output
  {
    self+=rhs;
    self
  }
}

impl<T,const N: Idx> Add for Tensor<T,N>
where T: Scalar
{
  type Output=Self;
  fn add(mut self, rhs: Self) -> Self::Output
  {
    self+=rhs;
    self
  }
}

impl<T,const N: Idx> Add for &Tensor<T,N>
where T: Scalar
{
  type Output=Tensor<T,N>;
  fn add(self, rhs: Self) -> Self::Output
  {
    let mut t: Tensor<T,N>=self.clone();
    t+=rhs;
    t
  }
}

impl<T,const N: Idx> Add<Tensor<T,N>> for &Tensor<T,N>
where T: Scalar
{
  type Output=Tensor<T,N>;
  fn add(self, rhs: Tensor<T,N>) -> Self::Output
  {
    let mut t: Tensor<T,N>=self.clone();
    t+=rhs;
    t
  }
}

impl<T,const N: Idx> Add<&Tensor<T,N>> for Tensor<T,N>
where T: Scalar
{
  type Output=Tensor<T,N>;
  fn add(self, rhs: &Self) -> Self::Output
  {
    let mut t: Tensor<T,N>=self.clone();
    t+=rhs;
    t
  }
}


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

  #[test]
  #[should_panic(expected="All dimensions of two tensors must be of the same size to add them.")]
  fn tensor_test_add_assign_tensor_1()
  {
    let mut t1: Tensor<f64,1>=Tensor::<f64,1>::new([5]);
    let t2: Tensor<f64,1>=Tensor::<f64,1>::new([4]);

    t1+=t2;
  }

  #[test]
  fn tensor_test_add_assign_tensor_2()
  {
    let mut t1: Tensor<f64,2>=Tensor::<f64,2>::new([2,3]);
    let mut t2: Tensor<f64,2>=Tensor::<f64,2>::new([2,3]);

    t1[[0,0]]=1.3;
    t1[[0,2]]=2.2;
    t1[[1,1]]=3.1;

    t2[[0,1]]=7.9;
    t2[[1,0]]=8.8;
    t2[[1,2]]=9.7;

    t1+=t2.clone();

    assert!(t1[[0,0]]==1.3);
    assert!(t1[[0,1]]==7.9);
    assert!(t1[[0,2]]==2.2);
    assert!(t1[[1,0]]==8.8);
    assert!(t1[[1,1]]==3.1);
    assert!(t1[[1,2]]==9.7);

    t1[[0,1]]=1.1;
    t1[[1,0]]=1.1;
    t1[[1,2]]=1.1;

    t1+=&t2;

    assert!(t1[[0,0]]==1.3);
    assert!(t1[[0,1]]==7.9+1.1);
    assert!(t1[[0,2]]==2.2);
    assert!(t1[[1,0]]==8.8+1.1);
    assert!(t1[[1,1]]==3.1);
    assert!(t1[[1,2]]==9.7+1.1);

    t1+=&t2;

    assert!(t1[[0,0]]==1.3);
    assert!(t1[[0,1]]==1.1+7.9+7.9);
    assert!(t1[[0,2]]==2.2);
    assert!(t1[[1,0]]==1.1+8.8+8.8);
    assert!(t1[[1,1]]==3.1);
    assert!(t1[[1,2]]==1.1+9.7+9.7);

    t1+=t2;
  }

  #[test]
  fn tensor_test_add_assign_scalar()
  {
    let mut t: Tensor<f64,1>=Tensor::<f64,1>::new([4]);
    t[0]=3.14;
    t[1]=1.618;
    t[2]=2.71;
    t[3]=1.414;

    let s: f64=1.202;

    t+=s;
    assert!(t[0]==3.14+s);
    assert!(t[1]==1.618+s);
    assert!(t[2]==2.71+s);
    assert!(t[3]==1.414+s);
    t+=&s;
    assert!(t[0]==3.14+s+s);
    assert!(t[1]==1.618+s+s);
    assert!(t[2]==2.71+s+s);
    assert!(t[3]==1.414+s+s);
  }

  #[test]
  fn tensor_test_add_tensor()
  {
    let mut t1: Tensor<f64,1>=Tensor::<f64,1>::new([3]);
    let mut t2: Tensor<f64,1>=Tensor::<f64,1>::new([3]);

    t1[0]=1.3;
    t1[1]=2.2;
    t1[2]=3.1;

    t2[0]=7.9;
    t2[1]=8.8;
    t2[2]=9.7;

    let t3: Tensor<f64,1>=t1+t2;

    assert!(t3[0]==1.3+7.9);
    assert!(t3[1]==2.2+8.8);
    assert!(t3[2]==3.1+9.7);
  }

  #[test]
  fn tensor_test_add_scalar()
  {
    let mut t1: Tensor<f64,1>=Tensor::<f64,1>::new([3]);
    t1[0]=1.3;
    t1[1]=2.2;
    t1[2]=3.1;

    let t2: Tensor<f64,1>=t1+3.14;

    assert!(t2[0]==1.3+3.14);
    assert!(t2[1]==2.2+3.14);
    assert!(t2[2]==3.1+3.14);
  }

  #[test]
  fn tensor_test_clone()
  {
    let mut t1: Tensor<f64,1>=Tensor::<f64,1>::new([3]);
    t1[0]=1.3;
    t1[1]=2.2;
    t1[2]=3.1;

    let t2: Tensor<f64,1>=t1.clone();
    assert!(t2[0]==1.3);
    assert!(t2[1]==2.2);
    assert!(t2[2]==3.1);
  }
}
