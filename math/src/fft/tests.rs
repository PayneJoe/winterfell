// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use alloc::vec::Vec;
use libc_print::std_name::{dbg, eprintln, println};

use rand_utils::rand_vector;

use crate::{
    fft::fft_inputs::FftInputs,
    field::{f128::BaseElement, StarkField},
    polynom,
    utils::get_power_series,
    FieldElement,
};

// CORE ALGORITHMS
// ================================================================================================

#[test]
fn fft_in_place() {
    // degree 3
    let n = 4;
    let mut p = rand_vector(n);
    let domain = build_domain(n);
    let expected = polynom::eval_many(&p, &domain);
    let twiddles = super::get_twiddles::<BaseElement>(n);
    p.fft_in_place(&twiddles);
    p.permute();
    assert_eq!(expected, p);

    // degree 7
    let n = 8;
    let mut p = rand_vector(n);
    let domain = build_domain(n);
    let twiddles = super::get_twiddles::<BaseElement>(n);
    let expected = polynom::eval_many(&p, &domain);
    p.fft_in_place(&twiddles);
    p.permute();
    assert_eq!(expected, p);

    // degree 15
    let n = 16;
    let mut p = rand_vector(n);
    let domain = build_domain(n);
    let twiddles = super::get_twiddles::<BaseElement>(16);
    let expected = polynom::eval_many(&p, &domain);
    p.fft_in_place(&twiddles);
    p.permute();
    assert_eq!(expected, p);

    // degree 1023
    let n = 1024;
    let mut p = rand_vector(n);
    let domain = build_domain(n);
    let expected = polynom::eval_many(&p, &domain);
    let twiddles = super::get_twiddles::<BaseElement>(n);
    p.fft_in_place(&twiddles);
    p.permute();
    assert_eq!(expected, p);
}

#[test]
fn permutate() {
    let s = 16 as usize;
    let mut data = (0..s).map(|i| BaseElement::new(i as u128)).collect::<Vec<_>>();
    data.permute();
    for i in (0..s).step_by(2) {
        println!("{}: {}-{}", i, data[i], data[i + 1]);
        assert_eq!(
            data[i].as_int().pow(2).checked_rem(s as u128).unwrap(),
            data[i + 1].as_int().pow(2).checked_rem(s as u128).unwrap()
        );
    }
}

#[test]
fn fft_get_twiddles() {
    let n = super::MIN_CONCURRENT_SIZE * 2;
    let g = BaseElement::get_root_of_unity(n.ilog2());
    println!("g^{} = {}", n, g.exp(n as u128));
    assert_eq!(g.exp(n as u128), BaseElement::ONE);

    let mut expected = get_power_series(g, n / 2);
    assert_eq!(expected[3], g.exp(3 as u128));
    assert_eq!(*expected.last().unwrap(), g.exp(((n / 2) - 1) as u128));
    let old = expected.clone();
    expected.permute();
    let new = expected.clone();
    for i in 0..10 {
        println!("{} - {}", old[i], new[(n / 2) - 1 - i]);
    }
    // assert_eq!(expected[0], g.exp(((n / 2) - 1) as u128));
    // assert_eq!(expected[(n / 4) - 1], g.exp((n / 4) as u128));

    let twiddles = super::get_twiddles::<BaseElement>(n);
    assert_eq!(expected, twiddles);
}

// HELPER FUNCTIONS
// ================================================================================================

fn build_domain(size: usize) -> Vec<BaseElement> {
    let g = BaseElement::get_root_of_unity(size.ilog2());
    get_power_series(g, size)
}
