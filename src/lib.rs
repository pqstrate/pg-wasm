use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use wasm_bindgen::prelude::*;
use winterfell::math::fft;
use winterfell::math::fields::f64::BaseElement;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn log_u32(a: u32);

    #[wasm_bindgen(js_namespace = ["performance"], js_name = now)]
    fn performance_now() -> f64;
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn bench_p3_goldilocks_mul(iterations: u32) {
    let mut a = Goldilocks::from_i64(12345);
    let b = Goldilocks::from_i64(67890);

    let start = performance_now();

    for _ in 0..iterations {
        a = a * b;
        // Prevent optimization by using black_box equivalent
        std::hint::black_box(&a);
    }

    let end = performance_now();

    console_log!(
        "[p3] goldilocks_mul: {} ms for {} iterations",
        end - start,
        iterations
    );
}

#[wasm_bindgen]
pub fn bench_winterfell_goldilocks_mul(iterations: u32) {
    let mut a = BaseElement::new(12345);
    let b = BaseElement::new(67890);

    let start = performance_now();

    for _ in 0..iterations {
        a = a * b;
        // Prevent optimization by using black_box equivalent
        std::hint::black_box(&a);
    }

    let end = performance_now();

    console_log!(
        "[wf] goldilocks_mul: {} ms for {} iterations",
        end - start,
        iterations
    );
}

#[wasm_bindgen]
pub fn bench_p3_fft(log_size: u32) {
    let size = 1 << log_size;
    let dft = Radix2DitParallel::default();

    {
        let values: Vec<Goldilocks> = (0..size).map(|i| Goldilocks::from_i64(i as i64)).collect();

        let matrix = RowMajorMatrix::new(values, 1);
        // console_log!("input: {:?}", matrix);
        let start = performance_now();

        let result = dft.dft_batch(matrix);
        std::hint::black_box(&result);

        let end = performance_now();
        // console_log!("output: {:?}", result);
        console_log!(
            "[p3]  1 poly fft: {} ms for size 2^{} = {}",
            end - start,
            log_size,
            size
        );
    }

    {
        let values: Vec<Goldilocks> = (0..size * 80)
            .map(|i| Goldilocks::from_i64(i as i64))
            .collect();

        let matrix = RowMajorMatrix::new(values, 80);
        // console_log!("input: {:?}", matrix);
        let start = performance_now();

        let result = dft.dft_batch(matrix);
        std::hint::black_box(&result);

        let end = performance_now();
        // console_log!("output: {:?}", result);
        console_log!(
            "[p3] 80 poly fft: {} ms for size 2^{} = {}",
            end - start,
            log_size,
            size
        );
    }
}

#[wasm_bindgen]
pub fn bench_winterfell_fft(log_size: u32) {
    let size = 1 << log_size;
    let mut values: Vec<BaseElement> = (0..size)
        .map(|i| BaseElement::new(i as u64))
        .collect();

    let start = performance_now();
    let twiddle=  fft::get_twiddles(size);
    let end = performance_now();
    console_log!("[wf] fft: {} ms to compute twiddles for size 2^{} = {}", end - start, log_size, size);

    let start = performance_now();

    fft::serial_fft(&mut values, &twiddle);
    std::hint::black_box(&values);

    let end = performance_now();

    console_log!("[wf] fft: {} ms for size 2^{} = {}", end - start, log_size, size);
}
