use blake3::Hasher;
use p3_blake3::Blake3;
use p3_commit::Mmcs;
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTree;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use std::mem::transmute;
use wasm_bindgen::prelude::*;
use winterfell::crypto::Hasher as WfHasher;
use winterfell::crypto::{hashers::Blake3_256, MerkleTree as WfMerkleTree};
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
    let mut values: Vec<BaseElement> = (0..size).map(|i| BaseElement::new(i as u64)).collect();

    let start = performance_now();
    let twiddle = fft::get_twiddles(size);
    let end = performance_now();
    console_log!(
        "[wf] fft: {} ms to compute twiddles for size 2^{} = {}",
        end - start,
        log_size,
        size
    );

    let start = performance_now();

    fft::serial_fft(&mut values, &twiddle);
    std::hint::black_box(&values);

    let end = performance_now();

    console_log!(
        "[wf] fft: {} ms for size 2^{} = {}",
        end - start,
        log_size,
        size
    );
}

#[wasm_bindgen]
pub fn bench_p3_merkle_tree(log_leaf_count: u32) {
    let leaf_count = 1 << log_leaf_count;

    // Benchmark Blake3 Merkle tree

    type Blake3FieldHash = SerializingHasher<Blake3>;
    type Blake3Compress = CompressionFunctionFromHasher<Blake3, 2, 32>;
    type Blake3ValMmcs = MerkleTreeMmcs<Goldilocks, u8, Blake3FieldHash, Blake3Compress, 32>;

    let blake3_hash = Blake3 {};
    let compress = Blake3Compress::new(blake3_hash);

    let field_hash = Blake3FieldHash::new(blake3_hash);
    let val_mmcs = Blake3ValMmcs::new(field_hash, compress);

    {
        let leaves: Vec<Goldilocks> = (0..leaf_count)
            .map(|i| Goldilocks::from_i64(i as i64))
            .collect();

        let leave_matrix = RowMajorMatrix::new(leaves, 1);

        let start = performance_now();
        let (_commitment, _prover_data) = val_mmcs.commit(vec![leave_matrix]);

        let end = performance_now();
        let elapsed = end - start;

        console_log!(
            "[p3] merkle tree  1 col: {} ms for {} leaves",
            elapsed,
            leaf_count
        );
    }
    {
        let leaves: Vec<Goldilocks> = (0..leaf_count * 80)
            .map(|i| Goldilocks::from_i64(i as i64))
            .collect();

        let leave_matrix = RowMajorMatrix::new(leaves, 80);

        let start = performance_now();
        let (_commitment, _prover_data) = val_mmcs.commit(vec![leave_matrix]);

        let end = performance_now();
        let elapsed = end - start;

        console_log!(
            "[p3] merkle tree 80 col: {} ms for {} leaves",
            elapsed,
            leaf_count
        );
    }
}

#[wasm_bindgen]
pub fn bench_winterfell_merkle_tree(log_leaf_count: u32) {
    let leaf_count = 1 << log_leaf_count;

    {
        let leaves: Vec<_> = (0..leaf_count)
            .map(|i| {
                let val = (1u64 << 55) + (i as u64);
                let bytes = val.to_le_bytes();
                let mut full_bytes = [0u8; 32];
                full_bytes[..8].copy_from_slice(&bytes);
                Blake3_256::<BaseElement>::hash(&full_bytes)
            })
            .collect();
        let start = performance_now();
        let _tree = WfMerkleTree::<Blake3_256<BaseElement>>::new(leaves).unwrap();
        let end = performance_now();
        let elapsed = end - start;
        console_log!(
            "[wf] merkle tree 1 col: {} ms for {} leaves",
            elapsed,
            leaf_count
        );
    }
    {
        let leaves_bases: Vec<_> = (0..leaf_count * 80)
            .map(|i| BaseElement::new((1u64 << 55) + (i as u64)))
            .collect();

        let start = performance_now();
        let leaves = leaves_bases
            .chunks(80)
            .map(|chunk| {
                // let hash_input = chunk.iter().flat_map(|el| el.to_bytes()).collect::<Vec<_>>();
                let hash_input = unsafe {
                    transmute::<[BaseElement; 80], [u8; 80 * 8]>(chunk.to_vec().try_into().unwrap())
                };
                Blake3_256::<BaseElement>::hash(&hash_input)
            })
            .collect::<Vec<_>>();
        let end = performance_now();
        let elapsed = end - start;
        console_log!(
            "[wf] merkle tree 80 col leave hash: {} ms for {} leaves",
            elapsed,
            leaf_count
        );

        let start = performance_now();
        let _tree = WfMerkleTree::<Blake3_256<BaseElement>>::new(leaves).unwrap();
        let end = performance_now();
        let elapsed = end - start;
        console_log!(
            "[wf] merkle tree 80 col: {} ms for {} leaves",
            elapsed,
            leaf_count
        );
    }
}