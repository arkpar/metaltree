use std::{io::{Read}};

use rand::RngCore;

const SHA256_DIGEST_SIZE: usize = 32;

const BENCHMARK_PERFECT_TREES: &[usize] = &[
        256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
        65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 33554432 * 32,
];
const BENCHMARK_IMPERFECT_TREES: &[usize] = &[
        250, 500, 1000, 2000, 4000, 8000, 16000, 32000,
        64000, 128000, 256000, 512000, 1024000, 2048000, 4096000, 8192000, 16384000, 32768000, 32768000 * 32,
];


static SORT_SRC: &str = include_str!("sort.metal");
//static SHA256_SRC: &str = include_str!("merkle.metal");
//static BLAKE2B_SRC: &str = include_str!("blake2b.metal");
static BLAKE2B_SRC: &str = include_str!("blake2b_opt.metal");

fn main() {
	let mut args = std::env::args();
    let _ = args.next();
	let command = match args.next() {
		Some(c) => c,
		None => {
			println!("Please specify a command!\n\n");
			print_help();
			return;
		}
    };

    if command == "info" {
        print_info();
        return;
    }

    let mut device = metal::Device::system_default().unwrap();

    println!("Compiling kernels");
    let library = device
        .new_library_with_source(BLAKE2B_SRC, &metal::CompileOptions::new())
        .unwrap();
    let mut kernel = library.get_function("merkle", None).unwrap();
    let library = device
        .new_library_with_source(SORT_SRC, &metal::CompileOptions::new())
        .unwrap();
    let mut sort_kernel = library.get_function("sort", None).unwrap();

    if command == "benchmark" {
        let iterations = args.next().map_or(10, |a| a.parse().unwrap());
        run_benchmark(&mut device, &mut kernel, &mut sort_kernel, iterations).unwrap();
        return;
    }
    if command == "benchmark-tree" {
        let iterations = args.next().map_or(10, |a| a.parse().unwrap());
        run_tree_benchmark(&mut device, &mut kernel, iterations).unwrap();
        return;
    }

    if command == "file" {
        let file = match args.next() {
            Some(f) => f,
            None => {
                println!("File command requires argument 'path'");
                return;
            }
        };

        run_file(&mut device, &mut kernel, &std::path::PathBuf::from(file)).unwrap();
        return;
    }

    print_help();
    return;
}

fn print_help(){
    println!("metaltree [command]");
    println!("");
    println!("Commands:");
    println!("  info              Prints OpenCL information");
    println!("  benchmark         Run benchmark");
    println!("  tree-benchmark    Run tree benchmark");
    println!("  file [path]       Calculate merkle tree from given file");
}

fn print_info() {
    let devices = metal::Device::all();
    for (j, device) in devices.into_iter().enumerate() {
        println!("Device #{},  Name: {}", j, device.name());
        println!("   TR={}, Max threads={:?}, Max buf={}", device.max_transfer_rate(), device.max_threads_per_threadgroup(), device.max_buffer_length());
    }
}

fn run_benchmark(device: &mut metal::Device, kernel: &mut metal::Function, sort_kernel: &mut metal::Function, iterations: usize) -> Result<(), String> {
    assert!(iterations > 0);

    println!("Running on device: {}", device.name());

    for i in 0 .. BENCHMARK_IMPERFECT_TREES.len() * 2 {
        let mut average_time = 0f64;
        let mut average_sort_time = 0f64;
        let leaves = if i % 2 == 0 {
            BENCHMARK_IMPERFECT_TREES[i / 2]
        } else {
            BENCHMARK_PERFECT_TREES[(i - 1) / 2]
        };

        let len = leaves * SHA256_DIGEST_SIZE;

        let mut buffer = device.new_buffer(
            len as u64,
            metal::MTLResourceOptions::CPUCacheModeDefaultCache | metal::MTLResourceOptions::StorageModeShared
        );

        for _j in 0 .. iterations {

            let sort_start = std::time::Instant::now();
            sort_tree(device, sort_kernel, leaves / 2, &mut buffer)?;
            let sort_end = std::time::Instant::now();


            let start = std::time::Instant::now();
            compute_tree(device, kernel, leaves, &mut buffer)?;
            let end = std::time::Instant::now();

            average_time += (end - start).as_millis() as f64 / iterations as f64;
            average_sort_time += (sort_end - sort_start).as_millis() as f64 / iterations as f64;
        }

        println!("{} leaves, {} iterations, Sort: {}ms avg, Compute: {}ms avg)", leaves, iterations, average_sort_time as u64, average_time as u64);
    }
    Ok(())
}
fn run_tree_benchmark(device: &mut metal::Device, kernel: &mut metal::Function, iterations: usize) -> Result<(), String> {
    assert!(iterations > 0);
    println!("Running on device: {}", device.name());
    let mut rng = rand::thread_rng();

    for i in 0 .. BENCHMARK_IMPERFECT_TREES.len() * 2 {
        let mut average_time = 0f64;
        let mut average_prep_time = 0f64;
        let leaves = if i % 2 == 0 {
            BENCHMARK_IMPERFECT_TREES[i / 2]
        } else {
            BENCHMARK_PERFECT_TREES[(i - 1) / 2]
        };

        type Key = [u8; 32];
        type Value = [u8; 32];

        let len = leaves * SHA256_DIGEST_SIZE;
        
        let tree_start = std::time::Instant::now();
        let mut tree: std::collections::BTreeMap<Key, Value> = Default::default();
        for _ in 0 .. leaves/2 {
            let mut key = Key::default();
            rng.fill_bytes(&mut key);
            let mut value = Value::default();
            rng.fill_bytes(&mut value);
            tree.insert(key, value);
        }
        let tree_end = std::time::Instant::now();
        let tree_time = (tree_end - tree_start).as_millis();

        let mut buffer = device.new_buffer(
            len as u64,
            metal::MTLResourceOptions::CPUCacheModeDefaultCache | metal::MTLResourceOptions::StorageModeShared
        );

        for _j in 0 .. iterations {
            let prep_start = std::time::Instant::now();
            let mut dst: *mut u8 = buffer.contents() as _;
            for (k, v) in tree.iter() {
                unsafe {
                    std::ptr::copy_nonoverlapping(k.as_ptr(), dst, 32);
                    dst = dst.add(32);
                    std::ptr::copy_nonoverlapping(v.as_ptr(), dst, 32);
                    dst = dst.add(32);
                }
            }
            let prep_end = std::time::Instant::now();
            average_prep_time += (prep_end - prep_start).as_millis() as f64 / iterations as f64;

            let start = std::time::Instant::now();
            compute_tree(device, kernel, leaves, &mut buffer)?;
            let end = std::time::Instant::now();

            average_time += (end - start).as_millis() as f64 / iterations as f64;
        }
        println!("{} Leaves, {} Iterations, Tree: {}ms, Prepare: {}ms avg, Compute: {}ms avg)\n", leaves, iterations, tree_time as u64, average_prep_time as u64, average_time as u64);
    }
    Ok(())
}
fn run_file(device: &mut metal::Device, program: &mut metal::Function, path: &std::path::Path) -> Result<(), String>{
    println!("Reading block file");

    let mut fp = std::fs::File::open(path).unwrap();
    let file_size = fp.metadata().unwrap().len() as usize;

    if file_size % SHA256_DIGEST_SIZE != 0 {
        println!("Invalid block file, invalid number of bytes");
        return Ok(());
    }

    let mut buffer = device.new_buffer(
        file_size as u64,
        metal::MTLResourceOptions::CPUCacheModeDefaultCache | metal::MTLResourceOptions::StorageModeShared,
    );

    let mut data: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(buffer.contents() as _, file_size) };
    fp.read_exact(&mut data).unwrap();
    let leaves = (file_size / SHA256_DIGEST_SIZE) as usize;

    println!("Starting computation\n");

    let start = std::time::Instant::now();
    compute_tree(device, program, leaves, &mut buffer)?;
    let end = std::time::Instant::now();

    print_results(&data, true, end - start);

    Ok(())
}

fn print_results(root_hash: &[u8], reverse_hash: bool, time: std::time::Duration){
    println!("  --> Tree computed!");
    println!("  --> Root hash: ");
    print_hash(root_hash, reverse_hash);
    println!("\n  --> Took: {}ms", time.as_millis());
}

fn print_hash(hash: &[u8], reverse: bool){
    for i in 0 .. SHA256_DIGEST_SIZE {
        if !reverse {
            print!("{:x}", hash[i]);
        } else {
            print!("{:x}", hash[SHA256_DIGEST_SIZE - 1 - i]);
        }
    }
}

fn sort_tree(device: &mut metal::Device, kernel: &mut metal::Function, leaves: usize, data: &mut metal::Buffer)-> Result<(), String> {
    objc::rc::autoreleasepool(|| {
        let command_queue = device.new_command_queue();

        let pipeline_state_descriptor = metal::ComputePipelineDescriptor::new();
        pipeline_state_descriptor.set_compute_function(Some(&kernel));

        let pipeline_state = device
            .new_compute_pipeline_state_with_function(
                pipeline_state_descriptor.compute_function().unwrap(),
            )
            .unwrap();

        let thread_group_count = metal::MTLSize { width: leaves as u64, height: 1, depth: 1 };
        let thread_group_size = metal:: MTLSize { width: 1, height: 1, depth: 1 };
        let log_n = (leaves as f64).log2() as u32;
    
        for p in 0 .. log_n {
            for q in 0.. p + 1 {
                
                let n1 = p;
                let n2 = q;
                
                let command_buffer = command_queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&pipeline_state);
                encoder.set_buffer(0, Some(data), 0);
                encoder.set_bytes(1, 4, std::ptr::addr_of!(n1) as _);
                encoder.set_bytes(2, 4, std::ptr::addr_of!(n2) as _);
                    
                encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
            }
        }
    });
    Ok(())
}

fn compute_tree(device: &mut metal::Device, kernel: &mut metal::Function, mut leaves: usize, data: &mut metal::Buffer)-> Result<(), String> {

    objc::rc::autoreleasepool(|| {
        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();

        let pipeline_state_descriptor = metal::ComputePipelineDescriptor::new();
        pipeline_state_descriptor.set_compute_function(Some(&kernel));

        let pipeline_state = device
            .new_compute_pipeline_state_with_function(
                pipeline_state_descriptor.compute_function().unwrap(),
            )
            .unwrap();


        let mut round: usize = 0;
        while leaves > 1 {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline_state);
            encoder.set_buffer(0, Some(data), 0);
            encoder.set_bytes(1, 8, std::ptr::addr_of!(round) as _);
            encoder.set_bytes(2, 8, std::ptr::addr_of!(leaves) as _);

            let threads: usize = if leaves % 2 == 0 { leaves } else { leaves + 1 } >> 1;

            let thread_group_count = metal::MTLSize {
                width: threads as u64,
                height: 1,
                depth: 1,
            };
    
            let thread_group_size = metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };

            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            encoder.end_encoding();

            if leaves % 2 != 0 {
                leaves += 1;
            }
            leaves >>= 1;
            round += 1;
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();


    });
    Ok(())
}

