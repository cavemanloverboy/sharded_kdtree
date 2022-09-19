use nabo::Neighbour;
use nabo::dummy_point::*;
use nabo::KDTree;
use sharded_kdtree::sharded::ShardedKDTree;
use rayon::prelude::*;
use std::time::Instant;
use integer_sqrt::IntegerSquareRoot;

fn main() {

    const NUM_THREADS: usize = 48;
    rayon::ThreadPoolBuilder::new().num_threads(NUM_THREADS).build_global().unwrap();

    const CLOUD_SIZE: u32 = 1_000_000;
    const QUERY_COUNT: u32 = 2_000_000;

    let pred = (CLOUD_SIZE * NUM_THREADS as u32 / QUERY_COUNT).integer_sqrt();
    println!("THEORETICAL MATCH IS {pred}");

    #[allow(non_snake_case)]
    for SHARDS in [1, 2, 4, 8, 16, 32] {


        println!("\n\n---- SHARDS = {SHARDS} ----");
        let cloud = random_point_cloud_3d(CLOUD_SIZE);

        let timer = Instant::now();
        let tree = std::sync::Arc::new(KDTree::<_,_,3>::new_with_bucket_size(&cloud, 32));
        let unsharded_build_time = timer.elapsed().as_micros();
        println!("unsharded tree took {unsharded_build_time} micros to build");

        let timer = Instant::now();
        let sharded_tree = ShardedKDTree::<_,_,3>::new_with_bucket_size_and_shards(&cloud, 32, SHARDS);
        let sharded_build_time = timer.elapsed().as_micros();
        println!("sharded tree took {sharded_build_time} micros to build");

        let queries = (0..QUERY_COUNT).map(|_| random_point_p3()).collect::<Vec<_>>();
        
        for k in [1] {// , 3, 4, 6, 8, 11, 16, 24] {

            let timer = Instant::now();
            let unsharded_result: Vec<Vec<Neighbour<_,_>>> = queries
                .par_iter()
                // .enumerate()
                .map_with(&tree, |t, q| {
                    let a = t.knn(k, q);
                    // if i == 0 { println!("{a:?}");}
                    a
                }).collect();
            let unsharded_query_time = timer.elapsed().as_micros();
            println!("unsharded tree took {unsharded_query_time} micros to query");

            let timer = Instant::now();
            let sharded_result = sharded_tree.knn(k, &queries);
            let sharded_query_time = timer.elapsed().as_micros();
            println!("sharded tree took {sharded_query_time} micros to query");

            assert_eq!(unsharded_result, sharded_result);

            // if sharded_query_time < unsharded_query_time {
            //     println!("SHARDED TREE WINS QUERY {:.2}%", 100.0*sharded_query_time as f64 / unsharded_query_time as f64)
            // } else {
            //     println!("UNSHARDED TREE WINS QUERY {:.2}%", 100.0*unsharded_query_time as f64 / sharded_query_time as f64)
            // }
            let min_time = sharded_query_time.min(unsharded_query_time);
            let sharded = sharded_query_time == min_time;
            let unsharded = unsharded_query_time == min_time;
            if sharded{ println!("SHARDED TREE WINS QUERY") }
            else if unsharded { println!("UNSHARDED TREE WINS QUERY") }

            let sharded_overall = sharded_build_time + sharded_query_time;
            let unsharded_overall = unsharded_build_time + unsharded_query_time;
            let min_time = sharded_overall.min(unsharded_overall);
            let sharded = sharded_overall == min_time;
            let unsharded = unsharded_overall == min_time;
            if sharded{ println!("SHARDED TREE WINS OVERALL") }
            else if unsharded { println!("UNSHARDED TREE WINS OVERALL") }
        }
    }

}
