use nabo::*;
use rayon::prelude::*;
use rayon::iter::{ParallelIterator, IntoParallelIterator};
use num_traits::Signed;
use std::collections::BinaryHeap;
use parking_lot::RwLock;
use std::sync::Arc;

pub struct ShardedKDTree<T: Scalar, P: Point<T>, const D: usize> {
    /// The leafsize to use for the leaves of the shards
    pub leafsize: u32,
    /// The root nodes of the shards
    root_nodes: Arc<Vec<KDTree<T, P, D>>>,
}


impl <T: Scalar + Signed + Send + Sync, P: Point<T> + Send + Sync, const D: usize> ShardedKDTree<T, P, D> {


    pub fn new_with_bucket_size_and_shards(points: &[P], leafsize: u32, shards: usize) -> Self
    where
    {
        // Fairly chunk the data across the shards and create a knn for each shard.
        let root_nodes = Arc::new(fair_chunks(points, shards).collect::<Vec<_>>()
            .into_par_iter()
            .map(|point_subset| {
                KDTree::new_with_bucket_size(point_subset, leafsize)
            })
            .collect());

        ShardedKDTree {
            leafsize,
            root_nodes
        }
    }

    pub fn knn(&self, k: u32, queries: &[P]) -> Vec<Vec<Neighbour<T, P>>> {

        queries
            .par_iter()
            .map_with(&self.root_nodes, |nodes, query| {

                // For every query point create a binary heap 
                let allocation = Arc::new(RwLock::new(BinaryHeap::with_capacity(k as usize*self.root_nodes.len())));

                nodes
                    .par_iter()
                    .for_each_with(Arc::clone(&allocation), |alc, root_node| {
                        let mut neighbors: BinaryHeap<Neighbour<T, P>> = BinaryHeap::from(root_node.knn(k, query));
                        alc.write().append(&mut neighbors);
                    });

                let mut allocation = Arc::try_unwrap(allocation).unwrap().into_inner().into_sorted_vec();
                allocation.truncate(k as usize);
                allocation
            }).collect()

    }

    pub fn knn_single(&self, k: u32, query: &P) -> Vec<Neighbour<T, P>> {

        // For every query point create a binary heap 
        let mut allocation = 
        // Arc::new(RwLock::new(
            BinaryHeap::with_capacity(k as usize*self.root_nodes.len());
        // ));

        self.root_nodes
            // .par_iter()
            // .for_each_with(Arc::clone(&allocation), |alc, root_node| {
            .iter()
            .for_each(|root_node| {
                let result = root_node.knn(k, query);
                let mut neighbors: BinaryHeap<Neighbour<T, P>> = BinaryHeap::from(result);
                // alc.write().append(&mut neighbors);
                allocation.append(&mut neighbors);
            });

        // let mut allocation = Arc::try_unwrap(allocation).unwrap().into_inner().into_sorted_vec();
        let mut allocation = allocation.into_sorted_vec();
        allocation.truncate(k as usize);
        
        allocation
    }
}



fn fair_chunks<T>(set: &[T], chunks: usize) -> impl Iterator<Item=&[T]> {

    // Calculate length of set
    let set_len = set.len();

    // Calculate chunking sizes
    let size = set_len / chunks;
    let remainder = set_len - size * chunks;

    (0..chunks)
        .scan(set, move |rem, i|  {
            if i < remainder {
                let chunk;
                (chunk, *rem) =  rem.split_at(size+1);
                Some(chunk)
            } else {
                let chunk;
                (chunk, *rem) =  rem.split_at(size);
                Some(chunk)
            }
        })
}


#[test]
fn test_fair_chunks_two() {
    
    let set = [1, 2, 3, 4, 5];
    let mut chunks = fair_chunks(&set, 2);

    assert_eq!(chunks.next(),Some([1, 2, 3].as_ref()));
    assert_eq!(chunks.next(),Some([4, 5].as_ref()));
}

#[test]
fn test_fair_chunks_four() {
    
    let set = [1, 2, 3, 4, 5];
    let mut chunks = fair_chunks(&set, 4);
    assert_eq!(chunks.next(),Some([1, 2].as_ref()));
    assert_eq!(chunks.next(),Some([3].as_ref()));
    assert_eq!(chunks.next(),Some([4].as_ref()));
    assert_eq!(chunks.next(),Some([5].as_ref()));
}

#[test]
fn test_fair_chunks_larger() {
    
    let set: Vec<usize> = (1..=17).collect();
    let mut chunks = fair_chunks(&set, 7);
    assert_eq!(chunks.next(),Some([1, 2, 3].as_ref()));
    assert_eq!(chunks.next(),Some([4, 5, 6].as_ref()));
    assert_eq!(chunks.next(),Some([7, 8, 9].as_ref()));
    assert_eq!(chunks.next(),Some([10, 11].as_ref()));
    assert_eq!(chunks.next(),Some([12, 13].as_ref()));
    assert_eq!(chunks.next(),Some([14, 15].as_ref()));
    assert_eq!(chunks.next(),Some([16, 17].as_ref()));
}