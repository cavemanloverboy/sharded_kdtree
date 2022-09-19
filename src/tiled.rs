use nabo::*;
use rayon::prelude::*;
use rayon::iter::{ParallelIterator, IntoParallelIterator};
use num_traits::{Signed, FromPrimitive, Zero};
use dashmap::{DashMap, ReadOnlyView};
use itertools::Itertools;
use crate::sharded::ShardedKDTree;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Index<const D: usize>(pub [u32; D]);

impl<const D: usize> Default for Index<D> {
    fn default() -> Self {
        Index([0; D])
    }
}

impl<const D: usize> Index<D> {
    fn from_vec(v: Vec<u32>) -> Index<D> {
        assert_eq!(v.len(), D, "Dimensionality mismatch");
       
        Index(v.try_into().unwrap())
    }
}

macro_rules! time_this_micros {
    ($lit:literal, $blk:block) => {
        {
            // let local_macro_timer = std::time::Instant::now();
            let local_macro_result = $blk;
            // println!("{} took {}", $lit, local_macro_timer.elapsed().as_micros());
            local_macro_result
        }
    };
}

pub struct TileTree<T: Scalar, P: Point<T>, const D: usize> {
    pub bounding_box: BoundingBox<T, P, D>,
    pub sharded_tree: ShardedKDTree<T, P, D>,
}

pub struct TiledKDTree<T: Scalar, P: Point<T>, const D: usize> {
    pub tile_trees: ReadOnlyView<Index<D>, TileTree<T, P, D>>,
    pub tile_bounding_boxes: ReadOnlyView<Index<D>, BoundingBox<T, P, D>>,
    pub deltas: [NotNan<T>; D],
    pub boxsize: Option<[T; D]>,
    bounding_box: BoundingBox<T, P, D>,
    pub tiles: [u32; D],
    pub shards: u32,
    pub leafsize: u32,
}

impl <T, P, const D: usize> TiledKDTree<T, P, D>
where
    T: Scalar + Signed + Send + Sync,
    P: Point<T> + Send + Sync,
{

    pub fn new(
        points: &[P],
        tiles: &[u32; D],
        shards: u32,
        leafsize: u32,
        boxsize: Option<&[T; D]>,
    ) -> Result<TiledKDTree<T, P, D>, ErrorCodes> {

        assert_eq!(P::DIM as usize, D, "Dimensionality mismatch");

        // Generate bounds for every tile
        let bounding_box;
        let (deltas, tile_bounding_boxes): ([NotNan<T>; D], DashMap<Index<D>, BoundingBox<T, P, D>>) = match boxsize {
            Some(boxsize) => {
                
                // Tile space according to the bounding box defined by boxsize
                bounding_box = BoundingBox::from_boxsize(boxsize);
                get_inner_bounding_boxes_and_deltas_for_tiles(bounding_box, tiles)
            },
            None => {

                // If no boxsize provided, find smallest axis-aligned
                // bounding box containing all points
                bounding_box = get_bounding_box(&points);

                // Then, tile this bounding box according to specified tiles
                get_inner_bounding_boxes_and_deltas_for_tiles(bounding_box, tiles)
            }
        };

        // Find which tiles the points belong in
        let point_tile_indices: Vec<(&P, Index<D>)> = {


            if let Some(boxsize) = boxsize {

                // If a boxsize is provided, i.e. if using periodic boundary coinditions,
                // check that all points are within boxsize while also getting indices
                get_indices_periodic(points, boxsize, tiles, &deltas)? 
            } else {

                get_indices(points, &bounding_box, tiles, &deltas)?
            }
            
        };

        // Now, partition the points into respective tiles.
        // Note: This clones the data when dereferencing *p.0 to create its own memory layout
        let point_partition: DashMap<Index<D>, Vec<P>> = DashMap::new();
        for p /* (&P, Index<D>) */ in point_tile_indices {

            // Get key into dashmap
            let key: Index<D> = p.1;

            // Append &P to vec of references (or initialize the vec)
            point_partition
                .entry(key)
                .and_modify(|v| v.push(*p.0)) // This clones because P: Copy + Clone
                .or_insert(Vec::with_capacity(21*points.len()/tiles.len()/20));
                        /* could do or_default() but this does a 
                            preemptive big allocation, which is better.
                                21/20 gives a 5% buffer for tiles with higher 
                                    than average number of points */
        }

        // Now build sharded kdtrees for every tile
        let tile_trees: DashMap<Index<D>, TileTree<T, P, D>> = point_partition
            .into_par_iter()
            .map(|partition| {

                // Get (key, value) pair
                let (index, points_in_partition) = partition;

                // Find bounding box corresponding to this index
                let bounding_box = *tile_bounding_boxes
                    .get(&index).unwrap().value();

                // Construct sharded tree
                let sharded_tree = ShardedKDTree::new_with_bucket_size_and_shards(
                    &points_in_partition,
                    leafsize,
                    shards as usize
                );

                (index, TileTree { bounding_box, sharded_tree })
            }).collect();
        
        // Build sharded kdtree
        // let tile_shards = ShardedKDTree::new_with_bucket_size_and_shards(points, leafsize, shards as usize);

        // Construct Tile
        Ok(TiledKDTree {
            tile_trees: tile_trees.into_read_only(),
            tile_bounding_boxes: tile_bounding_boxes.into_read_only(),
            deltas,
            shards,
            leafsize,
            bounding_box,
            boxsize: boxsize.map(|x| x.clone()),
            tiles: tiles.clone(),
        })
    }

    /// Finds the `k` nearest neighbour of `query`, using reasonable default parameters.
    ///
    /// If there are less than `k` points in the point cloud, the returned vector will be smaller than `k`.
    /// The default parameters are:
    /// Exact search, no max. radius, allowing self matching, sorting results, and not collecting statistics.
    /// If `k` <= 16, a linear vector is used to keep track of candidates, otherwise a binary heap is used.
    pub fn knn(&self, k: u32, queries: &[P]) -> Vec<Vec<Neighbour<T, P>>> {

        queries
            .par_iter()
            .map(|query| {

                #[allow(unused)]
                let mut query_count = 0;

                // Find which tile the box resides in
                let relevant_tree_index: Index<D> = self.find_index(query, &self.bounding_box, &self.tiles);

                // Grab relevant_sharded_tree
                let ref relevant_sharded_tree: ShardedKDTree<T,P,D> = self.tile_trees.get(&relevant_tree_index).unwrap().sharded_tree;

                // Collect neighbors in this tile
                let mut neighbors = Vec::with_capacity(k as usize * 4);
                let mut query_result = time_this_micros!("single", {relevant_sharded_tree.knn_single(k, query)});
                neighbors.append(&mut query_result);
                query_count += 1;

                // Check if we are done here
                let largest_dist2 = neighbors.iter().max().unwrap().dist2;
                let relevant_bounding_box = self.tile_bounding_boxes.get(&relevant_tree_index).unwrap();
                let (_, query_status) = relevant_bounding_box
                    .distance_to_sides(query, &largest_dist2);

                match query_status {
                    QueryStatus::Done => {
                        
                        // If the largest knn distance is less than the distance 
                        // to any side of the tile, we are done
                    },
                    QueryStatus::OneSide(side) => time_this_micros!("OneSide", {

                        // If only one side is touched, only one more tile needs to be queried (potentially)
                        let relevant_second_index = self.get_second_index(&relevant_tree_index, side);

                        // If we actually need a second query, do it
                        if let Some(relevant_second_index) = relevant_second_index {

                            // Get tree and do query
                            let ref second_tree = self.tile_trees.get(&relevant_second_index).unwrap().sharded_tree;
                            query_result = time_this_micros!("second query", { second_tree.knn_single(k, query) });
                            neighbors.append(&mut query_result);
                            
                            query_count += 1;

                            // Sort, truncate if not periodic
                            if self.boxsize.is_none() {
                                    neighbors.sort();
                                    neighbors.truncate(k as usize);
                            }
                        }
                    }),
                    QueryStatus::ManySides(sides, thin_tile) => {
                        
                        if let Some(_) = thin_tile {

                            // This is the most expensive branch, and can probably be optimized.
                            // We expect to enter this branch not that often unless num_tiles
                            // is within a few orders of magnitude of the number of data points.

                            // Get all relevant indices
                            let relevant_next_indices = self.get_next_indices_expensive(query, largest_dist2, &relevant_tree_index);

                            // Do all relevant queries
                            for relevant_next_index in relevant_next_indices {

                                // Get tree and do query
                                let ref next_tree = self.tile_trees.get(&relevant_next_index).unwrap().sharded_tree;
                                let mut next_query_results = next_tree.knn_single(k, query);
                                query_count += 1;

                                // Append, but wait to sort and truncate
                                neighbors.append(&mut next_query_results);

                            }
                        } else {

                            // Get all relevant indices
                            let relevant_next_indices = self.get_next_indices(&relevant_tree_index, sides);

                            // If we actually need more queries, do them
                            for relevant_next_index in relevant_next_indices {

                                // Get tree and do query
                                let ref next_tree = self.tile_trees.get(&relevant_next_index).unwrap().sharded_tree;
                                let mut next_query_results = next_tree.knn_single(k, query);
                                query_count += 1;

                                // Append, but wait to sort and truncate
                                neighbors.append(&mut next_query_results);

                            }
                        }

                        // Sort, truncate if not periodic
                        if self.boxsize.is_none() {
                            neighbors.sort();
                            neighbors.truncate(k as usize);
                        }
                    }
                }

                // Now we must deal with periodic boundary conditions if necessary
                if let Some(boxsize) = self.boxsize {

                    // Find closest dist2 to every side
                    let mut closest_side_dist2: [T; D] = [T::zero(); D];
                    for side in 0..D {

                        // Do a single index here. This is equal to distance to lower side
                        let query_component: T = *query.get(side as u32);

                        // Get distance to upper half
                        let upper = boxsize[side] - query_component;

                        // !negative includes zero
                        debug_assert!(!upper.is_negative()); 
                        debug_assert!(!query_component.is_negative());

                        // Choose lesser of two and then square
                        closest_side_dist2[side] = upper.min(query_component).powi(2);
                    }

                    // Find which images we need to check.
                    // Initialize vector with real image (which we will remove later)
                    let mut images_to_check = Vec::with_capacity(2_usize.pow(D as u32)-1);
                    for image in 1..2_usize.pow(D as u32) {
                        
                        // Closest image in the form of bool array
                        let closest_image = (0..D)
                            .map(|idx| ((image / 2_usize.pow(idx as u32)) % 2) == 1);

                        // Find distance to corresponding side, edge, vertex or other higher dimensional equivalent
                        let dist_to_side_edge_or_other: T = closest_image
                            .clone()
                            .enumerate()
                            .flat_map(|(side, flag)| if flag {
                                
                                // Get minimum of dist2 to lower and upper side
                                Some(closest_side_dist2[side])
                            } else { None })
                            .fold(T::zero(), |acc, x| acc + x);

                        if dist_to_side_edge_or_other < *largest_dist2 {

                            let mut image_to_check = query.clone();
                            
                            for (idx, flag) in closest_image.enumerate() {

                                // If moving image along this dimension
                                if flag {
                                    // Do a single index here. This is equal to distance to lower side
                                    let query_component: NotNan<T> = query.get(idx as u32);
                                    // Single index here as well
                                    let periodic_component = NotNan::<T>::new(boxsize[idx]).unwrap();

                                    if query_component < periodic_component / NotNan::<T>::from_u8(2_u8).unwrap() {
                                        // Add if in lower half of box
                                        image_to_check.set(idx as u32, query_component + periodic_component)
                                    } else {
                                        // Subtract if in upper half of box
                                        image_to_check.set(idx as u32, query_component - periodic_component)
                                    }
                                    
                                }
                            }

                            images_to_check.push(image_to_check);
                        }
                    }

                    // Then check all images
                    for image in &images_to_check {

                        // Find which tile the box resides in
                        let image_relevant_tree_index: Index<D> = self.find_index(image, &self.bounding_box, &self.tiles);

                        // Grab relevant_sharded_tree
                        let ref image_relevant_sharded_tree: ShardedKDTree<_,_,D> = self.tile_trees
                            .get(&image_relevant_tree_index).unwrap()
                            .sharded_tree;

                        // Collect neighbors in this tile
                        neighbors.append(&mut image_relevant_sharded_tree.knn_single(k, image));
                        query_count += 1;

                        // // Check if we are done here
                        // let image_largest_dist2 = image_neighbors.iter().max().unwrap().dist2;
                        // let image_relevant_bounding_box = self.tile_bounding_boxes.get(&image_relevant_tree_index).unwrap();
                        // let (_side_distances, query_status) = image_relevant_bounding_box
                        //     .distance_to_sides(image, &image_largest_dist2);

                        // match query_status {
                        //     QueryStatus::Done => {
                                
                        //         // If the largest knn distance is less than the distance 
                        //         // to any side of the tile, we are done
                        //     },
                        //     QueryStatus::OneSide(side) => { 

                        //         // If only one side is touched, only one more tile needs to be queried (potentially)
                        //         let image_relevant_second_index = self.get_second_index(&image_relevant_tree_index, side);

                        //         // If we actually need a second query, do it
                        //         if let Some(image_relevant_second_index) = image_relevant_second_index {

                        //             // Get tree and do query
                        //             let ref image_second_tree = self.tile_trees.get(&image_relevant_second_index).unwrap().sharded_tree;
                        //             let mut image_second_query_results = image_second_tree.knn_single(k, image, "image side");
                        //             query_count += 1;

                        //             // Append, sort, truncate, return
                        //             neighbors.append(&mut image_second_query_results);
                        //         }
                        //     },
                        //     QueryStatus::ManySides(sides, thin_tile) => {
                                
                        //         if let Some(_) = thin_tile {

                        //             // This is the most expensive branch, and can probably be optimized.
                        //             // We expect to enter this branch not that often unless num_tiles
                        //             // is within a few orders of magnitude of the number of data points.

                        //             // Get all relevant indices
                        //             let image_relevant_next_indices = self.get_next_indices_expensive(image, image_largest_dist2, &image_relevant_tree_index);

                        //             // Do all relevant queries
                        //             for image_relevant_next_index in image_relevant_next_indices {

                        //                 // Get tree and do query
                        //                 let ref image_next_tree = self.tile_trees.get(&image_relevant_next_index).unwrap().sharded_tree;
                        //                 let mut image_next_query_results = image_next_tree.knn_single(k, image, "image thin");
                        //                 query_count += 1;

                        //                 // Append, but wait to sort and truncate
                        //                 neighbors.append(&mut image_next_query_results);

                        //             }
                        //         } else {

                        //             // Get all relevant indices
                        //             let image_relevant_next_indices = self.get_next_indices(&image_relevant_tree_index, sides);

                        //             // If we actually need more queries, do them
                        //             for image_relevant_next_index in image_relevant_next_indices {

                        //                 // Get tree and do query
                        //                 let ref image_next_tree = self.tile_trees.get(&image_relevant_next_index).unwrap().sharded_tree;
                        //                 let mut image_next_query_results = image_next_tree.knn_single(k, image);
                        //                 query_count += 1;

                        //                 // Append, but wait to sort and truncate
                        //                 neighbors.append(&mut image_next_query_results);

                        //             }
                        //         }
                        //     }
                        // }
                    }

                    // Perform cleanup
                    neighbors.sort();
                    neighbors.dedup();
                    neighbors.truncate(k as usize);

                    neighbors
                } else {
                    // if query_count >= 4 { println!("{query_count}: {}", timer.elapsed().as_micros());}
                    // if not doing pbcs, just return neighbors
                    return neighbors
                }

            }).collect()

    }

    #[inline]
    #[allow(unused)]
    fn find_index(&self, point: &P, bounding_box: &BoundingBox<T, P, D>, tiles: &[u32; D]) -> Index<D> {
        
        // let lower = bounding_box.lower;
        // Index((0..P::DIM as i32)
        //     .map(|i| (((point.get(i as u32)-lower.get(i as u32)) / *self.deltas.get(i as usize).unwrap())
        //         .to_i64().unwrap()
        //         .max(0) as u32)
        //         .min(tiles[i as usize]-1))
        //     .collect::<Vec<_>>()
        //     .try_into()
        //     .unwrap())

        self.tile_bounding_boxes
            .iter()
            .fold((T::infinity(), Index::<D>::default()), |mut acc, x| {
                let dist = *x.1.distance_to_box(point);
                if dist < acc.0 {
                    acc = (dist, x.1.index.unwrap())
                }
                acc
            }).1
    }

    fn get_second_index(&self, current_index: &Index<D>, side: i32) -> Option<Index<D>> {

        // Sign bit encoded direction
        let lower = side.is_negative();
        let dim = side.abs() as usize;

        if lower {

            // Deal with lower edge case
            if current_index.0[dim] == 0 {

                // if let Some(boxsize) = self.boxsize {
                    
                //     // Deal with periodic boundary conditions

                //     // Get current index and wrap it 
                //     let mut new_index = current_index.clone();
                //     new_index.0[dim] = self.tiles[dim]-1;

                //     // Return index
                //     return Some(new_index)

                // } else {

                    // // If not using periodic boundary conditions, we are done
                    // return None
                // }

                // periodic bcs are dealt with elsewhere,
                // but the comment is here in case we revert.
                // for now, always return none for edge case
                return None
            } else {
                
                // If not an edge case, just decrement index
                let mut new_index = current_index.clone();
                new_index.0[dim] -= 1;

                Some(new_index)
            }
        } else {

             // Deal with upper edge case
             if current_index.0[dim] == self.tiles[dim]-1 {

                // if let Some(boxsize) = self.boxsize {
                    
                //     // Deal with periodic boundary conditions

                //     // Get current index and wrap it 
                //     let mut new_index = current_index.clone();
                //     new_index.0[dim] = 0;

                //     // Return index
                //     return Some(new_index)

                // } else {

                //     // If not using periodic boundary conditions, we are done
                //     return None
                // }


                // periodic bcs are dealt with elsewhere,
                // but the comment is here in case we revert.
                // for now, always return none for edge case
                return None
            } else {
                
                // If not an edge case, just increment index
                let mut new_index = current_index.clone();
                new_index.0[dim] += 1;

                Some(new_index)
            }

        }
    }


    fn get_next_indices(&self, current_index: &Index<D>, sides: Vec<i32>) -> Vec<Index<D>> {

        sides
            .iter()
            .powerset()
            .skip(1)
            .flat_map(|relevant_tile /* Vec<&i32> */|{

                // Initialize index
                let mut new_index = current_index.clone();
                for dim in relevant_tile {
                    
                    // Check if lower or upper side was reached
                    let lower: bool = dim.is_negative();
                    let i = dim.abs() as usize;

                    if lower {
                        if new_index.0[i] == 0 {
                            // Handle lower edge case, return early
                            return None 
                        } else {
                            // If not on lower edge, decrement by 1
                            new_index.0[i] -= 1;
                        }
                    } else {
                        if new_index.0[i] == self.tiles[i]-1 {
                            // Handle upper edge case, return early
                            return None 
                        } else {
                            // If not on upper edge, increment by 1
                            new_index.0[i] += 1;
                        }
                    }
                }

                // If not an edge case, return only after all dims
                // specified by relevant_tile are modified
                Some(new_index)
            }).collect()
    }

    fn get_next_indices_expensive(&self, point: &P, largest_dist2: NotNan<T>, current_index: &Index<D>) -> Vec<Index<D>> {

        // sides
        //     .iter()
        //     .powerset()
        //     .skip(1)
        //     .flat_map(|relevant_tile /* Vec<&i32> */|{

        //         // Initialize index
        //         let mut new_index = current_index.clone();
        //         for dim in relevant_tile {
                    
        //             // Check if lower or upper side was reached
        //             let lower: bool = dim.is_negative();
        //             let i = dim.abs() as usize;

        //             if lower {
        //                 if new_index.0[i] == 0 {
        //                     // Handle lower edge case, return early
        //                     return None 
        //                 } else {
        //                     // If not on lower edge, decrement by 1
        //                     new_index.0[i] -= 1;
        //                 }
        //             } else {
        //                 if new_index.0[i] == self.tiles[i]-1 {
        //                     // Handle upper edge case, return early
        //                     return None 
        //                 } else {
        //                     // If not on upper edge, increment by 1
        //                     new_index.0[i] += 1;
        //                 }
        //             }
        //         }

        //         // If not an edge case, return only after all dims
        //         // specified by relevant_tile are modified
        //         Some(new_index)
        //     }).collect()

        self.tile_bounding_boxes
            .iter()
            .flat_map(|key_value|{
                
                // Unpack the (key, value) pair
                let (index, bounding_box) = key_value;

                // If we are in the box, or if the box 
                if bounding_box.distance_to_box(point) < largest_dist2 && index != current_index {
                    Some(index.clone())
                } else {
                    None
                }
            }).collect()
    }
}


#[derive(Debug)]
pub enum ErrorCodes {
    InvalidBounds
}

pub enum QueryStatus {
    Done,
    OneSide(i32),
    ManySides(Vec<i32>, Option<Vec<usize>>),
}

use std::marker::PhantomData;
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct BoundingBox<T: Scalar, P: Point<T>, const D: usize> {
    pub lower: P,
    pub upper: P,
    pub index: Option<Index<D>>,
    phantom: PhantomData<T>,
}
impl<T: Scalar, P: Point<T>, const D: usize> BoundingBox<T, P, D> {
    pub fn from_boxsize(boxsize: &[T; D]) -> BoundingBox<T, P, D> {

        assert_eq!(P::DIM as usize, D, "Dimensionality mismatch");

        let mut upper = P::default();
        for d in 0..P::DIM {
            upper.set(d, NotNan::new(boxsize[d as usize]).expect("boxsize should have no nans"))
        }
        BoundingBox {
            lower: P::default(),
            upper,
            ..
            BoundingBox::default()
        }
    }
    
    pub fn distance_to_sides(&self, point: &P, largest_dist2: &NotNan<T>) -> ([[T; 2]; D], QueryStatus) {

        let mut distances = [[T::zero(); 2]; D];
        let mut count_close_sides = vec![];

        let mut thin_tile: Option<Vec<usize>> = None;
        for i in 0..D {

            // Gather relevant components
            let lower_side = self.lower.get(i as u32);
            let upper_side = self.upper.get(i as u32);
            let point_component = point.get(i as u32);

            // Calculate distance
            // TODO: Potential optimization here by switching dist2 --> dist
            // and only doing abs dist here since axis aligned p-norms are all equal
            distances[i][0] = (point_component-lower_side).powi(2);
            distances[i][1] = (point_component-upper_side).powi(2);

            // Push to counter if side is closer than largest distances
            let lower = &distances[i][0] < largest_dist2;
            let upper = &distances[i][1] < largest_dist2;
            if lower { count_close_sides.push(-(i as i32)) }
            if upper { count_close_sides.push(  i as i32) }

            // If both of these are active, then there may be a thin tile (or very low number density)
            if lower && upper {
                if let Some(ref mut thin_tile) = thin_tile {
                    // If already initialized, append
                    thin_tile.push(i);
                } else {
                    // If not, initialize.
                    thin_tile = Some(vec![i]);
                }
            }
        }

        // Determine query_status
        let query_status = match count_close_sides.len() {
            0 => QueryStatus::Done,
            1 => QueryStatus::OneSide(count_close_sides[0]),
            _ => QueryStatus::ManySides(count_close_sides, thin_tile),
        };

        (distances, query_status)
    }

    pub fn distance_to_box(&self, point: &P) -> NotNan<T> {

        // Construct closest point to `point` on the surface of the box
        let mut closest_point = point.clone();
        for i in 0..P::DIM {
            if point.get(i) > self.upper.get(i) {
                closest_point.set(i, self.upper.get(i));
            } else if point.get(i) < self.lower.get(i) {
                closest_point.set(i, self.lower.get(i));
            } else {
                closest_point.set(i, point.get(i));
            }
        }
        
        // Calculate distance between these two points
        let mut distance = NotNan::<T>::zero();
        for i in 0..P::DIM {
            distance += (closest_point.get(i) - point.get(i)).powi(2);
        }
        distance
    }
}

impl<T: Scalar, P: Point<T>, const D: usize> Default for BoundingBox<T, P, D> {
    fn default() -> Self {
        BoundingBox {
            lower: P::default(),
            upper: P::default(),
            index: None,
            phantom: PhantomData::<T>::default()
        }
    }
}

/// Given a set of points, this function finds the smallest axis-aligned
/// bounding box that contains all points. The first entry in [P; 2] is
/// the lower bounding box. The 
fn get_bounding_box<T: Scalar, P: Point<T>, const D: usize>(points: &[P]) -> BoundingBox<T, P, D> {
    points
        .iter()
        .fold(BoundingBox::default(), |mut bounding_box, point| {

            // For every dimension
            for d in 0..P::DIM {

                // Check lower
                if point.get(d) < bounding_box.lower.get(d) {
                    bounding_box.lower.set(d, point.get(d));
                }
                
                // Check upper
                if point.get(d) > bounding_box.upper.get(d) {
                    bounding_box.upper.set(d, point.get(d));
                }
            }

            bounding_box
        })
}

fn get_inner_bounding_boxes_and_deltas_for_tiles<T: Scalar, P: Point<T>, const D: usize>(
    bounding_box: BoundingBox<T, P, D>,
    tiles: &[u32; D]
) -> ( /* deltas */ [NotNan<T>; D], DashMap<Index<D>,BoundingBox<T, P, D>>) {

    // Sanity check
    assert_eq!(P::DIM as usize, D, "Dimensionality mismatch");

    // Get grid spacings in every dimension
    let (lower, upper) = (bounding_box.lower, bounding_box.upper);
    let deltas: [NotNan<T>; D] = tiles
        .iter()
        .enumerate()
        .map(|(d, num_tiles /* number of tilings in this dimension, not total */)|{

            // Get grid spacing in this dimension
            let delta = (upper.get(d as u32) - lower.get(d as u32)) / T::from(*num_tiles).unwrap();

            delta
        })
        .collect::<Vec<_>>()
        .try_into()
        .expect("dimensions should match");
    
    (deltas, tiles
        .iter()
        .map(|&x| 0..x)
        .multi_cartesian_product()
        .map(|vec_index| {

            // Convert vector index to array
            let index = Index::from_vec(vec_index);

            // Calculate lower bound array and convert to point
            #[allow(unused_mut)]//, reason = "This is actually modified in for_each via lower.set()")]
            let mut lower = P::default();
            index.0
                .iter()
                .enumerate()
                .map(|(i, idx)| {
                    NotNan::<T>::from_u32(*idx).unwrap()* deltas[i] 
                })
                .collect::<Vec<_>>()
                .iter()
                .enumerate()
                .for_each(|(i, idx)| lower.set(i as u32, *idx));
            

            // Calculate upper bound array
            #[allow(unused_mut)]//, reason = "This is actually modified in for_each via lower.set()")]
            let mut upper = P::default();
            index.0
                .iter()
                .enumerate()
                .map(|(i, idx)| {
                    NotNan::<T>::from_u32(*idx + 1).unwrap()* deltas[i] 
                })
                .collect::<Vec<_>>()
                .iter()
                .enumerate()
                .for_each(|(i, idx)| lower.set(i as u32, *idx));
            
            // Wrap index
            let index = Some(index);

            // Construct BoundingBox
            BoundingBox {
                lower,
                upper,
                index,
                ..
                BoundingBox::default()
            }
        })
        .collect::<Vec<_>>()
        .iter()
        .map(|x| (x.index.unwrap(), *x))
        .collect())
}

#[test]
fn test_bounding_box_3d() {

    use nabo::dummy_point::P3;
    
    // Define some set of points between (inclusive) [0,1]^3
    let points = [
        P3::new(0.0, 1.0, 0.0),
        P3::new(0.234, 0.543, 0.9234),
        P3::new(0.844, 0.918, 0.01),
        P3::new(1.0, 0.0, 1.0),
    ];

    // Get bounding box
    let bounding_box = get_bounding_box(&points);

    // Check bounding box
    assert_eq!(
        bounding_box,
        BoundingBox {
            lower: P3::new(0.0, 0.0, 0.0),
            upper: P3::new(1.0, 1.0, 1.0),
            ..
            BoundingBox::<_,_,3>::default()
        }
    )
}


/// This functions checks that a given set of points is consistent with 
/// the specifeid boxsize (assuming periodic boundary conditions).
fn get_indices_periodic<'a, T, P, const D: usize>(
    points: &'a [P],
    boxsize: &'a [T; D],
    // tile_bounding_boxes: &[BoundingBox<T, P, D>],
    tiles: &'a [u32; D],
    deltas: &'a [NotNan<T>; D],
) -> Result<Vec<(&'a P, Index<D>)>, ErrorCodes>
where
    T: Scalar,
    P: Point<T>, 
{

    points
        .iter()
        .map(|p| -> Result<(&P, Index<D>), ErrorCodes> {

            // Initialize index
            let mut index = [0_u32; D];

            for i in 0..P::DIM {

                // Get relevant component
                let p_component = p.get(i);

                // check that the component is positive and within the box
                if p_component.is_sign_negative() || *p_component > boxsize[i as usize] {
                    return Err(ErrorCodes::InvalidBounds)
                }
                
                // Find what tile it is in (in this dimension) and update index
                let dim_index = (p_component / deltas[i as usize]).to_u32().unwrap().min(tiles[i as usize]-1);
                index[i as usize] = dim_index;
            }

            // Return index
            Ok((p, Index(index)))
        }).collect()
}


/// This functions checks that a given set of points is consistent with 
/// the specified bounding box. Non-periodic boundary conditions. The bounding box
/// provided here should have been computed by `get_bounding_box` and thus should be
/// consistent, so no checks on `points` required here.
fn get_indices<'a, T, P, const D: usize>(
    points: &'a [P],
    bounding_box: &'a BoundingBox<T, P, D>,
    tiles: &'a [u32; D],
    deltas: &'a [NotNan<T>; D],
) -> Result<Vec<(&'a P, Index<D>)>, ErrorCodes>
where
    T: Scalar,
    P: Point<T>, 
{
    points
        .iter()
        .map(|p| -> Result<(&P, Index<D>), ErrorCodes> {

            // Initialize index
            let mut index = [0_u32; D];

            for i in 0..P::DIM {

                // Get relevant component
                let p_component = p.get(i);
                let distance_from_lower = p_component - bounding_box.lower.get(i);
                
                // Find what tile it is in (in this dimension) and update index
                let dim_index = (distance_from_lower / deltas[i as usize]).to_u32().unwrap().min(tiles[i as usize]-1);
                index[i as usize] = dim_index;
            }

            // Return index
            Ok((p, Index(index)))
        }).collect()
}

