// Copyright 2018 Mihail Mladenov
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//		http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::HashMap;

use super::error::Error;

use super::loader::FontData;
use super::loader::TTFCurve;
use super::loader::GlyphData;

use super::loader::create_vec_with_size_uninitialized;

#[derive(Default)]
pub struct GrayScaleSurface {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

pub struct FontRasterizer {
    font_data: FontData,
    glyph_data_cache: HashMap<char, GlyphData>,
    cache_enabled: bool,
}

impl FontRasterizer {
    pub fn from_font_data(font_data: FontData, cache_enabled: bool) -> FontRasterizer {
        FontRasterizer {
            font_data: font_data,
            glyph_data_cache: HashMap::<char, GlyphData>::new(),
            cache_enabled: cache_enabled,
        }
    }
    
    pub fn new(path: &str, cache_enabled: bool) -> Result<FontRasterizer, Error> {
		let font_data = FontData::load(path)?;
		
		Ok(FontRasterizer::from_font_data(font_data, cache_enabled))
	}
}

impl FontRasterizer {
    pub fn rasterize_glyph(&mut self, codepoint: char, height: u32) -> GrayScaleSurface {
		let expected_max_height = self.font_data.ascent as f32 - self.font_data.descent as f32;
		let font_scale = height as f32 / expected_max_height;
		
		// Have to use this ugly hack because else the borrow checker complains that we are keeping
		// mutable and immutable references to self at the same time :( .
		let cache_enabled = self.cache_enabled;
		
        if cache_enabled {
               	rasterize(font_scale,
                    self.glyph_data_cache.entry(codepoint).or_insert(
                        self.font_data.fetch_glyph_data_for_codepoint(codepoint),
                    ),
                )
        } 
        else {
            let g = self.font_data.fetch_glyph_data_for_codepoint(codepoint);
            rasterize(font_scale, &g)
        }
    }


}

struct Edge {
	lowermost_point: (f32, f32),
	uppermost_point: (f32, f32),
	// Needs to be +1/-1 but gets padded to 4 bytes anyway in order to preserve alignment 
	direction: f32,
}

use std::cmp::{Ordering, PartialOrd, PartialEq};



impl PartialOrd for Edge {
	fn partial_cmp(&self, other: &Edge)  -> Option<Ordering> {
		self.uppermost_point.1.partial_cmp(&other.uppermost_point.1)
	}
}

impl PartialEq for Edge {
	fn eq(&self, other: &Edge) -> bool {
		self.uppermost_point.1 == other.uppermost_point.1
	}
}

impl Edge {
	fn new(lowermost_point: (f32, f32), uppermost_point: (f32, f32), direction: f32) -> Edge {
		Edge {
			lowermost_point: lowermost_point,
			uppermost_point: uppermost_point,
			direction: direction,
		}
	}
}

fn rasterize(scale: f32, glyph_data: &GlyphData) -> GrayScaleSurface {
	let mut surface = GrayScaleSurface {
		width: 0,
		height: 0,
		data: Vec::new(),
	};
	
	let min_x = glyph_data.bounding_box_diagonal.start_point.0 as f32;
	let min_y = glyph_data.bounding_box_diagonal.start_point.1 as f32;
	let max_x = glyph_data.bounding_box_diagonal.end_point.0 as f32;
	let max_y = glyph_data.bounding_box_diagonal.end_point.1 as f32;

	surface.width = ((max_x - min_x + 1.0) * scale).ceil() as u32;
	surface.height = ((max_y - min_y + 1.0) * scale).ceil() as u32;
	
	// The next code can be confusing beause some operations are omitted due to mental algebra.
	// If S is the scaling matrix R is the reflection matrix t_o is translation of the lowermost 
	// bounding box point to the origin, then t_c is the translation of the resulting bounding box's
	// center to the origin and t_sc is the translation of the scaled bounding box's lowermost point
	// to the origin then the final transformation T on point p is Tp = RS(p + t_o + t_c) + t_sc.
	// Now when we writeout the explicit form of the final translation vector of this affine 
	// transformation namely RSt_o + RSt_c + t_sc some of the terms in the components will cencel 
	// out.
	// 
	// This operation is done only once per rasterization so it's irrelevant how efficient it is. 
	// I would've used more clear form but I've not implemented traits for 2d matrix algebra so I 
	// have to resort to this confusing componentwise form.
	
	let t_o = (-min_x, -min_y);
	let t_c_1 = min_y - max_y;

	let translation_vector = (scale * t_o.0 , - scale * (t_o.1 + t_c_1));
	
	let mut edges = linearize(scale, glyph_data);

	transform_edges_to_surface_space(scale, translation_vector, &mut edges);
	
	use std::cmp::Ordering::Equal;

	// Sort by the uppermost edges. Edges with uppermost points the higher up will be first.
	edges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Equal));

	rasterize_edges(&mut edges, &mut surface);

	surface
}

trait AddEdge {
	fn add_edge(&mut self, start_point: (f32, f32), end_point: (f32, f32));
}

impl AddEdge for Vec<Edge> {
	fn add_edge(&mut self, start_point: (f32, f32), end_point: (f32, f32)) {
		if start_point.1 < end_point.1 {
			self.push(Edge::new(start_point, end_point, 1.0));
		}
		else if start_point.1 > end_point.1 {
			self.push(Edge::new(end_point, start_point, -1.0));
		}
		else {
			// In case this happens we can safely drop those
		}
	}
}

const FLATNESS_CONSTANT_IN_PIXELS: f32 = 0.27;

fn linearize(scale: f32, glyph_data: &GlyphData) -> Vec<Edge> {
	let mut edges: Vec<Edge> = Vec::with_capacity(glyph_data.components.len() * 10);
	
	// Invert the scalar transform in order to get the flatness threshold in the glyph coordinate
	// system.
	let mut flatness_threshold = FLATNESS_CONSTANT_IN_PIXELS / scale;
	// We do this to avoid using square root during linearization of Bezier Curves.
	flatness_threshold = flatness_threshold * flatness_threshold;
	
	for i in glyph_data.components.iter() {
		match i {
			&TTFCurve::Line(ref line) => {
				edges.add_edge(
								(line.start_point.0 as f32, line.start_point.1 as f32),
								(line.end_point.0 as f32, line.end_point.1 as f32),
							   );		
			}

			&TTFCurve::QuadraticBezierCurve(ref curve) => {
				linearize_curve(
								 &mut edges,
								 flatness_threshold,
								 (curve.start_point.0 as f32, curve.start_point.1 as f32),
								 (curve.control_point.0 as f32, curve.control_point.1 as f32),
								 (curve.end_point.0 as f32, curve.end_point.1 as f32)
							   );
			}
		}
	}
	
	edges
}

// Use special case of De Casteljau's algorithm to turn the curve into polyline.
// https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm
fn linearize_curve(
				    edges: &mut Vec<Edge>, 
				    flatness_threshold: f32, 
				    point0: (f32, f32), 
				    point1: (f32, f32), 
				    point2: (f32, f32)
				   ) {
    // We can safely give hard limit to the depth of the "recursive" subdivision. This is because
    // depth of 15 giving us 2^15 == 32768 divisions for a single curve is insane even at very large
    // resolutions. For example even if the curve is bounded by a box of width 4000 pixels then for
    // the length of the curve is
    //  \[
    // 2\int\limits_0^1 \left| (p_2 - p_1)t + (p_1 - p_0) (1-t)\right|dt 
    //  =
    // 2\int\limits_0^1 \left| (p_2 - 2p_1 +p_0)t + (p_1 - p_0)\right|dt 
    // \leq
    // 2 \int\limits_0^1 8000t dt + 2 \int\limits_0^1 4000 dt 
    // = 
    // 8000 + 8000 
    // =
    // 16000
    // 	\]
    // This means that on average the length of each individual piece of this subdivided curve is 
    // going to be less than half a pixel!
    
    // Number of points times the division depth
    const STACK_SIZE: usize = 3 * 15;
    
    let mut stack: [(f32, f32); STACK_SIZE] = unsafe { ::std::mem::uninitialized() };
    stack[0] = point0;
    stack[1] = point1;
    stack[2] = point2;
    let mut stack_size: usize = 3;
    
    while 0 < stack_size {
		let beta02 = stack[stack_size - 1];
		let beta01 = stack[stack_size - 2];
		let beta00 = stack[stack_size - 3];
		stack_size = stack_size - 3;
		
		let beta10 = ((beta01.0 + beta00.0) / 2.0, (beta01.1 + beta00.1) / 2.0);
		let beta11 = ((beta02.0 + beta01.0) / 2.0, (beta02.1 + beta01.1) / 2.0);
		let beta20 = ((beta11.0 + beta10.0) / 2.0, (beta11.1 + beta10.1) / 2.0);
		
		let mid = ((beta02.0 + beta00.0) / 2.0, (beta02.1 + beta00.1) / 2.0);
		let height = (beta20.0 - mid.0, beta20.1 - mid.1);
		
		if height.0 * height.0 + height.1 * height.1 > flatness_threshold && stack_size + 6 < STACK_SIZE {
			// We still havent reached the desired flatness which means we have to subdivide
			
			stack[stack_size] = beta00;
			stack[stack_size + 1] = beta10;
			stack[stack_size + 2] = beta20;
			
			stack[stack_size + 3] = beta20;
			stack[stack_size + 4] = beta11;
			stack[stack_size + 5] = beta02;
			
			stack_size = stack_size + 6;
		}
		else {
			edges.add_edge(beta00, beta01);
			edges.add_edge(beta01, beta02);
		}
	}
}

fn transform_edges_to_surface_space(scale: f32, translation: (f32, f32), edges: &mut Vec<Edge>) {
	for i in 0..edges.len() {
		// We flip the second coordinate in order to match the coordinates of the pixel grid.
		// This doesn't change the direction of the edges but now the uppermost edge is with lower
		// numbers as vertical coordinates which is what we want.
		edges[i].lowermost_point.0 *= scale;
		edges[i].lowermost_point.1 *= -scale;
		edges[i].uppermost_point.0 *= scale;
		edges[i].uppermost_point.1 *= -scale;
		edges[i].lowermost_point.0 += translation.0;
		edges[i].lowermost_point.1 += translation.1;
		edges[i].uppermost_point.0 += translation.0;
		edges[i].uppermost_point.1 += translation.1;
		
		debug_assert!(edges[i].lowermost_point.0 >= -::std::f32::EPSILON);
		debug_assert!(edges[i].lowermost_point.1 >= -::std::f32::EPSILON);
		debug_assert!(edges[i].uppermost_point.0 >= -::std::f32::EPSILON);
		debug_assert!(edges[i].uppermost_point.1 >= -::std::f32::EPSILON);
	}
}

// Here it's more convenient to store the edge in slope-intercept form.
struct ActiveEdge {
	lowermost_point_1: f32,
	uppermost_point_1: f32,
	// The zeroth component of the intersection of the line passing through the edge's vertices 
	// with the current scanline. By having this point, the slope and the first components of the 
	// edge's vertices we can restore their original positions as well as easily compute the zeroth
	// component of the intersection of the edge with a certain scanline.
	scanline_top_intersection_0: f32,
	// The derivative with respect to the zeroth direction.
	dxdy: f32,
	direction: f32,
}

impl ActiveEdge {
	fn new(
			lowermost_point_1: f32,
			uppermost_point_1: f32,
	 		scanline_top_intersection_0: f32, 
	 		dxdy: f32,
	 		direction: f32,
	 	   ) -> ActiveEdge {
	    ActiveEdge {
			lowermost_point_1: lowermost_point_1,
			uppermost_point_1: uppermost_point_1,
			scanline_top_intersection_0: scanline_top_intersection_0,
			dxdy: dxdy,
			direction: direction,
		}
	}
}

fn rasterize_edges(edges: &mut Vec<Edge>, surface: &mut GrayScaleSurface) {
	// This function implements the high level functionality of the rasterizing algorithm
	
	surface.data = create_vec_with_size_uninitialized((surface.height * surface.width) as usize);
	
	// We store first component which represents the signed area of a pixel shadowed by an outline,
	// and then a second component that works as a commulative sum which will indicate that the 
	// area will be added to all the rest of the pixels on the right.
	let mut scanline: Vec<(f32,f32)> = Vec::with_capacity(surface.width as usize);
	scanline.resize(surface.width as usize, (0.0, 0.0));
	
	// We keep a set of edges that are relevant for the current scanline each iteration
	let mut active_edges: Vec<ActiveEdge> = Vec::with_capacity(surface.width as usize);
	
	let mut edges_idx = 0;
	
	for i in 0..surface.height {
		let scanline_top: f32 = i as f32;
		let scanline_bot: f32 = scanline_top + 1.0;
		
		// Remove edges that are not relevant for this scanline, hence no longer relevant in 
		// general.
		prune_active_edges(&mut active_edges, scanline_top);
		// Add all the new edges that have become relevant for the scanline.
		add_active_edges(&mut active_edges, &edges, &mut edges_idx, scanline_bot, scanline_top);
		// Fill the scanline array according to the edges intersecting the scanline
		process_active_edges(&active_edges, &mut scanline, scanline_bot, scanline_top);
		// Fill the surface scanline according to the scanline array
		draw_scanline(surface, &scanline, i);
		clear_scanline(&mut scanline);
		// Transform the representation of the active edges so that they are convenient for
		// pruning and processing next scanline.
		prepare_active_edges_for_next_scanline(&mut active_edges);
	}
}

fn clear_scanline(scanline: &mut Vec<(f32,f32)>) {
	for i in 0..scanline.len() {
		scanline[i].0 = 0.0;
		scanline[i].1 = 0.0;
	}
}

fn prune_active_edges(active_edges: &mut Vec<ActiveEdge>, scanline_top: f32) {
	let mut active_edges_count = active_edges.len();
	let mut idx: isize = 0;
	
	while (idx as usize) < active_edges_count {
		// We check if the edge is above the scanline in order to remove it
		if active_edges[idx as usize].lowermost_point_1 <= scanline_top {
			active_edges.swap_remove(idx as usize);
			idx = idx - 1;
			active_edges_count = active_edges_count - 1;
		}
		
		idx = idx + 1;
	}
}

const HORIZONTALITY_TOLERANCE: f32 = (1.0 / ((1 << 15) as f32)) * 4.0;

trait Activate {
	fn activate(&mut self, edge: &Edge, scanline_top: f32);
}

impl Activate for Vec<ActiveEdge> {
	fn activate(&mut self, edge: &Edge, scanline_top:f32) {
		let dx = edge.lowermost_point.0 - edge.uppermost_point.0;
		let dy = edge.lowermost_point.1 - edge.uppermost_point.1;
		
		debug_assert!(dy > 0.0);
		
		if dy <= HORIZONTALITY_TOLERANCE {
			// We dont want horizontal edges just drop them
		}
		else {
			let dxdy = dx/dy;
			
			let scanline_top_intersection_0 = edge.uppermost_point.0
											+ dxdy * (scanline_top - edge.uppermost_point.1);			
			
			self.push(ActiveEdge::new(
								       edge.lowermost_point.1,
								       edge.uppermost_point.1,
								       scanline_top_intersection_0,
								       dxdy,
								       edge.direction
								     ));			 
		}
	}
}

fn add_active_edges(
					 active_edges: &mut Vec<ActiveEdge>, 
					 edges: &Vec<Edge>, 
					 edges_idx: &mut usize, 
					 scanline_bot: f32,
					 scanline_top: f32,
				    ) {
	let edges_size = edges.len();
	
	// Add all the edges that have their uppermost point before the end of this scanline
	while *edges_idx < edges_size && edges[*edges_idx].uppermost_point.1 < scanline_bot {
		active_edges.activate(&edges[*edges_idx], scanline_top);
		*edges_idx = *edges_idx + 1;
	}
}

fn process_active_edges(
						 active_edges: &Vec<ActiveEdge>, 
						 scanline: &mut Vec<(f32, f32)>, 
						 scanline_bot: f32, 
						 scanline_top: f32
					   ) {
	for i in 0..active_edges.len() {
		process_active_edge(&active_edges[i], scanline, scanline_bot, scanline_top);
	}
}

fn process_active_edge(
						 edge: &ActiveEdge, 
						 scanline: &mut Vec<(f32, f32)>, 
						 scanline_bot: f32, 
						 scanline_top: f32
					   ) {
	// Now we need to find the highest point of the edge that is below the top of the scanline and 
	// the lowest point of the edge that is above the scanline bottom. In case that the edges go 
	// beyond the sanline the points we search for are intersections.
	
	let mut high_point: (f32, f32) = unsafe { ::std::mem::uninitialized() };
	let mut low_point: (f32, f32) = unsafe { ::std::mem::uninitialized() };
	
	if scanline_top < edge.uppermost_point_1 {
		high_point.0 = edge.scanline_top_intersection_0 
						+ edge.dxdy * (edge.uppermost_point_1 - scanline_top);
		high_point.1 = edge.uppermost_point_1;
	}
	else {
		// Here we know that since the actual edge is above the scanline, the 0th direction 
		// intersection has to be inside the surface since all edges are inside the surface
		high_point.0 = edge.scanline_top_intersection_0;
		high_point.1 = scanline_top;
	}
	
	if scanline_bot < edge.lowermost_point_1 {
		low_point.0 = edge.scanline_top_intersection_0 + edge.dxdy;
		low_point.1 = scanline_bot;
	}
	else {
		low_point.0 = edge.scanline_top_intersection_0
					  + edge.dxdy * (edge.lowermost_point_1 - scanline_top);
		low_point.1 = edge.lowermost_point_1;
	}
	
	// Once we have clipped the edge to the boundaries of the scanline there are two relevant 
	// possibilities to look for namely wether or not high_point.0 < low_point.0. In the two cases
	// the computation of the area will look differently because we process the pixels of the 
	// scanline from left to right (increasing in the 0th direction) hence we will care about the 
	// 0th coordinate of the edge. 
	//
	// However we can avoid that by simply recognizing that flipping the clipped edge around it's 
	// vertical center won't change the unsigned area.
	
	let mut dxdy = edge.dxdy;
	let sign = edge.direction;
	
	if high_point.0 > low_point.0 {
		::std::mem::swap(&mut (low_point.0), &mut (high_point.0));
		// The tangent also flips.
		dxdy = -dxdy;
	}
	
	debug_assert!(high_point.0 <= low_point.0);
	
	let start_pixel = high_point.0.floor().max(0.0);
	let mut start_pixel_idx = start_pixel as usize;
	let end_pixel = low_point.0.ceil();
	let height = low_point.1 - high_point.1;
	
	debug_assert!(height <= 1.0 && height >= 0.0);
    
	// Spans a single pixel and is trapezoid
	if end_pixel - start_pixel <= 1.00 {
		// We use start_pixel + 1.0 instead of end_pixel to handle vertical edges properly
		let area = height * ((start_pixel + 1.0 - low_point.0) + (start_pixel + 1.0 - high_point.0)) / 2.0;  
		
		scanline[start_pixel_idx].0 += sign * area;

		start_pixel_idx += 1;
		// It induces rectangles in all the other pixels on the right hence we put the height
		// times one (the width of the rectangles) in the next entry of the commulative sum 
		// array (second component of the scanline).
		if start_pixel_idx < scanline.len() {
			scanline[start_pixel_idx].1 += sign * height;
		}
	}
	else {
		// We need to find where the edge intersects the vertical pixel barier of the first
		// pixel containing it.
		
		let width = start_pixel + 1.0 - high_point.0;
		let dydx = 1.0 / dxdy;
		
		debug_assert!(dydx.is_finite());
		
		let intersection_1 = high_point.1 + width * dydx;
		let mut height = intersection_1 - high_point.1; 
		
		let area =  width * height / 2.0;
		
		debug_assert!(area >= 0.0);
		
		scanline[start_pixel_idx].0 += sign * area;
		start_pixel_idx += 1;
		
		let mut end_pixel_idx = (end_pixel - 1.0).round() as usize;
		
		debug_assert!(end_pixel_idx < scanline.len());

		while start_pixel_idx < end_pixel_idx {

			let area = (height + height + dydx) / 2.0;
			
			scanline[start_pixel_idx].0 += sign * area;
			
			height += dydx;
			start_pixel_idx += 1;
		}
		
		// This is a combination of a trapezoid and rectangle
		let end_width_rect = end_pixel - low_point.0;
		let end_width_trap = 1.0 - end_width_rect;
		let end_height = height + end_width_trap * dydx;
		let end_pixel_area = (height + end_height) / 2.0 * end_width_trap + end_height * end_width_rect;
		
		scanline[end_pixel_idx].0 += sign * end_pixel_area;

		end_pixel_idx += 1;
		
		if end_pixel_idx < scanline.len() {
			// All the remaining pixels are ocluded by rectangles with width 1 and height
			// end_height.
			scanline[end_pixel_idx].1 += sign * end_height;
		}
	}

}

fn draw_scanline(surface: &mut GrayScaleSurface, scanline: &Vec<(f32, f32)>, scanline_idx: u32) {
	let mut commulative_sum: f32 = 0.0;
	for i in 0..surface.width {
		commulative_sum += scanline[i as usize].1;
		
		let value = ((commulative_sum + scanline[i as usize].0) * 255.0).min(255.0).max(0.0);
		
		surface.data[surface.width as usize * scanline_idx as usize + i as usize] = 255 - value as u8;
	}
}


fn prepare_active_edges_for_next_scanline(active_edges: &mut Vec<ActiveEdge>) {
	for i in 0..active_edges.len() {
		// Find the value of x 1 unit further by multiplying by the slope (derivative of x with 
		// respect to y).
		active_edges[i].scanline_top_intersection_0 += active_edges[i].dxdy;
	}
}
