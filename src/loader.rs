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

use super::error::Error;
use std::io::Read;

fn read_struct<T>(buffer: &Vec<u8>, offset: usize) -> T {
    unsafe {
        let mut s: T = ::std::mem::uninitialized();
        let src_ptr: *const T = ::std::mem::transmute(buffer.as_ptr().offset(offset as isize));
        ::std::ptr::copy(src_ptr, &mut s as *mut T, 1);
        s
    }
}

// Unfortunately no template specialization in Rust :(
fn read_u16_be(buffer: &Vec<u8>, offset: usize) -> u16 {
    (buffer[offset] as u16) << 8 | buffer[offset + 1] as u16
}

fn read_i16_be(buffer: &Vec<u8>, offset: usize) -> i16 {
    let result = read_u16_be(buffer, offset);
    unsafe { *::std::mem::transmute::<*const u16,*const i16>(&result as *const u16) }
}

fn read_u32_be(buffer: &Vec<u8>, offset: usize) -> u32 {
    (buffer[offset] as u32) << 24
    |
    (buffer[offset + 1] as u32) << 16
    |
    (buffer[offset + 2] as u32) << 8
    |
    (buffer[offset + 3] as u32)
}

fn read_i32_be(buffer: &Vec<u8>, offset: usize) -> i32 {
    let result = read_u32_be(buffer, offset);
    unsafe { *::std::mem::transmute::<*const u32,*const i32>(&result as *const u32) }
}

pub fn create_vec_with_size_uninitialized<T>(size: usize) -> Vec<T> {
    unsafe {
        let mut tmp = Vec::<T>::new();
        tmp.reserve_exact(size);

        let result = Vec::<T>::from_raw_parts(tmp.as_mut_ptr(), size, size);

        ::std::mem::forget(tmp);

        result
    }
}

fn read_array<T>(buffer: &Vec<u8>, offset: usize, size: usize) -> Vec<T> {
    unsafe {
        let mut result = create_vec_with_size_uninitialized(size);

        let src_ptr: *const T = ::std::mem::transmute(buffer.as_ptr().offset(offset as isize));
        ::std::ptr::copy(src_ptr, result.as_mut_ptr(), size);

        result
    }
}

#[repr(C)]
struct OffsetTable {
    version: u32,
    num_tables: u16,
    search_range: u16,
    entry_selector: u16,
    range_shift: u16,
}

#[repr(C)]
struct TableDirectoryEntry {
    tag: u32,
    checksum: u32,
    offset: u32,
    length: u32,
}

enum FontVersion {
    AppleTTF,
    OpenType10,
    OpenTypeCCF,
    OldPostScript,
    Unsupported,
}

fn check_font_version(v: u32) -> FontVersion {
    if v == u32::from_le(0x00000100) {
        FontVersion::OpenType10
    } else if v == u32::from_le(0x65757274) {
        FontVersion::AppleTTF
    } else if v == 0x4F54544F {
        FontVersion::OpenTypeCCF
    } else if v == 0x31707974 {
        FontVersion::OldPostScript
    } else {
        FontVersion::Unsupported
    }
}

#[derive(Default)]
struct Location {
    offset: u32,
    length: u32,
}

#[derive(Default)]
pub struct FontData {
    // The contents of the .ttf file;
    data: Vec<u8>,

    table_count: u32,
    head_table: Location,
    loca_table: Location,
    hhea_table: Location,
    cmap_table: Location,
    maxp_table: Location,
    glyf_table: Location,
    hmtx_table: Location,
    kern_table: Location,
    name_table: Location,
    cff_table: Location,

    index_map_offset: u32,
    char_encoding_format: u32,
    long_loca_index: bool,

    pub units_per_em: u16,
    // The bounding box
    pub x_min: i16,
    pub y_min: i16,
    pub x_max: i16,
    pub y_max: i16,
    
    pub number_of_long_horizontal_metrics: u16,
    pub ascent: i16,
    pub descent: i16,
    pub line_gap: i16,
	pub advance_width_max: u16,
}

impl FontData {
    pub fn load(path: &str) -> Result<FontData, Error> {
        let mut font_data = FontData::default();

        let file = ::std::fs::File::open(path);

        if file.is_err() {
            Err(Error::FileReadError)
        } 
        else {
            let read_result = file.unwrap().read_to_end(&mut font_data.data);

            if read_result.is_err() {
                Err(Error::FileReadError)
            } 
            else {
                let parse_status = font_data.parse_contents();
                
                if parse_status == Error::Success {
                    Ok(font_data)
                } 
                else {
                    Err(parse_status)
                }
            }
        }

    }

    pub fn load_direct(data: Vec<u8>) -> Result<FontData, Error> {
        let mut font_data = FontData::default();

        font_data.data = data;

        let parse_status = font_data.parse_contents();

        if parse_status == Error::Success {
            Ok(font_data)
        } 
        else {
            Err(parse_status)
        }
    }

    fn parse_contents(&mut self) -> Error {
        let offset_table: OffsetTable = read_struct(&self.data, 0);

        let version = check_font_version(offset_table.version);

        self.table_count = u16::from_be(offset_table.num_tables) as u32;

        match version {
            FontVersion::OpenType10 |
            FontVersion::AppleTTF => self.parse_tt_outlines_font(),
            FontVersion::OpenTypeCCF => self.parse_cff_outlines_font(),
            _ => Error::UnsupportedFormat,
        }
    }
}

// These should work on little endian machine hence we need to
// use from_le() in order to abstract endianess.
const CFF_TAG_LE: u32 = 0x20464643;
const GLYF_TAG_LE: u32 = 0x66796C67;
const NAME_TAG_LE: u32 = 0x656D616E;
const LOCA_TAG_LE: u32 = 0x61636F6C;
const MAXP_TAG_LE: u32 = 0x7078616D;
const CMAP_TAG_LE: u32 = 0x70616D63;
const HEAD_TAG_LE: u32 = 0x64616568;
const HHEA_TAG_LE: u32 = 0x61656868;
const HMTX_TAG_LE: u32 = 0x78746D68;


impl FontData {
    fn parse_tt_outlines_font(&mut self) -> Error {
        let status = self.parse_ttf_contained_font();

        if status != Error::Success {
            return status;
        }

        self.glyf_table = self.find_table(u32::from_le(GLYF_TAG_LE));
        self.loca_table = self.find_table(u32::from_le(LOCA_TAG_LE));

        if self.glyf_table.offset == 0 {
            return Error::NoGlyfTable;
        }

        if self.loca_table.offset == 0 {
            return Error::NoLocaTable;
        }

        Error::Success
    }

    fn parse_cff_outlines_font(&mut self) -> Error {
        let status = self.parse_ttf_contained_font();

        if status != Error::Success {
            return status;
        }

        // TODO Add CFF outlines support.
        Error::UnsupportedFormat
    }

    fn find_table(&self, tag: u32) -> Location {
        // Unfortunately making those consts doesnt work on 1.18 because apparently size_of is not
        // compile time computed
        let init_offset: u32 = ::std::mem::size_of::<OffsetTable>() as u32;
        let stride: u32 = ::std::mem::size_of::<TableDirectoryEntry>() as u32;

        for k in 0..self.table_count {
            let current_offset: u32 = init_offset + k * stride;

            if read_struct::<u32>(&self.data, current_offset as usize) == tag {
                // TODO: provide support for computing table checksums
                let tde: TableDirectoryEntry = read_struct(&self.data, current_offset as usize);

                return Location {
                    offset: u32::from_be(tde.offset),
                    length: u32::from_be(tde.length),
                };
            }
        }

        Location {
            offset: 0,
            length: 0,
        }
    }

    fn parse_ttf_contained_font(&mut self) -> Error {
        macro_rules! get_required_tables
        {
            ($(($name:ident, $tag:ident,$err:path)),*) =>
            {
                $(
                    self.$name = self.find_table(u32::from_le($tag));
                    
                    if self.$name.offset == 0
                    {
                        return $err;
                    }
                 )*
            }
        }

        get_required_tables!((cmap_table, CMAP_TAG_LE, Error::NoCmapTable),
                             (name_table, NAME_TAG_LE, Error::NoNameTable),
                             (maxp_table, MAXP_TAG_LE, Error::NoMaxpTable),
                             (hhea_table, HHEA_TAG_LE, Error::NoHheaTable),
                             (head_table, HEAD_TAG_LE, Error::NoHeadTable),
                             (hmtx_table, HMTX_TAG_LE, Error::NoHmtxTable)
                             );

        let status = self.get_idx_data_table_from_cmap();

        if status != Error::Success {
            return status;
        }

        let status = self.fetch_global_info_from_head();
        
        if status != Error::Success {
			return status;
		}
		
		self.fetch_global_info_from_hhea()
    }

    fn fetch_global_info_from_head(&mut self) -> Error {
        #[repr(C)]
        struct HeadTable1 {
            version: u32,
            font_revision: u32,
            checksum_adjustment: u32,
            magic_number: u32,
            flags: u16,
            units_per_em: u16
        };
        
        struct HeadTable2 {
            created: i64,
            modified: i64,
            x_min: i16,
            y_min: i16,
            x_max: i16,
            y_max: i16,
            mac_style: u16,
            unused: u16,
            font_direction_hint: i16,
            index_to_loca_format: i16,
            glyph_data_format: i16,
        };

        let head_table1: HeadTable1 = read_struct(&self.data, self.head_table.offset as usize);
        let h1 = ::std::mem::size_of::<HeadTable1>();
        let head_table2: HeadTable2 = read_struct(&self.data, self.head_table.offset as usize + h1);
        
        if u32::from_be(head_table1.version) != 0x00010000 {
            Error::UnsupportedLocaTableVersion
        } 
        else {
            self.x_min = i16::from_be(head_table2.x_min);
            self.y_min = i16::from_be(head_table2.y_min);
            self.x_max = i16::from_be(head_table2.x_max);
            self.y_max = i16::from_be(head_table2.y_max);
            self.units_per_em = u16::from_be(head_table1.units_per_em);

            match i16::from_be(head_table2.index_to_loca_format) {
                0 => {
                    self.long_loca_index = false;
                    Error::Success
                }

                1 => {
                    self.long_loca_index = true;
                    Error::Success
                }

                _ => Error::UnsupportedLocaTableIndex,
            }
        }
    }
    
    fn fetch_global_info_from_hhea(&mut self) -> Error {
		#[repr(C)]
		struct HheaTable {
			version: u32,
			ascent: i16,
			descent: i16,
			line_gap: i16,
			advance_width_max: u16,
			min_left_side_bearing: i16,
			min_right_side_bearing: i16,
			// "The extent is the distance from the left side bearing to the right most positions 
			// in the glyph outline." 
			//    - https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6hhea.html
			x_max_extent: i16,
			caret_slope_rise: i16,
			caret_solo_run: i16,
			caret_offset: i16,
			reserved0: i16,
			reserved1: i16,
			reserved2: i16,
			reserved3: i16,
			metric_data_format: i16,
			number_of_long_horizontal_metrics: u16,
		};
		
		
		let hhea_table: HheaTable = read_struct(&self.data, self.hhea_table.offset as usize);
		
		if u32::from_be(hhea_table.version) != 0x00010000 {
			return Error::UnsupportedHheaTableVersion;
		}
		
		self.ascent = i16::from_be(hhea_table.ascent);
		self.descent = i16::from_be(hhea_table.descent);
		self.line_gap = i16::from_be(hhea_table.line_gap);
		self.advance_width_max = u16::from_be(hhea_table.advance_width_max);
		self.number_of_long_horizontal_metrics = u16::from_be(self.number_of_long_horizontal_metrics);
		
		Error::Success
	}
}

const PLATFORM_ID_UNICODE: u16 = 0;
const PLATFORM_ID_MICROSOFT: u16 = 3;

const PLATFORM_SPECIFIC_ID_MS_UCS2: u16 = 1;
const PLATFORM_SPECIFIC_ID_MS_UCS4: u16 = 10;

impl FontData {
    fn get_idx_data_table_from_cmap(&mut self) -> Error {
        
        #[repr(C)]
        struct CmapTableHeader {
            version: u16,
            subtable_count: u16,
        }

        #[repr(C)]
        struct CmapSubtable {
            platform_id: u16,
            platform_specific_id: u16,
            offset: u32,
        }

        let mut cmap_header: CmapTableHeader =
            read_struct(&self.data, self.cmap_table.offset as usize);

        if cmap_header.version != 0 {
            return Error::UnsupportedFormat;
        }

        cmap_header.subtable_count = u16::from_be(cmap_header.subtable_count);

        self.index_map_offset = 0;

        for k in 0..cmap_header.subtable_count {
            use std::mem::size_of;
            let offset = self.cmap_table.offset 
                         + size_of::<CmapTableHeader>() as u32
                         + k as u32 * size_of::<CmapSubtable>() as u32;
                
            let mut subtable: CmapSubtable = read_struct(&self.data, offset as usize);
            
            subtable.platform_id = u16::from_be(subtable.platform_id);
            subtable.platform_specific_id = u16::from_be(subtable.platform_specific_id);
            
            // We support only unicode encodings.
            
            if subtable.platform_id == PLATFORM_ID_UNICODE {
                self.index_map_offset = self.cmap_table.offset + u32::from_be(subtable.offset);
                break;
            }
            
            if subtable.platform_id == PLATFORM_ID_MICROSOFT &&
               (
                subtable.platform_specific_id == PLATFORM_SPECIFIC_ID_MS_UCS2 
                || 
                subtable.platform_specific_id == PLATFORM_SPECIFIC_ID_MS_UCS4
               ) {
                self.index_map_offset = self.cmap_table.offset + u32::from_be(subtable.offset);
            }
        }

        if self.index_map_offset == 0 {
            return Error::UnsupportedCharEncoding;
        }

        // We only support format 4, 6 and 12
        self.char_encoding_format = read_u16_be(&self.data, self.index_map_offset as usize) as u32;

        if self.char_encoding_format != 4 && self.char_encoding_format != 6 &&
            self.char_encoding_format != 12
        {
            return Error::UnsupportedFormat;
        }

        self.index_map_offset += 2;
        Error::Success
    }

    pub fn get_char_index(&self, codepoint: char) -> u32 {
        match self.char_encoding_format {
            4 => self.get_char_index_fmt4(codepoint),
            6 => self.get_char_index_fmt6(codepoint),
            12 => self.get_char_index_fmt12(codepoint),
            _ => 0, // This is impossible since we already checked for those formats
        }
    }

    fn get_char_index_fmt4(&self, codepoint: char) -> u32 {
        #[repr(C)]
        struct Table4 {
            length: u16, // Length of the whole table in bytes
            unused: u16, // Language code that's only used by Macintosh platform encodings.
            seg_count_x2: u16, // Segment count times 2
            search_range: u16, // The largest power of 2 less than or equal to segment count.
            entry_selector: u16, // The number of iterations we have to make
            range_shift: u16, // The remaining segments times 2 after search_count
        };
        
        let mut t4: Table4 = read_struct(&self.data, self.index_map_offset as usize);
        t4.seg_count_x2 = u16::from_be(t4.seg_count_x2);
        t4.search_range = u16::from_be(t4.search_range);
        t4.entry_selector = u16::from_be(t4.entry_selector);
        t4.range_shift = u16::from_be(t4.range_shift);

        let data_beginning = self.index_map_offset + ::std::mem::size_of::<Table4>() as u32;
        
        let mut search_offset = data_beginning;

        // We will test against 16 bit values only using this format
        let codepoint16 = codepoint as u16;

        // Binary search can be performed since segments are sorted by end codepoint

        if codepoint16 >= read_u16_be(&self.data, search_offset as usize + t4.range_shift as usize)
        {
            search_offset += t4.range_shift as u32;
        }
        
        // 
        search_offset -=2;
        
        for _ in 0..t4.entry_selector {
            t4.search_range /= 2;
            let end_codepoint: u16 = read_u16_be(
                                                  &self.data,
                                                  search_offset as usize + t4.search_range as usize
                                                );

            if codepoint16 > end_codepoint {
                search_offset += t4.search_range as u32;
            }
        }
        
        search_offset +=2;
        
        // Now the search_offset should be what we need.
        let segment = (search_offset - data_beginning) / 2;

        // Two bytes pad after the end codes
        let start_codes_offset = data_beginning + t4.seg_count_x2 as u32 +2;
        let deltas_offset = start_codes_offset + t4.seg_count_x2 as u32;
        let ranges_offset = deltas_offset + t4.seg_count_x2 as u32;

        let segment_start_code = read_u16_be(&self.data, (start_codes_offset + 2 * segment) as usize);
        let segment_range_offset = read_u16_be(&self.data, (ranges_offset + 2 * segment) as usize);
        let segment_delta_offset = read_u16_be(&self.data, (deltas_offset + 2 * segment ) as usize);

        if codepoint16 < segment_start_code {
            0
        } 
        else if segment_range_offset == 0 {

            (codepoint16.wrapping_add(segment_delta_offset)) as u32
        } 
        else {
            // According to the specification we need to use this obscure indexing trick
            let mut glyph_index_offset = segment_range_offset as u32;
            glyph_index_offset +=  2 * (codepoint16 - segment_start_code) as u32;
            glyph_index_offset += ranges_offset + 2 * segment as u32;
            
            let glyph_index = read_u16_be(&self.data, glyph_index_offset as usize);

            if glyph_index != 0 {
                (glyph_index.wrapping_add(segment_delta_offset)) as u32
            } 
            else {
                glyph_index as u32
            }
        }
    }

    fn get_char_index_fmt6(&self, codepoint: char) -> u32 {
        // Skip the first two entries.
        let mut data_beginning = self.index_map_offset + 4;
        
        let first_code = read_u16_be(&self.data, data_beginning as usize);
        data_beginning += 2;
        let code_count = read_u16_be(&self.data, data_beginning as usize);
        data_beginning += 2;
        
        if codepoint as u16 >= first_code + code_count {
            0
        } 
        else {
            let index_offset = data_beginning as usize + (codepoint as u16 - first_code) as usize;
            read_u16_be(&self.data, index_offset) as u32
        }
        
    }

    fn get_char_index_fmt12(&self, codepoint: char) -> u32 {
        #[repr(C)]
        struct Header {
            format: u32,
            length: u32,
            unused: u32,
            group_count: u32,
        };
        
        #[repr(C)]
        struct Group {
            start_codepoint: u32,
            end_codepoint: u32,
            start_codepoint_idx: u32,
        };
        let mut header = read_struct::<Header>(&self.data, self.index_map_offset as usize - 2);

        header.group_count = u32::from_be(header.group_count);

        let groups_offset = self.index_map_offset - 2 + ::std::mem::size_of::<Header>() as u32;

        let mut search_start: u32 = 0;
        let mut search_end: u32 = header.group_count;

        let group_size: u32 = ::std::mem::size_of::<Group>() as u32;

        while search_start < search_end {
            let mid = search_start + (search_end - search_end) / 2;

            let mut current_segment =
                read_struct::<Group>(&self.data, (groups_offset + mid * group_size) as usize);

            current_segment.start_codepoint = u32::from_be(current_segment.start_codepoint);
            current_segment.end_codepoint = u32::from_be(current_segment.end_codepoint);

            if current_segment.start_codepoint > codepoint as u32 {
                search_end = mid;
            } 
            else if current_segment.end_codepoint < codepoint as u32 {
                search_start = mid + 1;
            } 
            else {
                return u32::from_be(current_segment.start_codepoint_idx) + mid;
            }
        }

        0
    }

    fn get_glyph_offset(&self, glyph_index: u32) -> u32 {
        self.glyf_table.offset +
            if self.long_loca_index {
                read_u32_be(&self.data,(self.loca_table.offset + 4 * glyph_index) as usize)
            } 
            else {
                read_u16_be(&self.data,(self.loca_table.offset + 2 * glyph_index) as usize) as u32 * 2
            }
    }
}

pub type TTFScalar = i16;
pub type TTFPoint = (TTFScalar, TTFScalar);

#[derive(Default)]
pub struct QuadraticBezierCurve {
    pub start_point: TTFPoint,
    pub control_point: TTFPoint,
    pub end_point: TTFPoint,
}

#[derive(Default)]
pub struct Line {
    pub start_point: TTFPoint,
    pub end_point: TTFPoint,
}

use std::convert::Into;

impl Line {
    fn new<T: Into<TTFScalar>>(start_point: (T, T), end_point: (T, T)) -> Line {
        Line {
            start_point: (start_point.0.into(), start_point.1.into()),
            end_point: (end_point.0.into(), end_point.1.into()),
        }
    }
}


impl QuadraticBezierCurve {
    fn new<T: Into<TTFScalar>>(
        start_point: (T, T),
        control_point: (T, T),
        end_point: (T, T),
    ) -> QuadraticBezierCurve {
        QuadraticBezierCurve {
            start_point: (start_point.0.into(), start_point.1.into()),
            control_point: (control_point.0.into(), control_point.1.into()),
            end_point: (end_point.0.into(), end_point.1.into()),
        }
    }
}

pub enum TTFCurve {
    QuadraticBezierCurve(QuadraticBezierCurve),
    Line(Line),
}

#[derive(Default)]
pub struct GlyphData {
    pub components: Vec<TTFCurve>,
    pub bounding_box_diagonal: Line,
}

impl FontData {
    fn fetch_glyph_data(&self, glyph_index: u32) -> GlyphData {
        let mut glyph_data = GlyphData::default();

        let glyph_offset = self.get_glyph_offset(glyph_index);

        #[repr(C)]
        struct GlyfHeader {
            number_of_contours: i16,
            x_min: i16,
            y_min: i16,
            x_max: i16,
            y_max: i16,
        };

        let glyf_header: GlyfHeader = read_struct(&self.data, glyph_offset as usize);
        
        let d0 = (
            i16::from_be(glyf_header.x_min),
            i16::from_be(glyf_header.y_min),
        );
        let d1 = (
            i16::from_be(glyf_header.x_max),
            i16::from_be(glyf_header.y_max),
        );
        glyph_data.bounding_box_diagonal = Line::new(d0, d1);

        let mut current_offset = glyph_offset + ::std::mem::size_of::<GlyfHeader>() as u32;

        let number_of_contours = i16::from_be(glyf_header.number_of_contours);

        if number_of_contours >= 0
        // simple glyph
        {
            let mut end_points_of_contours = read_array::<u16>(
                &self.data,
                current_offset as usize,
                number_of_contours as usize,
            );
            end_points_of_contours.iter_mut().for_each(
                |x| *x = u16::from_be(*x),
            );

            current_offset += 2 * number_of_contours as u32;

            let instruction_length: u16 = read_u16_be(&self.data, current_offset as usize);

            // Skip instructions;
            current_offset += instruction_length as u32 + 2;

            let number_of_vertices = end_points_of_contours[(number_of_contours - 1) as usize] + 1;

            let mut flags = Vec::<u8>::with_capacity(number_of_vertices as usize);
            let mut repeat_flags: u8 = 0;
            let mut flag: u8 = 0;
            
            // Loading the flags first storing the repeating ones as well.
            for _ in 0..number_of_vertices {
                if repeat_flags == 0 {
                    flag = self.data[current_offset as usize];
                    current_offset += 1;

                    if flag & 0b00001000 > 0 {
                        repeat_flags = self.data[current_offset as usize];
                        current_offset += 1;
                    }
                } else {
                    repeat_flags -= 1;
                }

                flags.push(flag);
            }
            
            debug_assert!(repeat_flags == 0);
            
            let mut vertices: Vec<TTFPoint> =
                create_vec_with_size_uninitialized(number_of_vertices as usize);

            let mut prev_coord: i32 = 0;
            // Reading the first coordinates
            for i in 0..number_of_vertices as usize {
                 // coordinate is encoded as u8
                if flags[i] & 0b00000010u8 > 0 {
                    let tmp_coord = self.data[current_offset as usize];
                    current_offset += 1;
                    
                    // Holy shit!!! The documentation was so misleading here. I had so much trouble 
                    // finding this bug...
                    vertices[i].0 = (prev_coord as i32 + if flags[i] & 0b00010000u8 > 0 {
                        tmp_coord as i32
                    } 
                    else {
                        -(tmp_coord as i32)
                    }) as TTFScalar;
                    prev_coord = vertices[i].0 as i32;
                } 
                // coordinate is encoded as i16
                else { 
                    // checks if the coordinate is not repeated
                    if flags[i] & 0b00010000u8 == 0 {
                        let tmp_coord = read_i16_be(&self.data, current_offset as usize);
                        
                        current_offset += 2;
                        vertices[i].0 = (prev_coord + tmp_coord as i32) as TTFScalar;
                        prev_coord = vertices[i].0 as i32;
                    } 
                    else {
                        vertices[i].0 = prev_coord as TTFScalar;
                    }
                }
            }

            prev_coord = 0;
            // Reading the second coordinates
            for i in 0..number_of_vertices as usize {
                // coordinate is encoded as u8
                if flags[i] & 0b00000100u8 > 0 {
                    let tmp_coord = self.data[current_offset as usize];
                    current_offset += 1;
                    
                    // Holy mother of fuck why nobody mentioned that those were deltas as well?!?
                    // I'm looking at you 
                    // developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6glyf.html
                    vertices[i].1 = (prev_coord as i32 + if flags[i] & 0b00100000u8 > 0 {
                        tmp_coord as i32
                    } 
                    else {
                        -(tmp_coord as i32)
                    }) as TTFScalar;
                    prev_coord = vertices[i].1 as i32;
                } 
                // coordinate is encoded as i16
                else {
                    // checks if the coordinate is not repeated
                    if flags[i] & 0b00100000u8 == 0 {
                        let tmp_coord = read_i16_be(&self.data, current_offset as usize);
                        
                        current_offset += 2;
                        vertices[i].1 = (prev_coord + tmp_coord as i32) as TTFScalar;
                        prev_coord = vertices[i].1 as i32;
                    } 
                    else {
                        vertices[i].1 = prev_coord as TTFScalar;
                    }
                }
            }

            let mut start_index: usize = 0;

            for i in 0..end_points_of_contours.len() {
                start_index = FontData::load_contour(
                    &mut glyph_data,
                    &vertices,
                    &flags,
                    start_index,
                    end_points_of_contours[i] as usize,
                );
            }
        } else
        // compound glyph
        {
            // TODO: implement support for compound glyphs
        }

        glyph_data
    }

    fn load_contour(
        data: &mut GlyphData,
        vertices: &Vec<TTFPoint>,
        flags: &Vec<u8>,
        sidx: usize,
        eidx: usize,
    ) -> usize {
        let mut cidx = sidx;

        if flags[cidx] & 1u8 == 0
        // The first point is control point
        {
            let mut curve = QuadraticBezierCurve::default();

            if flags[eidx] & 1u8 == 0 {
                curve.start_point.0 = ((vertices[cidx].0 as i32 + vertices[eidx].0 as i32) / 2) as
                    TTFScalar;
                curve.start_point.1 = ((vertices[cidx].1 as i32 + vertices[eidx].1 as i32) / 2) as
                    TTFScalar;
            } 
            else {
                curve.start_point = vertices[eidx];
            }

            curve.control_point = vertices[cidx];
            // No bound checking at this level.
            cidx += 1;

            if flags[cidx] & 1u8 == 0 {
                curve.end_point.0 = ((curve.control_point.0 as i32 + vertices[cidx].0 as i32) /
                                         2) as TTFScalar;
                curve.end_point.1 = ((curve.control_point.1 as i32 + vertices[cidx].1 as i32) /
                                         2) as TTFScalar;
            } 
            else {
                curve.end_point = vertices[cidx];
            }
            
            data.components.push(TTFCurve::QuadraticBezierCurve(curve));
        }

        while cidx < eidx {
            // At least the first point has been added already so we can safely check the
            // previous point.

            if flags[cidx] & 1u8 == 0 {
                // The previous point must be a control point because else we would've
                // parsed the whole curve in the previous iteration
                let start_point =
                    (
                        ((vertices[cidx - 1].0 as i32 + vertices[cidx].0 as i32) / 2) as TTFScalar,
                        ((vertices[cidx - 1].1 as i32 + vertices[cidx].1 as i32) / 2) as TTFScalar,
                    );

                if flags[cidx + 1] & 1u8 == 0 {
                    let end_point =
                        (
                            ((vertices[cidx].0 as i32 + vertices[cidx + 1].0 as i32) / 2) as TTFScalar,
                            ((vertices[cidx].1 as i32 + vertices[cidx + 1].1 as i32) / 2) as TTFScalar,
                        );

                    let curve = QuadraticBezierCurve::new(start_point, vertices[cidx], end_point);
                    data.components.push(TTFCurve::QuadraticBezierCurve(curve));
                } 
                else {
                    let curve =
                        QuadraticBezierCurve::new(start_point, vertices[cidx], vertices[cidx + 1]);
                    data.components.push(TTFCurve::QuadraticBezierCurve(curve));
                }
            } 
            else {
                if flags[cidx + 1] & 1u8 == 0 {
                    let mut end_point = TTFPoint::default();

                    if cidx + 1 == eidx {
                        if flags[sidx] & 1u8 == 0 {
                            end_point.0 = ((vertices[sidx].0 as i32 + vertices[eidx].0 as i32) /
                                               2) as
                                TTFScalar;
                            end_point.1 = ((vertices[sidx].1 as i32 + vertices[eidx].1 as i32) /
                                               2) as
                                TTFScalar;
                        } 
                        else {
                            end_point = vertices[sidx];
                        }

                        let curve = QuadraticBezierCurve::new(
                            vertices[cidx],
                            vertices[cidx + 1],
                            end_point,
                        );
                        data.components.push(TTFCurve::QuadraticBezierCurve(curve));
                    } 
                    else {
                        // The next index is not the last so we can use at least one more point.
                        if flags[cidx + 2] & 1u8 == 0 {
                            end_point.0 =
                                ((vertices[cidx + 1].0 as i32 + vertices[cidx + 2].0 as i32) /
                                     2) as TTFScalar;
                            end_point.1 =
                                ((vertices[cidx + 1].1 as i32 + vertices[cidx + 2].1 as i32) /
                                     2) as TTFScalar;
                        } 
                        else {
                            end_point = vertices[cidx + 2];
                        }

                        let curve = QuadraticBezierCurve::new(
                            vertices[cidx],
                            vertices[cidx + 1],
                            end_point,
                        );
                        data.components.push(TTFCurve::QuadraticBezierCurve(curve));
                    }

                    // We used one more point here.
                    cidx += 1;
                } 
                else {
                    data.components.push(TTFCurve::Line(
                        Line::new(vertices[cidx], vertices[cidx + 1]),
                    ));
                }
            }

            cidx += 1;
        }

        if cidx == eidx {
            if flags[eidx] & 1u8 == 0 {
                // The previous point must be a control point because else we would've
                // parsed the whole curve before this part of the code
                let start_point =
                    (
                        ((vertices[eidx - 1].0 as i32 + vertices[eidx].0 as i32) / 2) as TTFScalar,
                        ((vertices[eidx - 1].1 as i32 + vertices[eidx].1 as i32) / 2) as TTFScalar,
                    );

                let mut end_point = TTFPoint::default();

                if flags[sidx] & 1u8 == 0 {
                    end_point.0 = ((vertices[sidx].0 as i32 + vertices[eidx].0 as i32) / 2) as
                        TTFScalar;
                    end_point.1 = ((vertices[sidx].1 as i32 + vertices[eidx].1 as i32) / 2) as
                        TTFScalar;
                } else {
                    end_point = vertices[sidx];
                }

                let curve = QuadraticBezierCurve::new(start_point, vertices[eidx], end_point);
                data.components.push(TTFCurve::QuadraticBezierCurve(curve));
            } 
            else {
                // If the start was control point we have already pushed this curve in the
                // beginning.
                if flags[sidx] & 1u8 > 0 {
                    data.components.push(TTFCurve::Line(
                        Line::new(vertices[eidx], vertices[sidx]),
                    ));
                }
            }
        }

        eidx + 1
    }

    pub fn fetch_glyph_data_for_codepoint(&self, codepoint: char) -> GlyphData {
        self.fetch_glyph_data(self.get_char_index(codepoint))
    }
}

