use ezttf::rasterizer::FontRasterizer;
use ezttf::rasterizer::GrayScaleSurface;

extern crate png;
extern crate ezttf;

use std::path::Path;
use std::fs::File;
use std::io::BufWriter;

use png::HasParameters;

const FONT : &'static str = "/usr/share/fonts/truetype/freefont/FreeSerif.ttf";
const LETTER: char = 'ℋ';

#[test]
fn test_0() {
    let mut rasterizer = FontRasterizer::new(FONT, false).unwrap();
    let surface = rasterizer.rasterize_glyph(LETTER, 500);
    
    let path = Path::new(r"./target/image.png");
    let file = File::create(path).unwrap();
    let ref mut bw = BufWriter::new(file);
    
    let mut encoder = png::Encoder::new(bw, surface.width, surface.height); 
    encoder.set(png::ColorType::Grayscale).set(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();

    writer.write_image_data(&surface.data).unwrap();
}
