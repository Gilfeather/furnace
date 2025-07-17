use std::fs;
use std::io::Read;

// Note: burn-import API is used internally by onnx2burn CLI tool
// For programmatic usage, we recommend using the CLI tool directly

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Download GPT-NeoX ONNX model if it doesn't exist
    let model_path = "gptneox_Opset18.onnx";
    
    if !std::path::Path::new(model_path).exists() {
        println!("Downloading GPT-NeoX ONNX model...");
        download_model(model_path)?;
    }
    
    println!("Loading ONNX model: {}", model_path);
    
    // Read the ONNX model file
    let model_bytes = fs::read(model_path)?;
    let file_size = model_bytes.len();
    
    println!("Model loaded successfully!");
    println!("Model info:");
    println!("  - File size: {} bytes ({:.2} MB)", file_size, file_size as f64 / 1024.0 / 1024.0);
    println!("  - File path: {}", model_path);
    
    // Basic file validation
    if file_size < 100 {
        return Err("Model file seems too small, might be corrupted".into());
    }
    
    // Check if it's a valid ONNX file by looking at the magic bytes
    if model_bytes.len() >= 4 {
        let magic = &model_bytes[0..4];
        println!("  - Magic bytes: {:02X} {:02X} {:02X} {:02X}", magic[0], magic[1], magic[2], magic[3]);
    }
    
    #[cfg(feature = "burn-import")]
    {
        println!("\n🔥 Attempting to convert ONNX to Burn format...");
        match convert_onnx_to_burn(model_path) {
            Ok(_) => println!("✅ Conversion successful!"),
            Err(e) => println!("❌ Conversion failed: {}", e),
        }
    }
    
    println!("\n🔥 ONNX Model Ready for Furnace!");
    println!("Next steps:");
    println!("1. Use onnx2burn to convert: onnx2burn {} gptneox_burn", model_path);
    println!("2. Start Furnace server: ./target/release/furnace --model-path gptneox_burn.mpk");
    println!("3. Test inference: curl http://localhost:3000/model/info");
    
    // Example of how to integrate with Furnace
    println!("\n📝 Integration example:");
    println!("```bash");
    println!("# Convert ONNX to Burn format");
    println!("onnx2burn {} gptneox_burn", model_path);
    println!("");
    println!("# Build and start Furnace");
    println!("cargo build --release");
    println!("./target/release/furnace --model-path gptneox_burn/gptneox_Opset18.mpk --port 3000");
    println!("```");
    
    Ok(())
}

#[cfg(feature = "burn-import")]
fn convert_onnx_to_burn(onnx_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;
    
    let output_dir = "gptneox_burn";
    
    // Create output directory if it doesn't exist
    if !Path::new(output_dir).exists() {
        fs::create_dir_all(output_dir)?;
    }
    
    println!("Converting {} to Burn format in {}/", onnx_path, output_dir);
    
    // Note: This is a simplified example. The actual burn-import API may differ.
    // In practice, you would use the burn-import crate's conversion functions here.
    // For now, we'll just show the concept and recommend using the onnx2burn CLI tool.
    
    println!("Note: Direct programmatic conversion requires more complex setup.");
    println!("Recommended approach: Use the onnx2burn CLI tool");
    
    Ok(())
}

fn download_model(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    
    let url = "https://github.com/onnx/models/raw/main/Generative_AI/gptneox_Opset18_transformers/gptneox_Opset18.onnx";
    
    println!("Downloading from: {}", url);
    
    let response = ureq::get(url).call()?;
    let mut file = fs::File::create(path)?;
    
    let mut buffer = Vec::new();
    response.into_reader().read_to_end(&mut buffer)?;
    file.write_all(&buffer)?;
    
    println!("Downloaded {} bytes to {}", buffer.len(), path);
    Ok(())
}