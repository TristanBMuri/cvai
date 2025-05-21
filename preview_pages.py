import fitz  # PyMuPDF
from pathlib import Path
import os

def preview_pdf(pdf_path):
    """Show first and last 3 pages of a PDF and return selected pages to remove."""
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    print(f"\nProcessing: {pdf_path.name}")
    print(f"Total pages: {total_pages}")
    
    # Create preview directory if it doesn't exist
    preview_dir = Path("preview_pages")
    preview_dir.mkdir(exist_ok=True)
    
    # Create subdirectory for this PDF
    pdf_preview_dir = preview_dir / pdf_path.stem
    pdf_preview_dir.mkdir(exist_ok=True)
    
    # Save first 3 pages
    for i in range(min(3, total_pages)):
        page = doc[i]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        pix.save(pdf_preview_dir / f"page_{i+1}.png")
        print(f"Saved page {i+1} as {pdf_preview_dir / f'page_{i+1}.png'}")
    
    # Save last 3 pages
    for i in range(max(0, total_pages-3), total_pages):
        page = doc[i]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        pix.save(pdf_preview_dir / f"page_{i+1}.png")
        print(f"Saved page {i+1} as {pdf_preview_dir / f'page_{i+1}.png'}")
    
    doc.close()
    
    # Ask user which pages to remove
    print("\nPlease check the preview pages in the 'preview_pages' directory")
    print("Enter the page numbers you want to remove (e.g., '1,2,3' or 'none' if you want to keep all):")
    pages_to_remove = input().strip()
    
    if pages_to_remove.lower() == 'none':
        return []
    
    try:
        return [int(x.strip()) for x in pages_to_remove.split(',')]
    except ValueError:
        print("Invalid input. Please enter numbers separated by commas or 'none'.")
        return []

def main():
    pdf_files = list(Path(".").glob("*.pdf"))
    pdf_files.sort()  # Sort alphabetically
    
    for pdf_path in pdf_files:
        pages_to_remove = preview_pdf(pdf_path)
        if pages_to_remove:
            print(f"Pages to remove from {pdf_path.name}: {pages_to_remove}")
        else:
            print(f"No pages to remove from {pdf_path.name}")
        
        # Ask if user wants to continue to next PDF
        if pdf_path != pdf_files[-1]:
            print("\nPress Enter to continue to next PDF, or type 'exit' to stop:")
            if input().strip().lower() == 'exit':
                break

if __name__ == "__main__":
    main() 