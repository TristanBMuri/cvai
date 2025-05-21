import fitz  # PyMuPDF
from pathlib import Path
import os

# Dictionary of PDFs and their pages to remove
# Format: {pdf_name: [pages_to_remove]}
# Note: Pages 1,2,3 are first 3 pages, pages 4,5,6 are last 3 pages
pages_to_remove = {
    "03 - Tintin In America.pdf": [2, 3, 6],  # Remove pages 2,3 from start and last page
    "04 - Cigars Of The Pharaoh 1934.pdf": [2, 3, 5, 6],  # Remove pages 2,3 from start and last 2 pages
    "05 - Tintin The Blue Lotus.pdf": [2, 3, 6],  # Remove pages 2,3 from start and last page
    "06 - TinTin The Broken Ear.pdf": [2, 3, 6],  # Remove pages 2,3 from start and last page
    "07 - TinTin The Black Island.pdf": [2],  # Remove page 2 from start
    "08 - TinTin King Ottokars Sceptre.pdf": [2, 3, 6],  # Remove pages 2,3 from start and last page
    "09 - TinTin The Crab with the Golden Claws.pdf": [2, 6],  # Remove page 2 from start and last page
    "10 - Tintin The Shooting Star.pdf": [2, 3, 6],  # Remove pages 2,3 from start and last page
    "11 - Tintin  Secret Of The Unicorn.pdf": [],  # none
    "12 - TinTin Red Backham's Treasure.pdf": [2],  # Remove page 2 from start
    "13 - TinTin  Seven Crystal Balls.pdf": [2, 6],  # Remove page 2 from start and last page
    "14 - TinTin Prisoners of the Sun.pdf": [6],  # Remove last page
    "15 - Tintin  Land of Black Gold.pdf": [2, 3, 4, 5, 6],  # Remove pages 2,3 from start and all last 3 pages
    "16 - TinTin Destination Moon.pdf": [2, 3, 6],  # Remove pages 2,3 from start and last page
    "17 - TinTin Explorers On The  Moon.pdf": [6],  # Remove last page
    "18 - Tintin The Calculus Affair.pdf": [2, 3],  # Remove pages 2,3 from start
    "19 - Tintin Red Sea Sharks.pdf": [2, 6],  # Remove page 2 from start and last page
    "20 - Tintin In Tibet.pdf": [2, 3, 5, 6],  # Remove pages 2,3 from start and last 2 pages
    "21 - Tintin Castafiore Emerald.pdf": [],  # none
    "22 - TinTin  Flight 714.pdf": [],  # none
    "23 - Tintin And The Picaros.pdf": [2, 3, 6],  # Remove pages 2,3 from start and last page
    "24 - Tintin And The Alph-art.pdf": [1, 5, 6]  # Remove first page and last 2 pages
}

def remove_pages(pdf_path, pages_to_remove):
    """Remove specified pages from a PDF and save the result."""
    if not pages_to_remove:
        print(f"No pages to remove from {pdf_path.name}")
        return
    
    print(f"\nProcessing {pdf_path.name}...")
    
    # Open the PDF
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    # Convert page numbers to actual page indices
    actual_pages_to_remove = []
    for page in pages_to_remove:
        if page <= 3:  # First 3 pages
            actual_pages_to_remove.append(page - 1)  # Convert to 0-based index
        else:  # Last 3 pages (4,5,6)
            # Convert 4,5,6 to actual last 3 pages
            actual_pages_to_remove.append(total_pages - (6 - page) - 1)
    
    print(f"Removing pages: {actual_pages_to_remove}")
    
    # Create a new PDF with the pages we want to keep
    new_doc = fitz.open()
    
    # Add all pages except the ones to remove
    for i in range(len(doc)):
        if i not in actual_pages_to_remove:
            new_doc.insert_pdf(doc, from_page=i, to_page=i)
    
    # Save the new PDF
    output_path = Path("cleaned_pdfs") / pdf_path.name
    output_path.parent.mkdir(exist_ok=True)
    new_doc.save(output_path)
    
    print(f"Saved cleaned PDF to {output_path}")
    
    # Close the documents
    doc.close()
    new_doc.close()

def main():
    # Create output directory
    output_dir = Path("cleaned_pdfs")
    output_dir.mkdir(exist_ok=True)
    
    # Process each PDF
    for pdf_name, pages in pages_to_remove.items():
        pdf_path = Path(pdf_name)
        if pdf_path.exists():
            remove_pages(pdf_path, pages)
        else:
            print(f"Warning: {pdf_name} not found")

if __name__ == "__main__":
    main() 