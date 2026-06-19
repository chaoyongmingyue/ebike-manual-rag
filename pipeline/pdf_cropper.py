import pypdfium2 as pdfium
from PIL import Image
from pathlib import Path
import io

class PDFCropper:
    """Crop regions from PDF pages using pypdfium2."""

    def __init__(self, pdf_path: str, dpi: int = 150):
        self.pdf_path = pdf_path
        self.dpi = dpi
        self.pdf = pdfium.PdfDocument(pdf_path)

    def get_page_size_pts(self, page_idx: int):
        """Get page dimensions in points (1/72 inch)."""
        page = self.pdf[page_idx]
        w, h = page.get_size()
        return w, h

    def crop_region(self, page_idx: int, bbox, output_path: str = None) -> Image.Image:
        """Crop a region from a PDF page.
        
        bbox: [x0, y0, x1, y1] in PDF points (72 DPI coordinate space).
        """
        pt_w, pt_h = self.get_page_size_pts(page_idx)
        scale = self.dpi / 72.0

        # Convert bbox (pts) to pixel coords at target DPI
        px_x0 = int(bbox[0] * scale)
        px_y0 = int((pt_h - bbox[3]) * scale)  # PDF y=0 is bottom, image y=0 is top
        px_x1 = int(bbox[2] * scale)
        px_y1 = int((pt_h - bbox[1]) * scale)

        pw = px_x1 - px_x0
        ph = px_y1 - px_y0
        if pw <= 0 or ph <= 0:
            pw = max(pw, 10)
            ph = max(ph, 10)

        # Render the page
        bitmap = self.pdf[page_idx].render(
            scale=scale,
            rotation=0,
            crop=(px_x0, px_y0, px_x0 + pw, px_y0 + ph),
        )
        pil_img = bitmap.to_pil()

        if output_path:
            pil_img.save(output_path, quality=90)
        return pil_img

    def close(self):
        self.pdf.close()
