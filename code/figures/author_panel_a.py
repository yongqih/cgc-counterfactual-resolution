"""Embed the preserved author SVG without replacing the schematic."""
from lxml import etree as ET
import pymupdf
from PIL import Image

def insert_author_panel(name, rect, width_pt, height_pt):
    svg=HERE/'assets/Figure_2a_author.svg'
    opened=pymupdf.open(svg);author=pymupdf.open(stream=opened.convert_to_pdf(),filetype='pdf');opened.close()
    x,y,w,h=rect;box=pymupdf.Rect(x*width_pt,(1-y-h)*height_pt,(x+w)*width_pt,(1-y)*height_pt)
    pdf_path=FIG/(name+'.pdf');doc=pymupdf.open(pdf_path)
    doc[0].show_pdf_page(box,author,0,keep_proportion=True)
    temporary=pdf_path.with_name(name+'.composed.pdf');doc.save(temporary,garbage=4,deflate=True);doc.close();author.close();temporary.replace(pdf_path)
    target=ET.parse(str(FIG/(name+'.svg')));fragment=ET.parse(str(svg)).getroot()
    fragment.set('x',str(box.x0));fragment.set('y',str(box.y0));fragment.set('width',str(box.width));fragment.set('height',str(box.height))
    target.getroot().append(fragment);target.write(str(FIG/(name+'.svg')),encoding='utf-8',xml_declaration=True)
    # Render all raster previews from the final Python-composed vector PDF.
    doc=pymupdf.open(pdf_path);pix=doc[0].get_pixmap(matrix=pymupdf.Matrix(600/72,600/72),alpha=False)
    im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
    im.save(FIG/(name+'.png'),dpi=(600,600));im.save(FIG/(name+'.tiff'),dpi=(600,600),compression='tiff_lzw');doc.close()
