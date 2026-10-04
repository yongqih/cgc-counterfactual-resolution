from lxml import etree as ET
import pymupdf
import re
from PIL import Image

def insert_figure3_artwork(placements,width_pt,height_pt):
    name='Figure_3';pdf_path=FIG/(name+'.pdf');doc=pymupdf.open(pdf_path)
    target=ET.parse(str(FIG/(name+'.svg')))
    for asset,rect in placements:
        svg=HERE/'assets'/asset
        opened=pymupdf.open(svg);author=pymupdf.open(stream=opened.convert_to_pdf(),filetype='pdf');opened.close()
        if asset=='Figure_3a_author.svg':
            # MuPDF batches adjacent author text lines into one PDF text trace.
            # Separate text objects retain identical appearance and make each
            # actual line independently selectable and collision-auditable.
            for xref in author[0].get_contents():
                stream=author.xref_stream(xref).decode('latin1')
                stream,count=re.subn(r'([-\d.]+) -6 TD',lambda m:'ET BT 1 0 0 -1 '+m[1]+' 6 Tm',stream)
                assert count==4
                author.update_stream(xref,stream.encode('latin1'))
        x,y,w,h=rect;box=pymupdf.Rect(x*width_pt,(1-y-h)*height_pt,(x+w)*width_pt,(1-y)*height_pt)
        doc[0].show_pdf_page(box,author,0,keep_proportion=True);author.close()
        fragment=ET.parse(str(svg)).getroot()
        for k,v in [('x',box.x0),('y',box.y0),('width',box.width),('height',box.height)]:fragment.set(k,str(v))
        target.getroot().append(fragment)
    temporary=pdf_path.with_name(name+'.composed.pdf');doc.save(temporary,garbage=4,deflate=True);doc.close();temporary.replace(pdf_path)
    target.write(str(FIG/(name+'.svg')),encoding='utf-8',xml_declaration=True)
    doc=pymupdf.open(pdf_path);pix=doc[0].get_pixmap(matrix=pymupdf.Matrix(600/72,600/72),alpha=False)
    im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
    im.save(FIG/(name+'.png'),dpi=(600,600));im.save(FIG/(name+'.tiff'),dpi=(600,600),compression='tiff_lzw');doc.close()
