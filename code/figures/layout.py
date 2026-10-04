"""Record and verify the explicitly shared plot edges of the approved layout."""
import json
from pathlib import Path

def require_matplotlib_panel_alignment(fig,*,json_out,overlay_svg=None,axes,panel_ids,
 row_groups=(),column_groups=(),exemptions=(),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True):
    fig.canvas.draw()
    width,height=fig.get_size_inches()*72
    boxes={label:[ax.get_position().x0*width,ax.get_position().y0*height,
                  ax.get_position().x1*width,ax.get_position().y1*height]
           for label,ax in zip(panel_ids,axes)}
    checks=[]
    for groups,indices,kind in [(row_groups,[1,3],'row'),(column_groups,[0,2],'column')]:
        for labels in groups:
            for edge in indices:
                values=[boxes[label][edge] for label in labels]
                checks.append({'kind':kind,'panels':labels,'edge':edge,'spread_pt':max(values)-min(values)})
    failures=[c for c in checks if c['spread_pt']>tolerance_pt]
    report={'verdict':'FAIL' if failures else 'PASS','plot_rectangles_pt':boxes,
            'checks':checks,'tolerance_pt':tolerance_pt,'recorded_design_exemptions':list(exemptions),
            'scope':'Declared shared plot edges; intentional unequal panel widths follow the approved layout.'}
    Path(json_out).write_text(json.dumps(report,indent=2),encoding='utf-8')
    if failures and strict:raise ValueError(f'Plot edge alignment failed: {failures}')
    return report
