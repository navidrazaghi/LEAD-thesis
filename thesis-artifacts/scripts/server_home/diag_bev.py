"""Which BEV cells get a calibrated image reference, and where do they sit?"""
import numpy as np, torch
from lead.config import load_lead_config
from lead.policy.transfuser.encoder import fusion_geometry as fg
from lead.policy.transfuser.encoder.deformable_attention import default_reference_points

c = load_lead_config(use_cli=False)
t = c.policy.transfuser
print('BEV extent x:', t.bev_min_x_meter, t.bev_max_x_meter, '| y:', t.bev_min_y_meter, t.bev_max_y_meter)
print('grid rows(y) x cols(x):', t.lidar_bev_grid_rows, t.lidar_bev_grid_cols, '| ref height', t.deformable_reference_height_meter)
print('cameras:', t.input_cameras)
for s in fg.stitched_camera_specs(c):
    print('  spec pos', s['pos'], 'rot', s['rot'], 'fov', s['fov'], s['width'], 'x', s['height'])
centres = fg.bev_cell_centres(c)
pix, ok = fg.bev_cells_in_image(c, t.deformable_reference_height_meter)
print('covered cells:', int(ok.sum()), 'of', len(ok))
print('x range of centres:', centres[:,0].min(), centres[:,0].max(), '| y range:', centres[:,1].min(), centres[:,1].max())
for label, m in (('covered', ok), ('uncovered', ~ok)):
    if m.any():
        print(f'{label}: x [{centres[m,0].min():.1f},{centres[m,0].max():.1f}] y [{centres[m,1].min():.1f},{centres[m,1].max():.1f}]')
# nearest cell to 10 m straight ahead
i = int(np.argmin(np.abs(centres[:,0]-10) + np.abs(centres[:,1])*2))
print('nearest-to-(10,0) cell:', centres[i], 'covered:', bool(ok[i]), 'pixel-norm:', pix[i])
# what does the projection say for that exact point?
pt = np.array([[centres[i,0], centres[i,1], t.deformable_reference_height_meter]])
for j, s in enumerate(fg.stitched_camera_specs(c)):
    px, inside = fg.project_to_camera(s, pt)
    print(f'  cam{j}: pixel {px[0]} inside={bool(inside[0])} (image {s["width"]}x{s["height"]})')
