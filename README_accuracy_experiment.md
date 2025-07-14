v1.0
- add "triangulate3DHPE_data.py", "triangulate3DHPE_show_result.py", "triangulate3DHPEcompute_mpjpe.py", "triangulate3DHPE_partial_accuracy"
    - triangulate3DHPE_data aggregate predicted landmarks
    - triangulate3DHPE_show_result show 3Dplots each frame
    - triangulate3DHPEcompute_mpjpe compute mpjpe between GT landmark and predicted landmark
    - triangulate3DHPE_partial_accuracy evaluate the accuracy of landmarks not present in the GT data (incomplete)

v0.2
- make concatenated images video using only 2 video

v0.1
- Be able to save annotate image and non-annotate image
- add "concatenation_images" function    
    - save and concatenate output images
- collect related files into "related_files" folder added

v0.0
- capture HD video
- dataset's internal params and external params input
- adjusted 3d-plot view init
- save 3d-plot